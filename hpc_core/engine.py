"""LLM engine: scheduler + continuous batching on top of the Qwen3 model.

Engine step (one iteration of the loop):
  1. admit waiting requests while capacity allows (continuous batching), or only
     when the batch is empty (`batching="static"`, used as a baseline);
  2. prefill the newly admitted sequences in one varlen forward pass;
  3. decode one token for *every* running sequence in one batched forward pass;
  4. finished sequences free their cache immediately, so a waiting request can
     take the slot at the very next step (no waiting for the whole batch).

Cache modes: "none" (recompute everything each step), "contiguous", "paged".
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence as Seq

import torch

from .kv_cache import (ContiguousDecodeBatch, ContiguousKVCache, ContiguousPrefillBatch, NoCacheBatch,
                       PagedDecodeBatch, PagedKVCache, PagedPrefillBatch)


@dataclass
class SamplingParams:
    max_tokens: int = 16
    min_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = False


@dataclass
class EngineConfig:
    cache_mode: str = "paged"          # none | contiguous | paged
    batching: str = "continuous"       # continuous | static
    max_num_seqs: int = 64
    max_model_len: int = 4096
    block_size: int = 16
    num_blocks: Optional[int] = None   # paged: explicit block count
    kv_cache_gb: Optional[float] = None  # KV memory budget (paged & contiguous)
    gpu_memory_utilization: float = 0.85
    max_num_batched_tokens: int = 8192  # prefill token budget per step


class Sequence:
    _ids = itertools.count()

    def __init__(self, prompt_ids: List[int], params: SamplingParams, callback: Callable):
        self.seq_id = next(Sequence._ids)
        self.prompt_ids = list(prompt_ids)
        self.output_ids: List[int] = []
        self.params = params
        self.callback = callback  # callback(token_id | None, finish_reason | None)
        self.block_table: List[int] = []
        self.slot: Optional[int] = None
        self.finished = False
        self.num_preemptions = 0
        self.arrival = time.perf_counter()

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)

    def all_tokens(self) -> List[int]:
        return self.prompt_ids + self.output_ids


def sample_tokens(logits: torch.Tensor, seqs: Seq[Sequence]) -> List[int]:
    toks = logits.argmax(dim=-1)
    if any(s.params.temperature > 0 for s in seqs):
        toks = toks.clone()
        for i, s in enumerate(seqs):
            p = s.params
            if p.temperature <= 0:
                continue
            l = logits[i] / p.temperature
            if p.top_k > 0:
                kth = torch.topk(l, min(p.top_k, l.numel())).values[-1]
                l = l.masked_fill(l < kth, float("-inf"))
            probs = torch.softmax(l, dim=-1)
            if p.top_p < 1.0:
                sp, si = torch.sort(probs, descending=True)
                cum = sp.cumsum(0)
                sp = sp.masked_fill(cum - sp > p.top_p, 0.0)
                probs = torch.zeros_like(probs).scatter(0, si, sp)
                probs = probs / probs.sum()
            toks[i] = torch.multinomial(probs, 1)[0]
    return toks.tolist()


class LLMEngine:
    def __init__(self, model, model_cfg, ecfg: EngineConfig, eos_token_ids, device: str, dtype: torch.dtype):
        self.model, self.mcfg, self.ecfg = model, model_cfg, ecfg
        self.device, self.dtype = device, dtype
        self.eos = set(eos_token_ids)
        self.waiting: deque = deque()
        self.running: List[Sequence] = []
        self._inbox: "queue.Queue" = queue.Queue()
        self._aborts: "queue.Queue" = queue.Queue()
        self._wake = threading.Event()
        self._stop = False
        self.stats = dict(steps=0, prefill_tokens=0, decode_tokens=0, decode_steps=0, batch_size_sum=0,
                          preemptions=0, finished=0, rejected=0)
        self.cache = self._build_cache()

    # ------------------------------------------------------------------ memory
    def _kv_budget_bytes(self) -> int:
        e = self.ecfg
        if e.kv_cache_gb is not None:
            return int(e.kv_cache_gb * 1024 ** 3)
        if self.device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            used = total - free
            return max(0, int(total * e.gpu_memory_utilization) - used)
        return 512 * 1024 ** 2

    def _build_cache(self):
        e, c = self.ecfg, self.mcfg
        if e.cache_mode == "none":
            return None
        budget = self._kv_budget_bytes()
        if e.cache_mode == "paged":
            n = e.num_blocks or budget // PagedKVCache.bytes_per_block(c, e.block_size, self.dtype)
            n = int(n)
            assert n >= 2, "KV budget too small for even two blocks"
            return PagedKVCache(c, n, e.block_size, self.device, self.dtype)
        if e.cache_mode == "contiguous":
            n = budget // ContiguousKVCache.bytes_per_slot(c, e.max_model_len, self.dtype)
            n = int(min(n, e.max_num_seqs))
            assert n >= 1, "KV budget too small for one max_model_len sequence"
            return ContiguousKVCache(c, n, e.max_model_len, self.device, self.dtype)
        raise ValueError(e.cache_mode)

    @property
    def max_concurrency(self) -> int:
        """Max sequences that can run at once given the cache mode and budget."""
        if self.ecfg.cache_mode == "contiguous":
            return self.cache.num_slots
        return self.ecfg.max_num_seqs

    # ------------------------------------------------------------------ public API
    def add_request(self, prompt_ids, params: SamplingParams, callback) -> Sequence:
        """Thread-safe. `callback(token_id, finish_reason)` is called from the engine thread."""
        seq = Sequence(prompt_ids, params, callback)
        self._inbox.put(seq)
        self._wake.set()
        return seq

    def abort(self, seq: Sequence):
        self._aborts.put(seq)
        self._wake.set()

    def stop(self):
        self._stop = True
        self._wake.set()

    def run_forever(self):
        while not self._stop:
            if not self.has_work():
                self._wake.wait(timeout=0.05)
                self._wake.clear()
            self.step()

    def has_work(self) -> bool:
        return bool(self.waiting or self.running or not self._inbox.empty() or not self._aborts.empty())

    # ------------------------------------------------------------------ scheduling
    def _drain(self):
        while True:
            try:
                seq = self._inbox.get_nowait()
            except queue.Empty:
                break
            L = seq.num_tokens
            e = self.ecfg
            limit = e.max_model_len
            if e.cache_mode == "paged":  # a sequence must always be able to fit in the pool on its own
                limit = min(limit, (self.cache.num_blocks - self.cache.watermark) * e.block_size)
            if L >= limit:
                self.stats["rejected"] += 1
                seq.finished = True
                seq.callback(None, "error")
                continue
            seq.params.max_tokens = min(seq.params.max_tokens, limit - L)
            self.waiting.append(seq)
        while True:
            try:
                seq = self._aborts.get_nowait()
            except queue.Empty:
                break
            if seq.finished:
                continue
            seq.finished = True
            if seq in self.running:
                self.running.remove(seq)
            elif seq in self.waiting:
                self.waiting.remove(seq)
            self._release(seq)

    def _release(self, seq):
        if self.cache is not None:
            self.cache.release(seq)

    def _schedule_prefill(self) -> List[Sequence]:
        e = self.ecfg
        if e.batching == "static" and self.running:
            return []
        admitted, tokens = [], 0
        while self.waiting and len(self.running) + len(admitted) < self.max_concurrency:
            seq = self.waiting[0]
            L = seq.num_tokens
            if admitted and tokens + L > e.max_num_batched_tokens:
                break
            if self.cache is not None and not self.cache.can_admit(L):
                break
            self.waiting.popleft()
            if self.cache is not None:
                self.cache.admit(seq, L)
            admitted.append(seq)
            tokens += L
        self.running.extend(admitted)
        return admitted

    def _preempt_for_decode(self):
        """Make sure every running sequence has room for this step's token; preempt (recompute) if not."""
        if self.ecfg.cache_mode != "paged":
            return
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            while not self.cache.ensure_decode_slot(seq):
                victim = self.running.pop()  # most recently admitted
                self.cache.release(victim)
                victim.num_preemptions += 1
                self.stats["preemptions"] += 1
                self.waiting.appendleft(victim)
                if victim is seq:
                    break
            if i < len(self.running) and self.running[i] is seq:
                i += 1

    # ------------------------------------------------------------------ execution
    def _t(self, x):
        return torch.tensor(x, device=self.device)

    def _forward_prefill(self, seqs: List[Sequence]) -> torch.Tensor:
        toks = [s.all_tokens() for s in seqs]
        lens = [len(t) for t in toks]
        ids = self._t([x for t in toks for x in t])
        pos = self._t([p for L in lens for p in range(L)])
        mode = self.ecfg.cache_mode
        if mode == "paged":
            batch = PagedPrefillBatch(self.cache, seqs, lens, self.device)
        elif mode == "contiguous":
            batch = ContiguousPrefillBatch(self.cache, seqs, lens, self.device)
        else:
            cu = [0]
            for L in lens:
                cu.append(cu[-1] + L)
            batch = NoCacheBatch(cu, self.device)
        self.stats["prefill_tokens"] += sum(lens)
        return self.model(ids, pos, batch)

    def _forward_decode(self, seqs: List[Sequence]) -> torch.Tensor:
        ids = self._t([s.output_ids[-1] for s in seqs])
        pos = self._t([s.num_tokens - 1 for s in seqs])
        if self.ecfg.cache_mode == "paged":
            batch = PagedDecodeBatch(self.cache, seqs, self.device)
        else:
            batch = ContiguousDecodeBatch(self.cache, seqs, self.device)
        return self.model(ids, pos, batch)

    def _append(self, seq: Sequence, tok: int):
        seq.output_ids.append(tok)
        p, n = seq.params, len(seq.output_ids)
        reason = None
        if tok in self.eos and not p.ignore_eos and n >= p.min_tokens:
            reason = "stop"
        elif n >= p.max_tokens:
            reason = "length"
        seq.callback(tok, reason)
        if reason is not None:
            seq.finished = True
            self.stats["finished"] += 1

    def _sample_and_append(self, seqs, logits):
        for s, t in zip(seqs, sample_tokens(logits, seqs)):
            self._append(s, t)

    def _reap(self):
        done = [s for s in self.running if s.finished]
        if done:
            self.running = [s for s in self.running if not s.finished]
            for s in done:
                self._release(s)

    def step(self):
        self._drain()
        self.stats["steps"] += 1
        new = self._schedule_prefill()
        if self.ecfg.cache_mode == "none":
            # No cache: every step recomputes the full context of all running sequences.
            if self.running:
                seqs = list(self.running)
                logits = self._forward_prefill(seqs)
                self.stats["decode_steps"] += 1
                self.stats["decode_tokens"] += len(seqs)
                self.stats["batch_size_sum"] += len(seqs)
                self._sample_and_append(seqs, logits)
                self._reap()
            return
        if new:
            self._sample_and_append(new, self._forward_prefill(new))
            self._reap()
        self._preempt_for_decode()
        # Sequences that were just prefilled already produced one token and decode next step
        # input is their latest token, so they join this decode step too.
        seqs = [s for s in self.running if not s.finished]
        if seqs:
            logits = self._forward_decode(seqs)
            self.stats["decode_steps"] += 1
            self.stats["decode_tokens"] += len(seqs)
            self.stats["batch_size_sum"] += len(seqs)
            self._sample_and_append(seqs, logits)
            self._reap()

    def snapshot(self) -> dict:
        s = dict(self.stats)
        s["avg_decode_batch"] = s["batch_size_sum"] / max(1, s["decode_steps"])
        s["waiting"], s["running"] = len(self.waiting), len(self.running)
        s["cache_mode"], s["batching"] = self.ecfg.cache_mode, self.ecfg.batching
        if self.cache is not None:
            s["kv_usage"] = self.cache.usage
        if self.ecfg.cache_mode == "paged":
            s["num_blocks"] = self.cache.num_blocks
        if self.ecfg.cache_mode == "contiguous":
            s["num_slots"] = self.cache.num_slots
        return s
