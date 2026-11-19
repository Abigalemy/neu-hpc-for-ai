"""OpenAI-compatible HTTP server around the CUDA Tile Qwen3 engine (no PyTorch).

  python server.py --model-dir /path/to/Qwen3-4B --lib ../build/libqwen3engine.so --port 8000 [--yarn 4] [--max-seqs 16]
  python server.py --fake --port 8000          # fake engine + fake tokenizer, for testing without a GPU

Endpoints: POST /v1/chat/completions, POST /v1/completions (both with "stream": true -> SSE), GET /v1/models, GET /health.
Request fields understood: max_tokens / max_completion_tokens, min_tokens, ignore_eos, stream, stream_options.include_usage,
temperature (only 0 / greedy is supported: anything else is rejected with 400), chat_template_kwargs.

Continuous batching: a scheduler thread owns the engine. Each iteration it admits waiting requests (prefill, which
yields their first token), then runs ONE decode step for all running sequences together; finished or disconnected
sequences free their slot and their KV blocks at once, so waiting requests join at the next iteration. A prefill pauses
the decode of the others while it runs (no chunked mixing of prefill and decode yet).

PagedAttention (Week 7 lecture; nano-vllm's Scheduler + BlockManager): the KV cache is a pool of fixed-size blocks
shared by all sequences, and a sequence takes blocks only as its length grows, so memory is not reserved for max_len
up front. Admission (first come, first served): the request at the head of the queue starts when a slot is free and
the pool has blocks for its prompt plus the first generated token. Before each decode step every running sequence
needs room for one more token; if a new block is needed and none is free, the most recently admitted sequence is
preempted: its blocks are freed and it goes back to the front of the queue, to be recomputed later (prefill of its
prompt plus the tokens it already produced), as in vLLM's recompute preemption. Its stream just pauses meanwhile.
"""
import argparse
import asyncio
import collections
import ctypes
import json
import os
import threading
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------- backends
class CBackend:
    """The real engine through the C interface of engine/qwen3_capi.cu."""

    def __init__(self, lib_path: str, model_dir: str, max_len: int, max_chunk: int, yarn: float, max_seqs: int,
                 kv_tokens: int = 0, rank: int = 0, world: int = 1, nccl_id: bytes = b""):
        lib = ctypes.CDLL(lib_path)
        lib.qe_create.restype = ctypes.c_void_p
        lib.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                  ctypes.c_longlong]
        lib.qe_create_ep.restype = ctypes.c_void_p
        lib.qe_create_ep.argtypes = lib.qe_create.argtypes + [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        lib.qe_idle_tick.restype = ctypes.c_int
        lib.qe_idle_tick.argtypes = [ctypes.c_void_p]
        for name in ("qe_vocab", "qe_max_len", "qe_max_chunk", "qe_max_seqs", "qe_block_size", "qe_num_blocks",
                     "qe_free_blocks"):
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        IntP = ctypes.POINTER(ctypes.c_int)
        lib.qe_prefill.restype = ctypes.c_int
        lib.qe_prefill.argtypes = [ctypes.c_void_p, ctypes.c_int, IntP, ctypes.c_int]
        lib.qe_decode_batch.restype = None
        lib.qe_decode_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, IntP, IntP, IntP, IntP]
        lib.qe_reserve.restype = ctypes.c_int
        lib.qe_reserve.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        lib.qe_release.restype = None
        lib.qe_release.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib = lib
        self.world = world
        if world > 1:        # expert parallelism: this process is rank `rank` of `world` (Part 3)
            self.h = lib.qe_create_ep(model_dir.encode(), max_len, max_chunk, yarn, max_seqs, kv_tokens, rank, world, nccl_id)
        else:
            self.h = lib.qe_create(model_dir.encode(), max_len, max_chunk, yarn, max_seqs, kv_tokens)
        if not self.h:
            raise RuntimeError("qe_create failed")
        self.vocab = lib.qe_vocab(self.h)
        self.max_len = lib.qe_max_len(self.h)
        self.max_seqs = lib.qe_max_seqs(self.h)
        self.block_size = lib.qe_block_size(self.h)
        self.num_blocks = lib.qe_num_blocks(self.h)

    def free_blocks(self) -> int:
        return self.lib.qe_free_blocks(self.h)

    def idle_tick(self) -> int:
        """No local work: take part in the other ranks' forward pass if there is one (1), else 0 (see ep_link.py)."""
        return self.lib.qe_idle_tick(self.h)

    def reserve(self, slot: int, n_tokens: int) -> bool:
        return bool(self.lib.qe_reserve(self.h, slot, n_tokens))

    def release(self, slot: int):
        self.lib.qe_release(self.h, slot)

    def prefill(self, slot: int, ids: List[int]) -> int:
        arr = (ctypes.c_int * len(ids))(*ids)
        return self.lib.qe_prefill(self.h, slot, arr, len(ids))

    def decode_batch(self, slots: List[int], tokens: List[int], positions: List[int]) -> List[int]:
        B = len(slots)
        A = ctypes.c_int * B
        out = A()
        self.lib.qe_decode_batch(self.h, B, A(*slots), A(*tokens), A(*positions), out)
        return list(out)


class FakeBackend:
    """Deterministic stand-in for tests: the next token is a function of the WHOLE sequence so far (a rolling hash of
    every token at every position), so a recompute that lost or reordered any earlier token gives other tokens; every
    decode must feed the position right after the slot's history. The `eos_after`-th generated token of a sequence
    (if set) is `eos`, once, so both stopping at EOS and ignoring it can be tested."""

    def __init__(self, vocab: int = 1000, max_len: int = 4096, eos: int = 2, eos_after: Optional[int] = None,
                 max_seqs: int = 4, delay: float = 0.0, block_size: int = 128, num_blocks: Optional[int] = None,
                 link=None, world: int = 1):
        self.vocab, self.max_len, self.eos, self.eos_after, self.max_seqs = vocab, max_len, eos, eos_after, max_seqs
        self.delay = delay            # seconds per engine call, to let concurrent requests overlap in tests
        self.calls = []
        self.prefill_ids = []         # (slot, ids) of every prefill, for the recompute test
        self.n_gen = {}
        self.hist = {}                # slot -> tokens in its "KV cache" (prompt, then every fed token)
        # block accounting like the engine's (default: every slot can reach max_len)
        self.block_size = block_size
        self.num_blocks = num_blocks if num_blocks is not None else max_seqs * -(-max_len // block_size)
        self.blocks = {}              # slot -> number of blocks it holds
        self.min_free = self.num_blocks
        # several ranks (Part 3): every engine call is a tick, answered by the coordinator like the engine's all-reduce
        self.link, self.world, self.passes = link, world, 0

    def _tick(self, have: int) -> int:
        if self.world == 1:
            return have
        m = self.link.fake_tick(have)
        if m:
            self.passes += 1
        return m

    def idle_tick(self) -> int:
        return self._tick(0)

    def free_blocks(self) -> int:
        return self.num_blocks - sum(self.blocks.values())

    def reserve(self, slot: int, n_tokens: int) -> bool:
        need = -(-n_tokens // self.block_size)
        have = self.blocks.get(slot, 0)
        if need <= have:
            return True
        if need > -(-self.max_len // self.block_size) or need - have > self.free_blocks():
            return False
        self.blocks[slot] = need
        self.min_free = min(self.min_free, self.free_blocks())
        return True

    def release(self, slot: int):
        self.blocks.pop(slot, None)

    def _check_reserved(self, slot: int, n_tokens: int):   # the engine exits if a written position has no block
        assert self.blocks.get(slot, 0) * self.block_size >= n_tokens, (slot, n_tokens, self.blocks.get(slot, 0))

    def _next(self, slot: int) -> int:
        self.n_gen[slot] = self.n_gen.get(slot, 0) + 1
        if self.eos_after is not None and self.n_gen[slot] == self.eos_after:
            return self.eos
        h = 0
        for t in self.hist[slot]:
            h = (h * 1000003 + t + 1) % 2147483647
        return 10 + h % (self.vocab - 10)

    def prefill(self, slot: int, ids: List[int]) -> int:
        self._tick(1)
        time.sleep(self.delay)
        self._check_reserved(slot, len(ids))
        self.calls.append(("prefill", slot, len(ids)))
        self.prefill_ids.append((slot, list(ids)))
        self.n_gen[slot] = 0
        self.hist[slot] = list(ids)
        return self._next(slot)

    def decode_batch(self, slots: List[int], tokens: List[int], positions: List[int]) -> List[int]:
        self._tick(1)
        time.sleep(self.delay)
        assert len(set(slots)) == len(slots), "a slot appears twice in one batch"
        for s_, p in zip(slots, positions):
            self._check_reserved(s_, p + 1)
        self.calls.append(("decode", tuple(slots), tuple(positions)))
        out = []
        for s_, t, p in zip(slots, tokens, positions):
            assert p == len(self.hist[s_]), f"slot {s_}: token fed at position {p}, history has {len(self.hist[s_])}"
            self.hist[s_].append(t)
            out.append(self._next(s_))
        return out


class FakeTokenizer:
    """Whitespace 'tokenizer' with a fixed vocabulary of word<i> strings, plus a tiny chat template."""
    eos_token_id = 2

    def encode(self, text: str) -> List[int]:
        return [10 + (sum(map(ord, w)) % 990) for w in text.split()] or [10]

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return "".join(f" w{i}" for i in ids if not (skip_special_tokens and i == self.eos_token_id))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        text = "".join(f"<|{m['role']}|> {m['content']} " for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")


# ---------------------------------------------------------------- generation
class GenRequest:
    def __init__(self, ids: List[int], max_new: int, min_new: int, ignore_eos: bool, loop):
        self.ids, self.max_new, self.min_new, self.ignore_eos = ids, max_new, min_new, ignore_eos
        self.loop, self.queue = loop, asyncio.Queue()
        self.slot = self.pos = self.last_tok = None   # pos: tokens in the KV cache (None: not prefilled yet)
        self.n_gen = 0               # tokens generated so far (an ignored EOS counts, a final EOS does not)
        self.gen: List[int] = []     # those tokens: a preempted request is recomputed from ids + gen
        self.cancelled = False       # set by the HTTP side when the client goes away

    def emit(self, tok: int, finish: Optional[str]):
        """Hand one event to the request's event loop. If that loop is already closed (RuntimeError: the client and its
        loop are gone), the request is marked cancelled instead of raising, so the scheduler's cleanup of OTHER requests
        goes on; other exceptions are not caught here."""
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, (tok, finish))
        except RuntimeError:
            self.cancelled = True


class Scheduler:
    """Owns the backend in one thread: admit -> prefill -> one batched decode step, repeated (see module docstring)."""

    def __init__(self, backend, eos_ids: List[int], link=None):
        self.backend, self.eos = backend, set(eos_ids)
        # several ranks (Part 3, ep_link.py): with no local work this thread keeps ticking (idle_tick) while any rank
        # has work, and parks when none has; a new request wakes the group through the coordinator
        self.world, self.link = getattr(backend, "world", 1), link
        self.idle_passes = 0
        self.free = list(range(backend.max_seqs))
        self.running: List[GenRequest] = []   # in admission order (the last one is preempted first)
        self.waiting = collections.deque()
        self.cv = threading.Condition()
        self.batch_sizes: List[int] = []      # decode batch size of every step (statistics, tests)
        self.preemptions = 0
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, req: GenRequest):
        with self.cv:
            self.waiting.append(req)
            self.cv.notify()
        if self.link is not None:
            self.link.want()

    def _accept(self, req: GenRequest, tok: int):
        """Deliver one generated token. Order when the request ends: update the request, release its slot and blocks,
        THEN emit the final event, so a client that has seen the end of its stream never finds them still held. The
        caller must not release the request again (it is out of self.running once finished)."""
        if tok in self.eos and not req.ignore_eos and req.n_gen >= req.min_new:
            self._release(req)               # a final EOS is not counted as generated
            req.emit(tok, "stop")
            return
        req.n_gen += 1
        req.gen.append(tok)
        req.last_tok = tok
        if req.n_gen == req.max_new:
            self._release(req)
            req.emit(tok, "length")
        else:
            req.emit(tok, None)

    def _release(self, req: GenRequest):
        self.running.remove(req)
        self.backend.release(req.slot)
        self.free.append(req.slot)

    def _preempt(self, req: GenRequest):
        """Recompute preemption: free the request's blocks and slot; it waits at the front of the queue."""
        self._release(req)
        req.slot = req.pos = None
        self.waiting.appendleft(req)
        self.preemptions += 1
        print(f"preempted a sequence of {len(req.ids) + len(req.gen)} tokens ({self.preemptions} so far, "
              f"{len(self.running)} running)", flush=True)

    def _loop(self):
        while True:
            try:
                self._iteration()
            except Exception:                       # never let the engine thread die silently: fail the requests
                import traceback
                traceback.print_exc()
                with self.cv:
                    for r in list(self.running):
                        self._release(r)        # release first, then tell the client (same order as _accept)
                        r.emit(-1, "error")

    def _iteration(self):
        with self.cv:
            while not self.waiting and not self.running and self.world == 1:
                self.cv.wait()
            idle = not self.waiting and not self.running       # only with several ranks
            admitted = []
            while not idle and self.waiting and self.free:    # first come, first served: stop at the first that must wait
                r = self.waiting[0]
                if r.cancelled:
                    self.waiting.popleft()
                    continue
                n = len(r.ids) + len(r.gen)      # tokens to prefill (prompt + already generated, if recomputed)
                if not self.backend.reserve(self.free[0], n + 1):   # its prompt and the first decode token
                    break
                self.waiting.popleft()
                r.slot = self.free.pop(0)
                self.running.append(r)       # running from now on, so an exception below still frees its slot
                admitted.append(r)
        if idle:                             # no local work: serve the other ranks' pass, or park if nobody has work
            if self.backend.idle_tick():
                self.idle_passes += 1
            else:
                self.link.park()
            return
        for r in admitted:                   # prefill: this request's first token = its TTFT
            ids = r.ids + r.gen
            tok = self.backend.prefill(r.slot, ids)
            with self.cv:
                r.pos = len(ids)
                self._accept(r, tok)            # releases r itself when it finishes
        with self.cv:
            for r in [r for r in self.running if r.cancelled]:
                self._release(r)
            # every running sequence writes one more position: take a block where one is needed, oldest first;
            # when the pool is empty, preempt the newest sequence (possibly the one asking) until it fits
            for r in list(self.running):
                while r in self.running and not self.backend.reserve(r.slot, r.pos + 1):
                    self._preempt(self.running[-1])
            batch = [r for r in self.running if r.pos is not None]
        if not batch:
            return
        outs = self.backend.decode_batch([r.slot for r in batch], [r.last_tok for r in batch], [r.pos for r in batch])
        self.batch_sizes.append(len(batch))
        with self.cv:
            for r, tok in zip(batch, outs):
                r.pos += 1
                self._accept(r, tok)            # releases r itself when it finishes

    async def tokens(self, prompt_ids: List[int], max_new: int, min_new: int, ignore_eos: bool):
        """Async generator of (token_id, finish_reason or None) for one request."""
        req = GenRequest(prompt_ids, max_new, min_new, ignore_eos, asyncio.get_running_loop())
        self.submit(req)
        try:
            while True:
                tok, fin = await req.queue.get()
                yield tok, fin
                if fin:
                    return
        finally:
            req.cancelled = True     # a finished request is already released; an unfinished one is dropped


class Detokenizer:
    """Incremental detokenisation with a sliding window (the prefix/read offsets vLLM uses): each step decodes only the
    last few tokens, so the cost per token stays constant instead of growing with the output length. An incomplete
    UTF-8 character (decoded as U+FFFD) is held back until the next token completes it."""

    def __init__(self, tok):
        self.tok, self.ids = tok, []
        self.prefix_offset = self.read_offset = 0

    def push(self, tid: int, is_eos: bool) -> str:
        if is_eos:
            return ""
        self.ids.append(tid)
        prefix = self.tok.decode(self.ids[self.prefix_offset:self.read_offset], skip_special_tokens=True)
        text = self.tok.decode(self.ids[self.prefix_offset:], skip_special_tokens=True)
        if len(text) <= len(prefix) or text.endswith("�"):
            return ""
        delta = text[len(prefix):]
        self.prefix_offset, self.read_offset = self.read_offset, len(self.ids)
        return delta


# ---------------------------------------------------------------- app
def create_app(backend, tokenizer, model_name: str, eos_ids: List[int], link=None) -> FastAPI:
    app = FastAPI()
    sched = Scheduler(backend, eos_ids, link)
    app.state.scheduler = sched

    def _check(body: dict, n_prompt: int, max_new: int):
        if float(body.get("temperature") or 0.0) != 0.0:
            raise HTTPException(400, "only greedy decoding (temperature 0) is supported")
        if max_new < 1:
            raise HTTPException(400, "max_tokens must be >= 1")
        if n_prompt < 1:
            raise HTTPException(400, "empty prompt")
        if n_prompt + max_new > backend.max_len:
            raise HTTPException(400, f"prompt ({n_prompt}) + max_tokens ({max_new}) exceeds the context ({backend.max_len})")
        if -(-(n_prompt + max_new) // backend.block_size) > backend.num_blocks:   # could never run, even alone
            raise HTTPException(400, f"prompt ({n_prompt}) + max_tokens ({max_new}) exceeds the KV cache "
                                     f"({backend.num_blocks * backend.block_size} tokens)")

    async def _respond(body: dict, prompt_ids: List[int], chat: bool):
        # explicit values win even when they are 0 (then rejected below); only a missing field takes the default
        if body.get("max_completion_tokens") is not None:
            max_new = int(body["max_completion_tokens"])
        elif body.get("max_tokens") is not None:
            max_new = int(body["max_tokens"])
        else:
            max_new = 256 if chat else 16
        min_new = int(body.get("min_tokens") or 0)
        ignore_eos = bool(body.get("ignore_eos", False))
        _check(body, len(prompt_ids), max_new)
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())
        obj = "chat.completion" if chat else "text_completion"
        n_prompt = len(prompt_ids)

        if not body.get("stream", False):
            detok, n_out, finish = Detokenizer(tokenizer), 0, None
            async for tid, fin in sched.tokens(prompt_ids, max_new, min_new, ignore_eos):
                if fin == "error":
                    raise HTTPException(500, "engine error (see the server log)")
                is_eos = fin == "stop"
                detok.push(tid, is_eos)
                n_out += 0 if is_eos else 1
                finish = fin
            text = tokenizer.decode(detok.ids, skip_special_tokens=True)
            choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish} if chat
                      else {"index": 0, "text": text, "finish_reason": finish})
            return JSONResponse({"id": rid, "object": obj, "created": created, "model": model_name, "choices": [choice],
                                 "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                                           "total_tokens": n_prompt + n_out}})

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        chunk_obj = "chat.completion.chunk" if chat else "text_completion"

        def chunk(text: Optional[str], finish: Optional[str], first: bool = False) -> str:
            if chat:
                delta = {"role": "assistant", "content": text or ""} if first else ({"content": text} if text else {})
                ch = {"index": 0, "delta": delta, "finish_reason": finish}
            else:
                ch = {"index": 0, "text": text or "", "finish_reason": finish}
            return "data: " + json.dumps({"id": rid, "object": chunk_obj, "created": created, "model": model_name,
                                          "choices": [ch]}) + "\n\n"

        async def stream():
            detok, n_out = Detokenizer(tokenizer), 0
            if chat:
                yield chunk("", None, first=True)
            async for tid, fin in sched.tokens(prompt_ids, max_new, min_new, ignore_eos):
                if fin == "error":
                    yield chunk("", "error")
                    break
                is_eos = fin == "stop"
                text = detok.push(tid, is_eos)
                n_out += 0 if is_eos else 1
                if text or fin:
                    yield chunk(text, fin)
            if include_usage:
                yield "data: " + json.dumps({"id": rid, "object": chunk_obj, "created": created, "model": model_name,
                                             "choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                                                                      "total_tokens": n_prompt + n_out}}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise HTTPException(400, "messages must be a non-empty list")
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             **(body.get("chat_template_kwargs") or {}))
        return await _respond(body, tokenizer.encode(text), chat=True)

    @app.post("/v1/completions")
    async def completions(request: Request):
        body = await request.json()
        prompt = body.get("prompt")
        if isinstance(prompt, list):     # list of token ids or a single-element list of strings
            if prompt and isinstance(prompt[0], int):
                # the engine reads the embedding row of every id without a check: reject bad ids here (400)
                if not all(type(t) is int and 0 <= t < backend.vocab for t in prompt):
                    raise HTTPException(400, f"prompt token ids must be integers in [0, {backend.vocab})")
                ids = prompt
            else:
                ids = tokenizer.encode(prompt[0] if prompt else "")
        else:
            ids = tokenizer.encode(prompt or "")
        return await _respond(body, ids, chat=False)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():   # for the launcher and tests: what this rank did
        return {"world": sched.world, "batch_steps": len(sched.batch_sizes), "idle_passes": sched.idle_passes,
                "parks": sched.link.parks if sched.link else 0, "running": len(sched.running),
                "waiting": len(sched.waiting)}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir")
    ap.add_argument("--served-model-name", default="Qwen/Qwen3-4B")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(__file__), "..", "build", "libqwen3engine.so"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-len", type=int, default=4096 + 1024 + 64)
    ap.add_argument("--max-chunk", type=int, default=4096)
    ap.add_argument("--yarn", type=float, default=0.0)
    ap.add_argument("--max-seqs", type=int, default=1, help="slots = most sequences decoded together")
    ap.add_argument("--kv-tokens", type=int, default=0,
                    help="KV-cache pool size in tokens, shared by all sequences (0: max-seqs x max-len)")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--fake-delay", type=float, default=0.0, help="seconds per fake engine call")
    ap.add_argument("--rank", type=int, default=0, help="this process's rank (expert parallelism, part3/launch.py)")
    ap.add_argument("--world", type=int, default=1, help="number of ranks = GPUs; > 1 needs --coord")
    ap.add_argument("--coord", default="", help="host:port of the launcher's coordinator (key in QWEN3_COORD_KEY)")
    a = ap.parse_args()
    link = None
    if a.world > 1:
        from ep_link import EpLink
        host, port = a.coord.rsplit(":", 1)
        link = EpLink((host, int(port)), bytes.fromhex(os.environ["QWEN3_COORD_KEY"]), a.rank)
    if a.fake:
        backend = FakeBackend(max_len=a.max_len, max_seqs=a.max_seqs, delay=a.fake_delay,
                              num_blocks=-(-a.kv_tokens // 128) if a.kv_tokens > 0 else None, link=link, world=a.world)
        tok, eos = FakeTokenizer(), [FakeTokenizer.eos_token_id]
    else:
        from transformers import AutoTokenizer   # tokenizer only; PyTorch is not needed for it
        tok = AutoTokenizer.from_pretrained(a.model_dir)
        eos = [tok.eos_token_id]
        gc = os.path.join(a.model_dir, "generation_config.json")
        if os.path.exists(gc):
            e = json.load(open(gc)).get("eos_token_id")
            eos += e if isinstance(e, list) else ([e] if e is not None else [])
        backend = CBackend(a.lib, a.model_dir, a.max_len, a.max_chunk, a.yarn, a.max_seqs, a.kv_tokens, a.rank, a.world,
                           bytes.fromhex(os.environ.get("QWEN3_NCCL_ID", "")))
        print(f"engine ready: vocab {backend.vocab}, max_len {backend.max_len}, max_seqs {backend.max_seqs}, "
              f"KV pool {backend.num_blocks} blocks x {backend.block_size} tokens, YaRN {a.yarn or 'off'}, "
              f"eos {sorted(set(eos))}", flush=True)
    import uvicorn
    uvicorn.run(create_app(backend, tok, a.served_model_name, eos, link), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
