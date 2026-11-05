"""KV-cache backends: none, contiguous (per-sequence slab), and paged (PagedAttention).

Each backend provides
  * a storage object (allocated once) with a resource manager (slots / blocks), and
  * `ForwardBatch` subclasses that the model calls via `batch.attend(layer, q, k, v)`.

All three produce the same attention result; they differ in *where K/V live* and
therefore in memory efficiency:

  none        : nothing is stored; every step re-runs the whole sequence.
  contiguous  : each running sequence reserves `max_model_len` tokens of K/V up
                front (internal fragmentation -> low concurrency).
  paged       : K/V live in fixed-size blocks allocated on demand and addressed
                through a per-sequence block table (PagedAttention).
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import torch
import torch.nn.functional as F

_HAS_GQA = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5)


def sdpa(q, k, v, is_causal=False, mask=None):
    """q: [B, nh, Lq, d]; k/v: [B, nkv, Lk, d]. Supports grouped-query attention."""
    nh, nkv = q.shape[1], k.shape[1]
    if nh != nkv:
        if _HAS_GQA:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal, enable_gqa=True)
        rep = nh // nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)


def _tile_attention(q, k, v, *, causal, q_start=0):
    """Lazy import keeps CPU-only use independent of the CUDA Tile package."""
    from .kernels.cutile_attention import flash_attention
    return flash_attention(q, k, v, causal=causal, q_start=q_start)


def varlen_causal_attention(q, k, v, cu_seqlens: List[int]):
    """Causal attention for concatenated sequences. q: [T, nh, d]; k/v: [T, nkv, d]."""
    out = torch.empty_like(q)
    for s, e in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        q_i = q[s:e].transpose(0, 1)[None]
        k_i = k[s:e].transpose(0, 1)[None]
        v_i = v[s:e].transpose(0, 1)[None]
        if q.is_cuda:
            o = _tile_attention(q_i, k_i, v_i, causal=True)
        else:
            o = sdpa(q_i, k_i, v_i, is_causal=True)
        out[s:e] = o[0].transpose(0, 1)
    return out


class ForwardBatch:
    logit_indices: torch.Tensor

    def attend(self, layer_idx, q, k, v):
        raise NotImplementedError


# --------------------------------------------------------------------------- none
class NoCacheBatch(ForwardBatch):
    """Full recompute of each sequence (prefill-style, nothing stored)."""

    def __init__(self, cu_seqlens: List[int], device):
        self.cu = cu_seqlens
        self.logit_indices = torch.tensor(cu_seqlens[1:], device=device) - 1

    def attend(self, layer_idx, q, k, v):
        return varlen_causal_attention(q, k, v, self.cu)


# --------------------------------------------------------------------- contiguous
class ContiguousKVCache:
    """Pool of `num_slots` slabs, each reserving `max_len` tokens of K/V per layer."""

    def __init__(self, cfg, num_slots, max_len, device, dtype):
        self.num_slots, self.max_len = num_slots, max_len
        shape = (num_slots, max_len, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.free_slots = deque(range(num_slots))

    @staticmethod
    def bytes_per_slot(cfg, max_len, dtype):
        return 2 * cfg.num_hidden_layers * max_len * cfg.num_key_value_heads * cfg.head_dim * torch.finfo(dtype).bits // 8

    def can_admit(self, num_tokens) -> bool:
        return len(self.free_slots) > 0

    def admit(self, seq, num_tokens):
        seq.slot = self.free_slots.popleft()

    def release(self, seq):
        if seq.slot is not None:
            self.free_slots.append(seq.slot)
            seq.slot = None

    def ensure_decode_slot(self, seq) -> bool:
        return True  # reserved up-front

    @property
    def usage(self):
        return 1.0 - len(self.free_slots) / self.num_slots


class ContiguousPrefillBatch(ForwardBatch):
    def __init__(self, cache: ContiguousKVCache, seqs, lens: List[int], device):
        self.cache = cache
        self.cu = [0]
        for L in lens:
            self.cu.append(self.cu[-1] + L)
        self.logit_indices = torch.tensor(self.cu[1:], device=device) - 1
        self.slot_ids = [s.slot for s in seqs]
        self.lens = lens

    def attend(self, layer_idx, q, k, v):
        for slot, s, e, L in zip(self.slot_ids, self.cu[:-1], self.cu[1:], self.lens):
            self.cache.k[layer_idx][slot, :L] = k[s:e]
            self.cache.v[layer_idx][slot, :L] = v[s:e]
        return varlen_causal_attention(q, k, v, self.cu)


class ContiguousDecodeBatch(ForwardBatch):
    def __init__(self, cache: ContiguousKVCache, seqs, device):
        self.cache = cache
        B = len(seqs)
        self.logit_indices = torch.arange(B, device=device)
        ctx = [s.num_tokens for s in seqs]  # includes the token being fed
        self.ctx = ctx
        self.max_ctx = max(ctx)
        self.slots = torch.tensor([s.slot for s in seqs], device=device)
        self.pos = torch.tensor([c - 1 for c in ctx], device=device)
        ar = torch.arange(self.max_ctx, device=device)[None]
        self.mask = (ar < torch.tensor(ctx, device=device)[:, None])[:, None, None, :]

    def attend(self, layer_idx, q, k, v):
        c = self.cache
        c.k[layer_idx][self.slots, self.pos] = k
        c.v[layer_idx][self.slots, self.pos] = v
        K = c.k[layer_idx][self.slots, : self.max_ctx].transpose(1, 2)  # [B, nkv, S, d]
        V = c.v[layer_idx][self.slots, : self.max_ctx].transpose(1, 2)
        if q.is_cuda:
            outs = [
                _tile_attention(q[i:i + 1, :, None, :], K[i:i + 1, :, :ctx, :],
                                V[i:i + 1, :, :ctx, :], causal=False)
                for i, ctx in enumerate(self.ctx)
            ]
            return torch.cat(outs, dim=0)[:, :, 0, :]
        o = sdpa(q[:, :, None, :], K, V, mask=self.mask)  # [B, nh, 1, d]
        return o[:, :, 0, :]


# ------------------------------------------------------------------------- paged
class PagedKVCache:
    """Block-structured KV store. Physical slot = block_id * block_size + offset."""

    def __init__(self, cfg, num_blocks, block_size, device, dtype):
        self.num_blocks, self.block_size = num_blocks, block_size
        shape = (num_blocks * block_size, cfg.num_key_value_heads, cfg.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(cfg.num_hidden_layers)]
        self.free_blocks = deque(range(num_blocks))
        self.watermark = max(1, int(0.01 * num_blocks))

    @staticmethod
    def bytes_per_block(cfg, block_size, dtype):
        return 2 * cfg.num_hidden_layers * block_size * cfg.num_key_value_heads * cfg.head_dim * torch.finfo(dtype).bits // 8

    def blocks_for(self, num_tokens):
        return -(-num_tokens // self.block_size)

    def can_admit(self, num_tokens) -> bool:
        return len(self.free_blocks) - self.blocks_for(num_tokens) >= self.watermark

    def admit(self, seq, num_tokens):
        seq.block_table = [self.free_blocks.popleft() for _ in range(self.blocks_for(num_tokens))]

    def release(self, seq):
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []

    def ensure_decode_slot(self, seq) -> bool:
        """Make sure a block exists for the token fed this step (position num_tokens-1)."""
        if len(seq.block_table) * self.block_size >= seq.num_tokens:
            return True
        if not self.free_blocks:
            return False
        seq.block_table.append(self.free_blocks.popleft())
        return True

    @property
    def usage(self):
        return 1.0 - len(self.free_blocks) / self.num_blocks


class PagedPrefillBatch(ForwardBatch):
    def __init__(self, cache: PagedKVCache, seqs, lens: List[int], device):
        self.cache = cache
        self.cu = [0]
        for L in lens:
            self.cu.append(self.cu[-1] + L)
        self.logit_indices = torch.tensor(self.cu[1:], device=device) - 1
        bs = cache.block_size
        slots = []
        for s, L in zip(seqs, lens):
            slots.extend(s.block_table[p // bs] * bs + p % bs for p in range(L))
        self.slot_mapping = torch.tensor(slots, device=device)

    def attend(self, layer_idx, q, k, v):
        self.cache.k[layer_idx].index_copy_(0, self.slot_mapping, k)
        self.cache.v[layer_idx].index_copy_(0, self.slot_mapping, v)
        return varlen_causal_attention(q, k, v, self.cu)


class PagedDecodeBatch(ForwardBatch):
    """One new token per sequence; K/V are gathered through the block tables."""

    def __init__(self, cache: PagedKVCache, seqs, device):
        self.cache = cache
        B, bs = len(seqs), cache.block_size
        self.logit_indices = torch.arange(B, device=device)
        ctx = [s.num_tokens for s in seqs]
        self.ctx = ctx
        max_blocks = max(len(s.block_table) for s in seqs)
        tables = torch.tensor([s.block_table + [0] * (max_blocks - len(s.block_table)) for s in seqs], device=device)
        self.block_tables = tables
        self.seq_lens = torch.tensor(ctx, device=device, dtype=torch.int32)
        self.slot_mapping = torch.tensor(
            [s.block_table[(c - 1) // bs] * bs + (c - 1) % bs for s, c in zip(seqs, ctx)], device=device)
        if not str(device).startswith("cuda") or bs not in (4, 8, 16, 32):
            ar_bs = torch.arange(bs, device=device)
            self.gather_idx = (tables[:, :, None] * bs + ar_bs).reshape(B, -1)
            S = self.gather_idx.shape[1]
            self.mask = (torch.arange(S, device=device)[None] < self.seq_lens[:, None])[:, None, None, :]

    def attend(self, layer_idx, q, k, v):
        c = self.cache
        c.k[layer_idx].index_copy_(0, self.slot_mapping, k)
        c.v[layer_idx].index_copy_(0, self.slot_mapping, v)
        if q.is_cuda and self.cache.block_size in (4, 8, 16, 32):
            from .kernels.cutile_attention import paged_decode_attention
            pool_shape = (c.num_blocks, c.block_size, c.k[layer_idx].shape[1], c.k[layer_idx].shape[2])
            return paged_decode_attention(q, c.k[layer_idx].view(pool_shape),
                                          c.v[layer_idx].view(pool_shape),
                                          self.block_tables, self.seq_lens)
        K = c.k[layer_idx][self.gather_idx].transpose(1, 2)  # [B, nkv, S, d]
        V = c.v[layer_idx][self.gather_idx].transpose(1, 2)
        o = sdpa(q[:, :, None, :], K, V, mask=self.mask)
        return o[:, :, 0, :]
