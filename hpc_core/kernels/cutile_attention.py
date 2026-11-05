"""Causal/GQA FlashAttention forward kernel implemented with CUDA Tile.

Inputs use PyTorch's [batch, heads, sequence, head_dim] layout. The kernel
streams K/V tiles and maintains online softmax statistics, so it never writes
the full attention-score matrix to global memory.
"""
from __future__ import annotations

import math

import torch
import cuda.tile as ct


_TILE_M = 64
_TILE_N = 64


@ct.kernel(occupancy=1)
def _flash_attention_kernel(
    q, k, v, out,
    q_start: int,
    scale_log2: float,
    query_group_size: ct.Constant[int],
    head_dim: ct.Constant[int],
    tile_m: ct.Constant[int],
    tile_n: ct.Constant[int],
    causal: ct.Constant[bool],
):
    """One program computes a query tile for one batch item and Q head."""
    tile_q = ct.bid(0)
    batch = ct.bid(1)
    q_head = ct.bid(2)
    kv_head = q_head // query_group_size

    q_rows = tile_q * tile_m + ct.arange(tile_m, dtype=ct.int32)
    k_cols = ct.arange(tile_n, dtype=ct.int32)
    q_tile = ct.load(
        q, index=(batch, q_head, tile_q, 0), shape=(1, 1, tile_m, head_dim),
        padding_mode=ct.PaddingMode.ZERO,
    ).reshape((tile_m, head_dim))

    max_score = ct.full((tile_m, 1), -float("inf"), dtype=ct.float32)
    denom = ct.full((tile_m, 1), 0.0, dtype=ct.float32)
    accum = ct.full((tile_m, head_dim), 0.0, dtype=ct.float32)
    key_len = k.shape[2]

    # Online softmax recurrence over K/V blocks. q_start aligns a short decode
    # query with its absolute position in the cached key sequence.
    max_key_pos = q_start + (tile_q + 1) * tile_m - 1
    key_tile_count = ct.cdiv(key_len, tile_n)
    if causal:
        key_tile_count = ct.cdiv(min(key_len, max_key_pos + 1), tile_n)

    for tile_k in range(key_tile_count):
        key_pos = tile_k * tile_n + k_cols
        k_tile = ct.load(
            k, index=(batch, kv_head, 0, tile_k), shape=(1, 1, head_dim, tile_n),
            order=(0, 1, 3, 2), padding_mode=ct.PaddingMode.ZERO,
        ).reshape((head_dim, tile_n))
        scores = ct.full((tile_m, tile_n), 0.0, dtype=ct.float32)
        scores = ct.mma(q_tile, k_tile, scores)

        valid = key_pos[None, :] < key_len
        if causal:
            valid = valid & ((q_start + q_rows)[:, None] >= key_pos[None, :])
        scores = ct.where(valid, scores, -float("inf"))
        scores = scores * scale_log2

        block_max = ct.max(scores, axis=1, keepdims=True)
        next_max = ct.maximum(max_score, block_max)
        old_scale = ct.exp2(max_score - next_max)
        weights = ct.exp2(scores - next_max)
        block_sum = ct.sum(weights, axis=1, keepdims=True)

        v_tile = ct.load(
            v, index=(batch, kv_head, tile_k, 0), shape=(1, 1, tile_n, head_dim),
            padding_mode=ct.PaddingMode.ZERO,
        ).reshape((tile_n, head_dim))
        block_accum = ct.full((tile_m, head_dim), 0.0, dtype=ct.float32)
        block_accum = ct.mma(weights.astype(q.dtype), v_tile, block_accum)
        accum = accum * old_scale + block_accum
        denom = denom * old_scale + block_sum
        max_score = next_max

    result = (accum / denom).astype(out.dtype)
    ct.store(out, index=(batch, q_head, tile_q, 0),
             tile=result.reshape((1, 1, tile_m, head_dim)))


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool,
    q_start: int = 0,
) -> torch.Tensor:
    """CUDA Tile attention for Qwen-style GQA.

    Args:
        q: [B, Hq, Sq, D]
        k, v: [B, Hkv, Sk, D]
        causal: apply a lower-triangular mask using absolute query positions.
        q_start: absolute position of q[:, :, 0, :] in the key sequence.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have [batch, heads, seq, head_dim] shape")
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("cuTile attention requires q, k, and v on one CUDA device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v batch dimensions must match")
    if k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("k/v shapes must match and q/k head dimensions must match")
    if q.shape[1] % k.shape[1]:
        raise ValueError("query heads must be divisible by key/value heads (GQA)")
    if q.shape[-1] not in (32, 64, 128, 256):
        raise ValueError(f"unsupported head dimension {q.shape[-1]}")
    if q.shape[2] < 1 or k.shape[2] < 1:
        raise ValueError("query and key sequences must be non-empty")
    if causal and (q_start < 0 or q_start + q.shape[2] > k.shape[2]):
        raise ValueError("causal query positions must lie within the key sequence")

    batch, q_heads, q_len, head_dim = q.shape
    # Store full query tiles, including the final partial one, into a padded
    # allocation and return a view of the requested sequence length.
    padded_q_len = ct.cdiv(q_len, _TILE_M) * _TILE_M
    out = torch.empty((batch, q_heads, padded_q_len, head_dim),
                      device=q.device, dtype=q.dtype)
    grid = (ct.cdiv(q_len, _TILE_M), batch, q_heads)
    ct.launch(
        torch.cuda.current_stream(), grid, _flash_attention_kernel,
        (q, k, v, out, q_start, (1.0 / math.sqrt(head_dim)) * (1.0 / math.log(2.0)),
         q_heads // k.shape[1], head_dim, _TILE_M, _TILE_N, causal),
    )
    return out[:, :, :q_len, :]
