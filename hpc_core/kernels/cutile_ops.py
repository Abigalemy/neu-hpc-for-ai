"""CUDA Tile kernels shared by the Qwen3 inference assignments.

The public functions are host-side PyTorch wrappers. The decorated functions are
cuTile kernels and therefore operate on tiles, not CUDA threads.
"""
from __future__ import annotations

import torch
import cuda.tile as ct


_SWIGLU_TILE = 256


@ct.kernel
def _swiglu_kernel(gate, up, out, tile_size: ct.Constant[int]):
    """Fuse SiLU(gate) * up for a matrix shaped [rows, intermediate]."""
    row = ct.bid(0)
    col_tile = ct.bid(1)
    g = ct.load(gate, (row, col_tile), shape=(1, tile_size),
                padding_mode=ct.PaddingMode.ZERO)
    u = ct.load(up, (row, col_tile), shape=(1, tile_size),
                padding_mode=ct.PaddingMode.ZERO)

    # Accumulate the activation in fp32, then round once to the model dtype.
    g32 = g.astype(ct.float32)
    u32 = u.astype(ct.float32)
    y = (g32 / (1.0 + ct.exp(-g32))) * u32
    ct.store(out, (row, col_tile), y.astype(g.dtype))


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor, tile_size: int = _SWIGLU_TILE) -> torch.Tensor:
    """Apply fused SwiGLU to same-shaped CUDA matrices using cuTile.

    The two-dimensional interface deliberately accepts strided tensors, which
    is needed for the gate/up views produced by Qwen3's fused gate-up projection.
    """
    if gate.shape != up.shape:
        raise ValueError(f"gate/up shape mismatch: {tuple(gate.shape)} != {tuple(up.shape)}")
    if gate.ndim != 2:
        raise ValueError(f"expected [rows, width] tensors, got {gate.ndim} dimensions")
    if gate.device.type != "cuda" or up.device != gate.device:
        raise ValueError("cuTile SwiGLU requires gate and up on the same CUDA device")
    if gate.dtype != up.dtype:
        raise ValueError(f"gate/up dtype mismatch: {gate.dtype} != {up.dtype}")
    if tile_size not in (128, 256, 512, 1024):
        raise ValueError("tile_size must be one of 128, 256, 512, or 1024")

    rows, width = gate.shape
    padded_width = ct.cdiv(width, tile_size) * tile_size
    out = torch.empty((rows, padded_width), device=gate.device, dtype=gate.dtype)
    grid = (rows, ct.cdiv(width, tile_size))
    ct.launch(torch.cuda.current_stream(), grid, _swiglu_kernel,
              (gate, up, out, tile_size))
    return out[:, :width]
