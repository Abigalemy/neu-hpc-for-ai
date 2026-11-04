"""Qwen3 decoder-only model (pure PyTorch) with pluggable attention/KV-cache backends.

The model itself is cache-agnostic. Every forward pass receives a `ForwardBatch`
(see kv_cache.py) that knows how to store K/V and how to compute attention for
the current batch (no cache / contiguous cache / paged cache).

Tokens are passed as a flat "varlen" tensor [T] so that prefill of several
sequences and batched decode share one code path.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool

    @classmethod
    def from_dict(cls, c: dict) -> "Qwen3Config":
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            # older configs: top-level rope_theta; transformers>=5 nests it in rope_parameters
            rope_theta=c.get("rope_theta") or (c.get("rope_parameters") or {}).get("rope_theta", 1_000_000.0),
            max_position_embeddings=c.get("max_position_embeddings", 40960),
            tie_word_embeddings=c.get("tie_word_embeddings", False),
        )

    @classmethod
    def from_json(cls, path: str) -> "Qwen3Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(q, k, cos, sin):
    # q: [T, nh, d]  k: [T, nkv, d]  cos/sin: [T, 1, d]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.d = cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.hidden_size, (self.nh + 2 * self.nkv) * self.d, bias=False)
        self.o_proj = nn.Linear(self.nh * self.d, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.d, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.d, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, batch):
        T = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.nh * self.d, self.nkv * self.d, self.nkv * self.d], dim=-1)
        q = self.q_norm(q.reshape(T, self.nh, self.d))
        k = self.k_norm(k.reshape(T, self.nkv, self.d))
        v = v.reshape(T, self.nkv, self.d)
        q, k = apply_rope(q, k, cos, sin)
        o = batch.attend(self.layer_idx, q, k, v)  # [T, nh, d]
        return self.o_proj(o.reshape(T, self.nh * self.d))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        g, u = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(g) * u)


class Qwen3Layer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_idx: int):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x, cos, sin, batch):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, batch)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config, max_positions: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Qwen3Layer(cfg, i) for i in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.max_positions = min(max_positions or cfg.max_position_embeddings, cfg.max_position_embeddings)
        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    def build_rope_cache(self, device, dtype):
        d = self.cfg.head_dim
        inv = 1.0 / (self.cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=device) / d))
        t = torch.arange(self.max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, batch) -> torch.Tensor:
        """input_ids/positions: [T]. Returns float32 logits [len(batch.logit_indices), vocab]."""
        x = self.embed_tokens(input_ids)
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        for layer in self.layers:
            x = layer(x, cos, sin, batch)
        x = self.norm(x[batch.logit_indices])
        return self.lm_head(x).float()

    # ------------------------------------------------------------------ loading
    def load_hf_state_dict(self, sd: dict):
        """Load a HuggingFace Qwen3 state dict (fusing qkv and gate/up)."""
        own = dict(self.named_parameters())
        used = set()

        def put(name, tensor):
            own[name].data.copy_(tensor)
            used.add(name)

        put("embed_tokens.weight", sd["model.embed_tokens.weight"])
        put("norm.weight", sd["model.norm.weight"])
        for i in range(self.cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            o = f"layers.{i}."
            put(o + "self_attn.qkv_proj.weight", torch.cat([
                sd[p + "self_attn.q_proj.weight"],
                sd[p + "self_attn.k_proj.weight"],
                sd[p + "self_attn.v_proj.weight"]], dim=0))
            put(o + "self_attn.o_proj.weight", sd[p + "self_attn.o_proj.weight"])
            put(o + "self_attn.q_norm.weight", sd[p + "self_attn.q_norm.weight"])
            put(o + "self_attn.k_norm.weight", sd[p + "self_attn.k_norm.weight"])
            put(o + "mlp.gate_up_proj.weight", torch.cat([
                sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]], dim=0))
            put(o + "mlp.down_proj.weight", sd[p + "mlp.down_proj.weight"])
            put(o + "input_layernorm.weight", sd[p + "input_layernorm.weight"])
            put(o + "post_attention_layernorm.weight", sd[p + "post_attention_layernorm.weight"])
        if self.cfg.tie_word_embeddings or "lm_head.weight" not in sd:
            self.lm_head.weight = self.embed_tokens.weight
        else:
            put("lm_head.weight", sd["lm_head.weight"])


def resolve_model_dir(name_or_path: str) -> str:
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(
        name_or_path,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
    )


def load_model(name_or_path: str, device: str, dtype: torch.dtype, max_positions: Optional[int] = None):
    """Build Qwen3ForCausalLM and load weights from a local dir or HF repo id."""
    from safetensors import safe_open

    model_dir = resolve_model_dir(name_or_path)
    cfg = Qwen3Config.from_json(os.path.join(model_dir, "config.json"))
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device(device):
            model = Qwen3ForCausalLM(cfg, max_positions)
    finally:
        torch.set_default_dtype(old)
    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for k in sf.keys():
                sd[k] = sf.get_tensor(k)
    model.load_hf_state_dict(sd)
    del sd
    model.build_rope_cache(device, dtype)
    model.eval()
    return model, cfg, model_dir
