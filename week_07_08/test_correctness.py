"""Correctness tests on a tiny random Qwen3 (CPU, float32) against HuggingFace transformers.

Checks, for every cache mode (none / contiguous / paged) and both batching policies:
  * forward logits match HuggingFace,
  * greedy generation of several concurrent requests of different lengths matches HuggingFace,
  * paged cache survives preemption (tiny block pool) and still produces identical text.

Run:  python test_correctness.py     (or: pytest -q test_correctness.py)
"""
import tempfile

import pytest
import torch
from transformers import Qwen3Config as HFConfig
from transformers import Qwen3ForCausalLM as HFModel

from engine import EngineConfig, LLMEngine, SamplingParams
from qwen3_model import load_model

torch.manual_seed(0)


@pytest.fixture(scope="module")
def setup():
    cfg = HFConfig(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
                   rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True)
    hf = HFModel(cfg).eval().float()
    with torch.no_grad():  # make norms non-trivial so weight-loading bugs are visible
        for n, p in hf.named_parameters():
            if "norm" in n:
                p.add_(torch.randn_like(p) * 0.1)
    d = tempfile.mkdtemp()
    hf.save_pretrained(d, safe_serialization=True)
    model, mcfg, _ = load_model(d, "cpu", torch.float32, max_positions=256)
    return hf, model, mcfg


def hf_greedy(hf, prompt, n):
    ids = list(prompt)
    for _ in range(n):
        with torch.no_grad():
            logits = hf(torch.tensor([ids])).logits[0, -1]
        ids.append(int(logits.argmax()))
    return ids[len(prompt):]


def run_engine(model, mcfg, ecfg, prompts, n_tokens):
    eng = LLMEngine(model, mcfg, ecfg, eos_token_ids=[], device="cpu", dtype=torch.float32)
    outs = [[] for _ in prompts]
    for i, p in enumerate(prompts):
        eng.add_request(p, SamplingParams(max_tokens=n_tokens[i], ignore_eos=True),
                        lambda tok, reason, i=i: outs[i].append(tok) if tok is not None else None)
    steps = 0
    while eng.has_work():
        eng.step()
        steps += 1
        assert steps < 5000, "engine did not terminate"
    return outs, eng


def test_forward_logits_match_hf(setup):
    from kv_cache import NoCacheBatch
    hf, model, _ = setup
    prompt = torch.randint(0, 200, (11,))
    ref = hf(prompt[None]).logits[0, -1]
    out = model(prompt, torch.arange(11), NoCacheBatch([0, 11], "cpu"))[0]
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.parametrize("mode", ["none", "contiguous", "paged"])
@pytest.mark.parametrize("batching", ["continuous", "static"])
def test_generation_matches_hf(setup, mode, batching):
    hf, model, mcfg = setup
    g = torch.Generator().manual_seed(1)
    lens = [3, 17, 9, 1, 25, 6]
    gens = [8, 5, 12, 10, 4, 9]
    prompts = [torch.randint(0, 200, (L,), generator=g).tolist() for L in lens]
    ecfg = EngineConfig(cache_mode=mode, batching=batching, max_num_seqs=3, max_model_len=64, block_size=4,
                        num_blocks=64, kv_cache_gb=0.1)
    outs, eng = run_engine(model, mcfg, ecfg, prompts, gens)
    for p, o, n in zip(prompts, outs, gens):
        assert o == hf_greedy(hf, p, n), f"{mode}/{batching} mismatch for prompt len {len(p)}"
    assert eng.stats["finished"] == len(prompts)
    if batching == "continuous":
        assert eng.snapshot()["avg_decode_batch"] > 1.0, "continuous batching should batch decode steps"


def test_paged_preemption(setup):
    hf, model, mcfg = setup
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 200, (L,), generator=g).tolist() for L in [5, 7, 9, 6, 8]]
    gens = [14] * 5
    ecfg = EngineConfig(cache_mode="paged", max_num_seqs=5, max_model_len=64, block_size=4, num_blocks=14)
    outs, eng = run_engine(model, mcfg, ecfg, prompts, gens)
    assert eng.stats["preemptions"] > 0, "test should exercise preemption"
    for p, o, n in zip(prompts, outs, gens):
        assert o == hf_greedy(hf, p, n)
    assert len(eng.cache.free_blocks) == eng.cache.num_blocks, "blocks leaked"


def test_contiguous_slots_released(setup):
    hf, model, mcfg = setup
    ecfg = EngineConfig(cache_mode="contiguous", max_num_seqs=2, max_model_len=64, kv_cache_gb=0.1)
    outs, eng = run_engine(model, mcfg, ecfg, [[1, 2, 3], [4, 5], [6]], [5, 5, 5])
    assert len(eng.cache.free_slots) == eng.cache.num_slots


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
