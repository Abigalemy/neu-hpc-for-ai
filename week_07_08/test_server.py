"""HTTP-level smoke test with a stub tokenizer (no model download): streaming + non-streaming."""
import json
import tempfile
import threading

import torch
from fastapi.testclient import TestClient
from transformers import Qwen3Config as HFConfig
from transformers import Qwen3ForCausalLM as HFModel

import server
from engine import EngineConfig, LLMEngine
from qwen3_model import load_model


class StubTok:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 190 + 5 for c in text]

    def decode(self, ids, **kw):
        return "".join(chr(97 + i % 26) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(m["content"] for m in messages)


def make_client():
    hf = HFModel(HFConfig(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=256, tie_word_embeddings=True)).eval()
    d = tempfile.mkdtemp()
    hf.save_pretrained(d)
    model, cfg, _ = load_model(d, "cpu", torch.float32, max_positions=256)
    eng = LLMEngine(model, cfg, EngineConfig(cache_mode="paged", max_model_len=128, num_blocks=64),
                    eos_token_ids=[], device="cpu", dtype=torch.float32)
    server.STATE.update(engine=eng, tok=StubTok(), name="tiny")
    threading.Thread(target=eng.run_forever, daemon=True).start()
    return TestClient(server.app), eng


def test_endpoints():
    c, eng = make_client()
    try:
        r = c.post("/v1/completions", json={"prompt": "hello", "max_tokens": 6, "ignore_eos": True}).json()
        assert r["usage"]["completion_tokens"] == 6 and r["choices"][0]["finish_reason"] == "length"
        assert len(r["choices"][0]["text"]) == 6

        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}],
                                                 "max_tokens": 5, "ignore_eos": True}).json()
        assert r["choices"][0]["message"]["role"] == "assistant"
        assert r["usage"]["completion_tokens"] == 5

        with c.stream("POST", "/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "hi"}], "max_tokens": 7, "ignore_eos": True,
                "stream": True, "stream_options": {"include_usage": True}}) as resp:
            lines = [l for l in resp.iter_lines() if l.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        chunks = [json.loads(l[6:]) for l in lines[:-1]]
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        assert chunks[-1]["usage"]["completion_tokens"] == 7
        assert "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"]) != ""

        # prompt longer than context window is rejected cleanly
        r = c.post("/v1/completions", json={"prompt": "x" * 500, "max_tokens": 4}).json()
        assert r["choices"][0]["finish_reason"] == "error"
        assert c.get("/metrics_json").json()["finished"] >= 3
    finally:
        eng.stop()
