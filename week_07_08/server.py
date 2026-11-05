"""OpenAI-compatible HTTP server (completions + chat completions, SSE streaming) for the Qwen3 engine.

Example:
  python server.py --model Qwen/Qwen3-0.6B --cache-mode paged --batching continuous --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
import uuid
from typing import List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer

try:
    from _qwen3_cpp_tokenizer import Qwen3Tokenizer as CppQwen3Tokenizer
except ImportError:
    CppQwen3Tokenizer = None

from engine import EngineConfig, LLMEngine, SamplingParams
from qwen3_model import load_model

app = FastAPI()
STATE = {}


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    prompt: Union[str, List[int], List[str], List[List[int]]] = ""
    max_tokens: Optional[int] = 16
    min_tokens: int = 0
    temperature: float = 0.0   # greedy by default (deterministic benchmarks)
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stream_options: Optional[dict] = None
    ignore_eos: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: Optional[str] = None
    messages: List[dict]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    min_tokens: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    stream_options: Optional[dict] = None
    ignore_eos: bool = False
    chat_template_kwargs: Optional[dict] = None


class IncrementalDetokenizer:
    """Streams text without emitting partial UTF-8 sequences."""

    def __init__(self, tok):
        self.tok, self.ids, self.prefix, self.read = tok, [], 0, 0

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        prefix_text = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=True)
        new_text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=True)
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text):]
        return ""


async def generate(prompt_ids: List[int], params: SamplingParams):
    """Async generator of (text_delta, finish_reason). Final item has finish_reason set."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def cb(tok, reason):
        loop.call_soon_threadsafe(q.put_nowait, (tok, reason))

    seq = STATE["engine"].add_request(prompt_ids, params, cb)
    detok = IncrementalDetokenizer(STATE["tok"])
    done = False
    try:
        while True:
            tok, reason = await q.get()
            text = detok.push(tok) if tok is not None else ""
            if reason is not None:
                done = True
            yield text, reason
            if done:
                return
    finally:
        if not done:  # client disconnected
            STATE["engine"].abort(seq)


def _params(req, max_tokens) -> SamplingParams:
    return SamplingParams(max_tokens=max_tokens, min_tokens=req.min_tokens, temperature=req.temperature,
                          top_p=req.top_p, top_k=req.top_k, ignore_eos=req.ignore_eos)


def _sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _prompt_to_ids(prompt) -> List[int]:
    tok = STATE["cpp_tok"]
    if isinstance(prompt, str):
        return tok.encode(prompt)
    if prompt and isinstance(prompt[0], str):
        return tok.encode(prompt[0])
    if prompt and isinstance(prompt[0], list):
        return list(prompt[0])
    return list(prompt)


async def _run(request_id, obj_type, prompt_ids, params, stream, include_usage, chat):
    model = STATE["name"]
    created = int(time.time())

    def chunk(text, finish):
        if chat:
            delta = {"content": text} if text else {}
            ch = {"index": 0, "delta": delta, "finish_reason": finish}
        else:
            ch = {"index": 0, "text": text, "finish_reason": finish}
        return {"id": request_id, "object": obj_type, "created": created, "model": model, "choices": [ch]}

    n_prompt = len(prompt_ids)
    if not stream:
        text, n_out, finish = "", 0, None
        async for t, reason in generate(prompt_ids, params):
            text += t
            n_out += 1 if t is not None else 0
            finish = reason
        n_out = n_out if finish != "error" else 0
        usage = {"prompt_tokens": n_prompt, "completion_tokens": n_out, "total_tokens": n_prompt + n_out}
        if chat:
            ch = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
        else:
            ch = {"index": 0, "text": text, "finish_reason": finish}
        return JSONResponse({"id": request_id, "object": obj_type.replace(".chunk", ""), "created": created,
                             "model": model, "choices": [ch], "usage": usage})

    async def stream_gen():
        n_out = 0
        if chat:
            first = chunk("", None)
            first["choices"][0]["delta"] = {"role": "assistant", "content": ""}
            yield _sse(first)
        finish = None
        async for t, reason in generate(prompt_ids, params):
            n_out += 1
            finish = reason
            if t and reason is None:
                yield _sse(chunk(t, None))
            elif reason is not None:
                if t:
                    yield _sse(chunk(t, None))
                n_out = n_out if reason != "error" else 0
        yield _sse(chunk("", finish))
        if include_usage:
            u = {"id": request_id, "object": obj_type, "created": created, "model": model, "choices": [],
                 "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_out,
                           "total_tokens": n_prompt + n_out}}
            yield _sse(u)
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_gen(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    ids = _prompt_to_ids(req.prompt)
    params = _params(req, req.max_tokens or 16)
    inc = bool((req.stream_options or {}).get("include_usage"))
    return await _run("cmpl-" + uuid.uuid4().hex[:12], "text_completion", ids, params, req.stream, inc, chat=False)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    kwargs = dict(req.chat_template_kwargs or {})
    kwargs.setdefault("enable_thinking", False)
    try:
        text = STATE["tok"].apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:  # tokenizer without a chat template
        text = "\n".join(f"{m['role']}: {m['content']}" for m in req.messages) + "\nassistant:"
    ids = STATE["cpp_tok"].encode(text)
    params = _params(req, req.max_completion_tokens or req.max_tokens or 256)
    inc = bool((req.stream_options or {}).get("include_usage"))
    return await _run("chatcmpl-" + uuid.uuid4().hex[:12], "chat.completion.chunk", ids, params, req.stream,
                      inc, chat=True)


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": STATE["name"], "object": "model", "owned_by": "local"}]}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/metrics_json")
async def metrics():
    return STATE["engine"].snapshot()


def build_engine(args):
    if CppQwen3Tokenizer is None:
        raise RuntimeError(
            "The C++ Qwen3 tokenizer extension is missing; build week_04_05/tokenizer_cpp first"
        )
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model, cfg, model_dir = load_model(args.model, args.device, dtype, max_positions=args.max_model_len)
    eos = {151645, 151643}
    gen_cfg = os.path.join(model_dir, "generation_config.json")
    if os.path.exists(gen_cfg):
        e = json.load(open(gen_cfg)).get("eos_token_id")
        if e is not None:
            eos = set(e if isinstance(e, list) else [e])
    ecfg = EngineConfig(cache_mode=args.cache_mode, batching=args.batching, max_num_seqs=args.max_num_seqs,
                        max_model_len=args.max_model_len, block_size=args.block_size,
                        num_blocks=args.num_blocks, kv_cache_gb=args.kv_cache_gb,
                        gpu_memory_utilization=args.gpu_memory_utilization,
                        max_num_batched_tokens=args.max_num_batched_tokens)
    engine = LLMEngine(model, cfg, ecfg, eos, args.device, dtype)
    tok = AutoTokenizer.from_pretrained(model_dir)
    cpp_tok = CppQwen3Tokenizer(os.path.join(model_dir, "tokenizer.json"))
    return engine, tok, cpp_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--cache-mode", default="paged", choices=["none", "contiguous", "paged"])
    ap.add_argument("--batching", default="continuous", choices=["continuous", "static"])
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--kv-cache-gb", type=float, default=None)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    engine, tok, cpp_tok = build_engine(args)
    STATE.update(engine=engine, tok=tok, cpp_tok=cpp_tok,
                 name=args.served_model_name or args.model)
    threading.Thread(target=engine.run_forever, daemon=True).start()
    print("engine ready:", json.dumps(engine.snapshot()), flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
