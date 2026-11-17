"""OpenAI-compatible HTTP server around the CUDA Tile Qwen3 engine (no PyTorch).

  python server.py --model-dir /path/to/Qwen3-4B --lib ../build/libqwen3engine.so --port 8000 [--yarn 4] [--max-seqs 16]
  python server.py --fake --port 8000          # fake engine + fake tokenizer, for testing without a GPU

Endpoints: POST /v1/chat/completions, POST /v1/completions (both with "stream": true -> SSE), GET /v1/models, GET /health.
Request fields understood: max_tokens / max_completion_tokens, min_tokens, ignore_eos, stream, stream_options.include_usage,
temperature (only 0 / greedy is supported: anything else is rejected with 400), chat_template_kwargs.

Continuous batching: a scheduler thread owns the engine. Each iteration it admits waiting requests into free KV-cache
slots (prefill, which yields their first token), then runs ONE decode step for all running sequences together; finished
or disconnected sequences free their slot at once, so waiting requests join at the next iteration. A prefill pauses the
decode of the others while it runs (no chunked mixing of prefill and decode yet).
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

    def __init__(self, lib_path: str, model_dir: str, max_len: int, max_chunk: int, yarn: float, max_seqs: int):
        lib = ctypes.CDLL(lib_path)
        lib.qe_create.restype = ctypes.c_void_p
        lib.qe_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        for name in ("qe_vocab", "qe_max_len", "qe_max_chunk", "qe_max_seqs"):
            getattr(lib, name).restype = ctypes.c_int
            getattr(lib, name).argtypes = [ctypes.c_void_p]
        IntP = ctypes.POINTER(ctypes.c_int)
        lib.qe_prefill.restype = ctypes.c_int
        lib.qe_prefill.argtypes = [ctypes.c_void_p, ctypes.c_int, IntP, ctypes.c_int]
        lib.qe_decode_batch.restype = None
        lib.qe_decode_batch.argtypes = [ctypes.c_void_p, ctypes.c_int, IntP, IntP, IntP, IntP]
        self.lib = lib
        self.h = lib.qe_create(model_dir.encode(), max_len, max_chunk, yarn, max_seqs)
        if not self.h:
            raise RuntimeError("qe_create failed")
        self.vocab = lib.qe_vocab(self.h)
        self.max_len = lib.qe_max_len(self.h)
        self.max_seqs = lib.qe_max_seqs(self.h)

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
    """Deterministic stand-in for tests: the next token is a function of the previous one; the `eos_after`-th generated
    token of a sequence (if set) is `eos`, once, so both stopping at EOS and ignoring it can be tested."""

    def __init__(self, vocab: int = 1000, max_len: int = 4096, eos: int = 2, eos_after: Optional[int] = None,
                 max_seqs: int = 4, delay: float = 0.0):
        self.vocab, self.max_len, self.eos, self.eos_after, self.max_seqs = vocab, max_len, eos, eos_after, max_seqs
        self.delay = delay            # seconds per engine call, to let concurrent requests overlap in tests
        self.calls = []
        self.n_gen = {}

    def _next(self, slot: int, tok: int) -> int:
        self.n_gen[slot] = self.n_gen.get(slot, 0) + 1
        if self.eos_after is not None and self.n_gen[slot] == self.eos_after:
            return self.eos
        return 10 + (tok * 7 + 3) % (self.vocab - 10)

    def prefill(self, slot: int, ids: List[int]) -> int:
        time.sleep(self.delay)
        self.calls.append(("prefill", slot, len(ids)))
        self.n_gen[slot] = 0
        return self._next(slot, ids[-1])

    def decode_batch(self, slots: List[int], tokens: List[int], positions: List[int]) -> List[int]:
        time.sleep(self.delay)
        assert len(set(slots)) == len(slots), "a slot appears twice in one batch"
        self.calls.append(("decode", tuple(slots), tuple(positions)))
        return [self._next(s_, t) for s_, t in zip(slots, tokens)]


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
        self.slot = self.pos = self.last_tok = None
        self.n_gen = 0               # tokens generated so far (an ignored EOS counts, a final EOS does not)
        self.cancelled = False       # set by the HTTP side when the client goes away

    def emit(self, tok: int, finish: Optional[str]):
        self.loop.call_soon_threadsafe(self.queue.put_nowait, (tok, finish))


class Scheduler:
    """Owns the backend in one thread: admit -> prefill -> one batched decode step, repeated (see module docstring)."""

    def __init__(self, backend, eos_ids: List[int]):
        self.backend, self.eos = backend, set(eos_ids)
        self.free = list(range(backend.max_seqs))
        self.running: List[GenRequest] = []
        self.waiting = collections.deque()
        self.cv = threading.Condition()
        self.batch_sizes: List[int] = []      # decode batch size of every step (statistics, tests)
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, req: GenRequest):
        with self.cv:
            self.waiting.append(req)
            self.cv.notify()

    def _accept(self, req: GenRequest, tok: int) -> bool:
        """Deliver one generated token; True when the request is finished."""
        if tok in self.eos and not req.ignore_eos and req.n_gen >= req.min_new:
            req.emit(tok, "stop")
            return True
        last = req.n_gen == req.max_new - 1
        req.emit(tok, "length" if last else None)
        req.n_gen += 1
        req.last_tok = tok
        return last

    def _release(self, req: GenRequest):
        self.running.remove(req)
        self.free.append(req.slot)

    def _loop(self):
        while True:
            try:
                self._iteration()
            except Exception:                       # never let the engine thread die silently: fail the requests
                import traceback
                traceback.print_exc()
                with self.cv:
                    for r in list(self.running):
                        r.emit(-1, "error")
                        self._release(r)

    def _iteration(self):
        with self.cv:
            while not self.waiting and not self.running:
                self.cv.wait()
            admitted = []
            while self.waiting and self.free:
                r = self.waiting.popleft()
                if r.cancelled:
                    continue
                r.slot = self.free.pop(0)
                self.running.append(r)       # running from now on, so an exception below still frees its slot
                admitted.append(r)
        for r in admitted:                   # prefill: this request's first token = its TTFT
            tok = self.backend.prefill(r.slot, r.ids)
            with self.cv:
                r.pos = len(r.ids)
                if self._accept(r, tok):
                    self._release(r)
        with self.cv:
            for r in [r for r in self.running if r.cancelled]:
                self._release(r)
            batch = [r for r in self.running if r.pos is not None]
        if not batch:
            return
        outs = self.backend.decode_batch([r.slot for r in batch], [r.last_tok for r in batch], [r.pos for r in batch])
        self.batch_sizes.append(len(batch))
        with self.cv:
            for r, tok in zip(batch, outs):
                r.pos += 1
                if self._accept(r, tok):
                    self._release(r)

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
def create_app(backend, tokenizer, model_name: str, eos_ids: List[int]) -> FastAPI:
    app = FastAPI()
    sched = Scheduler(backend, eos_ids)
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
            ids = prompt if prompt and isinstance(prompt[0], int) else tokenizer.encode(prompt[0] if prompt else "")
        else:
            ids = tokenizer.encode(prompt or "")
        return await _respond(body, ids, chat=False)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "local"}]}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

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
    ap.add_argument("--max-seqs", type=int, default=1, help="KV-cache slots = sequences decoded together")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.fake:
        backend, tok, eos = FakeBackend(max_len=a.max_len, max_seqs=a.max_seqs), FakeTokenizer(), [FakeTokenizer.eos_token_id]
    else:
        from transformers import AutoTokenizer   # tokenizer only; PyTorch is not needed for it
        tok = AutoTokenizer.from_pretrained(a.model_dir)
        eos = [tok.eos_token_id]
        gc = os.path.join(a.model_dir, "generation_config.json")
        if os.path.exists(gc):
            e = json.load(open(gc)).get("eos_token_id")
            eos += e if isinstance(e, list) else ([e] if e is not None else [])
        backend = CBackend(a.lib, a.model_dir, a.max_len, a.max_chunk, a.yarn, a.max_seqs)
        print(f"engine ready: vocab {backend.vocab}, max_len {backend.max_len}, max_seqs {backend.max_seqs}, "
              f"YaRN {a.yarn or 'off'}, eos {sorted(set(eos))}", flush=True)
    import uvicorn
    uvicorn.run(create_app(backend, tok, a.served_model_name, eos), host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
