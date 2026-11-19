"""Local tests of the HTTP server with the fake engine and fake tokenizers (no GPU, no model download).

  python test_server.py
"""
import asyncio
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server as server_mod  # noqa: E402
from server import Detokenizer, FakeBackend, FakeTokenizer, GenRequest, create_app  # noqa: E402

TOK = FakeTokenizer()


def make(eos_after=None, max_len=4096, max_seqs=4, delay=0.0, block_size=128, num_blocks=None):
    be = FakeBackend(max_len=max_len, eos=TOK.eos_token_id, eos_after=eos_after, max_seqs=max_seqs, delay=delay,
                     block_size=block_size, num_blocks=num_blocks)
    return be, create_app(be, TOK, "Qwen/Qwen3-4B", [TOK.eos_token_id])


async def sse(client, path, body):
    """POST with streaming; returns the list of parsed data objects and whether [DONE] came last."""
    events = []
    async with client.stream("POST", path, json=body) as r:
        assert r.status_code == 200, (r.status_code, await r.aread())
        assert r.headers["content-type"].startswith("text/event-stream")
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                events.append(line[6:])
    done = events and events[-1] == "[DONE]"
    return [json.loads(e) for e in events if e != "[DONE]"], done


def client_for(app):
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_health_models():
    _, app = make()
    async with client_for(app) as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        assert (await c.get("/v1/models")).json()["data"][0]["id"] == "Qwen/Qwen3-4B"


async def test_chat_stream_exact_length_and_usage():
    be, app = make(eos_after=3)   # the fake engine would emit EOS as the 3rd token; ignore_eos must override it
    msgs = [{"role": "user", "content": "explain tiling in shared memory"}]
    async with client_for(app) as c:
        ev, done = await sse(c, "/v1/chat/completions", {"model": "Qwen/Qwen3-4B", "messages": msgs, "stream": True,
                                                         "max_tokens": 8, "ignore_eos": True, "temperature": 0,
                                                         "stream_options": {"include_usage": True}})
    assert done, "stream must end with [DONE]"
    assert ev[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}, ev[0]
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e["choices"])
    finishes = [e["choices"][0]["finish_reason"] for e in ev if e["choices"]]
    assert finishes[-1] == "length" and all(f is None for f in finishes[:-1]), finishes
    usage = ev[-1]["usage"]
    n_prompt = len(TOK.encode(TOK.apply_chat_template(msgs)))
    assert usage == {"prompt_tokens": n_prompt, "completion_tokens": 8, "total_tokens": n_prompt + 8}, usage
    # with ignore_eos the EOS token id (2) is generated but has no text: 7 visible words + 1 silent EOS
    assert len(content.split()) == 7, content
    # engine calls: one prefill of the whole prompt into slot 0, then 7 single-sequence decodes at consecutive positions
    assert be.calls[0] == ("prefill", 0, n_prompt), be.calls[0]
    assert be.calls[1:] == [("decode", (0,), (n_prompt + i,)) for i in range(7)], be.calls


async def test_eos_stops_without_ignore_eos():
    be, app = make(eos_after=3)
    async with client_for(app) as c:
        ev, done = await sse(c, "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                         "stream": True, "max_tokens": 50})
    assert done
    content = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e["choices"])
    assert len(content.split()) == 2 and ev[-1]["choices"][0]["finish_reason"] == "stop", (content, ev[-1])
    assert len([c for c in be.calls if c[0] == "decode"]) == 2


async def test_max_completion_tokens_wins_and_non_stream():
    _, app = make()
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "a b"}],
                                                       "max_tokens": 50, "max_completion_tokens": 4})
    j = r.json()
    assert r.status_code == 200 and j["object"] == "chat.completion", j
    assert j["usage"]["completion_tokens"] == 4 and j["choices"][0]["finish_reason"] == "length", j
    assert len(j["choices"][0]["message"]["content"].split()) == 4


async def test_completions_string_and_ids():
    be, app = make()
    async with client_for(app) as c:
        ev, done = await sse(c, "/v1/completions", {"prompt": "one two three", "max_tokens": 5, "stream": True})
        assert done and ev[-1]["choices"][0]["finish_reason"] == "length"
        assert len("".join(e["choices"][0]["text"] for e in ev).split()) == 5
        r = await c.post("/v1/completions", json={"prompt": [11, 12, 13, 14], "max_tokens": 3})
        assert r.json()["usage"]["prompt_tokens"] == 4 and be.calls[-3][0] == "prefill" and be.calls[-3][2] == 4, (r.json(), be.calls[-3:])


async def test_rejections():
    _, app = make(max_len=64)
    async with client_for(app) as c:
        bad = [({"messages": [{"role": "user", "content": "x"}], "temperature": 0.7}, "greedy"),
               ({"messages": [{"role": "user", "content": " ".join(["w"] * 60)}], "max_tokens": 32}, "exceeds the context"),
               ({"messages": []}, "non-empty"),
               ({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}, ">= 1")]
        for body, why in bad:
            r = await c.post("/v1/chat/completions", json=body)
            assert r.status_code == 400 and why in r.json()["detail"], (body, r.status_code, r.text)


def visible(ev):
    return "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e["choices"])


async def test_requests_are_batched():
    be, app = make(max_seqs=4, delay=0.01)      # a small delay per engine call so the three requests overlap
    msgs = [{"role": "user", "content": "same prompt"}]
    body = {"messages": msgs, "stream": True, "max_tokens": 6, "ignore_eos": True}
    async with client_for(app) as c:
        alone, _ = await sse(c, "/v1/chat/completions", body)
        be.calls.clear()
        res = await asyncio.gather(*[sse(c, "/v1/chat/completions", body) for _ in range(3)])
    sched = app.state.scheduler
    assert all(visible(ev) == visible(alone) for ev, _ in res), "batched output must equal the output alone"
    decodes = [c for c in be.calls if c[0] == "decode"]
    assert max(len(c[1]) for c in decodes) == 3, decodes           # the three sequences were decoded together
    for slot in {s_ for c in decodes for s_ in c[1]}:              # every slot advances one position per step
        poss = [c[2][c[1].index(slot)] for c in decodes if slot in c[1]]
        assert poss == list(range(poss[0], poss[0] + len(poss))), (slot, poss)
    assert max(sched.batch_sizes) == 3


async def test_more_requests_than_slots():
    be, app = make(max_seqs=2, delay=0.01)
    body = {"messages": [{"role": "user", "content": "q"}], "stream": True, "max_tokens": 5, "ignore_eos": True}
    async with client_for(app) as c:
        res = await asyncio.gather(*[sse(c, "/v1/chat/completions", body) for _ in range(3)])
    assert all(done for _, done in res) and len({visible(ev) for ev, _ in res}) == 1
    assert max(len(c[1]) for c in be.calls if c[0] == "decode") <= 2      # never more sequences than slots
    assert sum(1 for c in be.calls if c[0] == "prefill") == 3


async def test_disconnect_frees_slot():
    be, app = make(max_seqs=1, delay=0.005)
    sched = app.state.scheduler
    gen = sched.tokens([11, 12, 13], 10_000, 0, True)
    got = [await gen.__anext__() for _ in range(3)]       # read 3 tokens, then the client "goes away"
    await gen.aclose()
    assert len(got) == 3
    second = [t async for t in sched.tokens([11, 12, 13], 4, 0, True)]   # must get the only slot back
    assert len(second) == 4 and second[-1][1] == "length", second


async def test_engine_error_fails_only_that_request():
    be, app = make(max_seqs=2)
    orig = be.decode_batch
    def broken(*a):
        raise RuntimeError("simulated engine failure")
    be.decode_batch = broken
    async with client_for(app) as c:
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 4})
        assert r.status_code == 500, (r.status_code, r.text)
        be.decode_batch = orig
        r = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}], "max_tokens": 4})
        assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 4, r.text
    assert sorted(app.state.scheduler.free) == [0, 1], app.state.scheduler.free   # no slot leaked


async def completion_tokens(c, prompt_ids, n):
    """Streamed /v1/completions with token ids; returns (text, finish reasons, usage)."""
    ev, done = await sse(c, "/v1/completions", {"prompt": prompt_ids, "max_tokens": n, "stream": True,
                                                "ignore_eos": True, "stream_options": {"include_usage": True}})
    assert done
    text = "".join(e["choices"][0]["text"] for e in ev if e["choices"])
    return text, [e["choices"][0]["finish_reason"] for e in ev if e["choices"]], ev[-1]["usage"]


async def test_paged_blocks_admit_more_than_contiguous():
    # a pool of exactly ONE max_len sequence (64 tokens = 16 blocks of 4): reserving max_len per sequence (contiguous
    # slots) would run one request at a time; with blocks taken on demand all four short requests run together
    be, app = make(max_len=64, max_seqs=8, delay=0.005, block_size=4, num_blocks=16)
    async with client_for(app) as c:
        alone = await completion_tokens(c, [11, 12, 13], 4)
        res = await asyncio.gather(*[completion_tokens(c, [11, 12, 13], 4) for _ in range(4)])
    sched = app.state.scheduler
    assert all(r[0] == alone[0] for r in res), "batched output must equal the output alone"
    assert max(sched.batch_sizes) == 4, sched.batch_sizes
    assert sched.preemptions == 0 and be.free_blocks() == 16 and not be.blocks, be.blocks   # nothing leaked


async def test_preemption_by_recompute():
    # 5 requests of 3 + 8 tokens need 3 blocks each (15) but the pool has 6: sequences are preempted when they need a
    # new block, recomputed later from prompt + generated tokens, and every stream still gets exactly the output alone
    be, app = make(max_len=64, max_seqs=8, delay=0.002, block_size=4, num_blocks=6)
    async with client_for(app) as c:
        alone = await completion_tokens(c, [11, 12, 13], 8)
        res = await asyncio.gather(*[completion_tokens(c, [11, 12, 13], 8) for _ in range(5)])
    sched = app.state.scheduler
    for text, fins, usage in res:
        assert text == alone[0], (text, alone[0])
        assert fins[-1] == "length" and all(f is None for f in fins[:-1]), fins
        assert usage["completion_tokens"] == 8, usage
    assert sched.preemptions > 0, "the pool is too small for all five: some sequence must have been preempted"
    recomputes = [ids for _, ids in be.prefill_ids if len(ids) > 3]        # prefill of prompt + generated tokens
    assert len(recomputes) == sched.preemptions, (recomputes, sched.preemptions)
    # every recompute re-feeds exactly the prompt and the tokens already sent, in order (the fake's output depends on
    # the whole history, so a lost or reordered token would also change the streamed text above)
    alone_ids = [int(w[1:]) for w in alone[0].split()]
    for ids in recomputes:
        assert ids == [11, 12, 13] + alone_ids[:len(ids) - 3], (ids, alone_ids)
    assert be.free_blocks() == 6 and not be.blocks and sorted(sched.free) == list(range(8)), (be.blocks, sched.free)


def test_fake_backend_depends_on_history():
    # the property the recompute test relies on: same last token, different earlier token -> different next token
    be = FakeBackend(max_seqs=2)
    be.reserve(0, 8); be.reserve(1, 8)
    assert be.prefill(0, [11, 12, 13]) != be.prefill(1, [11, 99, 13])


async def test_bad_token_ids_rejected():
    _, app = make()
    async with client_for(app) as c:
        for prompt in ([11, 1000], [11, -1], [11, 1.5], [11, True]):       # vocab of the fake engine is 1000
            r = await c.post("/v1/completions", json={"prompt": prompt, "max_tokens": 2})
            assert r.status_code == 400 and "token ids" in r.json()["detail"], (prompt, r.status_code, r.text)
        r = await c.post("/v1/completions", json={"prompt": [11, 999], "max_tokens": 2})
        assert r.status_code == 200, r.text


async def test_request_larger_than_kv_pool():
    _, app = make(max_len=64, max_seqs=2, block_size=4, num_blocks=4)   # 16 tokens of KV cache
    async with client_for(app) as c:
        r = await c.post("/v1/completions", json={"prompt": [11, 12, 13], "max_tokens": 14})
        assert r.status_code == 400 and "exceeds the KV cache" in r.json()["detail"], r.text
        r = await c.post("/v1/completions", json={"prompt": [11, 12, 13], "max_tokens": 13})   # 16 tokens: fits
        assert r.status_code == 200 and r.json()["usage"]["completion_tokens"] == 13, r.text


def test_detokenizer_utf8():
    class ByteTok:   # each id is one byte: a 3-byte character split over 3 tokens must not leak U+FFFD
        def decode(self, ids, skip_special_tokens=True):
            return bytes(ids).decode("utf-8", errors="replace")
    text = "ab好c"
    ids = list(text.encode("utf-8"))
    d = Detokenizer(ByteTok())
    out = [d.push(i, False) for i in ids]
    assert "".join(out) == text and not any("�" in o for o in out), out
    assert d.push(0, True) == ""   # an EOS adds no text


async def test_final_event_after_release():
    # the ORDER itself, observed at the moment the final event is handed to the client (inside the scheduler thread,
    # under its lock): the request must already be out of `running`, its slot back in `free` exactly once and its
    # blocks returned. Covers "length" ending at prefill (1 token) and in decode, "stop", and "error" with two
    # requests in flight (both released, and the scheduler still serves the next request).
    seen, orig_emit = [], GenRequest.emit
    def spy(req, tok, finish):
        if finish is not None:
            seen.append((finish, req in sched.running, sched.free.count(req.slot), be.blocks.get(req.slot, 0)))
        orig_emit(req, tok, finish)
    GenRequest.emit = spy
    try:
        for max_new, eos_after, want in ((1, None, "length"), (4, None, "length"), (4, 2, "stop")):
            seen.clear()
            be, app = make(eos_after=eos_after, max_seqs=2, block_size=4, num_blocks=8)
            sched = app.state.scheduler
            out = [t async for t in sched.tokens([11, 12, 13], max_new, 0, eos_after is None)]
            assert out[-1][1] == want and seen == [(want, False, 1, 0)], (max_new, want, out, seen)
        seen.clear()
        be, app = make(max_seqs=2, delay=0.002, block_size=4, num_blocks=8)
        sched, good = app.state.scheduler, be.decode_batch
        calls = []
        def broken(*a):
            calls.append(len(a[0]))
            if len(a[0]) == 2:                  # fail once both requests are in the same decode step
                raise RuntimeError("simulated engine failure")
            return good(*a)
        be.decode_batch = broken
        res = await asyncio.gather(*[collect(sched.tokens([11, 12, 13], 8, 0, True)) for _ in range(2)])
        assert all(r[-1][1] == "error" for r in res), res
        assert seen == [("error", False, 1, 0)] * 2, seen
        be.decode_batch = good
        out = [t async for t in sched.tokens([11, 12, 13], 3, 0, True)]
        assert out[-1][1] == "length" and sorted(sched.free) == [0, 1] and not be.blocks, (out, sched.free, be.blocks)
    finally:
        GenRequest.emit = orig_emit


async def collect(gen):
    return [t async for t in gen]


def test_emit_to_closed_loop_does_not_raise():
    # the scheduler's error cleanup emits to every failed request; one whose event loop is gone must not stop it
    loop = asyncio.new_event_loop()
    req = GenRequest([1], 1, 0, True, loop)
    loop.close()
    req.emit(-1, "error")
    assert req.cancelled


async def main():
    tests = [test_health_models, test_chat_stream_exact_length_and_usage, test_eos_stops_without_ignore_eos,
             test_max_completion_tokens_wins_and_non_stream, test_completions_string_and_ids, test_rejections,
             test_requests_are_batched, test_more_requests_than_slots, test_disconnect_frees_slot,
             test_engine_error_fails_only_that_request, test_paged_blocks_admit_more_than_contiguous,
             test_preemption_by_recompute, test_request_larger_than_kv_pool, test_bad_token_ids_rejected,
             test_final_event_after_release]
    for t in tests:
        await t()
        print("ok  ", t.__name__)
    test_detokenizer_utf8()
    print("ok   test_detokenizer_utf8")
    test_fake_backend_depends_on_history()
    print("ok   test_fake_backend_depends_on_history")
    test_emit_to_closed_loop_does_not_raise()
    print("ok   test_emit_to_closed_loop_does_not_raise")
    print("ALL SERVER TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
