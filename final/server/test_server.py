"""Local tests of the HTTP server with the fake engine and fake tokenizers (no GPU, no model download).

  python test_server.py
"""
import asyncio
import json
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import Detokenizer, FakeBackend, FakeTokenizer, create_app  # noqa: E402

TOK = FakeTokenizer()


def make(eos_after=None, max_len=4096, max_seqs=4, delay=0.0):
    be = FakeBackend(max_len=max_len, eos=TOK.eos_token_id, eos_after=eos_after, max_seqs=max_seqs, delay=delay)
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


async def main():
    tests = [test_health_models, test_chat_stream_exact_length_and_usage, test_eos_stops_without_ignore_eos,
             test_max_completion_tokens_wins_and_non_stream, test_completions_string_and_ids, test_rejections,
             test_requests_are_batched, test_more_requests_than_slots, test_disconnect_frees_slot,
             test_engine_error_fails_only_that_request]
    for t in tests:
        await t()
        print("ok  ", t.__name__)
    test_detokenizer_utf8()
    print("ok   test_detokenizer_utf8")
    print("ALL SERVER TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
