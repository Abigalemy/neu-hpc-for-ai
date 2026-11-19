"""Part 3 / Part 5: K data + expert parallel ranks of the server behind one HTTP endpoint, on one node.

  python part3/launch.py --world 2 --model-dir <snapshot> --lib build/libqwen3engine.so --port 8000 [server args]
  python part3/launch.py --world 3 --fake --port 8000          # CPU: fake engines, for testing

What it starts (vLLM's single-node DP + EP layout with TP = 1, reduced):
  - a coordinator (server/ep_link.py) that parks the ranks together when none has work and wakes them on a request
  - rank r = `server/server.py --rank r --world K` on GPU r only (CUDA_VISIBLE_DEVICES=r, so it is device 0 there),
    its own scheduler, KV cache and requests (data parallelism), experts [floor(r*128/K), floor((r+1)*128/K))
    (expert parallelism); the ranks talk to each other over NCCL (the NCCL id is made here and passed in the
    environment)
  - an HTTP load balancer on --port: each request goes to the next rank (round robin) and stays there to the end
  - a watchdog: if any rank exits, the whole group is stopped (the others would wait forever in a collective)
Rank r's own HTTP port is --port + 1 + r.
"""
import argparse
import itertools
import os
import secrets
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
sys.path.insert(0, os.path.join(HERE, "server"))
from ep_link import Coordinator  # noqa: E402


def nccl_unique_id(lib: str):
    """ncclGetUniqueId through the engine library, in a child process that sees no GPU (nothing allocated here).
    ncclGetUniqueId also starts NCCL's bootstrap root in the calling process, and every rank's ncclCommInitRank
    connects to it: the child must stay alive while the ranks start. It prints the id, then waits for its stdin to
    close, i.e. it lives exactly as long as the launcher. Returns (id hex, the child process)."""
    code = ("import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); n = lib.qe_nccl_id_bytes(); b = ctypes.create_string_buffer(n); "
            "assert lib.qe_nccl_unique_id(b) == 1; print(b.raw[:n].hex(), flush=True); sys.stdin.read()")
    p = subprocess.Popen([sys.executable, "-c", code, lib], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    hexid = p.stdout.readline().strip()
    if not hexid:
        raise RuntimeError(f"could not make an NCCL id (exit code {p.wait()})")
    return hexid, p


def make_balancer(ports, coord, procs):
    import httpx
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()
    rr = itertools.cycle(range(len(ports)))
    sent = [0] * len(ports)
    client = httpx.AsyncClient(timeout=None)

    async def forward(path: str, request: Request):
        body = await request.body()
        r = next(rr)
        sent[r] += 1
        url = f"http://127.0.0.1:{ports[r]}{path}"
        req = client.build_request("POST", url, content=body, headers={"content-type": "application/json"})
        resp = await client.send(req, stream=True)
        if resp.headers.get("content-type", "").startswith("text/event-stream"):
            async def relay():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()           # a client that leaves closes the rank's stream too
            return StreamingResponse(relay(), status_code=resp.status_code, media_type="text/event-stream")
        data = await resp.aread()
        await resp.aclose()
        return Response(content=data, status_code=resp.status_code, media_type=resp.headers.get("content-type"))

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await forward("/v1/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await forward("/v1/completions", request)

    @app.get("/v1/models")
    async def models():
        r = await client.get(f"http://127.0.0.1:{ports[0]}/v1/models")
        return JSONResponse(r.json(), status_code=r.status_code)

    @app.get("/health")
    async def health():
        try:
            for p in ports:
                if (await client.get(f"http://127.0.0.1:{p}/health", timeout=2)).status_code != 200:
                    return JSONResponse({"status": "starting"}, status_code=503)
        except httpx.HTTPError:
            return JSONResponse({"status": "starting"}, status_code=503)
        return {"status": "ok"}

    @app.get("/stats")
    async def stats():
        ranks = []
        for p in ports:
            try:
                ranks.append((await client.get(f"http://127.0.0.1:{p}/stats", timeout=2)).json())
            except httpx.HTTPError as e:
                ranks.append({"error": repr(e)})
        return {"requests_sent": sent, "coordinator": coord.stats, "ranks": ranks}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--lib", default=os.path.join(HERE, "build", "libqwen3engine.so"))
    a, rest = ap.parse_known_args()        # everything else goes to every rank's server.py
    if not 1 <= a.world <= 8:
        raise SystemExit("--world must be 1..8")
    key = secrets.token_bytes(16)
    coord = Coordinator(a.world, key)
    env = {**os.environ, "QWEN3_COORD_KEY": key.hex()}
    id_holder = None                       # keeps NCCL's bootstrap root alive (see nccl_unique_id)
    if not a.fake:
        env["QWEN3_NCCL_ID"], id_holder = nccl_unique_id(a.lib)
    ports = [a.port + 1 + r for r in range(a.world)]
    procs = []
    for r in range(a.world):
        cmd = [sys.executable, os.path.join(HERE, "server", "server.py"), "--rank", str(r), "--world", str(a.world),
               "--coord", f"{coord.address[0]}:{coord.address[1]}", "--port", str(ports[r])] + rest
        if a.fake:
            cmd.append("--fake")
        else:
            cmd += ["--lib", a.lib]
        procs.append(subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(r)}))
        print(f"rank {r}: pid {procs[-1].pid}, GPU {r}, port {ports[r]}", flush=True)

    def stop_all(why: str):
        print(f"launcher: {why}; stopping all ranks", flush=True)
        for p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 10
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        if id_holder is not None:
            id_holder.kill()
        os._exit(1)

    def watchdog():
        while True:
            for r, p in enumerate(procs):
                if p.poll() is not None:
                    stop_all(f"rank {r} exited with code {p.returncode}")
            if coord.error:
                stop_all(coord.error)
            if id_holder is not None and id_holder.poll() is not None:
                stop_all(f"the NCCL id process exited with code {id_holder.returncode}")
            time.sleep(0.2)

    threading.Thread(target=watchdog, daemon=True).start()
    coord.accept_all()                                  # every rank connects first, then the coordinator answers
    threading.Thread(target=coord.serve, daemon=True).start()
    print(f"launcher: {a.world} ranks connected; load balancer on port {a.port}", flush=True)
    import uvicorn
    try:
        uvicorn.run(make_balancer(ports, coord, procs), host="0.0.0.0", port=a.port, log_level="warning")
    finally:
        stop_all("load balancer stopped")


if __name__ == "__main__":
    main()
