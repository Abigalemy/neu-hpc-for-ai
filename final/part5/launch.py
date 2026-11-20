"""Part 5: vLLM's single-node multi-process layout for DP = EP = K, TP = 1 (design: part5/DESIGN.md).

  python part5/launch.py --world 2 --model-dir <snapshot> --lib build/libqwen3engine.so --port 8000 [server args]
  python part5/launch.py --world 3 --fake --port 8000          # CPU: fake engines, for testing

Processes (the Week 9 table, with TP = 1):
  - HTTP load balancer (this process, on --port): round robin over the K API servers
  - K API servers (server/api_server.py, CPU, port --port + 1 + i): HTTP, tokenizer, streaming; ZMQ to every engine core
  - K engine cores (server/engine_core.py, CPU): scheduler and KV cache management; each starts its GPU worker
  - K GPU workers (server/gpu_worker.py, one per GPU: CUDA_VISIBLE_DEVICES = r): weights and forward passes; the K
    workers run the MoE layers together over NCCL (Part 3)
  - the DP coordinator (server/ep_link.py, a thread here): park / wake of the engine cores (MoE lock step) and the load
    table the API servers route by. It also runs for K = 1, where it only collects statistics.
A watchdog stops everything if any process exits (an engine core exits when its worker dies, and a worker when its
engine core dies).
"""
import argparse
import os
import secrets
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # final/
sys.path.insert(0, os.path.join(HERE, "server"))
from ep_link import Coordinator  # noqa: E402


def _part3_launch():
    """part3/launch.py (its balancer and NCCL id helper), loaded by path: its module name equals this file's."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("part3_launch", os.path.join(HERE, "part3", "launch.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_p3 = _part3_launch()
make_balancer, nccl_unique_id = _p3.make_balancer, _p3.nccl_unique_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--lib", default=os.path.join(HERE, "build", "libqwen3engine.so"))
    a, rest = ap.parse_known_args()        # everything else goes to every API server and engine core
    if not 1 <= a.world <= 8:
        raise SystemExit("--world must be 1..8")
    key = secrets.token_bytes(16)
    roles = ("core", "api") + (("worker",) if a.fake and a.world > 1 else ())
    coord = Coordinator(a.world, key, roles=roles)
    caddr = f"{coord.address[0]}:{coord.address[1]}"
    ipc = tempfile.mkdtemp(prefix="qwen3-zmq-")
    env = {**os.environ, "QWEN3_COORD_KEY": key.hex()}
    id_holder = None                       # keeps NCCL's bootstrap root alive (see part3/launch.py)
    if not a.fake and a.world > 1:
        env["QWEN3_NCCL_ID"], id_holder = nccl_unique_id(a.lib)
    common = ["--world", str(a.world), "--ipc", ipc, "--coord", caddr] + rest + (["--fake"] if a.fake else ["--lib", a.lib])
    ports = [a.port + 1 + i for i in range(a.world)]
    procs = []                             # (name, Popen)
    for r in range(a.world):
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "server", "engine_core.py"), "--rank", str(r)] + common,
                             env={**env, "CUDA_VISIBLE_DEVICES": str(r)})
        procs.append((f"engine core {r}", p))
    for i in range(a.world):
        p = subprocess.Popen([sys.executable, os.path.join(HERE, "server", "api_server.py"), "--api", str(i),
                              "--port", str(ports[i])] + common, env={**env, "CUDA_VISIBLE_DEVICES": ""})
        procs.append((f"api server {i}", p))
    for name, p in procs:
        print(f"{name}: pid {p.pid}", flush=True)

    def stop_all(why: str):
        print(f"launcher: {why}; stopping all processes", flush=True)
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        deadline = time.time() + 10
        for _, p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                p.kill()
        if id_holder is not None:
            id_holder.kill()
        os._exit(1)

    def watchdog():
        while True:
            for name, p in procs:
                if p.poll() is not None:
                    stop_all(f"{name} exited with code {p.returncode}")
            if coord.error:
                stop_all(coord.error)
            if id_holder is not None and id_holder.poll() is not None:
                stop_all(f"the NCCL id process exited with code {id_holder.returncode}")
            time.sleep(0.2)

    threading.Thread(target=watchdog, daemon=True).start()
    coord.accept_all()
    threading.Thread(target=coord.serve, daemon=True).start()
    print(f"launcher: {a.world} API servers, {a.world} engine cores connected; load balancer on port {a.port}",
          flush=True)
    import uvicorn
    try:
        uvicorn.run(make_balancer(ports, coord, [p for _, p in procs]), host="0.0.0.0", port=a.port, log_level="warning")
    finally:
        stop_all("load balancer stopped")


if __name__ == "__main__":
    main()
