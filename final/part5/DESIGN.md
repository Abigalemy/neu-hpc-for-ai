# Part 5: vLLM's single-node multi-process architecture (DP = EP = K, TP = 1)

Sources:
- the Week 9 page ("vLLM Architecture": process types and counts, from the vLLM Developer Guide);
- the course walkthrough https://github.com/suhabe/vllm-zmq-moe (`moe_cutile_nccl_demo.py` and the ZMQ lessons in
  `educational/`).

## What vLLM runs (from the Week 9 page)

| Process | Count | Job |
|---|---|---|
| API server (CPU) | DP | HTTP (OpenAI API), tokenization, streaming results back. Talks to the engine cores over ZMQ; **many-to-many**: every API server connects to every engine core and can send a request to any of them |
| Engine core (CPU) | DP | Scheduler and KV cache management; a busy loop that schedules requests and dispatches work to its GPU worker |
| GPU worker | N = DP x PP x TP (= K here) | One per GPU: loads the weights, runs the forward passes, owns the GPU memory; talks only to the engine core that owns it |
| DP coordinator (CPU) | 1 if DP > 1 | Load balancing across DP ranks, and synchronized forward passes for MoE models |

The assignment adds an HTTP load balancer in front of the K API servers.

## How the course demo wires it (`moe_cutile_nccl_demo.py`)

- API server: ZMQ `ROUTER` (bound, `ipc://`) for requests and `PULL` (bound) for results. It waits until every engine
  has sent `READY`, then addresses a request to one engine by its identity.
- Engine: one `DEALER` per API server (identity = its rank) for requests, and one `PUSH` per API server for results.
  A `Poller` watches all the DEALERs; `replies[request_socket]` sends each result back to the API that sent the request.
- Engine -> GPU worker: a `multiprocessing.Pipe`. The engine starts its worker as a child process.
- GPU workers: NCCL between them (dispatch / expert compute / return); nothing else talks to the GPUs.

## What we have today (Part 3) and what changes

| Today | Part 5 |
|---|---|
| `part3/launch.py`: HTTP load balancer, coordinator, K rank processes, watchdog | Same launcher, starting the new processes; the balancer now sends to the K API servers |
| `server/server.py` rank process: HTTP + tokenizer + scheduler + engine (ctypes) in ONE process | Split into three processes: API server, engine core, GPU worker |
| `server/ep_link.py` Coordinator: park / wake of the ranks (lock step for MoE) | DP coordinator: the same park / wake, now with the engine cores; plus load reports for request routing |
| A request is pinned to one rank by the balancer | The balancer picks an API server; the API server picks an engine core (any of the K) |

New processes for K ranks: 1 balancer + K API servers + K engine cores + K GPU workers + 1 coordinator (+ the small
NCCL-id process from Part 3). K = 2: 8 processes.

### API server (`server/api_server.py`, K of them)
- The FastAPI app of today's `server.py`: `/v1/completions`, `/v1/chat/completions`, `/v1/models`, `/health`, SSE.
- Tokenizer (HF `tokenizers`, no PyTorch), chat template, request validation, and the incremental detokenizer.
- ZMQ: `ROUTER` for requests to the engine cores, `PULL` for their outputs (as in the demo).
- Picks an engine core for each new request: the least loaded one, from the coordinator's load table (fallback:
  round robin).

### Engine core (`server/engine_core.py`, K of them)
- Today's `Scheduler` unchanged: continuous batching, admission by free KV blocks, preemption, the idle / park loop.
- Its `backend` becomes a proxy with the same methods (`reserve`, `release`, `free_blocks`, `prefill`,
  `decode_batch`, `idle_tick`) that forwards each call over a Pipe to its GPU worker.
- ZMQ: one `DEALER` + one `PUSH` per API server; a `Poller` thread receives ADD / ABORT and the scheduler thread sends
  OUTPUT messages back to the API that sent the request.
- Reports its load (running + waiting requests) to the coordinator.

### GPU worker (`server/gpu_worker.py`, one per GPU)
- Started by its engine core; `CUDA_VISIBLE_DEVICES = r`; loads `libqwen3engine.so` with `qe_create_ep(rank, world,
  nccl_id)`; serves the backend calls from the Pipe. NCCL with the other workers inside the engine, as in Part 3.

### DP coordinator (the Part 3 coordinator, extended)
- Park / wake as today, with the engine cores as members (an engine core with no work keeps its worker in the
  collective with `idle_tick`; when nobody has work all park; a new request wakes all).
- Collects each engine core's load and answers the API servers' load queries.

### Messages (JSON over ZMQ)
| Direction | Message |
|---|---|
| API -> engine core | `ADD {id, prompt_ids, max_new, min_new, ignore_eos}`; `ABORT {id}` (client disconnected) |
| engine core -> API | `OUTPUT {id, tokens: [...], finish: null / "stop" / "length"}`; `ERROR {id, message}` |
| engine core <-> worker | Pipe: `(method, args)` -> result |

## Decisions (with the reason)

1. **KV cache management.** In vLLM the engine core owns the block allocator and sends block tables to the worker.
   Our block allocator lives in the C engine (`qe_reserve` / `qe_release`). First version: the engine core makes
   every allocation decision (admission, preemption) and the worker executes `reserve` / `release` as commands, so
   the GPU worker never decides anything. Moving the allocator itself into the engine core would need a C API that
   takes block tables; it is listed as a follow-up, not needed for the process structure.
2. **Engine core <-> worker channel: a Pipe**, as in the course demo. One round trip per call adds tens of
   microseconds per step against a decode step of 10+ ms; measured once the split works.
3. **Requests may go to any engine core** (many-to-many, as vLLM), not only "API i -> engine i" as in the demo's
   synthetic run.

## Implementation notes (as built)
- `server/server.py` keeps the single-process server (Parts 0, 1); `create_app` now takes the scheduler as an argument,
  so the API server reuses all of the HTTP code with an engine client in place of the scheduler.
- The worker sends its free KV block count with every reply, so the engine core's statistics never touch the Pipe
  from another thread.
- The coordinator also runs for K = 1 (vLLM has none there); it then only collects the engine core's statistics.
- Every child process exits by itself when its parent is gone (`ep_link.exit_with_parent`): uvicorn re-raises SIGTERM
  after its own shutdown, so the launcher's cleanup is not guaranteed to run, and a worker must never outlive its core.

## Tests (CPU first, fake engines; then one short GPU run)
`part5/test_part5.py` (runs locally, ~30 s):
1. K = 1 and K = 3 through all the processes (fake backend).
2. K = 3: concurrent requests equal the request run alone; requests land on every
   engine core; idle -> all parked, no ticks; a request wakes the group; the client disconnects -> ABORT -> KV blocks
   back to the starting count; killing ANY process (API, engine core, worker, coordinator) stops the whole group.
3. `--compat --world 2` with the real AIPerf and the 30B tokenizer (CPU container).
4. GPU smoke test on 2 GPUs: a few requests, outputs equal to the 1-GPU server.
