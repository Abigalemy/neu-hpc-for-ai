import modal
import os

app = modal.App("mlp-cutile-verification")
_here = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.dirname(_here)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11"
    )
    .pip_install("torch==2.10.0", "numpy", "cuda-tile==1.8.0",
                 "cuda-toolkit[tileiras,nvvm,nvcc]>=13.3,<13.6")
    .add_local_file(os.path.join(_here, "test_cases.db"), remote_path="/root/project/test_cases.db")
    .add_local_dir(os.path.join(_repo_dir, "hpc_core"), remote_path="/root/hpc_core")
)

@app.function(image=image, gpu="L4", timeout=300)
def run_database_cuda_test():
    import os
    import sqlite3
    import pickle
    import torch
    import numpy as np
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root")
    from hpc_core.kernels.cutile_ops import fused_swiglu
    from hpc_core.kernels.cutile_attention import flash_attention
    from hpc_core.engine import EngineConfig, LLMEngine, SamplingParams
    from hpc_core.kv_cache import NoCacheBatch
    from hpc_core.model.qwen3_model import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(2025)
    conn = sqlite3.connect("test_cases.db")
    cursor = conn.cursor()
    cursor.execute("SELECT weights_data, input_data, output_data FROM mlp_test_cases LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    assert row is not None, "No test cases found in test_cases.db!"
    weights_bytes, input_bytes, expected_output_bytes = row

    weights_dict = pickle.loads(weights_bytes)
    input_tensor = torch.from_numpy(np.frombuffer(input_bytes, dtype=np.float32).copy().reshape(2, 4, 256)).cuda()
    expected_output = torch.from_numpy(np.frombuffer(expected_output_bytes, dtype=np.float32).copy().reshape(2, 4, 256)).cuda()

    w_gate = weights_dict["gate_proj.weight"].cuda()
    w_up = weights_dict["up_proj.weight"].cuda()
    w_down = weights_dict["down_proj.weight"].cuda()

    gate_val = torch.matmul(input_tensor, w_gate.t())
    up_val = torch.matmul(input_tensor, w_up.t())

    fused_inter = fused_swiglu(
        gate_val.reshape(-1, gate_val.shape[-1]),
        up_val.reshape(-1, up_val.shape[-1]),
    ).reshape_as(gate_val)

    actual_output = torch.matmul(fused_inter, w_down.t())

    max_diff = torch.max(torch.abs(actual_output - expected_output)).item()
    print("Database verification max difference:", max_diff)
    assert max_diff < 1e-5, f"Verification failed with max diff {max_diff}"
    print("Test passed successfully via remote CUDA kernel against test_cases.db!")

    conn = sqlite3.connect("test_cases.db")
    attention_rows = conn.execute(
        """SELECT id, batch_size, q_heads, kv_heads, q_len, kv_len, head_dim,
                  causal, q_start, dtype, q_data, k_data, v_data, expected_data
           FROM cutile_attention_cases ORDER BY id"""
    ).fetchall()
    conn.close()
    assert attention_rows, "No deterministic cuTile attention cases found in test_cases.db"
    print("\nSQLite-reproduced CUDA Tile FlashAttention cases:")
    for (case_id, batch, q_heads, kv_heads, q_len, kv_len, head_dim,
         causal, q_start, dtype_name, q_data, k_data, v_data, expected_data) in attention_rows:
        assert dtype_name == "float32", f"unsupported SQLite case dtype: {dtype_name}"
        q = torch.from_numpy(np.frombuffer(q_data, dtype=np.float32).copy()).reshape(
            batch, q_heads, q_len, head_dim).cuda()
        k = torch.from_numpy(np.frombuffer(k_data, dtype=np.float32).copy()).reshape(
            batch, kv_heads, kv_len, head_dim).cuda()
        v = torch.from_numpy(np.frombuffer(v_data, dtype=np.float32).copy()).reshape(
            batch, kv_heads, kv_len, head_dim).cuda()
        expected = torch.from_numpy(np.frombuffer(expected_data, dtype=np.float32).copy()).reshape(
            batch, q_heads, q_len, head_dim).cuda()
        actual = flash_attention(q, k, v, causal=bool(causal), q_start=q_start)
        max_diff = (actual - expected).abs().max().item()
        print(f"  case {case_id}: B={batch}, Hq/Hkv={q_heads}/{kv_heads}, "
              f"Sq/Sk={q_len}/{kv_len}, causal={bool(causal)}, max diff={max_diff:.3e}")
        assert torch.allclose(actual, expected, atol=2e-4, rtol=2e-4), (
            f"SQLite attention case {case_id} mismatch: {max_diff}"
        )

    print("\nCUDA Tile FlashAttention checks (GQA, causal prefill/decode, partial tiles):")
    for dtype in (torch.float16, torch.bfloat16):
        cases = [(65, 65, 0, True), (1, 73, 72, True), (33, 73, 0, False)]
        for q_len, kv_len, q_start, causal in cases:
            q = torch.randn(2, 4, q_len, 128, device="cuda", dtype=dtype)
            k = torch.randn(2, 2, kv_len, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            # Exercise non-contiguous inputs as produced by the model's [T,H,D] layout.
            q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
            k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
            v = v.transpose(-1, -2).contiguous().transpose(-1, -2)

            key_pos = torch.arange(kv_len, device="cuda")
            query_pos = q_start + torch.arange(q_len, device="cuda")
            mask = (key_pos[None, None, None, :] <=
                    query_pos[None, None, :, None]) if causal else None
            ref = torch.nn.functional.scaled_dot_product_attention(
                q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1),
                attn_mask=mask,
            )
            actual = flash_attention(q, k, v, causal=causal, q_start=q_start)
            max_diff = (actual.float() - ref.float()).abs().max().item()
            tolerance = 3e-2 if dtype == torch.bfloat16 else 5e-3
            print(f"  {dtype} q={q_len}, kv={kv_len}, causal={causal}: max diff={max_diff:.3e}")
            if not torch.allclose(actual.float(), ref.float(), atol=tolerance, rtol=tolerance):
                raise AssertionError(
                    f"FlashAttention mismatch for {dtype}, q={q_len}, kv={kv_len}: {max_diff}"
                )

    print("\nShared Qwen3 model integration (Tile attention/SwiGLU + cache modes):")
    import copy
    import json
    conn = sqlite3.connect("test_cases.db")
    row = conn.execute(
        """SELECT config_json, input_ids_json, seq_lens_json, state_dict_pickle,
                  expected_logits, logits_rows, vocab_size
           FROM qwen3_e2e_cases WHERE id=1"""
    ).fetchone()
    conn.close()
    assert row is not None, "No reproducible Qwen3 end-to-end case in test_cases.db"
    config_json, ids_json, lens_json, state_blob, expected_blob, n_rows, vocab_size = row
    cfg = Qwen3Config.from_dict(json.loads(config_json))
    model_gpu = Qwen3ForCausalLM(cfg, max_positions=32).cuda().eval()
    state = pickle.loads(state_blob)
    model_gpu.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    model_gpu.cuda().eval()
    model_gpu.build_rope_cache("cuda", torch.float32)
    model_cpu = copy.deepcopy(model_gpu).cpu().eval()
    prompts = []
    offset = 0
    flat_ids = json.loads(ids_json)
    for length in json.loads(lens_json):
        prompts.append(flat_ids[offset:offset + length])
        offset += length
    packed = torch.tensor(flat_ids, dtype=torch.long)
    positions = torch.cat([torch.arange(len(p), dtype=torch.long) for p in prompts])
    cu = [0, len(prompts[0]), len(packed)]
    expected = torch.from_numpy(np.frombuffer(expected_blob, dtype=np.float32).copy())
    expected = expected.reshape(n_rows, vocab_size)
    with torch.inference_mode():
        ref_logits = model_cpu(packed, positions, NoCacheBatch(cu, "cpu"))
        tile_logits = model_gpu(
            packed.cuda(), positions.cuda(), NoCacheBatch(cu, "cuda")
        ).cpu()
    logits_diff = (tile_logits - ref_logits).abs().max().item()
    db_diff = (ref_logits - expected).abs().max().item()
    print(f"  SQLite reference logits: max diff={db_diff:.3e}")
    print(f"  CUDA Tile vs CPU logits: max diff={logits_diff:.3e}")
    assert torch.allclose(ref_logits, expected, atol=1e-6, rtol=1e-6), db_diff
    assert torch.allclose(tile_logits, expected, atol=2e-4, rtol=2e-4), logits_diff

    for mode in ("none", "contiguous", "paged"):
        def generate(model, device):
            engine = LLMEngine(
                model, cfg,
                EngineConfig(cache_mode=mode, max_num_seqs=2, max_model_len=32,
                             block_size=4, num_blocks=32, kv_cache_gb=0.1),
                eos_token_ids=[], device=device, dtype=torch.float32,
            )
            outputs = [[] for _ in prompts]
            for i, prompt in enumerate(prompts):
                engine.add_request(
                    prompt, SamplingParams(max_tokens=3, ignore_eos=True),
                    lambda token, reason, i=i: outputs[i].append(token) if token is not None else None,
                )
            while engine.has_work():
                engine.step()
            return outputs

        cpu_tokens = generate(model_cpu, "cpu")
        gpu_tokens = generate(model_gpu, "cuda")
        print(f"  {mode} greedy decode: {'PASS' if gpu_tokens == cpu_tokens else 'FAIL'}")
        assert gpu_tokens == cpu_tokens, (mode, cpu_tokens, gpu_tokens)

    from hpc_core.kernel_bench import run_kernel_matrix
    run_kernel_matrix()

@app.local_entrypoint()
def main():
    run_database_cuda_test.remote()
