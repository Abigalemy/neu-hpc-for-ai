"""Generate deterministic, end-to-end Qwen3 prefill cases for the SQLite suite.

The tiny model keeps the reference run practical on CPU. Its complete state dict,
inputs, config, sequence boundaries, and reference logits are stored so the same
forward pass can be replayed by the CUDA Tile verification script.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hpc_core.kv_cache import NoCacheBatch
from hpc_core.model.qwen3_model import Qwen3Config, Qwen3ForCausalLM


DB_PATH = Path(__file__).with_name("test_cases.db")
SEED = 20251104
CONFIG = {
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 32,
    "tie_word_embeddings": True,
}
INPUT_IDS = [1, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16, 18]
SEQ_LENS = [7, 9]


def main() -> None:
    torch.manual_seed(SEED)
    cfg = Qwen3Config.from_dict(CONFIG)
    model = Qwen3ForCausalLM(cfg, max_positions=32).cpu().eval()
    model.build_rope_cache("cpu", torch.float32)
    state = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    ids = torch.tensor(INPUT_IDS, dtype=torch.long)
    positions = torch.cat([torch.arange(n, dtype=torch.long) for n in SEQ_LENS])
    cu = [0]
    for n in SEQ_LENS:
        cu.append(cu[-1] + n)
    with torch.inference_mode():
        expected = model(ids, positions, NoCacheBatch(cu, "cpu")).cpu().numpy()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS qwen3_e2e_cases")
        conn.execute("""CREATE TABLE qwen3_e2e_cases (
            id INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            config_json TEXT NOT NULL,
            input_ids_json TEXT NOT NULL,
            seq_lens_json TEXT NOT NULL,
            state_dict_pickle BLOB NOT NULL,
            expected_logits BLOB NOT NULL,
            logits_rows INTEGER NOT NULL,
            vocab_size INTEGER NOT NULL
        )""")
        conn.execute(
            "INSERT INTO qwen3_e2e_cases VALUES (1,?,?,?,?,?,?,?,?)",
            (SEED, json.dumps(CONFIG, sort_keys=True), json.dumps(INPUT_IDS),
             json.dumps(SEQ_LENS), pickle.dumps(state, protocol=4),
             np.asarray(expected, dtype=np.float32).tobytes(),
             expected.shape[0], expected.shape[1]),
        )
    print(f"Wrote 1 reproducible Qwen3 end-to-end case to {DB_PATH}; logits={expected.shape}")


if __name__ == "__main__":
    main()
