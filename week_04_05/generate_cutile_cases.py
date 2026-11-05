"""Create deterministic Q/K/V cases for reproducing the CUDA Tile attention tests."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import torch
import torch.nn.functional as F


DB_PATH = Path(__file__).with_name("test_cases.db")
SEED = 20251104
CASES = (
    # Prefill: non-multiple query/key length exercises both causal and tail masks.
    (1, 4, 2, 65, 65, 32, True, 0),
    # Decode: one query at absolute position 72 attends to a longer cached prefix.
    (2, 4, 2, 1, 73, 32, True, 72),
    # Non-causal case exercises a query/key length mismatch.
    (1, 4, 2, 33, 73, 32, False, 0),
)


def tensor_blob(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def main() -> None:
    torch.manual_seed(SEED)
    rows = []
    for case_id, (batch, q_heads, kv_heads, q_len, kv_len, head_dim, causal, q_start) in enumerate(CASES, 1):
        q = torch.randn(batch, q_heads, q_len, head_dim, dtype=torch.float32)
        k = torch.randn(batch, kv_heads, kv_len, head_dim, dtype=torch.float32)
        v = torch.randn_like(k)
        mask = None
        if causal:
            q_pos = q_start + torch.arange(q_len)
            k_pos = torch.arange(kv_len)
            mask = k_pos[None, None, None, :] <= q_pos[None, None, :, None]
        expected = F.scaled_dot_product_attention(
            q, k.repeat_interleave(q_heads // kv_heads, dim=1),
            v.repeat_interleave(q_heads // kv_heads, dim=1), attn_mask=mask,
        )
        rows.append((case_id, SEED, batch, q_heads, kv_heads, q_len, kv_len,
                     head_dim, int(causal), q_start, "float32", tensor_blob(q),
                     tensor_blob(k), tensor_blob(v), tensor_blob(expected)))

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DROP TABLE IF EXISTS cutile_attention_cases")
        conn.execute("""CREATE TABLE cutile_attention_cases (
            id INTEGER PRIMARY KEY,
            seed INTEGER NOT NULL,
            batch_size INTEGER NOT NULL,
            q_heads INTEGER NOT NULL,
            kv_heads INTEGER NOT NULL,
            q_len INTEGER NOT NULL,
            kv_len INTEGER NOT NULL,
            head_dim INTEGER NOT NULL,
            causal INTEGER NOT NULL,
            q_start INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            q_data BLOB NOT NULL,
            k_data BLOB NOT NULL,
            v_data BLOB NOT NULL,
            expected_data BLOB NOT NULL
        )""")
        conn.executemany(
            "INSERT INTO cutile_attention_cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
    print(f"Wrote {len(rows)} deterministic cuTile attention cases to {DB_PATH}")


if __name__ == "__main__":
    main()
