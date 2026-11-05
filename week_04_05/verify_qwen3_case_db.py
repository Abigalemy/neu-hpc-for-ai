"""Check that the real-weight Qwen3-4B reference case is intact and replayable."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np


DB_PATH = Path(__file__).with_name("test_cases.db")


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert integrity == "ok", integrity
        row = conn.execute(
            """SELECT model_id, model_revision, model_config_json, runtime_json, input_ids_json,
                      positions_json, dtype, causal, expected_logits, logits_rows,
                      vocab_size, max_abs_diff, matches_huggingface
               FROM qwen3_4b_prefill_cases WHERE id=1"""
        ).fetchone()

    assert row is not None, "The real-weight Qwen3-4B case is missing"
    (model_id, revision, config_json, runtime_json, ids_json, positions_json, dtype, causal,
     logits_blob, rows, vocab, max_diff, matches) = row
    config = json.loads(config_json)
    runtime = json.loads(runtime_json)
    ids = json.loads(ids_json)
    positions = json.loads(positions_json)
    logits = np.frombuffer(logits_blob, dtype=np.float32)

    assert model_id == "Qwen/Qwen3-4B", model_id
    assert revision and len(revision) >= 7, revision
    assert runtime.get("torch") and runtime.get("transformers") and runtime.get("cuda_tile")
    assert dtype == "float32", dtype
    assert causal == 1
    assert len(ids) == len(positions) == rows
    assert len(logits) == rows * vocab
    assert vocab == config["vocab_size"]
    assert matches == 1 and max_diff <= 2e-4
    assert np.isfinite(logits).all()
    print(f"SQLite integrity: {integrity}")
    print(f"Model: {model_id}@{revision}")
    print(f"Runtime: {runtime}")
    print(f"Prompt tokens: {len(ids)}; reference logits: [{rows}, {vocab}]; dtype={dtype}")
    print(f"CUDA Tile vs Hugging Face max abs diff: {max_diff:.3e}; match={bool(matches)}")


if __name__ == "__main__":
    main()
