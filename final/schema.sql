-- Reference test cases for Qwen3 modules (forward inference only).
-- PyTorch/HF generates them; the CUDA Tile implementation is checked against them.
-- Configuration and scalar arguments are relational columns (no JSON); tensors are raw little-endian bytes.
-- bfloat16 tensors are stored as their 16-bit pattern (int16), float32 as float32. Fixed comparison tolerances live
-- in the runner (run_tests.py), not in the database.

-- One initialized module (e.g. one RMSNorm with its random weights). Many cases may share it.
CREATE TABLE module (
    id    INTEGER PRIMARY KEY,
    kind  TEXT NOT NULL,          -- 'rmsnorm', 'rope', 'mlp', ...
    dtype TEXT NOT NULL           -- 'float32' | 'bfloat16' : dtype of weights and activations
);

-- Constructor arguments of the module (hidden_size, eps, ...). Exactly one of the value columns is set.
CREATE TABLE module_param (
    module_id  INTEGER NOT NULL REFERENCES module(id),
    name       TEXT NOT NULL,
    int_value  INTEGER,
    real_value REAL,
    PRIMARY KEY (module_id, name)
);

-- Saved weights (random for synthetic cases, copied from the checkpoint for real-weight cases).
CREATE TABLE module_tensor (
    module_id INTEGER NOT NULL REFERENCES module(id),
    name      TEXT NOT NULL,      -- 'weight', 'gate_proj', ...
    dtype     TEXT NOT NULL,
    shape     TEXT NOT NULL,      -- e.g. '2560' or '9728x2560'
    data      BLOB NOT NULL,
    PRIMARY KEY (module_id, name)
);

-- One forward call of a module.
CREATE TABLE test_case (
    id        INTEGER PRIMARY KEY,
    module_id INTEGER NOT NULL REFERENCES module(id),
    label     TEXT NOT NULL       -- human-readable: 'normal_t3', 'large_t33', ...
);

-- Scalar arguments of the forward call (e.g. start position). Exactly one value column is set.
CREATE TABLE case_param (
    case_id    INTEGER NOT NULL REFERENCES test_case(id),
    name       TEXT NOT NULL,
    int_value  INTEGER,
    real_value REAL,
    PRIMARY KEY (case_id, name)
);

-- Inputs and expected outputs of the call.
CREATE TABLE case_tensor (
    case_id INTEGER NOT NULL REFERENCES test_case(id),
    role    TEXT NOT NULL CHECK (role IN ('input', 'expected')),
    name    TEXT NOT NULL,
    dtype   TEXT NOT NULL,
    shape   TEXT NOT NULL,
    data    BLOB NOT NULL,
    PRIMARY KEY (case_id, role, name)
);
