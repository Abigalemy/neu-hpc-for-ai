"""Check tests/moe_layer_test.cu (one real Qwen3-30B-A3B MoE layer on the GPU) against numpy.

  python tools/moe_layer_check.py <model_dir> <layer> <tokens> <dir with x.bin idx.bin w.bin y.bin>

The router, the 128 experts of the layer and the HF semantics (bf16 logits, fp32 softmax, top-8, renormalise; each
expert with the bf16 roundings of HF's SwiGLU) are recomputed here from the safetensors files themselves.
  routing: per token the same 8 experts; a different choice is accepted only as a tie, i.e. the swapped experts' fp32
           logits differ by at most 2 bf16 ulps (the GPU and numpy can round a logit to different bf16 neighbours)
  weights: the bf16 rule of run_tests.compare, for the experts both chose
  y:       compare_moe (run_tests.py) with the terms of the experts the GPU chose
Exit code 0 = pass.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_tests import _bf16, _expert_out, compare, compare_moe  # noqa: E402


def tensors(model_dir, names):
    """bf16 tensors by name, read with numpy from the safetensors shards (as fp32)."""
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    out, by_file = {}, {}
    for n in names:
        by_file.setdefault(index[n], []).append(n)
    for f, ns in by_file.items():
        path = os.path.join(model_dir, f)
        with open(path, "rb") as fh:
            hlen = int.from_bytes(fh.read(8), "little")
            header = json.loads(fh.read(hlen))
        mm = np.memmap(path, dtype=np.uint8, mode="r")
        for n in ns:
            meta = header[n]
            assert meta["dtype"] == "BF16", (n, meta["dtype"])
            b, e = meta["data_offsets"]
            raw = np.frombuffer(mm[8 + hlen + b: 8 + hlen + e].tobytes(), dtype=np.uint16)
            out[n] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(meta["shape"])
    return out


def bf16_ulp(v):
    """Spacing of bf16 numbers at |v| (8 significant bits)."""
    v = np.maximum(np.abs(v), 1e-30)
    return 2.0 ** (np.floor(np.log2(v)) - 7)


def main(model_dir, layer, T, d):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    E, K, H, I = cfg["num_experts"], cfg["num_experts_per_tok"], cfg["hidden_size"], cfg["moe_intermediate_size"]
    pre = f"model.layers.{layer}.mlp."
    names = [pre + "gate.weight"] + [f"{pre}experts.{e}.{w}_proj.weight" for e in range(E) for w in ("gate", "up", "down")]
    W = tensors(model_dir, names)
    wr = W[pre + "gate.weight"]
    wg = np.stack([W[f"{pre}experts.{e}.gate_proj.weight"] for e in range(E)])
    wu = np.stack([W[f"{pre}experts.{e}.up_proj.weight"] for e in range(E)])
    wd = np.stack([W[f"{pre}experts.{e}.down_proj.weight"] for e in range(E)])
    rd16 = lambda f, shape: (np.fromfile(os.path.join(d, f), np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
    x = rd16("x.bin", (T, H))
    y = rd16("y.bin", (T, H))
    gidx = np.fromfile(os.path.join(d, "idx.bin"), np.int32).reshape(T, K)
    gw = rd16("w.bin", (T, K))

    l32 = x @ wr.T                                   # fp32 logits; HF rounds them to bf16 before the softmax
    lg = _bf16(l32)
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    ridx = np.argsort(-p, axis=1, kind="stable")[:, :K]
    rw = np.take_along_axis(p, ridx, 1); rw = _bf16(rw / rw.sum(1, keepdims=True))

    bad = ties = 0
    w_ok = True
    for t in range(T):
        g, r = set(gidx[t].tolist()), set(ridx[t].tolist())
        if len(g) != K or not all(0 <= e < E for e in g):
            bad += 1
            continue
        if g != r:
            if len(g - r) == len(r - g):
                sw = list(g ^ r)
                spread = l32[t, sw].max() - l32[t, sw].min()
                if spread <= 2 * bf16_ulp(l32[t, sw]).max():
                    ties += 1
                else:
                    bad += 1
                    print(f"  token {t}: experts {sorted(g - r)} instead of {sorted(r - g)}, logits {l32[t, sw]}")
                    continue
            else:
                bad += 1
                continue
        for j, e in enumerate(ridx[t]):
            if e in g:
                gj = list(gidx[t]).index(e)
                ok, _, _ = compare("bfloat16", np.array([gw[t, gj]], np.float32), np.array([rw[t, j]], np.float32))
                w_ok &= ok or g != r      # after a tie swap the renormalisation differs slightly
    # reference terms for the experts the GPU chose, grouped by expert (numpy matmuls over all its tokens)
    eo = np.zeros((T, K, H), np.float32)
    for e in range(E):
        rows = np.argwhere(gidx == e)
        if len(rows) == 0:
            continue
        xt = x[rows[:, 0]]
        gg = _bf16(xt @ wg[e].T); uu = _bf16(xt @ wu[e].T)
        eo[rows[:, 0], rows[:, 1]] = _bf16(_bf16(_bf16(gg / (1 + np.exp(-gg))) * uu) @ wd[e].T)
    # self-check of the grouped reference against the per-token one: different summation order in numpy can put an
    # fp32 value on the other side of a bf16 rounding boundary, so allow a few bf16 steps of the row's scale
    one = _expert_out(x[0], int(gidx[0, 0]), wg, wu, wd)
    dev = float(np.abs(eo[0, 0] - one).max()) / (float(np.abs(one).max()) or 1.0)
    if dev > 2e-2:
        print(f"  reference self-check: grouped vs per-token differ by {dev:.2e} of the row maximum")
        return 1
    ok_y, dmax, frac, eref = compare_moe(y, eo, gw)
    m = float(np.abs(eref).max()) or 1.0
    print(f"MOE_LAYER_CHECK layer={layer} tokens={T}: experts {T - bad - ties}/{T} same, {ties} tie (<= 2 bf16 ulps of the "
          f"logit), {bad} wrong; weights {'ok' if w_ok else 'WRONG'}; y max err {dmax / m:.2e} of max|y| = {frac:.2f} of "
          f"the bound ({'ok' if ok_y else 'FAIL'})")
    return 0 if (bad == 0 and w_ok and ok_y) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
