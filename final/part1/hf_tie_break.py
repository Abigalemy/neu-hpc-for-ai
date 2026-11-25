"""HF reference with the engine's tie rule in the MoE router: among experts with equal router probabilities, the lower
expert id is taken (the engine's moe_router_kernel does a repeated arg-max that keeps the lowest id among the maxima).
HF's Qwen3MoeTopKRouter uses torch.topk, which does not specify the order of equal values, and the router logits are
rounded to bf16 before the softmax, so equal values among the 128 experts are common (Part 1: every primary routing
difference was such a tie). The patch replaces only the top-k: a stable descending sort keeps equal values in index
order, so its first K entries are torch.topk's values with ties broken towards the lower id; everything else is HF's
code unchanged.

  python part1/hf_tie_break.py      # local check (CPU, a few seconds): untied rows unchanged, tied rows -> lower ids
"""
import inspect

ORIGINAL_TOPK = "torch.topk(router_probs, self.top_k, dim=-1)"


def stable_topk(x, k):
    import torch
    vals, idx = torch.sort(x, dim=-1, descending=True, stable=True)
    return vals[..., :k], idx[..., :k]


def patch():
    """Patch transformers' Qwen3MoeTopKRouter.forward; stops if the installed version's router is not the code this
    patch copies (so a new transformers cannot silently change what the reference computes)."""
    import torch
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mq
    src = inspect.getsource(mq.Qwen3MoeTopKRouter.forward)
    for line in ("router_logits = F.linear(hidden_states, self.weight)",
                 "router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)",
                 ORIGINAL_TOPK, "router_top_value /= router_top_value.sum(dim=-1, keepdim=True)",
                 "router_top_value = router_top_value.to(router_logits.dtype)"):
        assert line in src, f"Qwen3MoeTopKRouter.forward changed (missing {line!r}); update part1/hf_tie_break.py"
    original = mq.Qwen3MoeTopKRouter.forward

    def forward(self, hidden_states):   # HF's forward (transformers 5.x) with torch.topk -> stable_topk
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = stable_topk(router_probs, self.top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    mq.Qwen3MoeTopKRouter.forward = forward
    return original


def self_check():
    """Runs the patched and the original router on the same inputs (bf16, 128 experts, top-8, normalised as Qwen3):
    rows without ties at the top-8 boundary must give identical outputs; rows built with ties must pick lower ids.
    Returns a list of result lines; raises AssertionError on a failure."""
    import torch
    from transformers.models.qwen3_moe import modeling_qwen3_moe as mq
    from transformers import Qwen3MoeConfig
    torch.manual_seed(0)
    E, K, H = 128, 8, 64
    cfg = Qwen3MoeConfig(num_experts=E, num_experts_per_tok=K, hidden_size=H, norm_topk_prob=True)
    r = mq.Qwen3MoeTopKRouter(cfg).to(torch.bfloat16)
    with torch.no_grad():
        r.weight.normal_(0, 0.5)
    x = torch.randn(4096, H, dtype=torch.bfloat16)
    original = patch()
    with torch.no_grad():
        lp, wp, ip = r(x)
        lo, wo, io = original(r, x)
        # rows with a tie across the top-K boundary (the K-th and (K+1)-th probabilities equal): there the two may differ
        p = torch.softmax(lo, dtype=torch.float, dim=-1)
        top = torch.sort(p, dim=-1, descending=True).values
        boundary_tie = top[:, K - 1] == top[:, K]
        # compare (expert id, weight) pairs: order each row by expert id and carry the weights along, so a weight
        # attached to the wrong expert cannot pass as the same set of ids plus the same set of weights
        ip_s, op = torch.sort(ip, -1)
        io_s, oo = torch.sort(io, -1)
        same_set = (ip_s == io_s).all(-1)
        same_w = (torch.gather(wp, -1, op) == torch.gather(wo, -1, oo)).all(-1)
        assert torch.equal(lp, lo)
        assert same_set[~boundary_tie].all(), "patched router chose a different expert set on a row without a tie"
        assert same_w[~boundary_tie].all(), "patched router gave an expert a different weight on a row without a tie"
        lines = [f"{len(x)} random rows: {int(boundary_tie.sum())} with a tie at the top-{K} boundary; every other row "
                 f"takes the same {K} experts as torch.topk, each with the same weight"]
        # built ties: a router whose rows for experts 5, 9, 40, 77, 100 are identical gives those experts equal logits
        w = r.weight.clone()
        for e in (9, 40, 77, 100):
            r.weight[e] = w[5]
        r.weight[5] *= 4                                  # make the tied group large so it reaches the top-8
        for e in (9, 40, 77, 100):
            r.weight[e] = r.weight[5]
        lg, _, it = r(x[:256])
        tied = [5, 9, 40, 77, 100]
        rows = 0
        for t in range(256):
            chosen = set(it[t].tolist())
            n_in = len(chosen & set(tied))
            if 0 < n_in < len(tied):                       # part of the tied group fits: it must be the lowest ids
                assert chosen & set(tied) == set(tied[:n_in]), (t, sorted(chosen), tied)
                rows += 1
            assert all(lg[t, e] == lg[t, 5] for e in tied)
        assert rows > 0, "no row cut the tied group; the built case tests nothing"
        lines.append(f"built ties (experts {tied} with equal logits): {rows} of 256 rows split the group at the top-{K} "
                     f"boundary, each keeps the lowest ids")
    mq.Qwen3MoeTopKRouter.forward = original
    return lines


if __name__ == "__main__":
    import transformers
    print(f"transformers {transformers.__version__}")
    for line in self_check():
        print(line)
    print("TIE BREAK CHECK PASSED")
