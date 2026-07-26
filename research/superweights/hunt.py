#!/usr/bin/env python3
"""
RANGER Experiment D — the super-weight hunt.

THE claim under test (Super Weight, arXiv 2411.07191, + report §3.9): a literal
HANDFUL of individual weights — not 0.5% ≈ hundreds of thousands — create the
massive-activation pathway, and protecting just those few during low-bit
quantization recovers most of the benefit of blanket outlier protection at
~1000x less full-precision storage.

Method (paper's recipe): super weights live in mlp.down_proj and betray
themselves through activation spikes. If output channel j spikes massively at
some token, and input channel i spikes at the same token, then y_j ≈ W[j,i]·x_i
is dominated by ONE weight: W[j,i]. Detect it, zero it, re-run, repeat — the
spike decay curve shows how many真 super weights the model has.

Three-part protocol:
  1. HUNT     iterative spike-tracing over all down_proj layers -> coordinates,
              values, and the dominance ratio |W[j,i]·x_i| / |y_j| per find.
  2. ABLATE   zeroing the found super weights should crater PPL; zeroing the
              same NUMBER of random weights should do nothing (control).
  3. PROTECT  3-bit RTN with {none, top-3 SW, all-found SW, 0.5% blanket}
              exact-fp protection at equal quant settings. The falsifiable
              claim: all-found (~10 weights) captures most of blanket's
              (~hundreds of thousands) benefit.

Usage:
    python hunt.py --model HuggingFaceTB/SmolLM2-135M
    python hunt.py --selftest            # network-free, with a PLANTED super weight
"""
import argparse, copy, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import load, wikitext_ppl, quantize_weight, linears     # noqa: E402


# ------------------------------------------------------------------ hunt
@torch.no_grad()
def spike_scan(model, ids):
    """One forward pass; returns per-down_proj-layer spike info:
    (layer_name, out_spike, out_ch, in_ch, token_idx, dominance)."""
    caps, handles = {}, []
    targets = [(n, m) for n, m in linears(model) if "down_proj" in n]
    for n, m in targets:
        def mk(name):
            def hook(mod, inp, out):
                caps[name] = (inp[0].detach(), out.detach())
            return hook
        handles.append(m.register_forward_hook(mk(n)))
    model(ids)
    for h in handles:
        h.remove()

    rows = []
    for n, m in targets:
        x, y = caps[n]                       # [1, T, in], [1, T, out]
        x, y = x[0].float(), y[0].float()
        flat = y.abs().argmax()
        t, j = int(flat // y.shape[1]), int(flat % y.shape[1])
        out_spike = float(y[t, j].abs())
        i = int(x[t].abs().argmax())
        contrib = float((m.weight[j, i] * x[t, i]).abs())
        dominance = contrib / max(out_spike, 1e-9)
        rows.append(dict(layer=n, out_spike=out_spike, out_ch=j, in_ch=i,
                         token=t, dominance=dominance,
                         weight_val=float(m.weight[j, i])))
    return sorted(rows, key=lambda r: -r["out_spike"])


@torch.no_grad()
def hunt(model, ids, rounds=10):
    """Iterative detection: find top spike, zero its weight, repeat.
    Restores all weights afterwards. Returns the find list (spike-decay order)."""
    saved = {}
    finds = []
    mods = dict(linears(model))
    for r in range(rounds):
        top = spike_scan(model, ids)[0]
        m = mods[top["layer"]]
        key = (top["layer"], top["out_ch"], top["in_ch"])
        if key in saved:                      # same coord re-found -> stop
            break
        saved[key] = m.weight.data[top["out_ch"], top["in_ch"]].clone()
        finds.append(top)
        m.weight.data[top["out_ch"], top["in_ch"]] = 0.0
    for (name, j, i), v in saved.items():     # restore
        mods[name].weight.data[j, i] = v
    return finds


# ------------------------------------------------------------------ ablate
@torch.no_grad()
def ablation_test(model, finds, eval_fn, n_random_controls=None, seed=0):
    """PPL with super weights zeroed vs same-count random weights zeroed."""
    mods = dict(linears(model))
    base = eval_fn(model)

    saved = []
    for f in finds:
        m = mods[f["layer"]]
        saved.append((m, f["out_ch"], f["in_ch"],
                      m.weight.data[f["out_ch"], f["in_ch"]].clone()))
        m.weight.data[f["out_ch"], f["in_ch"]] = 0.0
    ppl_sw = eval_fn(model)
    for m, j, i, v in saved:
        m.weight.data[j, i] = v

    rng = np.random.default_rng(seed)
    saved = []
    names = list(mods)
    for _ in range(n_random_controls or len(finds)):
        m = mods[names[rng.integers(len(names))]]
        j, i = int(rng.integers(m.weight.shape[0])), int(rng.integers(m.weight.shape[1]))
        saved.append((m, j, i, m.weight.data[j, i].clone()))
        m.weight.data[j, i] = 0.0
    ppl_rand = eval_fn(model)
    for m, j, i, v in saved:
        m.weight.data[j, i] = v
    return dict(base=base, sw_zeroed=ppl_sw, random_zeroed=ppl_rand)


# ------------------------------------------------------------------ protect
@torch.no_grad()
def protection_test(model, finds, eval_fn, bits=3, blanket_pct=0.005):
    """3-bit RTN with different exact-fp protection sets, at equal settings."""
    fp_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    mods = dict(linears(model))

    def quantize_all():
        for _, m in linears(model):
            m.weight.data = quantize_weight(m.weight.data, bits, per="out")\
                            .to(m.weight.dtype)

    def restore():
        model.load_state_dict(fp_state)

    out = {}
    arms = {"none": [], "sw_top3": finds[:3], "sw_all": finds}
    for arm, fl in arms.items():
        restore()
        originals = [(mods[f["layer"]], f["out_ch"], f["in_ch"],
                      mods[f["layer"]].weight.data[f["out_ch"], f["in_ch"]].clone())
                     for f in fl]
        quantize_all()
        for m, j, i, v in originals:          # restore exact fp super weights
            m.weight.data[j, i] = v
        out[arm] = {"ppl": eval_fn(model), "n_fp_weights": len(fl)}

    restore()                                  # blanket: top-pct |W| per layer
    n_blanket = 0
    keep = []
    for _, m in linears(model):
        W = m.weight.data
        k = max(1, int(blanket_pct * W.numel()))
        thr = torch.kthvalue(W.abs().flatten(), W.numel() - k + 1).values
        mask = W.abs() >= thr
        keep.append((m, mask, W[mask].clone()))
        n_blanket += int(mask.sum())
    quantize_all()
    for m, mask, vals in keep:
        m.weight.data[mask] = vals
    out["blanket_0.5pct"] = {"ppl": eval_fn(model), "n_fp_weights": n_blanket}
    restore()
    return out


# ------------------------------------------------------------------ selftest
def selftest():
    """Network-free, with GROUND TRUTH: plant a super weight at known
    coordinates in a tiny random model and verify the hunt finds exactly it,
    the ablation shows it matters, and all protection arms run."""
    from transformers import LlamaConfig, LlamaForCausalLM
    print("SELFTEST — tiny random model with a PLANTED super weight\n")
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).eval()
    PLANT = ("model.layers.1.mlp.down_proj", 5, 7)
    dict(linears(model))[PLANT[0]].weight.data[PLANT[1], PLANT[2]] = 8.0
    ids = torch.randint(0, 512, (1, 128))

    print("1. hunt finds the planted coordinate first")
    finds = hunt(model, ids, rounds=4)
    f0 = finds[0]
    got = (f0["layer"], f0["out_ch"], f0["in_ch"])
    print(f"   planted {PLANT} -> found {got}  "
          f"(spike {f0['out_spike']:.1f}, dominance {f0['dominance']:.2f})")
    assert got == PLANT, "hunt missed the planted super weight"
    assert finds[0]["out_spike"] > 2 * finds[1]["out_spike"], "no spike decay"

    print("2. ablation: zeroing it moves the loss; random zeroing doesn't")
    def eval_fn(m):
        with torch.no_grad():
            return float(m(ids, labels=ids).loss)
    ab = ablation_test(model, finds[:1], eval_fn)
    d_sw = abs(ab["sw_zeroed"] - ab["base"])
    d_rand = abs(ab["random_zeroed"] - ab["base"])
    print(f"   loss delta  sw {d_sw:.4f}  vs random {d_rand:.4f}")
    assert d_sw > 5 * max(d_rand, 1e-6), "planted SW ablation not distinguishable"

    print("3. protection arms all run at 3-bit")
    prot = protection_test(model, finds[:1], eval_fn, bits=3)
    for arm, r in prot.items():
        print(f"   {arm:14s}: loss {r['ppl']:.4f}  fp-weights {r['n_fp_weights']}")
    assert set(prot) == {"none", "sw_top3", "sw_all", "blanket_0.5pct"}

    print("\nSELFTEST OK — hunt/ablate/protect validated with ground truth. "
          "Real coordinates need real weights.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--out", default="superweights.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    print(f"loading {args.model} ...")
    model, tok, device = load(args.model, dtype=torch.float32)
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[:, :512].to(device)
    eval_fn = lambda m: wikitext_ppl(m, tok, device, max_windows=args.windows)

    print(f"\n=== 1. HUNT ({args.rounds} rounds) ===")
    finds = hunt(model, ids, rounds=args.rounds)
    print(f"  {'#':>2s} {'layer':40s} {'coord':>12s} {'spike':>9s} {'dom':>5s} {'W':>8s}")
    for k, f in enumerate(finds):
        print(f"  {k:2d} {f['layer'][-40:]:40s} ({f['out_ch']:4d},{f['in_ch']:4d}) "
              f"{f['out_spike']:9.1f} {f['dominance']:5.2f} {f['weight_val']:8.4f}")
    print("  (spike-decay elbow = how many TRUE super weights this model has; "
        "dominance ≈1 means one weight explains the spike)")

    print(f"\n=== 2. ABLATE ===")
    ab = ablation_test(model, finds, eval_fn)
    print(f"  fp PPL {ab['base']:.2f} | SW-zeroed {ab['sw_zeroed']:.2f} | "
          f"{len(finds)} random-zeroed {ab['random_zeroed']:.2f}")
    catastrophic = ab["sw_zeroed"] > 2 * ab["base"] and \
        ab["random_zeroed"] < 1.1 * ab["base"]
    print(f"  verdict: {'CONFIRMED — a handful of weights are load-bearing' if catastrophic else 'weak effect — SW story does not hold for this model'}")

    print(f"\n=== 3. PROTECT (W{args.bits} RTN) ===")
    prot = protection_test(model, finds, eval_fn, bits=args.bits)
    ppl_none = prot["none"]["ppl"]; ppl_blanket = prot["blanket_0.5pct"]["ppl"]
    for arm, r in prot.items():
        rec = ((ppl_none - r["ppl"]) / max(ppl_none - ppl_blanket, 1e-9)
               if arm != "none" else 0.0)
        print(f"  {arm:14s}: PPL {r['ppl']:8.2f} | fp weights {r['n_fp_weights']:>7d} "
              f"| {100*rec:5.1f}% of blanket's recovery")
    json.dump({"model": args.model, "finds": finds, "ablation": ab,
               "protection": {k: v for k, v in prot.items()}},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
