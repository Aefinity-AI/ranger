#!/usr/bin/env python3
"""
RANGER Experiment A — cross-family outlier census.

THE claim under test (the spine of the theory, §4.1 of the report):
    activation-outlier structure is caused by architecture choices, and a
    5-minute kurtosis measurement predicts post-training-quantization damage
    across model families better than parameter count does.

For each model we measure:
  * architecture flags — DETECTED at runtime from the loaded modules/config
    (QK-Norm, sandwich norms, sliding window, norm class, activation), not
    hardcoded from memory;
  * activation outlier signature — per-layer residual-stream excess kurtosis,
    incoherence, crest factor; peak + emergence layer (relative depth);
  * weight kurtosis (control variable — damage should track ACTIVATION
    outliers beyond what weight statistics explain);
  * the PTQ bit cliff — WikiText-2 PPL at fp and weight-only RTN W4/W3/W2.

Then it computes Spearman rank correlations across families:
    corr(peak activation kurtosis, W3 log-damage)   <- theory says HIGH
    corr(log param count,          W3 log-damage)   <- theory says lower
and compares QK-Norm vs no-QK-Norm group means.

Fault-isolated per model (a gated/OOM model is recorded and skipped) and
resumable: results accumulate in census.json; rerun to fill gaps.

Usage:
    python census.py                       # default ungated roster
    python census.py --models Qwen/Qwen3-0.6B-Base HuggingFaceTB/SmolLM2-135M
    python census.py --selftest            # network-free pipeline validation
"""
import argparse, copy, gc, json, os, sys, traceback
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import (load, wikitext_ppl, collect_hidden_stats,   # noqa: E402
                    quantize_weight, linears, tensor_stats)

# Ungated, Apache/MIT-ish small models spanning the architecture knobs.
DEFAULT_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",    # Llama-style, no QK-Norm
    "HuggingFaceTB/SmolLM2-360M",    # same family, scale control
    "Qwen/Qwen3-0.6B-Base",          # QK-Norm
    "EleutherAI/pythia-410m",        # GPT-NeoX: LayerNorm, learned-pos era
    "allenai/OLMo-2-0425-1B",        # QK-Norm + reordered norms
]
# Worth adding manually if you have accepted their licenses on HF:
GATED_EXTRAS = ["meta-llama/Llama-3.2-1B", "google/gemma-3-1b-pt"]


def detect_arch(model):
    """Read architecture facts from the loaded model, not from memory."""
    names = [n for n, _ in model.named_modules()]
    classes = {type(m).__name__ for _, m in model.named_modules()}
    cfg = model.config
    return {
        "qk_norm": any(n.endswith("q_norm") or n.endswith("k_norm") for n in names),
        "sandwich_norm": any("pre_feedforward_layernorm" in n
                             or "post_feedforward_layernorm" in n for n in names),
        "sliding_window": bool(getattr(cfg, "sliding_window", None)),
        "norm": ("RMSNorm" if any("RMSNorm" in c for c in classes)
                 else "LayerNorm" if any("LayerNorm" in c for c in classes) else "?"),
        "hidden_act": str(getattr(cfg, "hidden_act",
                          getattr(cfg, "hidden_activation", "?"))),
        "num_layers": int(cfg.num_hidden_layers),
        "hidden_size": int(cfg.hidden_size),
    }


def weight_kurtosis(model, max_per_layer=200_000):
    """Excess kurtosis over (a sample of) all linear weights — control variable."""
    chunks = []
    g = torch.Generator().manual_seed(0)
    for _, m in linears(model):
        w = m.weight.detach().float().flatten()
        if w.numel() > max_per_layer:
            w = w[torch.randperm(w.numel(), generator=g)[:max_per_layer]]
        chunks.append(w)
    x = torch.cat(chunks)
    mu, sd = x.mean(), x.std().clamp_min(1e-9)
    return float((((x - mu) / sd) ** 4).mean() - 3.0)


def bit_cliff(model, ppl_fn, bits_list=(4, 3, 2)):
    """Weight-only RTN PPL at each bit level, restoring fp weights between."""
    fp_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    out = {"fp": ppl_fn(model)}
    for b in bits_list:
        with torch.no_grad():
            for _, m in linears(model):
                m.weight.data = quantize_weight(m.weight.data, b, per="out")\
                                .to(m.weight.dtype)
        out[f"w{b}"] = ppl_fn(model)
        model.load_state_dict(fp_state)
    return out


def census_one(model, tok, device, ppl_fn, hidden_fn):
    row = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    row["arch"] = detect_arch(model)
    stats = hidden_fn(model)                      # list over layers (+embed)
    kurts = [s["kurtosis"] for s in stats]
    kmax = int(np.argmax(kurts))
    row["act"] = {
        "peak_kurtosis": float(max(kurts)),
        "emergence_rel_depth": kmax / max(len(kurts) - 1, 1),
        "peak_incoherence": float(max(s["incoherence"] for s in stats)),
        "mean_crest": float(np.mean([s["max_abs"] / (s["rms"] + 1e-9) for s in stats])),
        "per_layer_kurtosis": [float(k) for k in kurts],
    }
    row["weight_kurtosis"] = weight_kurtosis(model)
    row["ppl"] = bit_cliff(model, ppl_fn)
    # scale-free damage: log(PPL_b / PPL_fp)
    row["damage"] = {k: float(np.log(v / row["ppl"]["fp"]))
                     for k, v in row["ppl"].items() if k != "fp"}
    return row


def spearman(a, b):
    """Rank correlation, hand-rolled (numpy only)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(census):
    ok = {k: v for k, v in census.items() if "error" not in v}
    if len(ok) < 3:
        print(f"\nonly {len(ok)} successful models — need >=3 for correlations")
        return {}
    names = list(ok)
    kurt = [ok[n]["act"]["peak_kurtosis"] for n in names]
    dmg = [ok[n]["damage"]["w3"] for n in names]
    par = [np.log(ok[n]["params_M"]) for n in names]
    wk = [ok[n]["weight_kurtosis"] for n in names]

    print("\n" + "=" * 78)
    print(f"{'model':34s} {'params':>7s} {'QKN':>4s} {'peak-kurt':>10s} "
          f"{'w3-damage':>10s} {'w2-damage':>10s}")
    print("-" * 78)
    for n in sorted(names, key=lambda n: ok[n]["act"]["peak_kurtosis"]):
        r = ok[n]
        print(f"{n.split('/')[-1]:34s} {r['params_M']:6.0f}M "
              f"{'yes' if r['arch']['qk_norm'] else 'no':>4s} "
              f"{r['act']['peak_kurtosis']:10.1f} "
              f"{r['damage']['w3']:10.3f} {r['damage'].get('w2', float('nan')):10.3f}")

    res = {
        "spearman_kurt_vs_w3damage": spearman(kurt, dmg),
        "spearman_params_vs_w3damage": spearman(par, dmg),
        "spearman_weightkurt_vs_w3damage": spearman(wk, dmg),
        "n_models": len(names),
    }
    qkn = [ok[n]["act"]["peak_kurtosis"] for n in names if ok[n]["arch"]["qk_norm"]]
    noq = [ok[n]["act"]["peak_kurtosis"] for n in names if not ok[n]["arch"]["qk_norm"]]
    if qkn and noq:
        res["mean_peak_kurt_qk_norm"] = float(np.mean(qkn))
        res["mean_peak_kurt_no_qk_norm"] = float(np.mean(noq))
    print("-" * 78)
    print(f"Spearman  peak-activation-kurtosis vs W3 damage : "
          f"{res['spearman_kurt_vs_w3damage']:+.2f}   <- theory: strongly positive")
    print(f"Spearman  log-params               vs W3 damage : "
          f"{res['spearman_params_vs_w3damage']:+.2f}   <- theory: weaker")
    print(f"Spearman  weight-kurtosis (control) vs W3 damage: "
          f"{res['spearman_weightkurt_vs_w3damage']:+.2f}")
    if qkn and noq:
        print(f"mean peak kurtosis  QK-Norm {res['mean_peak_kurt_qk_norm']:.1f}  "
              f"vs no-QK-Norm {res['mean_peak_kurt_no_qk_norm']:.1f}   "
              f"<- theory: QK-Norm lower")
    print(f"(n={len(names)} — treat correlations as directional, not conclusive)")
    return res


# --------------------------------------------------------------- selftest
def selftest():
    """Network-free: run the full pipeline on tiny random models with different
    injected outlier severities, so every code path (stats, cliff, analysis,
    json) executes. Random weights carry no real signal — this validates the
    HARNESS, not the theory."""
    from transformers import LlamaConfig, LlamaForCausalLM
    print("SELFTEST — tiny random models, synthetic outlier severities\n")
    census = {}
    for name, gain in [("tiny-mild", 2.0), ("tiny-medium", 8.0), ("tiny-wild", 30.0)]:
        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                          num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=4, max_position_embeddings=256)
        model = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():   # inject channel outliers of varying severity
            for li in [1, 2]:
                w = model.model.layers[li].mlp.down_proj.weight
                w[torch.randperm(w.shape[0])[:3], :] *= gain
        ids = torch.randint(0, 512, (1, 256))
        def ppl_fn(m):
            with torch.no_grad():
                return float(torch.exp(m(ids, labels=ids).loss))
        def hidden_fn(m):
            with torch.no_grad():
                hs = m(ids, output_hidden_states=True).hidden_states
            return [tensor_stats(h[0]) for h in hs]
        census[name] = census_one(model, None, "cpu", ppl_fn, hidden_fn)
        print(f"  {name:12s} peak-kurt {census[name]['act']['peak_kurtosis']:8.1f}  "
              f"w3-damage {census[name]['damage']['w3']:+.3f}  "
              f"arch-detect qk_norm={census[name]['arch']['qk_norm']}")
    res = analyze(census)
    assert res and not np.isnan(res["spearman_kurt_vs_w3damage"])
    print("\nSELFTEST OK — full pipeline (stats, cliff, correlation, table) runs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--windows", type=int, default=20, help="PPL eval windows")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--out", default="census.json")
    ap.add_argument("--force", action="store_true", help="re-measure cached models")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    census = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for mid in args.models:
        if mid in census and "error" not in census[mid] and not args.force:
            print(f"[cached] {mid}")
            continue
        print(f"\n=== {mid} ===")
        try:
            model, tok, device = load(mid, dtype=torch.float32)
            ppl_fn = lambda m: wikitext_ppl(m, tok, device, seqlen=args.seqlen,
                                            max_windows=args.windows)
            hidden_fn = lambda m: collect_hidden_stats(m, tok, device,
                                                       seqlen=args.seqlen)
            census[mid] = census_one(model, tok, device, ppl_fn, hidden_fn)
            print(f"  peak-kurt {census[mid]['act']['peak_kurtosis']:.1f} | "
                  f"fp PPL {census[mid]['ppl']['fp']:.2f} | "
                  f"w3 damage {census[mid]['damage']['w3']:+.3f}")
            del model
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            census[mid] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  SKIPPED — {census[mid]['error'].splitlines()[0][:120]}")
            traceback.print_exc(limit=1)
        json.dump(census, open(args.out, "w"), indent=2)   # save after each model

    census["_analysis"] = analyze({k: v for k, v in census.items()
                                   if not k.startswith("_")})
    json.dump(census, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out} — run plot_census.py for the figure")


if __name__ == "__main__":
    main()
