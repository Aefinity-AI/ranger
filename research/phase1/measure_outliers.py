#!/usr/bin/env python3
"""
Phase-1 · Experiment 0 (instrumentation): does a REAL model have the massive-
activation / outlier pathology that RANGER is built around?

Loads a small open model, runs WikiText-2 text through it, and reports per-layer
residual-stream outlier statistics (excess kurtosis, incoherence, max-abs, the
top outlier channels). If the theory's premise is right we should see kurtosis
spike at a specific "massive emergence layer" (2605.08504) and a handful of
channels dominate.

Usage:
    python measure_outliers.py --model HuggingFaceTB/SmolLM2-135M
    python measure_outliers.py --model Qwen/Qwen3-0.6B-Base
"""
import argparse, json
import numpy as np
from common import load, collect_hidden_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--seqlen", type=int, default=1024)
    ap.add_argument("--out", default="outliers.json")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    model, tok, device = load(args.model)
    nl = model.config.num_hidden_layers
    hid = model.config.hidden_size
    print(f"  layers={nl}  hidden={hid}  device={device}")

    print("running WikiText-2 through the model (residual-stream stats) ...")
    stats = collect_hidden_stats(model, tok, device, seqlen=args.seqlen)

    print("\n layer |  kurtosis |  incoherence |  max|x|  |   rms   |  crest(max/rms)")
    print(" " + "-" * 68)
    rows = []
    for i, s in enumerate(stats):
        crest = s["max_abs"] / (s["rms"] + 1e-9)
        rows.append({"layer": i, **s, "crest": crest})
        tag = "  <- residual stream (post-embed)" if i == 0 else ""
        print(f"  {i:3d}  | {s['kurtosis']:9.1f} | {s['incoherence']:11.2f} | "
              f"{s['max_abs']:7.1f} | {s['rms']:7.3f} | {crest:8.1f}{tag}")

    kmax = max(range(len(stats)), key=lambda i: stats[i]["kurtosis"])
    imax = max(range(len(stats)), key=lambda i: stats[i]["incoherence"])
    print("\n  massive-emergence layer (peak kurtosis): "
          f"L{kmax}  kurtosis={stats[kmax]['kurtosis']:.0f}")
    print(f"  peak incoherence:                         L{imax}  "
          f"μ={stats[imax]['incoherence']:.1f}")
    print(f"  top outlier channels at L{kmax}: "
          f"{[round(c,1) for c in stats[kmax]['top_channels']]}")

    verdict = ("PATHOLOGY CONFIRMED — real model has concentrated outliers; RANGER's premise holds"
               if max(s["kurtosis"] for s in stats) > 20 else
               "weak/absent outliers — this model may already be quantization-friendly")
    print(f"\n  VERDICT: {verdict}")

    json.dump({"model": args.model, "num_layers": nl, "hidden": hid,
               "emergence_layer": kmax, "peak_kurtosis": stats[kmax]["kurtosis"],
               "peak_incoherence": stats[imax]["incoherence"], "layers": rows},
              open(args.out, "w"), indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
