#!/usr/bin/env python3
"""Phase 1a — first contact with REAL model tensors.

Everything so far (E1–E6) ran on synthetic tensors. This script re-asks the
core mechanism questions against actual pretrained weights, streamed one
tensor at a time from safetensors (mmap) so it runs inside the 6 GB dev VM:

  Q1  Do real weight matrices show the heavy-tailed outlier structure the
      theory assumes? (per-tensor kurtosis + max/std census)
  Q2  Do super-weights exist where the literature says (early mlp.down_proj),
      and how large are they relative to the tensor's std?
  Q3  On real tensors, does the E5 orthogonal-axis rule hold: split
      super-weights FIRST, then rotate — vs rotate-only, vs raw — measured by
      4-bit RTN per-channel quantization MSE?

Usage:
  python analyze_real_weights.py <model_repo_id> [--max-tensors N]
Output: phase1_results_<model>.json + printed census table.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

RNG = np.random.default_rng(0)


def kurtosis(x):
    x = x.ravel().astype(np.float64)
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - m) ** 4).mean() / s**4 - 3.0)


def orthogonal(n, rng):
    """Randomized Hadamard when n is a power of 2 (O(n^2) apply, cheap build);
    else a seeded random orthogonal from QR."""
    if n & (n - 1) == 0:
        h = np.array([[1.0]])
        while h.shape[0] < n:
            h = np.block([[h, h], [h, -h]])
        d = rng.choice([-1.0, 1.0], size=n)
        return (h / math.sqrt(n)) * d  # H @ diag(d), columns stay orthonormal
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def rtn_quant_mse(w, bits=4):
    """Per-output-channel symmetric round-to-nearest INT quantization MSE."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(w).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(w / scale), -qmax - 1, qmax) * scale
    return float(((w - q) ** 2).mean())


def analyze_tensor(name, w, n_super=8):
    """Census + the raw / rotate / split-then-rotate comparison (Q3)."""
    out = {
        "name": name,
        "shape": list(w.shape),
        "kurtosis": kurtosis(w),
        "max_over_std": float(np.abs(w).max() / (w.std() + 1e-12)),
    }
    # Q3 only on manageable matrices (rotation cost is cols^2).
    rows, cols = w.shape
    if cols > 4096 or rows > 32768:
        return out
    w = w.astype(np.float32)
    rot = orthogonal(cols, RNG).astype(np.float32)

    # split: remove the n_super largest-|w| entries (kept fp16 on the side)
    flat = np.abs(w).ravel()
    idx = np.argpartition(flat, -n_super)[-n_super:]
    w_split = w.copy()
    w_split.ravel()[idx] = 0.0

    out["super_weights"] = [
        {"index": [int(i // cols), int(i % cols)],
         "value_over_std": float(flat[i] / (w.std() + 1e-12))}
        for i in sorted(idx, key=lambda i: -flat[i])
    ]
    out["mse_raw"] = rtn_quant_mse(w)
    out["mse_rotated"] = rtn_quant_mse(w @ rot)
    out["mse_split_rotated"] = rtn_quant_mse(w_split @ rot)
    out["kurtosis_rotated"] = kurtosis(w @ rot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo_id")
    ap.add_argument("--max-tensors", type=int, default=0,
                    help="analyze only the first N 2-D weights (0 = all)")
    args = ap.parse_args()

    path = snapshot_download(args.repo_id, allow_patterns=["*.safetensors"])
    files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    results, n = [], 0
    for f in files:
        with safe_open(os.path.join(path, f), framework="pt") as st:
            for name in st.keys():
                t = st.get_tensor(name)
                if t.ndim != 2 or "embed" in name or "lm_head" in name:
                    continue
                r = analyze_tensor(name, t.to(torch.float32).numpy())
                results.append(r)
                flag = "  <-- Q3" if "mse_raw" in r else ""
                print(f"{name:60s} kurt={r['kurtosis']:8.2f} "
                      f"max/std={r['max_over_std']:7.1f}{flag}", flush=True)
                n += 1
                if args.max_tensors and n >= args.max_tensors:
                    break
        if args.max_tensors and n >= args.max_tensors:
            break

    tag = args.repo_id.split("/")[-1].replace(".", "_")
    with open(f"phase1_results_{tag}.json", "w") as fh:
        json.dump({"model": args.repo_id, "tensors": results}, fh, indent=1)

    q3 = [r for r in results if "mse_raw" in r]
    if q3:
        gain_rot = np.mean([r["mse_raw"] / r["mse_rotated"] for r in q3])
        gain_split = np.mean([r["mse_raw"] / r["mse_split_rotated"] for r in q3])
        print(f"\n== Q3 across {len(q3)} tensors: "
              f"rotate-only mean MSE gain {gain_rot:.3f}x | "
              f"split-then-rotate {gain_split:.3f}x")
    print(f"wrote phase1_results_{tag}.json")


if __name__ == "__main__":
    main()
