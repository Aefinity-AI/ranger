#!/usr/bin/env python3
"""
Phase-1 · Experiment 1+2 (PTQ on a REAL model, weight-only, no training):

  A. Bit-width sweep — FP baseline vs RTN weight quant at {4,3,2} bits. End-to-end
     WikiText-2 perplexity. Establishes where the real model breaks (the ~3-bit
     cliff ParetoQ predicts).
  B. Pillar 2 — importance-ordered mixed precision vs uniform at EQUAL average bits.
     Split each weight's output channels by importance (||row|| proxy for the
     trained MatFormer ordering); give the important half +1 bit, the rest -1 bit.
     If ordered < uniform PPL, Pillar 2 holds on a real model.
  C. Pillar 1 (mechanism) — per-layer rotation check. On real calibration
     activations, compare reconstruction error of x·W vs its quantized version,
     naive vs Hadamard-rotated in-dim. Rotation should cut the error.

Everything is weight-side / measurement-side so it runs without a full QuaRot
inference rewrite (that + QAT is Phase 2 — see README). CPU-safe on 135M.

Usage:
    python ptq_eval.py --model HuggingFaceTB/SmolLM2-135M
"""
import argparse, copy, json
import numpy as np
import torch
from common import (load, wikitext_ppl, quantize_weight, quantize_act,
                    hadamard_pad, linears)


@torch.no_grad()
def quantize_model_uniform(model, bits):
    for _, m in linears(model):
        m.weight.data = quantize_weight(m.weight.data, bits, per="out").to(m.weight.dtype)


@torch.no_grad()
def quantize_model_ordered(model, b_avg):
    """Importance-ordered mixed precision at equal average bits (Pillar 2)."""
    hi, lo = b_avg + 1, b_avg - 1
    for _, m in linears(model):
        W = m.weight.data
        imp = W.abs().sum(dim=1)                      # per-output-channel importance
        order = torch.argsort(imp, descending=True)
        n_hi = W.shape[0] // 2                        # top half -> hi bits, avg = b_avg
        idx_hi = order[:n_hi]
        Wq = quantize_weight(W, lo, per="out")
        Wq[idx_hi] = quantize_weight(W[idx_hi], hi, per="out")
        m.weight.data = Wq.to(m.weight.dtype)


@torch.no_grad()
def rotation_mechanism_check(model, tok, device, bits=4, n_layers=6):
    """Pillar 1 on real tensors: does rotating the in-dim cut weight-quant error?"""
    # capture real inputs to a set of MLP down_proj layers
    caps, handles = {}, []
    targets = [(n, m) for n, m in linears(model) if "down_proj" in n][:n_layers]
    for n, m in targets:
        def mk(name):
            def hook(mod, inp, out): caps.setdefault(name, inp[0].detach())
            return hook
        handles.append(m.register_forward_hook(mk(n)))
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(ds["text"][:50]), return_tensors="pt").input_ids[:, :1024].to(device)
    model(ids)
    for h in handles:
        h.remove()

    rows = []
    for n, m in targets:
        x = caps[n].reshape(-1, caps[n].shape[-1]).float()      # [tokens, in]
        W = m.weight.data.float()                                # [out, in]
        Y = x @ W.t()
        # naive: quantize W and x directly
        Yq_naive = quantize_act(x, bits) @ quantize_weight(W, bits, per="out").t()
        # rotated in-dim: x' = x H, W' = W H ; x'·W'^T == x·W^T (H orthogonal)
        xr, _ = hadamard_pad(x)
        Wr, _ = hadamard_pad(W)
        Yq_rot = quantize_act(xr, bits) @ quantize_weight(Wr, bits, per="out").t()
        e_naive = (Y - Yq_naive).norm() / (Y.norm() + 1e-9)
        e_rot = (Y - Yq_rot).norm() / (Y.norm() + 1e-9)
        rows.append((n, float(e_naive), float(e_rot), float(e_naive / e_rot)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--windows", type=int, default=20, help="WikiText windows for PPL (speed)")
    ap.add_argument("--out", default="ptq_results.json")
    args = ap.parse_args()

    print(f"loading {args.model} ...")
    model, tok, device = load(args.model)
    fp_state = copy.deepcopy(model.state_dict())

    def restore():
        model.load_state_dict(fp_state)

    res = {"model": args.model}
    print("\n=== A. bit-width sweep (weight-only RTN) ===")
    ppl_fp = wikitext_ppl(model, tok, device, max_windows=args.windows)
    print(f"  FP baseline      : PPL {ppl_fp:.2f}")
    res["fp"] = ppl_fp
    res["uniform"] = {}
    for b in [4, 3, 2]:
        restore()
        quantize_model_uniform(model, b)
        p = wikitext_ppl(model, tok, device, max_windows=args.windows)
        res["uniform"][b] = p
        print(f"  RTN W{b}           : PPL {p:.2f}   (+{100*(p-ppl_fp)/ppl_fp:.0f}% vs FP)")

    print("\n=== B. Pillar 2 — importance-ordered vs uniform at equal avg bits ===")
    res["pillar2"] = {}
    for b in [3, 2]:
        restore(); quantize_model_uniform(model, b)
        pu = wikitext_ppl(model, tok, device, max_windows=args.windows)
        restore(); quantize_model_ordered(model, b)
        po = wikitext_ppl(model, tok, device, max_windows=args.windows)
        res["pillar2"][b] = {"uniform": pu, "ordered": po}
        tag = "ordered WINS" if po < pu else "no gain"
        print(f"  avg {b} bits: uniform PPL {pu:.2f}  |  ordered PPL {po:.2f}  -> {tag}")

    print("\n=== C. Pillar 1 (mechanism) — rotation cuts weight-quant error on real acts ===")
    restore()
    rows = rotation_mechanism_check(model, tok, device, bits=4)
    res["pillar1"] = [{"layer": n, "e_naive": a, "e_rot": b, "cut_x": c} for n, a, b, c in rows]
    for n, a, b, c in rows:
        print(f"  {n[-40:]:40s}  naive {a:.4f}  rot {b:.4f}  cut {c:.2f}x")
    avg_cut = float(np.mean([c for *_, c in rows]))
    print(f"  average rotation cut: {avg_cut:.2f}x  "
          f"({'rotation helps on real tensors' if avg_cut > 1.05 else 'no benefit'})")
    res["pillar1_avg_cut"] = avg_cut

    json.dump(res, open(args.out, "w"), indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
