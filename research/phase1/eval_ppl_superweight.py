#!/usr/bin/env python3
"""E7 — real-model, end-to-end test of Pillar 2 (super-weight preservation).

The synthetic runs (E2, E5) showed super-weight splitting matters at the
tensor-MSE level. This is the first END-TO-END check on a real pretrained
model: quantize every linear weight to W4 (per-channel symmetric RTN), then
restore the top-K globally largest |w| entries to full precision, and measure
wikitext-2 perplexity as K sweeps 0 -> 64.

If P2 holds on real models, ppl(K=small) should recover a large fraction of
the W4 degradation, with a step at the model's true super-weight count.

Runs on CPU inside the 6 GB VM: model is SmolLM2-135M (fp32 ~540 MB).

Usage:
  python eval_ppl_superweight.py [--model HuggingFaceTB/SmolLM2-135M]
                                 [--tokens 32768] [--ks 0,1,4,16,64]
"""
import argparse
import json
import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)


def get_eval_ids(tokenizer, n_tokens):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds)
    ids = tokenizer(text, return_tensors="pt").input_ids[0][:n_tokens]
    return ids


def perplexity(model, ids, ctx=1024):
    nll, count = 0.0, 0
    for i in range(0, len(ids) - 1, ctx):
        chunk = ids[i:i + ctx + 1]
        if len(chunk) < 2:
            break
        out = model(chunk[:-1].unsqueeze(0))
        logp = torch.log_softmax(out.logits[0].float(), dim=-1)
        nll -= logp[torch.arange(len(chunk) - 1), chunk[1:]].sum().item()
        count += len(chunk) - 1
    return math.exp(nll / count)


def target_linears(model):
    """All transformer-block linear weights (skip embeddings / lm_head)."""
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "lm_head" not in n]


def rtn_w4_(weight):
    """In-place per-output-channel symmetric 4-bit RTN (math in fp32,
    result stored back in the weight's own dtype)."""
    qmax = 7
    w = weight.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / qmax
    scale[scale == 0] = 1.0
    weight.copy_((torch.clamp(torch.round(w / scale), -8, 7) * scale).to(weight.dtype))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--ks", default="0,1,4,16,64")
    args = ap.parse_args()
    ks = [int(k) for k in args.ks.split(",")]

    # bf16 everywhere: native checkpoint dtype, and 3 weight copies must fit
    # in the 6 GB VM. Logits are upcast to fp32 inside perplexity().
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.eval()
    ids = get_eval_ids(tok, args.tokens)

    print("bf16 baseline ...", flush=True)
    results = {"model": args.model, "tokens": len(ids),
               "ppl_bf16": perplexity(model, ids)}
    print(f"  ppl = {results['ppl_bf16']:.3f}", flush=True)

    # Save originals of every target weight, and build the global |w| top list.
    linears = target_linears(model)
    originals = {n: m.weight.clone() for n, m in linears}
    all_mags = torch.cat([m.weight.abs().ravel() for _, m in linears])
    kmax = max(ks)
    if kmax > 0:
        top_global = torch.topk(all_mags, kmax).values
        thresholds = {k: top_global[k - 1].item() for k in ks if k > 0}

    for n, m in linears:
        rtn_w4_(m.weight)

    results["sweep"] = []
    for k in ks:
        # restore, on top of W4, every original entry with |w| >= threshold(k)
        for n, m in linears:
            if not hasattr(m, "_w4"):
                m._w4 = m.weight.clone()
            m.weight.copy_(m._w4)
            if k > 0:
                mask = originals[n].abs() >= thresholds[k]
                m.weight[mask] = originals[n][mask]
        ppl = perplexity(model, ids)
        n_restored = 0 if k == 0 else int(sum(
            (originals[n].abs() >= thresholds[k]).sum().item()
            for n, _ in linears))
        results["sweep"].append({"k": k, "restored": n_restored, "ppl": ppl})
        print(f"  W4 + top-{k:3d} restored ({n_restored} entries): "
              f"ppl = {ppl:.3f}", flush=True)

    tag = args.model.split("/")[-1].replace(".", "_")
    with open(f"e7_superweight_{tag}.json", "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"wrote e7_superweight_{tag}.json")


if __name__ == "__main__":
    main()
