#!/usr/bin/env python3
"""E9 — was E7's null an artifact? Corrected super-weight protocol at W4.

E7 restored top-|w| values AFTER quantization with an outlier-inflated
per-channel absmax scale — provably a no-op (the row max is bit-exact under
that scheme; workflow wf_4b49c9dc-3d0). This is the corrected experiment:

  arm 1  W4 per-channel RTN            — must reproduce E7's 36.435 anchor
  arm 2  W4 g=128 RTN                  — the actual field baseline
  arm 3  g=128 + per-group MSE clip grid (c in 1.0..0.5)
  arm 4  g=128 + HOLD-OUT top-K global |w| from the scale + restore,
         K in {64, 512} (exact-K by topk index)
  arm 5  g=128 + hold-out the E8 activation-identified super-weight
         coordinates + clip (only if --e8-json given and E8 found any)
  arm 6  per-channel RTN + restore exactly K=64 by topk INDEX — clean
         re-run of E7's own arm with the >=-threshold tie bug fixed

Equal-bits accounting is written into the JSON. Every protection arm prints
n_changed (must be > 0) and the measured per-group scale shrinkage.

Prediction: arm 2 recovers most of the 16.15 -> 36.44 gap (<= ~22);
arms 4/5 beat arm 3 by >= 0.15 PPL if a genuine super-weight lever exists.
Kill: arm 2 > 30 -> quantizer bug, halt. Arms 4/5 within noise of arm 3 ->
REAL null at 135M (scale caveat; escalates E12).
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (checkpoint_json, effective_bpw, get_eval_ids,
                    global_topk_indices, load_or_init, perplexity,
                    rtn_w4_grouped_, rtn_w4_perchannel_, target_linears)

torch.set_grad_enabled(False)

CLIP_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


def restore_(linears, originals):
    if originals is None:  # --no-copy single-arm mode: weights are pristine
        return
    for n, m in linears:
        m.weight.copy_(originals[n])


def masks_from_flat(linears, flat_idx):
    """{name: flat LongTensor} -> {name: bool mask} built lazily per call."""
    out = {}
    for n, m in linears:
        if n + ".weight" in flat_idx or n in flat_idx:
            idx = flat_idx.get(n, flat_idx.get(n + ".weight"))
            mask = torch.zeros(m.weight.numel(), dtype=torch.bool)
            mask[idx] = True
            out[n] = mask.view_as(m.weight)
    return out


def quantize_all(linears, originals, g=None, holdout=None, clip=None):
    """Restore originals, then quantize every linear. Returns agg stats."""
    restore_(linears, originals)
    agg = {"n_held": 0, "n_changed": 0, "shrink_sum": 0.0, "shrink_n": 0,
           "bpw_by_cols": {}}
    for n, m in linears:
        if g is None:
            rtn_w4_perchannel_(m.weight)
            cols = m.weight.shape[1]
            agg["bpw_by_cols"][cols] = effective_bpw(cols, 1)
            continue
        hm = holdout.get(n) if holdout else None
        st = rtn_w4_grouped_(m.weight, g=g, holdout_mask=hm, clip_grid=clip)
        agg["n_held"] += st["n_held"]
        agg["n_changed"] += st["n_changed"]
        if st["scale_shrinkage"] is not None:
            agg["shrink_sum"] += st["scale_shrinkage"]
            agg["shrink_n"] += 1
        cols = m.weight.shape[1]
        agg["bpw_by_cols"][cols] = effective_bpw(cols, st["n_groups_per_row"])
    agg["scale_shrinkage_mean"] = (
        agg["shrink_sum"] / agg["shrink_n"] if agg["shrink_n"] else None)
    del agg["shrink_sum"], agg["shrink_n"]
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--e8-json", default="")
    ap.add_argument("--arms", default="1,2,3,4,5,6")
    ap.add_argument("--holdout-ks", default="64,512")
    ap.add_argument("--no-copy", action="store_true",
                    help="skip the originals copy (large models). At most "
                         "ONE weight-mutating arm per invocation; use the "
                         "resume-safe JSON across invocations.")
    args = ap.parse_args()
    arms = set(args.arms.split(","))
    holdout_ks = [int(k) for k in args.holdout_ks.split(",") if k]

    if args.no_copy:
        n_mutating = (sum(a in arms for a in ("1", "2", "3", "5"))
                      + ("4" in arms) * len(holdout_ks))
        if n_mutating > 1 or "6" in arms:
            raise SystemExit("--no-copy allows at most one weight-mutating "
                             "arm per invocation (and not arm 6)")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16)
    model.eval()
    ids = get_eval_ids(tok, args.tokens)
    linears = target_linears(model)
    originals = (None if args.no_copy
                 else {n: m.weight.clone() for n, m in linears})

    tag = args.model.split("/")[-1].replace(".", "_")
    out_path = f"e9_holdout_w4_{tag}.json"
    results = load_or_init(out_path, {"model": args.model,
                                      "tokens": int(len(ids)),
                                      "group": args.group,
                                      "clip_grid": CLIP_GRID})
    done = {a["arm"] for a in results["arms"] if "ppl" in a}
    if done:
        print(f"resuming: {sorted(done)} already recorded", flush=True)

    def record(name, ppl, extra=None):
        rec = {"arm": name, "ppl": ppl, **(extra or {})}
        results["arms"].append(rec)
        checkpoint_json(out_path, results)
        print(f"  {name}: ppl = {ppl:.3f}"
              + (f"  n_changed={extra.get('n_changed')}"
                 f" shrink={extra.get('scale_shrinkage_mean')}"
                 if extra and "n_changed" in extra else ""), flush=True)

    if "bf16" not in done:
        print("bf16 baseline ...", flush=True)
        record("bf16", perplexity(model, ids))

    if "1" in arms and "w4_perchannel" not in done:
        print("arm 1: W4 per-channel RTN (E7 anchor) ...", flush=True)
        st = quantize_all(linears, originals)
        record("w4_perchannel", perplexity(model, ids), st)

    if "2" in arms and f"w4_g{args.group}" not in done:
        print(f"arm 2: W4 g={args.group} RTN ...", flush=True)
        st = quantize_all(linears, originals, g=args.group)
        record(f"w4_g{args.group}", perplexity(model, ids), st)

    if "3" in arms and f"w4_g{args.group}_clip" not in done:
        print(f"arm 3: g={args.group} + clip grid ...", flush=True)
        st = quantize_all(linears, originals, g=args.group, clip=CLIP_GRID)
        record(f"w4_g{args.group}_clip", perplexity(model, ids), st)

    if "4" in arms:
        for k in holdout_ks:
            if f"w4_g{args.group}_holdout_top{k}" in done:
                continue
            print(f"arm 4: g={args.group} + hold-out top-{k} ...", flush=True)
            restore_(linears, originals)
            flat = global_topk_indices(linears, k)
            holdout = masks_from_flat(linears, flat)
            st = quantize_all(linears, originals, g=args.group,
                              holdout=holdout)
            st["k"] = k
            record(f"w4_g{args.group}_holdout_top{k}",
                   perplexity(model, ids), st)

    if ("5" in arms and args.e8_json and os.path.exists(args.e8_json)
            and f"w4_g{args.group}_holdout_e8coords_clip" not in done):
        e8 = json.load(open(args.e8_json))
        assert e8.get("model") == args.model, (
            f"--e8-json is for {e8.get('model')}, not {args.model}")
        cap_truncated = e8.get("superweight_terminated") == "max_rounds_cap"
        coords = [e for e in e8.get("superweight_rounds", [])
                  if e.get("out_max_over_median", 0) >= 30
                  and "stopped" not in e]
        if coords:
            print(f"arm 5: g={args.group} + hold-out {len(coords)} "
                  "E8 super-weight coords + clip ...", flush=True)
            holdout = {}
            for e in coords:
                n = f"model.layers.{e['layer']}.mlp.down_proj"
                m = dict(linears)[n]
                hm = holdout.setdefault(
                    n, torch.zeros_like(m.weight, dtype=torch.bool))
                hm[e["row"], e["col"]] = True
            st = quantize_all(linears, originals, g=args.group,
                              holdout=holdout, clip=CLIP_GRID)
            st["coords"] = [[e["layer"], e["row"], e["col"]] for e in coords]
            st["e8_coords_cap_truncated"] = cap_truncated
            record(f"w4_g{args.group}_holdout_e8coords_clip",
                   perplexity(model, ids), st)
        else:
            print("arm 5 skipped: E8 found no >=30x super-weight coords",
                  flush=True)
            results["arms"].append({"arm": "holdout_e8coords",
                                    "skipped": "no E8 coords"})
            checkpoint_json(out_path, results)

    if "6" in arms and "w4_perchannel_restore_exact64" not in done:
        print("arm 6: per-channel + restore exact-64 by topk index ...",
              flush=True)
        restore_(linears, originals)
        flat = global_topk_indices(linears, 64)  # from ORIGINAL weights
        st = quantize_all(linears, originals)  # plain per-channel W4
        n_changed = 0
        for n, m in linears:
            if n not in flat:
                continue
            idx = flat[n]
            before = m.weight.ravel()[idx].clone()
            after = originals[n].ravel()[idx]
            n_changed += int((before != after).sum().item())
            m.weight.view(-1)[idx] = after
        st["n_changed"] = n_changed
        record("w4_perchannel_restore_exact64",
               perplexity(model, ids), st)

    restore_(linears, originals)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
