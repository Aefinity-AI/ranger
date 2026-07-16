#!/usr/bin/env python3
"""E8 — activation census + super-weight identification on a real model.

Does SmolLM2-135M have massive activations (2402.17762) / a super weight
(2411.07191), and do the activation-outlier channels coincide with the
weight-census residual-highway channels {100, 247, 260, 371, 446, 507}?

Arms (single script run, ~12 min on the 6GB CPU box):
  1. wikitext 4096-token pass: residual-stream per-channel max & median per
     layer (block outputs — where 2402.17762 defines massive activations)
  2. same pass: mlp.down_proj INPUT (1536-d SwiGLU intermediate) and OUTPUT
     (576-d) per-channel max & median, plus per-token argmax-channel
     stability of the intermediate (static-exemption feasibility for E11)
  3. near-empty prompt (BOS + newline): input-independence check
  4. iterative super-weight extraction per the paper: find the down_proj
     out-spike layer/row + in-spike col at the same token, zero that weight,
     re-run, <= 5 rounds

Kill signals (from the Phase 1b design):
  - no residual channel reaches 30x max/median in any layer -> no
    massive-activation structure at 135M (scale caveat)
  - near-empty spikes differ from wikitext spikes beyond rank noise ->
    static channel exemption (E11) is dead on arrival

Output: e8_activation_census_<model>.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import get_eval_ids, checkpoint_json

torch.set_grad_enabled(False)

# SmolLM2-135M weight-census recurring residual channels (default);
# override with --census-channels for other models
CENSUS_TOP10 = [100, 507, 446, 371, 247, 260, 17, 8, 261, 162]


def block_out(out):
    return out[0] if isinstance(out, (tuple, list)) else out


class ChannelStats:
    """Per-channel max|x| and median|x| (exact within window, median across
    windows) — tiny memory, no full activation storage."""

    def __init__(self):
        self.mx = None
        self.win_medians = []

    def update(self, h):  # h: [tokens, dim]
        a = h.abs().float()
        m = a.amax(0)
        self.mx = m if self.mx is None else torch.maximum(self.mx, m)
        self.win_medians.append(a.median(0).values)

    def summary(self, top=10):
        med = torch.stack(self.win_medians).median(0).values
        ratio = self.mx / (med + 1e-6)
        vals, idx = torch.sort(ratio, descending=True)
        mvals, midx = torch.sort(self.mx, descending=True)
        return {
            "max_abs": float(self.mx.max()),
            "top_channels": [
                {"ch": int(i), "max_over_median": float(v),
                 "max_abs": float(self.mx[i])}
                for v, i in zip(vals[:top].tolist(), idx[:top].tolist())
            ],
            # ranked by raw max|x| — the quantity that actually inflates a
            # per-token absmax scale (max/median saturates on ~0 medians)
            "top_by_maxabs": [
                {"ch": int(i), "max_abs": float(v),
                 "max_over_median": float(ratio[i])}
                for v, i in zip(mvals[:top].tolist(), midx[:top].tolist())
            ],
        }


def run_census(model, ids, ctx=1024):
    layers = model.model.layers
    res = [ChannelStats() for _ in layers]
    dp_in = [ChannelStats() for _ in layers]
    dp_out = [ChannelStats() for _ in layers]
    argmax_counts = [{} for _ in layers]  # intermediate per-token argmax ch

    handles = []
    for li, layer in enumerate(layers):
        def mk_res(li):
            def hook(mod, inp, out):
                res[li].update(block_out(out)[0])
            return hook

        def mk_dp(li):
            def hook(mod, inp, out):
                x = inp[0][0]  # [tokens, 1536]
                dp_in[li].update(x)
                dp_out[li].update(out[0])
                am = x.abs().argmax(dim=-1)
                for c in am.tolist():
                    argmax_counts[li][c] = argmax_counts[li].get(c, 0) + 1
            return hook

        handles.append(layer.register_forward_hook(mk_res(li)))
        handles.append(layer.mlp.down_proj.register_forward_hook(mk_dp(li)))

    for i in range(0, len(ids), ctx):
        chunk = ids[i:i + ctx]
        if len(chunk) < 2:
            break
        model(chunk.unsqueeze(0))
    for h in handles:
        h.remove()

    per_layer = []
    for li in range(len(layers)):
        top_am = sorted(argmax_counts[li].items(), key=lambda kv: -kv[1])[:5]
        n_tok = sum(argmax_counts[li].values())
        per_layer.append({
            "layer": li,
            "residual": res[li].summary(),
            "down_proj_in": dp_in[li].summary(),
            "down_proj_out": dp_out[li].summary(),
            "dp_in_argmax_top5": [
                {"ch": c, "frac_tokens": n / n_tok} for c, n in top_am],
        })
    return per_layer


def superweight_rounds(model, tok, max_rounds=20, spike_floor=30.0):
    """2411.07191 iterative identification: locate the down_proj out-spike
    (layer, row) and the in-spike col at the same token, zero that single
    weight, re-run."""
    prompt = "Apple Inc. is a worldwide tech company."
    ids = tok(prompt, return_tensors="pt").input_ids
    layers = model.model.layers
    zeroed = []
    seen = set()  # (layer, row, col) already zeroed — never re-identify
    for rnd in range(max_rounds):
        rec = [None] * len(layers)  # (out_maxabs, out_med, row, tokpos, col)

        def mk(li):
            def hook(mod, inp, out):
                x, y = inp[0][0].abs().float(), out[0].abs().float()
                v, flat = y.ravel().max(0)
                tpos, row = divmod(int(flat), y.shape[1])
                # largest input channel at that token whose (row, col) is
                # not already zeroed
                for col in x[tpos].argsort(descending=True).tolist():
                    if (li, row, col) not in seen:
                        break
                rec[li] = (float(v), float(y.median()), row, tpos, col)
            return hook

        hs = [l.mlp.down_proj.register_forward_hook(mk(i))
              for i, l in enumerate(layers)]
        model(ids)
        for h in hs:
            h.remove()

        spikes = [(r[0] / (r[1] + 1e-6), li) + r for li, r in enumerate(rec)]
        ratio, li, v, med, row, tpos, col = max(spikes)
        entry = {"round": rnd, "layer": li, "row": row, "col": col,
                 "token_pos": tpos, "out_maxabs": v,
                 "out_max_over_median": ratio,
                 "weight_value": float(
                     layers[li].mlp.down_proj.weight[row, col])}
        zeroed.append(entry)
        if ratio < spike_floor:
            entry["stopped"] = "spike below floor"
            break
        seen.add((li, row, col))
        layers[li].mlp.down_proj.weight[row, col] = 0.0
    # restore
    for e in zeroed:
        if "weight_value" in e:
            layers[e["layer"]].mlp.down_proj.weight[
                e["row"], e["col"]] = e["weight_value"]
    terminated = ("converged" if zeroed and "stopped" in zeroed[-1]
                  else "max_rounds_cap")
    return zeroed, terminated


def overlap_analysis(per_layer, census_top10):
    """Model-wide residual ranking vs the weight-census top-10."""
    agg = {}
    for rec in per_layer:
        for t in rec["residual"]["top_channels"]:
            a = agg.setdefault(t["ch"], {"best_ratio": 0.0, "layers": 0})
            a["best_ratio"] = max(a["best_ratio"], t["max_over_median"])
            a["layers"] += 1
    ranked = sorted(agg.items(), key=lambda kv: -kv[1]["best_ratio"])
    top10 = [{"ch": c, **v} for c, v in ranked[:10]]
    act_set = {c for c, _ in ranked[:10]}
    return {
        "activation_top10": top10,
        "census_top10": census_top10,
        "overlap": sorted(act_set & set(census_top10)),
        "overlap_count": len(act_set & set(census_top10)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=int, default=4096)
    ap.add_argument("--census-channels", default="",
                    help="comma list of weight-census residual channels for "
                         "the overlap analysis (default: SmolLM2-135M's)")
    args = ap.parse_args()
    census_top10 = ([int(c) for c in args.census_channels.split(",") if c]
                    or CENSUS_TOP10)
    if (census_top10 is CENSUS_TOP10
            and "SmolLM2-135M" not in args.model):
        print("WARNING: overlap analysis uses SmolLM2-135M census channels; "
              "pass --census-channels for this model", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16)
    model.eval()
    tag = args.model.split("/")[-1].replace(".", "_")
    out_path = f"e8_activation_census_{tag}.json"

    ids = get_eval_ids(tok, args.tokens)
    print(f"arm 1+2: census over {len(ids)} wikitext tokens ...", flush=True)
    per_layer = run_census(model, ids)
    results = {"model": args.model, "tokens": int(len(ids)),
               "per_layer": per_layer,
               "overlap": overlap_analysis(per_layer, census_top10)}
    checkpoint_json(out_path, results)
    print(json.dumps(results["overlap"], indent=1), flush=True)

    print("arm 3: near-empty prompt ...", flush=True)
    empty = [tok.bos_token_id] if tok.bos_token_id is not None else []
    empty_ids = torch.tensor(
        empty + tok("\n", add_special_tokens=False).input_ids +
        tok(" ", add_special_tokens=False).input_ids, dtype=torch.long)
    results["near_empty"] = run_census(model, empty_ids, ctx=8)
    checkpoint_json(out_path, results)

    print("arm 4: iterative super-weight extraction ...", flush=True)
    rounds, terminated = superweight_rounds(model, tok)
    results["superweight_rounds"] = rounds
    results["superweight_terminated"] = terminated
    print(f"  terminated: {terminated}", flush=True)
    for e in results["superweight_rounds"]:
        print(f"  round {e['round']}: L{e['layer']} down_proj"
              f"[{e['row']},{e['col']}] ratio {e['out_max_over_median']:.1f}"
              f" w={e['weight_value']:.4f}", flush=True)

    # headline verdicts
    best = max(t["max_over_median"] for rec in per_layer
               for t in rec["residual"]["top_channels"][:1])
    results["verdicts"] = {
        "best_residual_max_over_median": best,
        "massive_activations_at_100x": best >= 100,
        "massive_activations_at_30x": best >= 30,
    }
    checkpoint_json(out_path, results)
    print(f"best residual max/median = {best:.1f}  "
          f"(100x gate: {best >= 100}, 30x kill-floor: {best >= 30})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
