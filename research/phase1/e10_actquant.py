#!/usr/bin/env python3
"""E10 + E11 — Pillar 1 end-to-end on a real model: per-token activation
fake-quant, per-linear rotation sandwich, and static channel exemption.

Definitions (QuaRot/SpinQuant convention, audit-verified):
  AN = dynamic per-token symmetric absmax fake-quant on the INPUT of every
  transformer-block nn.Linear; norms, softmax, residual adds, embeddings,
  lm_head, KV stay float.

Rotation sandwich (exact identity in fp): quantize x@Q against W@Q, one
shared seeded QR-orthogonal per input width (576 residual/head, 1536
intermediate). NOT a fast Hadamard — deployment-cost claims qualified.

The WHOLE model runs fp32 here (540 MB — affordable) so that W@Q carries no
bf16 storage rounding: activation effects are unconfounded by weight dtype.
The fp32 no-hook anchor is measured as arm 0a; arm 0b (bits=16 passthrough
hooks) must match it exactly, proving the harness inert. The historical
bf16 anchor 16.15 is reported for continuity but not used as the gate.

E10 arms: 0a fp32_nohooks | 0b passthrough | a8 | a4 | a4_rot |
  a4_down_only | a4_qkv_only | a4_o_only | a4_gateup_only |
  w4g128_a8 | w4g128rot_a4rot
E11 arms: a4_exempt_census | a4_exempt_census_down (needs --e8-json) |
  a4_exempt_random

Kill signals: 0b vs 0a mismatch -> harness bug, stop. a8 off anchor by >5%
-> harness bug. a4_rot recovering <20% of the a4 collapse (log-PPL) ->
Pillar 1 core mechanism fails on real weights (record as major theory hit).
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (checkpoint_json, fake_quant_pertoken, get_eval_ids,
                    load_or_init, orthogonal_q, perplexity,
                    rtn_w4_grouped_, target_linears)

torch.set_grad_enabled(False)

CENSUS8 = [100, 507, 446, 371, 247, 260, 17, 8]
RESIDUAL_INPUT_ROLES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")


def role_of(name):
    for r in ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj"):
        if name.endswith(r):
            return r
    return None


class Harness:
    """Forward-pre hooks on every block linear; per-arm config on modules."""

    def __init__(self, model, no_snapshot=False):
        self.linears = target_linears(model)
        self.originals = None  # lazy copy before first weight mutation
        self.no_snapshot = no_snapshot  # large models: one mutating arm,
        self.handles = []               # run it LAST, never restore
        for n, m in self.linears:
            m._act_bits = 16
            m._rot_q = None
            m._exempt = None
            self.handles.append(m.register_forward_pre_hook(self._hook))

    @staticmethod
    def _hook(mod, args):
        x = args[0]
        if mod._rot_q is not None:
            x = (x.float() @ mod._rot_q).to(x.dtype)  # rotate in fp32
        if mod._act_bits < 16 or mod._exempt is not None:
            x = fake_quant_pertoken(x, mod._act_bits, mod._exempt)
        return (x,) + args[1:]

    def snapshot(self):
        if self.no_snapshot:
            if getattr(self, "_mutated", False):
                raise RuntimeError("--no-snapshot: second weight-mutating "
                                   "arm requested; weights are dirty")
            self._mutated = True
            return
        if self.originals is None:
            self.originals = {n: m.weight.clone() for n, m in self.linears}

    def restore_weights(self):
        if self.originals is not None:
            for n, m in self.linears:
                m.weight.copy_(self.originals[n])

    def configure(self, bits_by_role=None, rot=False, exempt_map=None,
                  qcache=None):
        """bits_by_role: {role: bits}, default 16. rot: sandwich all block
        linears. exempt_map: {role: LongTensor of channels}."""
        self.restore_weights()
        if rot:
            self.snapshot()  # once per configure, BEFORE any mutation
        for n, m in self.linears:
            r = role_of(n)
            m._act_bits = (bits_by_role or {}).get(r, 16)
            m._exempt = (exempt_map or {}).get(r)
            m._rot_q = None
            if rot:
                q = qcache[m.weight.shape[1]]
                m._rot_q = q
                m.weight.copy_(
                    (m.weight.float() @ q).to(m.weight.dtype))

    def quantize_weights_g128(self):
        self.snapshot()
        for n, m in self.linears:
            rtn_w4_grouped_(m.weight, g=128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--dtype", default="float32",
                    choices=["float32", "bfloat16"],
                    help="bfloat16 for large models (rotated-weight arms "
                         "then carry bf16 storeback rounding — reported)")
    ap.add_argument("--census-channels", default="",
                    help="weight-census exemption channels (default: "
                         "SmolLM2-135M's CENSUS8)")
    ap.add_argument("--act-channels", default="",
                    help="activation-census exemption channels; enables the "
                         "a4_exempt_act arm")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="skip the originals copy (large models); at most "
                         "one weight-mutating arm, run it last")
    ap.add_argument("--e8-json", default="")
    ap.add_argument("--arms", default=(
        "fp32_nohooks,passthrough,a8,a4,a4_rot,a4_down_only,a4_qkv_only,"
        "a4_o_only,a4_gateup_only,w4g128_a8,w4g128rot_a4rot,"
        "a4_exempt_census,a4_exempt_census_down,a4_exempt_random"))
    args = ap.parse_args()
    arm_list = args.arms.split(",")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    ids = get_eval_ids(tok, args.tokens)

    hidden = model.config.hidden_size
    census8 = ([int(c) for c in args.census_channels.split(",") if c]
               or CENSUS8)[:8]
    act8 = [int(c) for c in args.act_channels.split(",") if c][:8]
    census_arms_requested = any("census" in a or "random" in a
                                for a in arm_list)
    if (census_arms_requested and hidden != 576
            and not args.census_channels):
        raise SystemExit(
            f"CENSUS8/random exemption channels are SmolLM2-135M-specific "
            f"(hidden 576); this model has hidden {hidden}. Pass "
            f"--arms without exemption arms or supply model-specific "
            f"channels.")
    unroled = [n for n, _ in
               [(n, m) for n, m in model.named_modules()
                if isinstance(m, torch.nn.Linear) and "lm_head" not in n]
               if role_of(n) is None]
    if unroled:
        raise SystemExit(f"linears with unrecognized role (would silently "
                         f"stay unquantized): {unroled}")

    tag = args.model.split("/")[-1].replace(".", "_")
    out_path = f"e10_e11_actquant_{tag}.json"
    results = load_or_init(out_path, {"model": args.model,
                                      "tokens": int(len(ids)),
                                      "dtype": args.dtype})
    results["ctx"] = args.ctx
    results["census8"] = census8
    if act8:
        results["act8"] = act8
    done = {a["arm"] for a in results["arms"] if "ppl" in a}
    if done:
        print(f"resuming: {sorted(done)} already recorded", flush=True)

    def record(name, ppl, extra=None):
        results["arms"].append({"arm": name, "ppl": ppl, **(extra or {})})
        checkpoint_json(out_path, results)
        print(f"  {name}: ppl = {ppl:.3f}", flush=True)

    # fp32 anchor BEFORE hooks exist
    if "fp32_nohooks" in arm_list and "fp32_nohooks" not in done:
        print("arm 0a: fp32 anchor, no hooks ...", flush=True)
        record("fp32_nohooks", perplexity(model, ids, ctx=args.ctx))

    h = Harness(model, no_snapshot=args.no_snapshot)
    widths = sorted({m.weight.shape[1] for _, m in h.linears})
    qcache = {w: orthogonal_q(w, seed=20260716) for w in widths}

    all4 = {r: 4 for r in ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")}
    all8 = {r: 8 for r in all4}

    # E8-derived down_proj-input exemption channels (top-8 stable
    # intermediate channels), if available
    down_exempt = None
    if args.e8_json and os.path.exists(args.e8_json):
        e8 = json.load(open(args.e8_json))
        assert e8.get("model") == args.model, (
            f"--e8-json is for {e8.get('model')}, not {args.model}")
        # Rank candidates by raw max|x| — what actually inflates a per-token
        # absmax scale (max/median saturates on ~0-median channels, review
        # finding wf_1d7a9bc7). Pool = top_by_maxabs union the per-token
        # argmax channels (the measured scale-setters).
        chans = {}
        for rec in e8.get("per_layer", []):
            dp = rec["down_proj_in"]
            for t in dp.get("top_by_maxabs", dp["top_channels"]):
                chans[t["ch"]] = max(chans.get(t["ch"], 0.0), t["max_abs"])
            for a in rec.get("dp_in_argmax_top5", []):
                if a["frac_tokens"] >= 0.5:  # persistent scale-setter
                    chans[a["ch"]] = max(chans.get(a["ch"], 0.0), 1e9)
        top = sorted(chans.items(), key=lambda kv: -kv[1])[:8]
        down_exempt = torch.tensor(sorted(c for c, _ in top),
                                   dtype=torch.long)
        results["down_exempt_channels"] = down_exempt.tolist()

    gen = torch.Generator().manual_seed(20260716)
    pool = [c for c in range(hidden) if c not in census8]
    rand8 = torch.tensor(
        sorted(torch.tensor(pool)[torch.randperm(len(pool), generator=gen)[:8]]
               .tolist()), dtype=torch.long)
    results["random8"] = rand8.tolist()
    census_t = torch.tensor(sorted(census8), dtype=torch.long)
    act_t = torch.tensor(sorted(act8), dtype=torch.long) if act8 else None

    ARMS = {
        "passthrough": dict(bits_by_role={r: 16 for r in all4}),
        "a8": dict(bits_by_role=all8),
        "a4": dict(bits_by_role=all4),
        "a4_rot": dict(bits_by_role=all4, rot=True),
        "a4_down_only": dict(bits_by_role={"down_proj": 4}),
        "a4_qkv_only": dict(bits_by_role={"q_proj": 4, "k_proj": 4,
                                          "v_proj": 4}),
        "a4_o_only": dict(bits_by_role={"o_proj": 4}),
        "a4_gateup_only": dict(bits_by_role={"gate_proj": 4, "up_proj": 4}),
        "a4_exempt_census": dict(
            bits_by_role=all4,
            exempt_map={r: census_t for r in RESIDUAL_INPUT_ROLES}),
        "a4_exempt_census_down": dict(
            bits_by_role=all4,
            exempt_map=(
                {r: census_t for r in RESIDUAL_INPUT_ROLES}
                | ({"down_proj": down_exempt}
                   if down_exempt is not None else {}))),
        "a4_exempt_random": dict(
            bits_by_role=all4,
            exempt_map={r: rand8 for r in RESIDUAL_INPUT_ROLES}),
        "a4_exempt_act": dict(
            bits_by_role=all4,
            exempt_map=({r: act_t for r in RESIDUAL_INPUT_ROLES}
                        if act_t is not None else {})),
    }

    for name in arm_list:
        if name == "fp32_nohooks" or name in done:
            continue
        if name == "w4g128_a8":
            print("arm: W4 g=128 + A8 ...", flush=True)
            h.configure(bits_by_role=all8)
            h.quantize_weights_g128()
            record("w4g128_a8", perplexity(model, ids, ctx=args.ctx))
            continue
        if name == "w4g128rot_a4rot":
            print("arm: W4 g=128 rotated + A4 rotated ...", flush=True)
            h.configure(bits_by_role=all4, rot=True, qcache=qcache)
            for n, m in h.linears:  # quantize the ROTATED weights
                rtn_w4_grouped_(m.weight, g=128)
            record("w4g128rot_a4rot", perplexity(model, ids, ctx=args.ctx))
            continue
        if name not in ARMS:
            print(f"unknown arm {name}, skipping", flush=True)
            continue
        if name == "a4_exempt_census_down" and down_exempt is None:
            print("a4_exempt_census_down skipped: no --e8-json", flush=True)
            continue
        if name == "a4_exempt_act" and act_t is None:
            print("a4_exempt_act skipped: no --act-channels", flush=True)
            continue
        cfg = ARMS[name]
        print(f"arm: {name} ...", flush=True)
        h.configure(qcache=qcache, **cfg)
        record(name, perplexity(model, ids, ctx=args.ctx))

    h.restore_weights()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
