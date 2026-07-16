#!/usr/bin/env python3
"""
RANGER Experiment C — quantized-drafter speculative decoding.

THE claim under test (report §2.9, roadmap C): speculative decoding is exact by
construction — with greedy decoding, a drafted token is accepted iff it equals
the target model's argmax, and rejected drafts are simply replaced. A bad
drafter can therefore only cost SPEED, never QUALITY. That makes the drafter
the one place in the system where extreme quantization (down to ternary) is
provably risk-free.

Falsifiable predictions:
  P-C1  greedy acceptance (argmax agreement) degrades slowly down to ~3-bit,
        then cliffs at 2-bit/ternary (PTQ ternary is expected to be terrible —
        ParetoQ says sub-3-bit needs QAT; the curve should SHOW that cliff).
  P-C2  even a heavily quantized drafter nets positive end-to-end speedup vs
        no drafting on memory-bound decode — until acceptance collapses.
  P-C3  outputs are bit-identical to plain greedy decoding for EVERY drafter
        precision (the exactness guarantee, demonstrated live).

Measurements per drafter config {fp, w8, w4, w3, w2, ternary}:
  * argmax agreement rate on real text (teacher-forced) — this IS the greedy
    acceptance probability — plus expected accepted-drafts-per-round for k=5
  * end-to-end assisted-generation tok/s vs the no-drafter baseline
  * exactness: assisted output == plain greedy output, token for token

Usage:
    python drafter_bench.py                    # 135M drafts for 1.7B (GPU best)
    python drafter_bench.py --target HuggingFaceTB/SmolLM2-360M   # CPU-friendly
    python drafter_bench.py --selftest         # network-free pipeline check
"""
import argparse, copy, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import load, quantize_weight, linears          # noqa: E402


# ------------------------------------------------------------ drafter quantizers
@torch.no_grad()
def quantize_drafter(model, config):
    """In-place weight-only PTQ of the drafter. config in {fp,w8,w4,w3,w2,ternary}."""
    if config == "fp":
        return
    for _, m in linears(model):
        W = m.weight.data
        if config == "ternary":                       # BitNet-style PTQ ternary
            s = W.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
            m.weight.data = (torch.clamp((W / s).round(), -1, 1) * s).to(W.dtype)
        else:
            bits = int(config[1:])
            m.weight.data = quantize_weight(W, bits, per="out").to(W.dtype)


# ------------------------------------------------------------ acceptance probe
@torch.no_grad()
def acceptance_probe(target, drafter, ids, k=5):
    """Greedy acceptance statistics on real text, teacher-forced.

    With greedy decoding, a draft is accepted iff drafter-argmax == target-argmax,
    so per-position argmax agreement IS the acceptance probability. We also
    report the empirical mean accepted-run length and the expected accepted
    drafts per round of k under an iid approximation (sum_{i=1..k} p^i).
    """
    lt = target(ids).logits[:, :-1].argmax(-1)
    ld = drafter(ids).logits[:, :-1].argmax(-1)
    match = (lt == ld)[0].cpu().numpy()
    p = float(match.mean())
    runs, run = [], 0
    for m in match:
        if m:
            run += 1
        else:
            runs.append(run); run = 0
    runs.append(run)
    exp_accept_k = float(sum(p ** i for i in range(1, k + 1)))
    return {"agreement": p, "mean_run": float(np.mean(runs)),
            "exp_accepted_per_round_k5": exp_accept_k}


# ------------------------------------------------------------ speed + exactness
@torch.no_grad()
def timed_generate(target, tok, prompts, device, new_tokens, drafter=None):
    """Greedy generation over prompts; returns (tok/s, list of output id tensors)."""
    outs, total_new, t_total = [], 0, 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        kw = dict(max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                  do_sample=False, pad_token_id=tok.eos_token_id or 0)
        if drafter is not None:
            kw["assistant_model"] = drafter
        target.generate(ids, max_new_tokens=8, do_sample=False,
                        pad_token_id=kw["pad_token_id"])       # warmup
        t0 = time.perf_counter()
        out = target.generate(ids, **kw)
        t_total += time.perf_counter() - t0
        total_new += out.shape[1] - ids.shape[1]
        outs.append(out[0].cpu())
    return total_new / max(t_total, 1e-9), outs


def run_bench(target, drafter_fp, tok, device, prompts, probe_ids,
              new_tokens, configs):
    results = {}
    print(f"\n--- no-drafter baseline ---")
    base_tps, base_outs = timed_generate(target, tok, prompts, device, new_tokens)
    results["baseline"] = {"tok_s": base_tps}
    print(f"  plain greedy: {base_tps:.2f} tok/s")

    fp_state = {k: v.cpu().clone() for k, v in drafter_fp.state_dict().items()}
    for cfg in configs:
        drafter_fp.load_state_dict(fp_state)
        quantize_drafter(drafter_fp, cfg)
        acc = acceptance_probe(target, drafter_fp, probe_ids)
        tps, outs = timed_generate(target, tok, prompts, device, new_tokens,
                                   drafter=drafter_fp)
        exact = all(torch.equal(a, b) for a, b in zip(outs, base_outs))
        results[cfg] = {**acc, "tok_s": tps, "speedup_x": tps / base_tps,
                        "exact_output": bool(exact)}
        print(f"  drafter {cfg:8s}: agreement {acc['agreement']:.3f} | "
              f"E[accept/round k=5] {acc['exp_accepted_per_round_k5']:.2f} | "
              f"{tps:6.2f} tok/s ({tps/base_tps:.2f}x) | "
              f"exact={'YES' if exact else 'NO'}")
    drafter_fp.load_state_dict(fp_state)
    return results


# ------------------------------------------------------------ selftest
def selftest():
    """Network-free: tiny random target+drafter with a shared vocab. Validates
    quantizer paths, the acceptance probe, assisted generation, and the
    exactness guarantee. Random models say nothing about REAL acceptance."""
    from transformers import LlamaConfig, LlamaForCausalLM
    print("SELFTEST — tiny random target + drafter, no network\n")
    torch.manual_seed(0)
    tcfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                       num_hidden_layers=4, num_attention_heads=4,
                       num_key_value_heads=4, max_position_embeddings=256)
    dcfg = LlamaConfig(vocab_size=512, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=2, num_attention_heads=2,
                       num_key_value_heads=2, max_position_embeddings=256)
    target, drafter = LlamaForCausalLM(tcfg).eval(), LlamaForCausalLM(dcfg).eval()

    print("1. quantizer paths change weights; ternary is 3-valued")
    W0 = drafter.model.layers[0].mlp.down_proj.weight.data.clone()
    st = {k: v.clone() for k, v in drafter.state_dict().items()}
    quantize_drafter(drafter, "w4")
    assert not torch.equal(W0, drafter.model.layers[0].mlp.down_proj.weight.data)
    drafter.load_state_dict(st); quantize_drafter(drafter, "ternary")
    u = torch.unique(drafter.model.layers[0].mlp.down_proj.weight.data[0])
    assert u.numel() <= 3, f"ternary row has {u.numel()} values"
    drafter.load_state_dict(st)
    print("   OK")

    print("2. acceptance probe returns sane stats")
    ids = torch.randint(0, 512, (1, 128))
    acc = acceptance_probe(target, drafter, ids)
    assert 0.0 <= acc["agreement"] <= 1.0
    print(f"   agreement={acc['agreement']:.3f} (random models ≈ chance)  OK")

    print("3. assisted generation runs and is EXACT vs plain greedy")
    ids = torch.randint(0, 512, (1, 16))
    kw = dict(max_new_tokens=24, min_new_tokens=24, do_sample=False, pad_token_id=0)
    plain = target.generate(ids, **kw)
    assisted = target.generate(ids, assistant_model=drafter, **kw)
    assert torch.equal(plain, assisted), "assisted output diverged from greedy!"
    print(f"   {plain.shape[1]-16} tokens, bit-identical  OK")

    print("4. timing harness end-to-end")
    class TinyTok:      # minimal stand-in for the tokenizer interface used
        eos_token_id = 0
        def __call__(self, text, return_tensors=None):
            class R: input_ids = torch.randint(1, 512, (1, 12))
            return R()
    tps, outs = timed_generate(target, TinyTok(), ["a", "b"], "cpu", 16,
                               drafter=drafter)
    assert tps > 0 and len(outs) == 2
    print(f"   assisted {tps:.1f} tok/s on tiny models  OK")

    print("\nSELFTEST OK — pipeline validated. Real acceptance/speed numbers "
          "need real weights (run on your machine).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="HuggingFaceTB/SmolLM2-1.7B")
    ap.add_argument("--drafter", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--configs", nargs="*",
                    default=["fp", "w8", "w4", "w3", "w2", "ternary"])
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--probe-tokens", type=int, default=2048,
                    help="real-text tokens for the acceptance probe")
    ap.add_argument("--out", default="drafter_results.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    print(f"loading target  {args.target} ...")
    target, tok, device = load(args.target)
    print(f"loading drafter {args.drafter} ...")
    drafter, dtok, _ = load(args.drafter, device=device)
    assert tok.get_vocab() == dtok.get_vocab(), \
        "target and drafter must share a tokenizer for speculative decoding"

    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    probe_ids = tok(text, return_tensors="pt").input_ids[:, :args.probe_tokens].to(device)
    prompts = [t[:300] for t in ds["text"] if len(t) > 300][:3]

    results = {"target": args.target, "drafter": args.drafter,
               "new_tokens": args.new_tokens, "device": str(device)}
    results["runs"] = run_bench(target, drafter, tok, device, prompts,
                                probe_ids, args.new_tokens, args.configs)

    r = results["runs"]
    print("\n--- verdicts ---")
    ok_exact = all(v.get("exact_output", True) for k, v in r.items() if k != "baseline")
    print(f"  P-C3 exactness at every precision : {'CONFIRMED' if ok_exact else 'VIOLATED'}")
    if "w4" in r and "fp" in r:
        drop4 = r["fp"]["agreement"] - r["w4"]["agreement"]
        print(f"  P-C1 agreement drop fp->w4        : {drop4:+.3f} "
              f"({'slow degradation' if drop4 < 0.05 else 'CLIFF at 4-bit — claim wounded'})")
    gains = {k: v["speedup_x"] for k, v in r.items() if k != "baseline"}
    print(f"  P-C2 speedup by drafter precision : " +
          "  ".join(f"{k}:{v:.2f}x" for k, v in gains.items()))
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
