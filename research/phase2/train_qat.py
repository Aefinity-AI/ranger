#!/usr/bin/env python3
"""
RANGER Phase-2 — QAT fine-tuning loop.

The experiment ParetoQ says is mandatory below ~3 bits: take a pretrained
model, fake-quantize its weights (STE), and fine-tune so the weights learn to
tolerate rounding — optionally with the RANGER extras:

  --superweight-pct 0.005   constraint G: top 0.5%% |W| kept in a fp sparse path
  --nested                  Pillar 2: importance-ordered bits±1 at equal avg
  --protect-downproj        Pillar 3: down_proj (FC2) gets +1 bit
  --kurt-lambda 0.05        Pillar 1: kurtosis hinge on the residual stream
  --ternary                 BitNet b1.58 (needs the most training of all)
  --rotate-downproj         exact online Hadamard on down_proj (R4-style)

Protocol per run: eval PPL + peak kurtosis BEFORE (fp), AFTER conversion
(PTQ-equivalent, step 0), and AFTER training. The three-point comparison is the
result: (fp -> ptq) is the damage, (ptq -> qat) is the recovery.

Examples (on your GPU box; 135M works on CPU too, slowly):
    python train_qat.py --model HuggingFaceTB/SmolLM2-135M --bits 3 --steps 300
    python train_qat.py --model HuggingFaceTB/SmolLM2-135M --bits 2 --steps 800 \
        --superweight-pct 0.005 --nested --protect-downproj --kurt-lambda 0.05
    python train_qat.py --synthetic --steps 30        # network-free smoke test
"""
import argparse, json, math, sys, os, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase1"))
from common import load, wikitext_ppl, collect_hidden_stats            # noqa: E402
from qat_modules import convert_to_qat, kurtosis_loss, average_bits    # noqa: E402


def get_batches(tok, device, seqlen, batch, steps, synthetic=False, vocab=None):
    """Yield [batch, seqlen] token blocks. WikiText-2 train split, or synthetic."""
    if synthetic:
        g = torch.Generator().manual_seed(0)
        for _ in range(steps):
            yield torch.randint(0, vocab, (batch, seqlen), generator=g).to(device)
        return
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n_blocks = ids.numel() // seqlen
    blocks = ids[: n_blocks * seqlen].view(n_blocks, seqlen)
    perm = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(0))
    i = 0
    for _ in range(steps):
        if i + batch > n_blocks:
            i = 0
        yield blocks[perm[i:i + batch]].to(device)
        i += batch


def make_tiny_model():
    """Random tiny Llama for the network-free smoke test."""
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=512)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg), None, "cpu"


def peak_kurtosis(model, tok, device, synthetic, vocab):
    """Max excess kurtosis over the residual stream (the outlier signature)."""
    with torch.no_grad():
        if synthetic:
            ids = torch.randint(0, vocab, (1, 256)).to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            stats = []
            for h in hs:
                x = h.float().reshape(-1)
                stats.append(float((((x - x.mean()) / x.std().clamp_min(1e-9)) ** 4).mean() - 3))
            return max(stats)
        return max(s["kurtosis"] for s in collect_hidden_stats(model, tok, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--ternary", action="store_true")
    ap.add_argument("--superweight-pct", type=float, default=0.0)
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--protect-downproj", action="store_true")
    ap.add_argument("--rotate-downproj", action="store_true")
    ap.add_argument("--kurt-lambda", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--windows", type=int, default=20, help="PPL eval windows")
    ap.add_argument("--synthetic", action="store_true", help="no-network smoke test")
    ap.add_argument("--save", default=None, help="dir to save the QAT checkpoint")
    ap.add_argument("--out", default="qat_results.json")
    args = ap.parse_args()

    # ---- load -------------------------------------------------------------
    if args.synthetic:
        model, tok, device = make_tiny_model()
    else:
        print(f"loading {args.model} ...")
        model, tok, device = load(args.model, dtype=torch.float32)  # QAT wants fp32 master
    vocab = model.config.vocab_size
    res = {"args": vars(args)}

    # ---- BEFORE: fp baseline ----------------------------------------------
    if not args.synthetic:
        res["fp_ppl"] = wikitext_ppl(model, tok, device, max_windows=args.windows)
        print(f"[before] fp PPL          : {res['fp_ppl']:.2f}")
    res["fp_peak_kurt"] = peak_kurtosis(model, tok, device, args.synthetic, vocab)
    print(f"[before] fp peak kurtosis: {res['fp_peak_kurt']:.1f}")

    # ---- convert ------------------------------------------------------------
    n = convert_to_qat(model, bits=args.bits, ternary=args.ternary,
                       superweight_pct=args.superweight_pct, nested=args.nested,
                       protect_downproj=args.protect_downproj,
                       rotate_downproj=args.rotate_downproj)
    res["avg_bits"] = average_bits(model)
    print(f"converted {n} linears | effective avg weight bits = {res['avg_bits']:.2f}")

    # ---- step-0 = PTQ-equivalent damage ------------------------------------
    if not args.synthetic:
        res["ptq_ppl"] = wikitext_ppl(model, tok, device, max_windows=args.windows)
        print(f"[step 0] PTQ-equiv PPL   : {res['ptq_ppl']:.2f}   "
              f"(+{100*(res['ptq_ppl']-res['fp_ppl'])/res['fp_ppl']:.0f}% vs fp)")

    # ---- train --------------------------------------------------------------
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    def lr_fn(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    need_hs = args.kurt_lambda > 0

    t0, lm_ema, kt_ema = time.time(), None, 0.0
    for step, ids in enumerate(get_batches(tok, device, args.seqlen, args.batch,
                                           args.steps * args.grad_accum,
                                           synthetic=args.synthetic, vocab=vocab)):
        out = model(ids, labels=ids, output_hidden_states=need_hs)
        lm = out.loss
        loss = lm
        if need_hs:
            kt = kurtosis_loss(out.hidden_states)
            loss = lm + args.kurt_lambda * kt
            kt_ema = 0.95 * kt_ema + 0.05 * float(kt)
        (loss / args.grad_accum).backward()
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        # EMA of the PURE LM loss — comparable across --kurt-lambda ablations
        lm_ema = float(lm) if lm_ema is None else 0.95 * lm_ema + 0.05 * float(lm)
        if step % max((args.steps * args.grad_accum) // 10, 1) == 0:
            extra = f"  kurt(ema) {kt_ema:6.2f}" if need_hs else ""
            print(f"  step {step:5d}  lm(ema) {lm_ema:7.4f}{extra}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:6.0f}s")
    model.eval()
    res["final_lm_loss_ema"] = lm_ema
    if need_hs:
        res["final_kurt_loss_ema"] = kt_ema

    # ---- AFTER --------------------------------------------------------------
    if not args.synthetic:
        res["qat_ppl"] = wikitext_ppl(model, tok, device, max_windows=args.windows)
        rec = (res["ptq_ppl"] - res["qat_ppl"]) / max(res["ptq_ppl"] - res["fp_ppl"], 1e-9)
        res["recovery_frac"] = rec
        print(f"[after ] QAT PPL         : {res['qat_ppl']:.2f}   "
              f"(recovered {100*rec:.0f}% of the PTQ damage)")
    res["qat_peak_kurt"] = peak_kurtosis(model, tok, device, args.synthetic, vocab)
    print(f"[after ] peak kurtosis   : {res['qat_peak_kurt']:.1f}  "
          f"(was {res['fp_peak_kurt']:.1f})")

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.save, "qat_state.pt"))
        print(f"saved checkpoint -> {args.save}/qat_state.pt")
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
