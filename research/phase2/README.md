# RANGER Phase-2 — QAT scaffold (the sub-4-bit rescue)

Phase-1 established (and E1/ParetoQ predicted) that **below ~3 bits, post-training tricks
stop working — you must train**. This scaffold is that training: STE fake-quant fine-tuning
with every RANGER mechanism as a flag, so each pillar can be ablated independently.

Reuses the Phase-1 environment — no new dependencies:

```bash
cd research/phase2
source ../phase1/.venv/bin/activate
python selftest_phase2.py          # network-free: validates all 8 mechanisms (~1 min, CPU)
```

## The experiment (three-point protocol)

Every run measures PPL + peak residual-stream kurtosis at three points:
**fp → step-0 (= PTQ damage) → after QAT (= recovery)**. The result you care about is
`recovery_frac`: how much of the PTQ damage training clawed back.

```bash
# baseline QAT at 3 bits (~30 min on a laptop GPU for 135M, 300 steps)
python train_qat.py --model HuggingFaceTB/SmolLM2-135M --bits 3 --steps 300

# the full RANGER stack at 2 bits — the money run
python train_qat.py --model HuggingFaceTB/SmolLM2-135M --bits 2 --steps 800 \
    --superweight-pct 0.005 --nested --protect-downproj --kurt-lambda 0.05

# BitNet-style ternary (needs the most steps of anything here)
python train_qat.py --model HuggingFaceTB/SmolLM2-135M --ternary --steps 2000

# network-free smoke test of the full loop (no model download)
python train_qat.py --synthetic --steps 30
```

Scale up with `--model HuggingFaceTB/SmolLM2-1.7B` / `Qwen/Qwen3-1.7B-Base` on a
24 GB+ GPU (fp32 masters: budget ~4 bytes/param + AdamW state ≈ 12×params in bytes;
1.7B wants gradient accumulation and `--batch 1-2 --grad-accum 8`).

## Flags ↔ theory map

| Flag | Pillar / constraint | Mechanism |
|---|---|---|
| `--bits` + STE (always on) | ParetoQ | learn weights that tolerate rounding — the only route below ~3 bits |
| `--superweight-pct 0.005` | **constraint G** (E5) | top 0.5% \|W\| kept in a trainable fp sparse path, split **before** quantization so outliers can't inflate the dense scale |
| `--nested` | **Pillar 2** | per-channel importance ordering: important half bits+1, rest bits−1, equal average |
| `--protect-downproj` | **Pillar 3** | down_proj (FC2) gets +1 bit — QAT scaling law says FC2 dominates the error budget |
| `--kurt-lambda 0.05` | **Pillar 1** | log-hinge on hidden-state excess kurtosis → trains the residual stream incoherence-native (S2D/KurTail, the 2601.22966 result) |
| `--ternary` | BitNet b1.58 | weights ∈ {−s, 0, +s}, s = mean\|W\| |
| `--rotate-downproj` | QuaRot R4 | exact online Hadamard on down_proj input (x and W rotated by the same H) |

## The ablation grid that would settle the theory (GPU, ~a weekend)

At `--bits 2`, run: none / +sw / +nested / +downproj / +kurt / all — each vs the same
fp and PTQ anchors. RANGER predicts the full stack recovers the most damage at equal
average bits, and that `--kurt-lambda` measurably drops `qat_peak_kurt` vs its ablation.
If the kurtosis flag lowers kurtosis but does NOT improve `qat_ppl`, Pillar 1's
training-time story is in trouble — that is the falsifiable part.

## Honest scope

- STE with dynamic max-abs scales (ParetoQ-style simplicity) — no learned scales (LSQ),
  no distillation loss; both are natural upgrades if the ablation grid looks promising.
- `--rotate-downproj` is the only rotation here; full residual-stream rotation folding
  (QuaRot Q1) requires absorbing RMSNorm scales into adjacent linears — use
  `llm-compressor`'s SpinQuant/QuaRot path for that rather than reimplementing.
- Checkpoints save `state_dict` only (fake-quant modules must be re-created by
  `convert_to_qat` with the same flags before loading).
