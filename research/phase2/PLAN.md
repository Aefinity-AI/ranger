# RANGER Phase 2 — training-time pillars (QAT side)

**Status:** plan only (2026-07-16). Phase 1 (PTQ side, `../phase1/RESULTS-phase1.md`)
is complete; everything below needs GPU training the 6 GB dev VM cannot do.
Budget class: one 24–48 GB GPU (theory doc §6); models ≤ 1.5 B.

## What Phase 1 established that reshapes Phase 2

1. Rotation is a scale/architecture-robust ~92% fix for A4 activations —
   but it is post-hoc. The open Pillar-1 question is now purely
   training-time: **can a model be trained so the outliers never form**,
   making even the one irreducible online Hadamard (down_proj input)
   unnecessary?
2. Massive activations exist at 135M (332×) — so the cheapest possible
   QAT testbed (SmolLM2-135M) exhibits the pathology. No need for 1B+
   models to study the mechanism.
3. Static channel exemption works exactly as far as the outliers are
   channel-concentrated (59% at 135M, 25% at 0.6B). Training-time
   suppression attacks the same structure at the source.
4. The super-weight lever is scale-gated (null at 135M, real at 0.6B)
   and clip flips sign with scale — any Phase 2 result must be measured
   at ≥2 scales before generalizing.

## Experiments (cheapest-first, each gates the next)

### P2.1 — Outlier-suppression finetune (the decisive P1 test)
Take SmolLM2-135M as-is. Add (a) learnable gated residual rescaling
(2601.22966), (b) QK-norm if absent, (c) a kurtosis/incoherence penalty
on the residual stream (KurTail-style), active during a short continued
pretrain (0.5–2 B tokens, LR warm restart). Measure before/after:
per-layer residual kurtosis + max/median (the E8 census, reusable as-is),
A4-plain PPL, A4-rot PPL, W4A4 PPL.
- **Prediction:** kurtosis collapses; A4-plain recovers ≥ the rotation
  arm's 92% *without any rotation*. The E8 census channels {100,507,...}
  lose their spikes.
- **Kill:** if the finetune costs >1% bf16 PPL or the sinks migrate to
  new channels instead of vanishing (2603.17771's "the model wants
  them"), record the failure mode — that is itself the Pillar-1 verdict.
- **Cost:** ~1–2 GPU-days. All eval harnesses already exist in phase1/.

### P2.2 — Fold-the-rotation QAT
Same base, but instead of suppressing outliers, bake a fixed Hadamard
into the weights (fold points from the E10 sandwich) and QAT with W4A4
fake-quant so the model *adapts to the rotated basis*. Compare endpoints
of P2.1 vs P2.2 vs plain W4A4-QAT at equal tokens.
- **Prediction:** P2.2 ≥ plain QAT; P2.1 ≥ P2.2 if suppression works
  (cleaner basis beats adapted basis).
- **Cost:** ~2 GPU-days.

### P2.3 — MatFormer ordering vs Hessian (the P2 experiment)
Train a small MatFormer (nested FFN granularities, joint loss) at
~135M–360M from a warm start, then: quantize outer shells hard / inner
slice light (trained ordering) vs KronQ-Hessian mixed precision at equal
average bits vs uniform. This is the "trained ordering ≈ free Hessian"
test the theory doc calls the most interesting single experiment.
- **Kill:** if uniform matches both at equal bits, Pillar 2 is dead at
  this scale for PTQ *and* the ordering claim.
- **Cost:** ~3–4 GPU-days (the MatFormer train dominates).

### P2.4 — AltUp width-vs-bits on a real model (P4)
Only if P2.1–P2.3 leave budget: AltUp-K widen SmolLM2 (K∈{1,2}) with
short QAT, measure kurtosis-vs-K and the bits-at-equal-quality trade
against the synthetic −0.60 bits/2× coefficient (E2) and the E6
redundancy rule (width buys averaging, not a second outlier fix).

## Infrastructure notes
- Eval: reuse phase1 harnesses verbatim (E8 census, e9/e10 PPL arms) so
  Phase 2 numbers are directly comparable to the Phase 1 anchors.
- Provenance: same rules — pre-registered predictions and kill signals
  per run, JSONs + logs committed, n_changed-style tripwires wherever a
  no-op is possible, disjoint-slice confirmation for any small effect.
- Two-scale rule: any positive result at 135M gets a 0.6B replication
  before it enters the theory doc (Phase 1's clip sign-flip is the
  cautionary tale).
- Candidate platforms: single A100/L40S on RunPod/Lambda, or Colab Pro
  A100 for P2.1. HF training stack (transformers Trainer or nanoGPT-style
  loop) — decide per platform.
