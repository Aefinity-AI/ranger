# RANGER mock run — results

**What this is:** mechanism tests of the RANGER theory on synthetic tensors engineered to
reproduce the documented pathologies of real LLMs — massive-activation channels
(2402.17762), attention-sink tokens (2601.22966), super-weights (2411.07191). No GPU, no
real model; pure numpy, fixed seed (`20260714`). Run with `python3 research/experiments/mock_run.py`.

**Why it counts:** every RANGER prediction rests on a mathematical mechanism. If a mechanism
fails on clean synthetic data, the corresponding pillar is in trouble. So these can *falsify*,
and in fact one sub-hypothesis was falsified (E1, below) and two design constraints were
discovered (E5, E1). That is the point — a mock run that can't kill a darling isn't a test.

## Scorecard

| Test | Mechanism | Result | Headline number |
|---|---|---|---|
| **E0** | quantizer sanity: MSE ∝ R²/2²ᵇ | **PASS** | fit slope −2.20 (theory −2.00) |
| **E1** | Pillar 1 — rotation kills channel outliers | **PASS** (scope corrected) | incoherence 8.4×↓, kurtosis 156×↓, wins W4A4 1.23× |
| **E2** | Pillar 4 — capacity–precision duality | **PASS** | **−0.60 bits per 2× width** (predicted −0.50) |
| **E3** | Pillar 2 — importance-ordered precision | **PASS** | Hessian-opt 2.3×, noisy MatFormer-proxy 1.8× |
| **E4** | VQ blessing of dimensionality | **PASS** | +9.4 dB (2-D), +10.5 dB (4-D) over scalar |
| **E5** | combined stack under pressure | **PASS** | RANGER 0.927 < AWQ 0.941 < QuaRot 0.995 @ W2A4 |

## The three things we actually learned

### 1. Pillar 4 (the riskiest claim) is VALIDATED — and I had to fix my own test to see it
The capacity–precision duality says AltUp-style widening by `K` buys ≈ ½·log₂K fewer bits.
My first test embedded the target through a ReLU with a linear "undo," so the widened network
had 90–920% full-precision error — it wasn't computing the target at all. Rebuilt as an **exact
linear redundant embedding** (`Hact = Y·R`, readout `W2 = pinv(R)`, fp error ≈ 1e-14), the
measured bit-saving is **−0.60 bits per doubling of width** — within striking distance of the
−0.50 prediction, and if anything slightly *better*. Width and bits are genuinely on one frontier.

### 2. Rotation and sparse-outlier protection are BASIS-INCOMPATIBLE (new design constraint)
E5 v1 combined a Hadamard rotation with "keep the top 0.5% of the *rotated* weights in FP16."
It failed — because **rotation delocalizes the super-weights**: after rotating, the concentrated
outliers are smeared across the whole matrix, so there is nothing sparse left to protect.
Fix: **split the super-weights off *before* rotating**, rotate the outlier-free dense remainder,
and run the sparse part as a parallel FP path on unrotated activations. With that ordering RANGER
wins (0.927 vs QuaRot 0.995). *Consequence for the theory:* Pillars 1 (rotation) and 2/3
(sparse/nested precision) must operate on **orthogonal axes** — rotate the dense bulk, protect the
sparse tail separately. This tightens the compatibility matrix.

### 3. Rotation is a 4-bit-activation enabler, NOT a sub-4-bit one (a sub-hypothesis, FALSIFIED)
I predicted rotation's advantage would *grow* as bits drop. The sweep says otherwise: rotation
wins at **W4A4 (1.23×)** but is slightly *worse* than plain per-token quant at W3A3 and W2A2
(0.92×). Why: when outlier channels carry real signal that dominates the output, per-token quant
*accidentally preserves them accurately*, while rotation democratizes precision across all
channels — hurting at ultra-low bits. This is not a bug; it's the truth, and it matches **ParetoQ
(2502.02631)**: below ~3 bits you must do **QAT**, not post-hoc tricks. So Pillar 1's scope is
"make 4-bit activations viable" — sub-4-bit activation quality is a training-time (QAT) problem.

## Honest limitations
- Synthetic tensors, not a real model. These test *mechanisms*, not end-to-end LLM quality.
- No QAT loop here (STE training is Phase 1–2 of the plan and needs a real small model).
- The outlier model (gain, count, sink strength) is hand-set; real distributions vary by family.
- E5's parallel sparse path uses two activation quantizations for clarity; a real kernel fuses them.

## E6 — the P1 × P4 compound test (`sweep_width_bits.py`, `sweep.png`)

**Question:** do rotation (Pillar 1) and width (Pillar 4) *compound* against the 4-bit activation
floor, or overlap? This needed two redesigns — v1 used a dense random embedding that *pre-Gaussianized*
the outliers (so there was no floor for rotation to break; confounded result), and v1 also inherited
E2's ill-conditioned K=1 square-inverse artifact. v2 uses orthonormal embeddings, a true raw-outlier
baseline, kurtosis tracking, and continuous-dB measurement.

**Answer: PARTIAL — complementary across error *types*, redundant within the outlier type.**

- **Outlier axis → redundant.** Rotation and width are both "spread energy across the basis," so both
  crush activation kurtosis: raw 58.7 → rotation 1.6 (38×) → width ~5 (11×). At a W?A4 budget, rotation
  alone buys **+5.3 dB**, width alone **+9.3 dB**, and **both together +9.4 dB — rotation adds only
  +0.1 dB on top of width.** Once one spreader has run, the other is nearly a no-op on outliers.
- **Averaging axis → width is unique.** On a clean (outlier-free) signal, where rotation does nothing,
  width still buys **+2.3 dB per 2× (~0.4 bit/2×)** from redundancy-averaging of the residual error.
- **This corrects the theory.** The compatibility matrix had marked *AltUp ⊕ rotation* as a clean
  synergy (🔗 "width→incoherence"). It is actually **largely redundant on that axis**. Width's genuine,
  non-overlapping value is averaging + capacity — not a second outlier fix.
- **Design rule that falls out:** use **one** spreader for outliers (rotation ≪ 16× width in cost),
  and spend width on averaging + capacity. Folded into report §4.3 (coupling 2) and §6.5.

## Net effect on the theory
Three of four pillars have their core mechanism confirmed on synthetic data (P1 at 4-bit, P2
including a noisy trained-ordering proxy, P4 with the coefficient in range); the fourth (VQ shaping,
supporting the per-role formats of Pillar 3) is confirmed. Two refinements now fold back into the
main report and compatibility matrix: the **orthogonal-axis constraint** and the **4-bit scope of
rotation**. Next real step remains Phase 1 on a ≤1.5 B open model (SmolLM2 / Qwen3), where the QAT
mechanisms E1/E2 can't fully exercise here get their end-to-end test.
