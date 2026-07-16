# RANGER Phase 1 — real-model results (SmolLM2-135M; Qwen3-0.6B replication)

**What this is:** the first end-to-end tests of the RANGER mechanisms on real
pretrained models, run entirely on a 6 GB CPU-only dev VM. Methodology is
E7-compatible throughout: teacher-forced perplexity over non-overlapping
ctx-1024 windows on the same 16,384-token WikiText-2 slice. Anchors: bf16
16.150 / fp32 16.140; E7's per-channel W4 = 36.435 (reproduced exactly).
Pre-registered predictions and kill gates were fixed before each run
(workflow `wf_4b49c9dc-3d0`); the harness passed a 15-check self-test and a
12-agent adversarial code review (7 verified findings fixed before any
compute burned). Effects are claimed only above 3× the ~0.05 PPL run noise.

## Headline findings

1. **E7's null result was an artifact by construction — retract it.**
   Under per-channel symmetric absmax RTN, each row's largest weight is
   reconstructed bit-exactly (proven exhaustively over all 65,278 finite
   nonzero bf16 values), so E7's "restore top-|w|" arm restored weights that
   carried no quantization error. Its small K-sweep wiggle is residual
   cleanup on the ~40/64 entries that weren't their row's max (arm 6:
   36.171, n_changed=40). **Nothing about super-weight protection can be
   concluded from E7.**

2. **A pure weight-space census predicts a real model's activation
   outliers.** The recurring super-weight coordinates of the weight census
   (no data, no calibration, no forward pass) — residual channels
   {100, 507, 446, 371, 247, 260} — are the model's measured
   massive-activation channels: activation top-4 {507, 260, 100, 371} all
   sit in the census top-6 (overlap 6/10). The model's dominant super
   weight, found by the activation-spike method of 2411.07191, is
   `layers.11.mlp.down_proj[507, 1229]` (output spike 25,817× median,
   w = 5.875) — the same coordinate the census listed as that tensor's #2
   entry, in the census's write-dominant channel 507.

3. **Massive activations exist at 135M** — well below the ~1B floor
   documented in the literature (residual ch 507 reaches 332× max/median,
   |x| ≈ 19,600) — and they are input-independent (present on a BOS+newline
   prompt), so the residual-side channels are statically exemptable. The
   SwiGLU intermediate is the opposite: its per-token argmax channel moves
   constantly (top channel covers 2–12% of tokens), which is precisely why
   QuaRot must keep its down_proj-input Hadamard online.

4. **Rotation as a 4-bit-activation enabler replicates end-to-end** (the
   synthetic E1 result, now on a real model): A4 per-token quantization
   collapses PPL 16.14 → 13,773; a per-linear folded rotation sandwich
   recovers **92.3%** of the log-PPL collapse (→ 27.11). A8 is near-free
   (16.78).

5. **The census channels causally carry the A4 failure mode — but static
   exemption cannot replace rotation.** Exempting just the 8 census
   channels (1.4% of channels, +0.17 avg activation bits, zero online
   transforms, zero calibration): 13,773 → 254 (**59.1%** log-recovery).
   Control: 8 random channels recover 1.0% (12,864) — full specificity.
   Adding the E8-derived down_proj set: 63.1%. Rotation's 92.3% stands
   apart; the gap is the token-dependent tail (finding 3). **Pillar 1's
   "skip the rotation" upgrade is refuted; "fold the rotation" stands,
   with census exemption as a hardware-trivial partial alternative.**

6. **At 135M there is no exploitable super-weight lever over an honest W4
   baseline — and MSE-clipping actively hurts.** With the corrected
   protocol (exclusion from scale, exact-K by index, n_changed tripwire):

   | E9 arm (equal ~4.13 bpw for g=128 arms) | PPL | note |
   |---|---|---|
   | bf16 | 16.150 | anchor |
   | W4 per-channel (E7 quantizer, 4.03 bpw) | 36.435 | E7 anchor reproduced |
   | W4 g=128 | **24.896** | the honest baseline: grouping alone ≈ half the log-gap |
   | g=128 + MSE clip grid | 26.917 | clip **hurts** end-to-end (weight-MSE ≠ loss) |
   | g=128 + hold-out top-64 (n_changed=63, scales ↓3.0×) | 24.870 | within noise of baseline |
   | g=128 + hold-out top-512 | 25.342 | *worse* than baseline |
   | g=128 + hold-out 20 E8 coords + clip | 26.541 | rescues 0.38 of the clip damage (>3× noise) |

   The Super Weight paper's mechanism is visible exactly where it claims —
   protection rescues aggressive clipping — but at this scale nothing beats
   plain g=128. Scale caveat: super-weight literature is documented from
   ~1B up; hence the Qwen3-0.6B replication below.

7. **A4 failure-site attribution refutes the "down_proj input dominates"
   expectation on this model:** A4 restricted to the residual-stream
   readers collapses hardest (gate/up-only 766, qkv-only 296) vs down-only
   66.8 and o-only 17.2 — consistent with finding 3: the biggest outliers
   live in the residual stream, not the SwiGLU intermediate.

## E10/E11 full table (fp32, hooks proven inert: passthrough == 16.140)

| arm | PPL | log-recovery of A4 collapse |
|---|---|---|
| fp32 anchor / passthrough | 16.140 | — |
| A8 | 16.780 | — |
| A4 | 13,773.5 | 0% |
| A4 + rotation sandwich | 27.114 | **92.3%** |
| A4 + census-8 exemption | 254.28 | 59.1% |
| A4 + census-8 + down-set | 194.40 | 63.1% |
| A4 + 8 random channels | 12,863.8 | 1.0% |
| W4g128 + A8 | 25.627 | deployable near-term point |
| W4g128(rot) + A4(rot) | 53.868 | end-to-end W4A4 point |

Qualifiers: per-token dynamic fake-quant upper-bounds deployable A4; the
QR orthogonal is not a fast Hadamard; rotated arms carry no weight-dtype
confound (whole model fp32).

## Corrections fed back into the theory doc

- §5 Pillar 2: E7 protocol invalid (finding 1); real-model evidence at
  135M shows no lever vs g=128 (finding 6) — scale-gated pending E12.
- §6.5/E5's "split super-weights before rotating" does not transfer to
  real weights at top-8 (census: split-then-rotate gain 1.002× mean,
  max 1.025×) — real weight tensors are too Gaussian for the split to
  matter at the weight level; the constraint remains real for the
  *activation*-side exemption path.
- §5 Pillar 1 scope: "no online rotation needed" is refuted as stated;
  the honest form is "fold what folds; a census exemption buys 59% of
  the A4 recovery for free, rotation buys 92%".
- §3.8: the s_j formula is SmoothQuant's, not AWQ's (AWQ grid-searches
  s = s_x^α); fixed in place.

## E12 — Qwen3-0.6B cross-model replication

*Pending; results appended when the chain completes
(`run_e12_qwen.sh`: weight census → channel extraction → activation
census → bf16 / W4g128 / hold-out-64 / E8-coords arms).*

## Reproduction

```
python selftest_phase1b.py                     # 15 harness checks
python e8_activation_census.py --tokens 4096   # census + super weights
python e9_holdout_w4.py --e8-json e8_activation_census_SmolLM2-135M.json
python e10_actquant.py --e8-json e8_activation_census_SmolLM2-135M.json
./run_e12_qwen.sh                              # Qwen3-0.6B replication
```

All results JSONs are checkpointed per-arm and committed alongside the
run logs. One torch process at a time on the 6 GB VM.
