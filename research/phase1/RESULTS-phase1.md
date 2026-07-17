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
   SwiGLU intermediate is the opposite: in most layers its per-token argmax
   channel moves constantly (top channel covers median 6.3% of tokens;
   22/30 layers below 12%), though a concentrated tail exists (layers at
   27.7%, 52.6%, 55.3%, and one at 98.5% — per-layer values in
   e8_activation_census_SmolLM2-135M.json dp_in_argmax_top5). A static
   exemption must hold at EVERY layer to remove the online transform, so
   the hopping majority still forces it online — which is precisely why
   QuaRot keeps its down_proj-input Hadamard online. (Wording corrected
   2026-07-16: an earlier draft said "2–12%," which understated the tail.)

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

## E12 — Qwen3-0.6B cross-model replication (2026-07-16)

Methodology deltas, all recorded in the JSONs: ppl over **8,192** tokens
at **ctx 512** (large-vocab logits transient forced both; within-model
deltas only — never compare absolute PPLs across models), weights bf16,
E9 arms one-per-invocation (`--no-copy`), rotation arm in bf16 with fp32
rotate-then-cast (storeback rounding noted). Qwen3-0.6B is chat-tuned,
hence the high absolute wikitext PPL (34.23 bf16).

**What replicates (the robust core):**

| finding | SmolLM2-135M | Qwen3-0.6B |
|---|---|---|
| massive activations (max/median) | 332× | **1,390×** (ch 35, 25 layers) |
| A8 activation quant | ~free (+4%) | ~free (+2.5%) |
| A4 collapse (vs anchor) | 16.14 → 13,773 | 34.23 → 146,845 |
| **rotation sandwich recovery** | **92.3%** | **92.5%** |
| random-channel exemption | ~0% | ~0% |
| top-\|w\| hold-out vs g=128 | null (Δ0.03) | null (Δ0.03) |
| SW-coords increment | +0.38 over clip | +0.45 over clip |

The rotation number matching to within 0.2 points across a 4.4×
model-size gap and two architectures is the strongest Pillar-1 evidence
this phase produced. The |w|-ranked hold-out null also replicates: the
global-magnitude selection is simply not how super weights work.

**What does NOT replicate (architecture/scale-dependent):**

- **Census→activation overlap collapses: 6/10 → 2/10.** Qwen's #1
  weight-census channel (35) IS its #1 activation channel — the model's
  dominant super weights all write into `down_proj[35, *]` (L2 spike
  51,458×) — but the rest of the census set misses. The weight census
  predicts the *top* coordinate, not the channel *set*, on this
  architecture (Qwen has QK-norm and a 1024-wide residual sampled by
  only top-8 super-weights/tensor — the census may simply be too shallow
  there).
- **Static channel exemption weakens: 59% → 25% → 0%.** With
  activation-measured channels it still recovers a real, specific 25%
  (17,973 vs 146,845; random ≈ 0%), but the census-derived set does
  nothing on Qwen. Qwen's A4 failure is far less channel-concentrated.
  Rotation's advantage over exemption grows from 1.6× (log terms) to
  3.7× — QuaRot-style rotation is the only mechanism that traveled.
- **MSE clipping flips sign: hurts at 135M (+2.0 PPL), helps at 0.6B
  (−6.2 PPL: 48.06 → 41.82).** The Super Weight paper's clip+protect
  recipe becomes net-positive at 0.6B (41.37 total), supporting the
  scale-dependence reading of the 135M null rather than a mechanism
  failure. The coordinate increment on top of clip is small but
  consistent at both scales (+0.38 / +0.45).

**E12 verdict on the round's headline:** "a weight census can replace
online rotation" is dead — but the sharpened, evidence-backed claims
that survive are (1) rotation is a scale- and architecture-robust ~92%
A4 fix; (2) activation-outlier channels are causally identifiable and a
handful of them carry a measurable share of the A4 failure (59% at
135M, 25% at 0.6B); (3) the weight census finds the single dominant
super-weight coordinate on both models for free. All three feed
RANGER's Pillar 1/3 design directly.

## Confirmation pass — disjoint eval slices (2026-07-16)

Per the pre-registered noise rule ("re-run the decisive pair on a second
disjoint slice before believing any small win"), the two small effects
were re-measured on fresh data (SmolLM2: tokens 16,384–32,768; Qwen:
tokens 8,192–16,384; own results files `*_off*.json`):

| claim | original slice | disjoint slice | verdict |
|---|---|---|---|
| clip hurts at 135M | +2.02 (24.90→26.92) | +0.62 (26.97→27.59) | **CONFIRMED** (direction) |
| clip helps at 0.6B | −6.24 (48.06→41.82) | −2.40 (26.46→24.06) | **CONFIRMED** (direction) |
| SW-coords increment, 135M | +0.38 | **−0.015** | **REFUTED — noise** |
| SW-coords increment, 0.6B | +0.45 | +0.24 | **CONFIRMED** (direction) |

Corrections this forces on the findings above: finding 6's "rescues 0.38
of the clip damage" at 135M does not survive fresh data — at 135M the
super-weight coordinate lever is null even on top of clip. At 0.6B it is
real but small. The clip sign-flip with scale stands on both slices of
both models. (Slice-to-slice magnitude variation is large — direction,
not magnitude, is the replicated quantity.)

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
