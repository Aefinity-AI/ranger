# RANGER — experiment roadmap (beyond Phases 1–2)

Eight follow-on experiments, each with the theory link, the falsifiable claim, and the cost.
Ordered roughly by (evidence value ÷ effort). A–D are inference-only or nearly free;
E–H need training or device work.

---

## A. Cross-family outlier census ★ (tests the theory's spine) — BUILT: `research/census/`

**What:** run `phase1/measure_outliers.py` + the `ptq_eval.py` bit-cliff across ~8 small
models that differ in exactly the architecture knobs the causal chain (§4.1) says matter:

| Model | QK-Norm | Post-norm | Notes |
|---|---|---|---|
| SmolLM2-135M/360M | no | no | Llama-style baseline |
| Llama-3.2-1B | no | no | second no-QKN family |
| Qwen3-0.6B | **yes** | no | the QK-Norm test |
| OLMo-2-1B | **yes** | reordered | second yes-QKN family |
| Gemma-3-1B | yes (G3) | **yes** | + 5:1 local:global |
| Pythia-410M | no | no | GeLU/learned-pos control |

**Falsifiable claim:** peak residual kurtosis (and the emergence-layer spike) is
systematically lower in QK-Norm/post-norm families, and **kurtosis measured in 5 minutes
predicts the 4→3→2-bit PPL cliff** better than parameter count does. If kurtosis and PTQ
damage don't correlate across families, the causal chain — the spine of RANGER — is wrong.
**Cost:** inference only; CPU-viable; an afternoon. **This is the highest evidence-per-FLOP
experiment on the list.**

## B. KV-cache quantization lab (the untested novel coupling)

**What:** inference-time KV quant via the transformers cache API / hooks:
sink-token exemption (first N tokens fp), pre-RoPE vs post-RoPE key quant, per-head vs
per-token scales — and on Gemma-3-1B, **split precision by layer type** (our §2.5 coupling:
local sliding-window KV quantized hard, global KV gently).
**Falsifiable claims:** (i) exempting ≤4 sink tokens rescues most of 2-bit-KV damage;
(ii) local-attention layers tolerate 2-bit KV at ≲half the PPL damage of global layers at
equal bits. Nobody has published the local/global KV-precision split — this is unclaimed
territory. **Cost:** medium implementation, inference only.

## C. Quantized-drafter speculative decoding (risk-free extreme quant, made real)

**What:** SmolLM2-135M drafts for SmolLM2-1.7B (same tokenizer) via transformers'
built-in assisted generation. Quantize the *drafter* progressively 8→4→3→2→ternary;
measure acceptance rate and end-to-end tok/s. Speculative decoding is exact by
construction — a bad drafter can only cost speed, never quality — so this is the one
place ternary is provably risk-free (§2.9).
**Falsifiable claim:** acceptance degrades slowly to ~3-bit then cliffs; even a ternary
drafter nets positive speedup on memory-bound decode. If acceptance collapses at 4-bit,
the "risk-free ternary drafter" story dies. **Cost:** small — assisted generation is
already in transformers; mostly a measurement script.

## D. Super-weight hunt (protect 10 weights, not 0.5%)

**What:** reproduce 2411.07191 on SmolLM2: trace max-|activation| spikes through
down_proj to locate the literal handful of super-weights; confirm by ablation
(zeroing them should crater PPL); then in `ptq_eval.py`, protect only those ~10 weights
during 3-bit RTN vs protecting the top-0.5% blanket.
**Falsifiable claim:** ~10 targeted weights capture most of the benefit of ~700k blanket
ones (≈1000× less fp storage) — which would let Phase-2's `--superweight-pct` shrink to
near-zero cost. **Cost:** small; inference + a short script.

## E. Emergence-layer-targeted regularization (cheaper Pillar 1)

**What:** use A's per-layer outlier map to apply the Phase-2 kurtosis reg only at/around
the massive-emergence layer (2605.08504 says the pathology is localized), instead of all
layers; likewise key per-layer bit allocation to the measured outlier map (Pillar 3 on a
real model). **Falsifiable claim:** targeted reg keeps ≥80% of full-reg benefit at a
fraction of the optimization pressure. **Cost:** trivial code change to phase2 + QAT runs.

## F. Distillation-assisted QAT

**What:** add `--distill` to phase2: KL on logits from the fp teacher (cache teacher
logits for the training slice offline to avoid holding two models). KD is the
best-documented booster for 2-bit recovery; it's the obvious next flag once the ablation
grid runs. **Cost:** moderate memory bookkeeping; known-positive expected value.

## G. Downstream evals beyond PPL

**What:** add ARC-Easy / HellaSwag / BoolQ (small subsets, log-likelihood scoring) to
`phase1/common.py` so the bit cliff and QAT recovery are measured on reasoning, not just
perplexity — PPL is known to hide (or exaggerate) downstream damage at low bits.
**Cost:** small; improves every other experiment's credibility.

## H. Edge-reality check (GGUF + device)

**What:** export QAT checkpoints to GGUF, run llama.cpp bench: actual tok/s, RAM, and
energy on a real edge device (phone/Pi/laptop-iGPU). Also the ExecuTorch path LFM2 uses.
Research numbers mean nothing on the edge until a device confirms them. **Cost:** tooling
work, no training.

---

## The one gap only training can close: Pillar 4 on a real model

Width→quantizability (E6's −0.60 bits per 2× width) has only synthetic evidence, because
it needs width-controlled checkpoints. Cheapest real test: **from-scratch pair on
TinyStories** (~10–30M params — TinyStories makes tiny models coherent): same param count,
one narrow-deep, one 2×-wide-shallow, then run both through the Phase-1 bit cliff.
Falsifiable: the wide one quantizes ~0.5 bit lower at equal fp quality. A weekend GPU job,
and it would complete real-model coverage of all four pillars.

## Suggested order

**A → C → D** (three afternoons, all inference-only, each kills-or-confirms a distinct
theory claim) → **B** (novel coupling, publishable if it holds) → **G** alongside →
**E/F** as Phase-2 upgrades → **TinyStories pair** → **H** when deploying.
