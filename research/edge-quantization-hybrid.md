# RANGER: An Incoherence‑Native, Elastic‑Precision Hybrid for Legendary Edge Intelligence

**A deep cross‑reference of bleeding‑edge (mid‑2026) edge‑model architecture and quantization, the math that makes each work, their convergence points, and a concrete, testable hybrid recipe.**

Author: research synthesis prepared on branch `claude/edge-quantization-hybrid-research-9clrdh`
Date: 2026‑07‑14
Status: theory + experiment plan (nothing here is trained yet — every quantitative claim is either cited to a primary source or flagged as a *prediction*)

---

## 0. TL;DR (the legendary claim in one breath)

> **Quantizability is a *trainable architectural property*, not a post‑hoc treatment.** Every winning quantization method and every winning edge‑architecture trick is, underneath, attacking the **same** scalar quantity: the **outlier / heavy‑tail structure of the weight and activation tensors** (massive activations, attention/residual sinks, super‑weights), weighted by **local loss curvature** (the Hessian). If you *co‑design* the model so that (1) its residual stream is **incoherence‑native** (outliers structurally suppressed and a fixed rotation folded into the weights, so there is **zero inference‑time rotation cost**), (2) its capacity is **elastically nested** (a high‑precision inner core with progressively lower‑precision outer shells, using the model's *own trained importance ordering* instead of an estimated one), and (3) its per‑role bit budget is **curvature‑matched**, then you reach a strictly better **accuracy‑per‑bit‑per‑joule** frontier than architecture tricks or quantization tricks can reach alone.

I call the recipe **RANGER — Rotation‑Aligned, Nested‑precision, Gaussianized Elastic Representation.** The four pillars each make a falsifiable prediction and each is runnable on a single 24–48 GB GPU using ≤1.5 B‑parameter open models. The rest of this document is the evidence, the math, the compatibility analysis, the predictions, and the experiment plan — followed by the honest list of bottlenecks, framed in plain language, where I want your help.

---

## 1. The landscape as of mid‑2026 — what is *actually* bleeding edge

### 1.1 Gemma 4 (Google DeepMind, arXiv 2607.02770, 2 Jul 2026) — the reference point

This is the model the prompt calls "Gemma4." It is real and twelve days old at the time of writing. Full architecture, from the primary source:

| Model | Total / Active | Notes |
|---|---|---|
| E2B | 5 B total / **2.3 B effective** | Per‑Layer Embeddings (PLE), 4:1 local:global |
| E4B | 8 B total / **4.5 B effective** | PLE, 5:1 local:global |
| 12B | 12 B dense | **encoder‑free** (raw image patches + raw 40 ms audio → single matmul) |
| 26B‑A4B | 26 B total / **3.8 B active** | **Mixture‑of‑Experts** |
| 31B | 31 B dense | leading *open dense* model on Arena |

Design choices that matter for us:

- **Per‑Layer Embeddings (PLE)** — inherited from Gemma 3n. The embedder for E2B carries ~2.34 B of the 5 B parameters; these are *per‑layer* embedding tables that can live in CPU / fast storage and be streamed in per layer, so the accelerator only ever holds the 2.3 B "effective" compute weights. **This decouples knowledge capacity (total params) from accelerator memory & compute (active params).**
- **Local : global attention = 5:1** (4:1 for E2B), sliding‑window local layers, few global layers. Cuts KV‑cache size dramatically because only the global layers keep full‑context KV.
- **KV‑cache sharing** (Shazeer MQA lineage) **+ "reuse keys as values" in global layers** (`values = keys`, citing Kayyam et al. 2026, *"Do transformers need three projections?"*). Together with **p‑RoPE** (partial RoPE, `p = 0.25` on global layers; Barbero et al. 2025) this **reduces the global KV cache by 37.5 %.** RoPE base frequencies: 1 M (global) / 10 k (local).
- **QK‑Norm**, **pre‑norm *and* post‑norm** with RMSNorm. (Both matter for quantization — see §4.)
- **Thinking mode** (reasoning traces before answering).
- **Multi‑Token‑Prediction (MTP) drafter head** for speculative decoding — a 4‑layer transformer that cross‑attends to the main model's KV; for E2B/E4B it does top‑k over token *clusters* so the final projection shrinks from `d × 262 000` to `d × 4096`.
- **Quantization‑Aware Training (QAT)** shipped in two formats:
  - **mobile**: per‑channel **mix of int2 and int4 weights + int8 activations**;
  - **Q4_0**: block‑wise 4‑bit (llama.cpp format).
  - Crucially: *"to enable stable inference in fp16, we introduce a scalar scale at each block in order to bound the activation ranges to fit fp16."* — this is Google openly admitting Gemma's **activation‑outlier problem** and patching it with per‑block rescaling. Hold that thought; it is the whole ballgame in §4.
- **QAT on the encoders**: image encoder W8A8 → 2× memory, −44 % latency; audio encoder → **per‑layer‑cluster {2,4,8}‑bit weights** + int8 activations → **−78 % on‑disk** (390 MB → 87 MB) *while transcription/translation quality went up 10–17 % vs Gemma 3n.* This is a live, shipping example of **curvature‑matched mixed precision** — pillar 3 of RANGER already exists inside Gemma 4's audio stack.

Memory footprint (text‑only, from the report): E2B 4.6 GB bf16 → **0.8 GB** mobile‑quant; 31B 64 GB bf16 → **19.2 GB** Q4_0. KV at 32k adds 0.05–1.1 GB.

Benchmark leaps (thinking mode): E2B ≈ Gemma 3 27B at **~10× fewer params**; 31B hits MMLU‑Pro 85.2, AIME‑2026 89.2, LiveCodeBench v6 80.0, GPQA‑Diamond 84.3.

> **Aside worth stating plainly:** the paper's own Arena‑Text leaderboard (19 Jun 2026) puts **Claude Fable 5 at rank 1, Elo 1508** — the top of the table, above every open model including Gemma 4 31B (rank 43, Elo 1451). The "legendary status" target in the prompt is, per Google's own measurement, already the reference ceiling. The interesting question is therefore not "can a small model beat Fable 5" — it is **"how far down the size/precision curve can we push while holding the quality frontier,"** which is exactly what RANGER attacks.

### 1.2 The rival stack

| Family | Key architectural idea | Why it matters here |
|---|---|---|
| **Qwen3** (2505.09388) | unified thinking/non‑thinking, dense + MoE, "thinking budget" | MoE small variants; strong small‑model quality baseline |
| **Phi‑4‑mini** (2503.01743) | GQA + **Mixture‑of‑LoRAs** for multimodal | LoRA adapters = a natural high‑precision residual path (see §5, pillar 2) |
| **SmolLM2 / 3** (2502.02737) | data‑centric overtraining of a 1.7 B model | best fully‑open small baseline for *our* experiments |
| **LFM2 (Liquid)** (2511.23404) | **gated short convolutions + GQA blocks**, hardware‑in‑the‑loop architecture search, CPU‑first | the strongest non‑attention‑heavy edge design; constant‑size state |
| **Mamba‑2‑Hybrid / Zamba / Nemotron‑H / Apriel‑H1** (2406.07887, 2405.16712, 2504.11409, 2511.02651) | interleave **SSM (constant‑state recurrence)** with a few attention layers | kills the KV‑cache‑grows‑with‑length problem; but SSM has *its own* quantization pathology (MambaQuant, 2501.13484) |
| **MobileLLM / PhoneLM / BlueLM‑V** (2411.05046, 2411.10640, 2507.05934) | deep‑and‑thin, weight sharing, on‑device co‑design | evidence that depth‑over‑width + sharing is edge‑optimal |

The convergent trend across *all* of them: **thinking mode + long‑context KV compression + shipped low‑bit weights.** Everyone is standing on the same three legs. The differentiation is *how* they compress the KV cache (sliding window vs SSM state vs sharing) and *how* they quantize (PTQ vs QAT).

---

## 2. Architecture atoms and their math

Each "atom" is a reusable mechanism. I give the mechanism, the one‑line math, the win, and — the part everybody skips — **its effect on quantizability** (developed fully in §4).

### 2.1 Per‑Layer Embeddings (PLE) — Gemma 3n/4
- **Mechanism:** factor the token representation into a shared trunk plus a *per‑layer* learned embedding `e_ℓ(token)` streamed from CPU at layer ℓ.
- **Math:** `h_ℓ = f_ℓ(h_{ℓ-1}) + P_ℓ · e_ℓ(x)`, where `e_ℓ` lives off‑accelerator. Capacity grows with `Σ_ℓ |e_ℓ|` while accelerator FLOPs/memory track only `f_ℓ`.
- **Win:** 5 B knowledge at 2.3 B compute cost.
- **Quantizability:** PLE tables are **read‑only lookups on the memory‑bound path** → they can be quantized *independently and very aggressively* (they never enter a matmul that accumulates), and because they are CPU‑resident, their bit‑width trades disk/RAM, not accelerator SRAM. **PLE is a free low‑bit surface.**

### 2.2 MatFormer nested elasticity — Devvrit et al. 2310.07707, used in Gemma 3n/4
- **Mechanism:** train one network so that many nested sub‑networks (`Mix'n'Match`) are simultaneously valid models. E2B is literally the inner slice of E4B.
- **Math:** for FFN width `d_ff`, train on a set of granularities `g ∈ {d_ff/8, d_ff/4, d_ff/2, d_ff}` with a joint loss `Σ_g L(model_g)`. The inner slice is optimized to be a standalone model.
- **Win:** one training run → a whole elastic family; pick the size at deploy time.
- **Quantizability (this is the key latent insight):** MatFormer **induces a trained importance ordering on the weights** — inner slices are more important because they must function alone. That ordering is *exactly what mixed‑precision quantization needs and normally has to estimate with a Hessian.* RANGER pillar 2 exploits this.

### 2.3 AltUp (Alternating Updates) — Baykal et al. 2301.13310, used in Gemma 3n
- **Mechanism:** widen the token embedding by a factor `K` but only process one `1/K` block per layer, *predicting* the rest and *correcting* it cheaply (predict‑and‑correct).
- **Math:** representation `x ∈ ℝ^{K·d}` split into K blocks; layer updates block `i`, a lightweight linear predictor updates the others: `x̂_{j≠i} = A x_i`, then a correction mixes. Cost ≈ one width‑`d` layer; capacity ≈ width‑`Kd`.
- **Win:** up to **+87 % effective capacity at ~flat latency** (paper's number on SuperGLUE/SQuAD).
- **Quantizability (the duality — RANGER pillar 4):** widening the representation by `K` spreads the same signal energy over `K×` more coordinates. Per‑coordinate variance drops ~`1/K`; the vector becomes **more Gaussian and more incoherent** (max‑abs/`‖·‖₂` shrinks toward the `1/√(Kd)` incoherent floor). **Wider representations are intrinsically more quantizable.** AltUp gives that width *cheaply* — so it is the ideal partner for aggressive bit reduction.

### 2.4 LAuReL (Learned Augmented Residual Layer) — Menghani et al. 2411.07501
- **Mechanism:** generalize the residual `x + f(x)` to `x + g·f(x) + low‑rank learned cross‑layer mixing`.
- **Math:** `h_{ℓ+1} = α·h_ℓ + f(h_ℓ) + (U Vᵀ) h_ℓ` with `U,V` low‑rank; α, g learned scalars.
- **Win:** better quality per parameter; can replace a chunk of FFN width.
- **Quantizability:** the learned residual gain `α`/`g` **bounds residual‑stream growth**, which is where massive activations accumulate (see §4.1). A well‑behaved residual gain is a mild *outlier suppressor*. (This is the same lever that "A Unified View of Attention and Residual Sinks," 2601.22966, isolates as the thing to make learnable/gated.)

### 2.5 Local:global attention interleaving + sliding window — Gemma 2/3/4
- **Mechanism:** most layers attend within a window `w`; a few attend globally.
- **Math:** KV memory `≈ (n_global·L_ctx + n_local·w)·d_kv`, vs `n_layers·L_ctx·d_kv` for full attention.
- **Win:** near‑linear KV in context length for the common case.
- **Quantizability:** local layers have **bounded, stationary** attention statistics → far fewer KV outliers than global layers. The *global* layers are where RoPE‑induced KV outliers live (RotateKV 2501.16383, PolarQuant 2502.00527) — so you can **quantize local KV hard and global KV gently**, a natural mixed‑precision split.

### 2.6 KV‑cache sharing + keys‑as‑values + p‑RoPE — Gemma 4
- **Mechanism:** share KV heads across query heads (MQA/GQA), reuse keys as values in global layers, and rotate only a fraction `p` of dims (p‑RoPE).
- **Math:** p‑RoPE applies position rotation to `p·d_head` dims and leaves `(1−p)·d_head` position‑agnostic. Gemma 4: `p=0.25`.
- **Win:** −37.5 % global KV.
- **Quantizability (a genuine synergy for RANGER pillar 1):** RoPE is the main source of KV‑cache outliers because it injects high‑frequency, position‑dependent structure that fights uniform quantization. **The `(1−p)` un‑rotated fraction is a subspace you are free to Hadamard‑rotate for incoherence** without disturbing positions. **p‑RoPE literally *enlarges the rotatable subspace* that QuaRot/SpinQuant need.** Nobody has (publicly) exploited this coupling yet — it is one of the crossover points the prompt asked for.

### 2.7 MoE — Gemma 4 26B‑A4B, Qwen3
- **Mechanism:** route each token to `k` of `N` experts.
- **Math:** capacity `∝ N`, compute `∝ k`.
- **Quantizability caveat:** experts see fewer tokens each → **higher per‑expert curvature variance and router fragility** → they quantize *worse* than dense layers. Practice: keep router + any shared/always‑on expert at higher bits. (This is a *conflict*, catalogued in §4.4.)

### 2.8 SSM / linear‑attention hybrids — LFM2, Mamba‑2‑Hybrid, Nemotron‑H
- **Mechanism:** replace most attention with a constant‑size recurrent state (SSM) or gated conv; keep a few attention layers.
- **Math:** state update `s_t = A s_t‑1 + B x_t`, output `y_t = C s_t`; memory `O(d_state)` independent of length.
- **Win:** flat memory in context length — the ultimate long‑context edge trick.
- **Quantizability caveat:** the parallel‑scan and gate projections have **channel‑variance pathologies** (MambaQuant 2501.13484 fixes them with variance‑aligned rotations + Karhunen‑Loève). SSMs are *not* free lunch under low bits; they need their own rotation.

### 2.9 MTP drafter / speculative decoding — Gemma 4
- **Mechanism:** a tiny model drafts several tokens; the big model verifies in parallel.
- **Win:** 2–3× decode throughput, *lossless* (exact same distribution).
- **Quantizability:** orthogonal — but the drafter itself can be ternary/2‑bit since draft errors are corrected by verification. **The drafter is the one place extreme quantization is genuinely risk‑free**, because wrong drafts are simply rejected.

---

## 3. Quantization atoms and their math

### 3.1 The one equation everything orbits

Uniform quantizer, step `Δ`, `b` bits over dynamic range `R = max|x|`:

```
Δ = R / 2^b        MSE ≈ Δ²/12 = R² / (12 · 2^{2b})
```

Two dials, and only two: **`R` (set by outliers)** and **`b` (bits)**. Error falls 4× per bit but rises with the *square* of the largest value. **A single outlier that doubles `R` costs you a full bit of precision for every other weight sharing that scale.** This is why the entire field is, at bottom, an outlier‑management field. Everything below is a different way to shrink `R`, spend `b` where curvature demands it, or change the *geometry* so `R` is the wrong quantity.

### 3.2 Quantization‑Aware Training (QAT) — Gemma 4 ships it
- **Mechanism:** insert fake‑quant `x → Δ·round(x/Δ)` in the forward pass, backprop through it with a straight‑through estimator (STE); the network *learns weights that tolerate rounding*.
- **Scaling laws (new, load‑bearing):**
  - **"Scaling Law for QAT"** (2505.14302): quant error decomposes predictably in (model size, training tokens, granularity); bigger models and finer granularity shrink it; the FC2 (second FFN) layer dominates the error budget.
  - **"Compute‑Optimal QAT"** (2509.22935): the optimal fraction of training spent in QAT is predictable from **tokens‑per‑parameter‑byte**; more compute → more QAT.
  - **ParetoQ** (2502.02631): there is a **sharp phase boundary near 3 bits**. Above it (≥4‑bit) PTQ is ~free. Below it (≤2‑bit, ternary), you *must* do QAT, and there is a Pareto‑stable frontier where **1.58‑bit ternary and 2‑bit are genuinely competitive** — but only via QAT, not PTQ.
- **Win:** recovers most of the low‑bit gap; the only way into the ≤2‑bit regime.
- **Cost/conflict:** needs training compute; ternary‑from‑scratch is the expensive extreme (§4.4, bottleneck B).

### 3.3 Learned rotations / incoherence processing — QuaRot, SpinQuant, QuIP#, KurTail, DuQuant
- **Mechanism:** insert an orthogonal `Q` and its inverse around a linear op: `y = (xQ)(Qᵀ W)`. The math is unchanged (`QQᵀ=I`) but `xQ` and `QᵀW` are **incoherent** — energy spread across coordinates, outliers gone.
- **Math (the crux — "incoherence"):** a matrix `W ∈ ℝ^{m×n}` is **μ‑incoherent** if `max_{ij}|W_{ij}| ≤ μ·‖W‖_F/√(mn)`. Multiplying by a random/Hadamard rotation makes any matrix μ‑incoherent with `μ = O(√(log(mn)))` w.h.p. — i.e. the max entry drops to ~`√(log n)/√n` of the Frobenius norm. Since `R = max|·|`, **rotation shrinks `R` from O(1) toward O(1/√n)** → the `R²` in §3.1 collapses. QuaRot uses randomized Hadamard (free, structured); SpinQuant *learns* `Q` on the Stiefel manifold via Cayley optimization to squeeze out the last bit.
- **Win:** end‑to‑end **W4A4** (weights + activations + KV all 4‑bit) at small loss — previously impossible because activations wouldn't go below 8‑bit.
- **Cost/conflict:** the Hadamard transforms that *cannot* be folded into weights (the ones inside the residual/attention) must run **online at inference** → 5–10 % latency, extra kernels. And rotation **does not commute with RoPE** (§4.4, bottleneck A). RANGER pillar 1's whole point is to make the rotation *foldable and free.*

### 3.4 Vector / codebook / trellis quantization — AQLM, VPTQ, QuIP#, QTIP, GPTVQ, Grouped‑Lattice‑VQ
- **Mechanism:** quantize *groups* of `d` weights jointly to the nearest entry in a learned/lattice codebook, instead of `d` independent scalars.
- **Math (the "blessing of dimensionality," GPTVQ 2402.15319):** a scalar quantizer tiles ℝ^d with a cubic `Zᐟ` lattice; that is provably suboptimal. Joint VQ can use a *better* lattice (e.g. **E8**, QuIP#) or a **trellis** (QTIP — effectively infinite dimension). Two gains stack:
  - **shaping gain** (round cells → spheres): up to ~**1.53 dB** (~0.25 bit) in the high‑dim limit;
  - **space‑filling/packing gain** from the lattice.
  - Net: VQ reaches accuracy at **~2 bits** that scalar needs ~3 bits for (AQLM, VPTQ, QuIP# all Pareto‑dominate GPTQ below 3 bits).
- **Trellis (QTIP 2406.11235):** a bit‑shift trellis gives ultra‑high effective dimension with `O(1)` state → best published rate‑distortion, and it is *compute*, not *lookup*, so it dodges VQ's memory‑bound gather.
- **The counter‑punch (GSQ 2604.18556, Sep 2026):** carefully optimized **scalar** quantization with Gumbel‑Softmax grid learning can *match* VQ at low bits **while staying hardware‑friendly** (no codebook gather). This matters for RANGER — it means we can get most of the VQ gain without the edge‑hostile lookup.
- **Cost/conflict:** codebook **gather is memory‑bound and slow on edge NPUs** without fast random access (§4.4, bottleneck C).

### 3.5 Extreme low‑bit ternary — BitNet b1.58, ParetoQ
- **Mechanism:** weights `∈ {−1, 0, +1}` (1.58 bit = log₂3), activations int8. Matmul → **add/subtract only** (no multiplies).
- **Math:** `W̃ = clip(round(W/γ), −1, 1)`, `γ = mean|W|`; forward uses accumulation, so on a ternary accelerator (VitaLLM 2604.27396, LUT designs 2604.25183) energy per op collapses.
- **Win:** at ≥3 B params, BitNet b1.58 ≈ FP16 on perplexity + downstream, at a fraction of the energy; ternary hardware is now being taped out.
- **Cost/conflict:** must be **trained ternary from scratch** (QAT); you cannot PTQ your way to good ternary. HGF (2602.05269) shows a fix: a **ternary backbone + a low‑rank FP16 correction path with adaptive gates** recovers quality and stabilizes training — a hybrid that RANGER borrows directly.

### 3.6 Microscaling formats — MXFP4, NVFP4, IF4 / Adaptive Block‑Scaled
- **Mechanism:** a small block of elements shares a scale.
  - **MXFP4:** block of 32, shared 8‑bit **E8M0** (power‑of‑two) scale, each element **E2M1** (4‑bit float).
  - **NVFP4:** block of **16**, shared **E4M3** (FP8) scale **+** a second per‑tensor FP32 scale → finer granularity, less error (ARCQuant 2601.07475 pushes it further with augmented residual channels; Adaptive Block‑Scaled / IF4 2603.28765 picks int‑vs‑float per block).
- **Math:** shrinking the block from 32→16 and using a *floating* block scale reduces the effective `R` seen by each element (each block gets its own range) — it is **fine‑grained per‑block outlier control in hardware.**
- **Win:** **hardware‑native on Blackwell**‑class parts → real speed, not just memory. This is where the industry is standardizing for 4‑bit.
- **Relation to rotation:** microscaling and rotation are *complements* — rotate to kill the worst outliers, then let per‑block scales mop up the residual heavy tail. (QuantVGGT 2509.21302 and many 2026 recipes do exactly this pairing.)

### 3.7 Hessian / curvature‑aware error minimization — GPTQ, OBQ, KronQ, VPTQ, GWQ
- **Mechanism:** minimize not raw weight error but **output** error, weighted by the layer‑input Hessian `H = XXᵀ`: `min_{Ŵ} (W−Ŵ) H (W−Ŵ)ᵀ`.
- **Math:** GPTQ solves this greedily with the OBQ closed form, using `H⁻¹` to *compensate* remaining weights for each rounding. **KronQ (2607.07964, 8 Jul 2026, six days old)** makes `H` tractable at scale by approximating it as a **Kronecker product** `H ≈ H_1 ⊗ H_2`, giving second‑order accuracy at near‑diagonal cost.
- **Win:** the standard "smart rounding" that every good PTQ pipeline starts from.
- **RANGER use:** the *cheap* Kronecker‑factored Hessian is exactly the sensitivity signal pillar 3 needs to set per‑role bits — but MatFormer (pillar 2) gives a *trained* ordering that may make the Hessian estimate unnecessary for the coarse split. Testable which wins.

### 3.8 Activation‑aware scaling — AWQ, SmoothQuant
- **Mechanism:** migrate difficulty from (hard‑to‑quantize) activations to (easy) weights via a per‑channel scale `s`: `y = (x/s)(sW)`; choose `s` to protect the salient channels AWQ finds via activation magnitude.
- **Math:** `s_j = (max_i|x_{ij}|)^α / (max_i|W_{ij}|)^{1−α}` — balance the ranges.
- **Win:** cheap, calibration‑light, ubiquitous; the baseline weight‑only 4‑bit method.
- **Relation:** a *diagonal* (per‑channel) special case of the rotation idea — scaling is a diagonal transform; rotation is the full orthogonal generalization. RANGER's folded rotation subsumes it.

### 3.9 Mixed‑precision / low‑rank residual — ResQ, OWQ, Super‑Weight, HGF
- **Mechanism:** keep a **tiny** high‑precision path alongside the low‑bit bulk: outlier channels (OWQ), a low‑rank residual (ResQ), a handful of **super‑weights** (2411.07191 — literally *one* weight can dominate PPL), or a gated FP16 correction (HGF).
- **Math:** `W ≈ Q_low(W) + L`, `L = U Vᵀ` rank‑`r` in FP16, `r ≪ d`. The low‑rank part captures the top singular directions that carry the outliers.
- **Win:** a few % of parameters in FP16 buys back most of the extreme‑quant loss.
- **RANGER use:** this *is* pillar 2's outer‑to‑inner precision gradient, and it is where Phi‑4‑mini's **Mixture‑of‑LoRAs** becomes a gift — the LoRA adapters are already the high‑precision residual path.

### 3.10 Calibration‑free — SINQ, PolarQuant
- **SINQ** (2509.22944): Sinkhorn‑Knopp dual‑axis normalization removes the need for calibration data by balancing row/column scales — matters for edge, where you may not have representative calibration data.
- **PolarQuant** (2603.29078): block normalize → Walsh‑Hadamard rotate → Gaussian‑matched quantize; near‑lossless, no calibration.

---

## 4. The cross‑reference — the unifying variable, and who plays nice with whom

### 4.1 The shared bottleneck: outliers are one phenomenon with one cause

A remarkable convergence in the 2024–2026 interpretability literature: **massive activations, attention sinks, residual sinks, and super‑weights are the same phenomenon.**

- **Massive Activations** (Sun et al. 2402.17762): a few activation dimensions are 1000–10000× larger than the rest; they act as **learned constant biases** and drive attention concentration.
- **Attention sinks** (StreamingLLM lineage; Active‑Dormant heads 2410.13835): the model dumps "no‑op" attention onto a few tokens (often BOS). Those sink tokens carry the massive activations.
- **A Unified View of Attention and Residual Sinks** (2601.22966, Jan 2026): both are **outlier‑driven rescaling** interacting with RMSNorm; make the rescaling **learnable / gated** and you **improve W4A4 quantization robustness.** ← direct architecture→quantization causal link.
- **A Single Layer to Explain Them All** (2605.08504, May 2026): massive activations emerge at a *specific* "Massive Emergence Layer" via a **RMSNorm × FFN** interaction — so the pathology is *localizable* and therefore *fixable at that layer.*
- **Super Weight** (2411.07191): a *tiny* set of weights create the massive activations; protect them (FP16) and low‑bit quant suddenly works.
- **Attention Sinks Induce Gradient Sinks** (2603.17771): in pre‑norm nets the sinks are an *adaptive response to gradient concentration* — i.e. the model *wants* them for training stability. **You cannot just delete them; you must give the model a cleaner mechanism for the same job.** This is why "just clip the outliers" fails and *architectural* fixes (QK‑Norm, gated attention) succeed.

**The mechanism chain (this is the spine of the whole document):**

```
pre‑norm training stability need
      → gradient concentration on a few tokens (gradient sinks)
      → model parks a "constant bias" there (attention/residual sink)
      → that bias shows up as MASSIVE ACTIVATIONS in a few residual dims
      → those dims blow up R = max|x|
      → uniform quantization error ∝ R² explodes
      → low‑bit quantization fails
```

Every quantization method is a different intervention on this chain:
- **AWQ/SmoothQuant** rescale after the fact (diagonal).
- **QuaRot/SpinQuant/QuIP#** rotate the outliers away (orthogonal).
- **Mixed‑precision/Super‑Weight/OWQ** exempt the outlier channels (sparse FP16).
- **VQ/trellis** change the *geometry* so `R` isn't the right measure (lattice shaping).
- **QAT** teaches the weights to tolerate the rounding.

And every *architecture* fix intervenes **earlier in the chain**, before the outlier forms:
- **QK‑Norm** (Gemma 4) bounds attention logits → smaller sinks.
- **Gated/clipped attention** ("Quantizable Transformers" 2306.12929) lets heads "do nothing" without a massive activation → **full INT8 for free.**
- **Learnable residual gain (LAuReL / gated rescaling)** bounds residual growth.
- **p‑RoPE** removes the RoPE high‑frequency that seeds KV outliers.
- **AltUp width** dilutes per‑dim magnitude toward the incoherent floor.
- **S2D spectral decay** (2602.14432) regularizes the top singular directions that carry the outliers, *during* fine‑tuning.

> **This is the convergence the prompt asked for.** Architecture tricks and quantization tricks are the same fight at two ends of one causal chain. The optimal system does not pick a side — it **suppresses the outlier at the source *architecturally*, folds a rotation to clean the residue *for free*, and spends bits by curvature.** That is RANGER.

### 4.2 The capacity–precision duality (why "outside the box" is literally a real axis)

Two knobs buy quality:
- **width / capacity** (AltUp, PLE, MatFormer, MoE) — cheap params/FLOPs;
- **precision / bits** (everything in §3).

They are **duals through the incoherence coefficient.** Widen by `K` (AltUp) → per‑dim variance ↓ ~`1/K` → incoherence `μ` ↓ → the `R` in §3.1 ↓ → you can afford ~`½·log₂K` fewer bits at equal error. Conversely, spend bits and you can afford less width. The **efficient frontier** is a curve in (width, bits) space, and the **"safe crossover point"** the prompt asked about is where the two marginal returns balance:

```
minimize   L(width w, bits b, nesting depth s)
subject to  Compute(w,s) + Memory(w,b,s) ≤ Budget
KKT / equal‑marginal‑returns condition at the optimum:
   ∂L/∂w · (∂Budget/∂w)⁻¹  =  ∂L/∂b · (∂Budget/∂b)⁻¹  =  ∂L/∂s · (∂Budget/∂s)⁻¹
```

In words: **at the legendary point, one more bit, one more slice of AltUp width, and one more nesting shell all buy the same quality per joule.** Most current systems are *far* off this balance — they max one knob (e.g. "just do 4‑bit") and leave the others on the table. RANGER co‑optimizes all three.

### 4.3 Compatibility matrix — what stacks and what fights

Legend: ✅ synergistic · ➖ neutral/orthogonal · ⚠️ conflict (needs care) · 🔗 novel coupling RANGER exploits

| | QAT | Rotation | VQ/trellis | Ternary | Microscale FP4 | Mixed‑prec / low‑rank | KV‑quant |
|---|---|---|---|---|---|---|---|
| **PLE** | ✅ | ➖ | ✅ (offload table) | ✅ | ➖ | ✅ | ➖ |
| **MatFormer** | ✅ | ➖ | ✅ | ⚠️ | ✅ | 🔗 **nested precision** | ➖ |
| **AltUp** | ✅ | 🔗 **width→incoherence** | ✅ | ✅ | ✅ | ✅ | ➖ |
| **QK‑Norm / gated attn** | ✅ | 🔗 **less rotation needed** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **local:global** | ✅ | ✅ | ✅ | ✅ | ✅ | 🔗 **split KV precision** | ✅ |
| **p‑RoPE** | ✅ | 🔗 **enlarges rotatable subspace** | ➖ | ➖ | ➖ | ➖ | ✅ |
| **MoE** | ⚠️ router | ✅ | ⚠️ per‑expert | ⚠️ | ✅ | ✅ (router FP) | ➖ |
| **SSM/LFM2** | ✅ | ⚠️ **needs variance‑aligned rotation (MambaQuant)** | ✅ | ⚠️ | ✅ | ✅ | n/a (small state) |
| **MTP drafter** | ✅ | ➖ | ✅ | 🔗 **risk‑free extreme quant** | ✅ | ➖ | ➖ |

The four 🔗 couplings are the paper‑worthy, under‑exploited crossover points:
1. **p‑RoPE ⊕ rotation** — partial RoPE hands rotation a clean subspace to work in.
2. **AltUp ⊕ low‑bit** — width buys incoherence *and* an averaging bonus. ⚠️ **Corrected by the mock run (§6.5):** width's incoherence gain is *redundant* with rotation (both are "spread energy across the basis," so once one has run, the other adds ~0 dB on the outlier axis). Width's *unique*, non‑redundant contribution is (a) **redundancy‑averaging** of the residual clean quant error (~0.4 bit / 2× width, which rotation cannot give) and (b) **capacity** (the original AltUp point). So the honest rule is: use **one** spreader for outliers (rotation is far cheaper than 16× width for that job), and spend width on averaging + capacity — *not* as a second outlier fix. This is complementarity across error *types*, not a compounding stack on the outlier axis.
3. **MatFormer ⊕ mixed‑precision** — trained importance ordering *is* the bit‑allocation prior.
4. **QK‑Norm/gated attention ⊕ rotation** — kill the sink at the source, so the rotation is smaller/foldable and the online‑Hadamard latency largely disappears.

### 4.4 The conflicts, stated honestly (these are §7's plain‑language bottlenecks)
- **A. Rotation ⟂ RoPE.** RoPE rotates dim‑pairs by a position‑dependent angle; a fixed incoherence rotation `Q` on those same dims does not commute, so you scramble position. QuaRot works around it by rotating only the residual/value/output spaces. p‑RoPE's un‑rotated fraction is the clean fix.
- **B. Ternary needs from‑scratch QAT.** You cannot PTQ into good 1.58‑bit; ParetoQ is explicit. So ternary is a *design‑time* choice, not a deploy‑time one — costly on a modest GPU. HGF's low‑rank FP correction is the mitigation.
- **C. VQ codebook gather is memory‑bound on edge NPUs.** Great compression, poor edge throughput unless the hardware has fast gather. QTIP's trellis (compute not lookup) and GSQ's hardware‑friendly scalar are the escapes.
- **D. MoE experts quantize worse.** Fewer tokens/expert → higher curvature variance + router fragility. Keep router + shared expert at higher bits.
- **E. SSM has its own outliers.** The parallel scan/gate projections need variance‑aligned rotation (MambaQuant) before low‑bit.
- **F. Online‑Hadamard latency.** Rotations that can't fold cost 5–10 % decode. RANGER pillar 1 is precisely the attempt to make them foldable/free.
- **G. Rotation ⟂ sparse‑outlier protection (found empirically — see §6.5).** A Hadamard rotation *delocalizes* the super‑weights: after rotating, the concentrated outliers are smeared across the whole matrix, so there is nothing sparse left to keep in FP16. **Pillars 1 (rotation) and 2/3 (sparse/nested precision) must therefore live on orthogonal axes** — split the super‑weights off *before* rotating, rotate the outlier‑free dense bulk, and carry the sparse tail on a parallel FP path over unrotated activations. Ignoring this makes the two mechanisms *fight* (the combined stack lost to plain AWQ until the ordering was fixed).

---

## 5. RANGER — the hybrid recipe and its testable predictions

**RANGER = Rotation‑Aligned · Nested‑precision · Gaussianized · Elastic Representation.** Four pillars, each a falsifiable hypothesis.

### Pillar 1 — Incoherence‑Native residual stream (kill the outlier at the source; make the rotation free)
- **Build:** QK‑Norm + **gated/clipped attention** (2306.12929) + **learnable gated residual rescaling** (2601.22966 / LAuReL) so sinks never need to become massive activations; **p‑RoPE** so a `(1−p)` subspace is position‑free; a **fixed Hadamard mixing folded into the adjacent linear weights** (so it costs *nothing* at inference, unlike online QuaRot Hadamards); an **outlier/kurtosis regularizer** (S2D spectral decay + KurTail‑style kurtosis penalty) active during the QAT phase.
- **Prediction P1:** the incoherence‑native model reaches **W4A4 and W2A8 at <2 % MMLU/PPL degradation with *no online rotation*,** vs >8 % degradation (or a mandatory online Hadamard) for a vanilla same‑size model. *Falsified if* it still needs online rotation to hit W4A4, or if the reg penalty costs >1 % full‑precision quality.
- **Cheap proxy metric:** track **activation excess kurtosis** and **max‑abs/‖·‖₂ (incoherence)** per layer during training; P1 predicts these fall monotonically with the reg and correlate (r < −0.7) with post‑quant accuracy.

### Pillar 2 — Nested elastic precision (MatFormer ordering = the mixed‑precision prior)
- **Build:** train a **MatFormer** family; keep the **inner slice at 8‑bit (or FP)**, wrap **progressively lower‑precision shells** (mid 4‑bit, outer 2‑bit/ternary). Because the inner slice is trained to stand alone, it is *provably the most important* — so bits and importance are aligned **by construction**, no Hessian search needed for the coarse split. Add a **low‑rank FP16 residual (HGF/ResQ/Mixture‑of‑LoRAs)** to catch super‑weights.
- **Prediction P2:** at **equal average bits**, nested‑precision ≥ uniform precision **and** ≥ Hessian‑sensitivity mixed‑precision, on MMLU/PPL — because the ordering is trained‑in, not estimated. *Falsified if* uniform or estimated‑sensitivity matches it within noise.

### Pillar 3 — Curvature‑matched per‑role formats
- **Build:** allocate formats by tensor role, using a **cheap Kronecker‑factored Hessian (KronQ)** for the fine split:
  - **FFN weights** → VQ/trellis or GSQ scalar (blessing of dimensionality; they're dense/Gaussian after Pillar 1).
  - **Attention/global KV** → rotation + microscale FP4; **local KV** → harder low‑bit (they're benign).
  - **PLE tables** → aggressive low‑bit codebook, CPU‑resident (free surface).
  - **Router / shared expert / super‑weights** → FP16 exceptions.
  - **MTP drafter** → ternary (verification makes it risk‑free).
- **Prediction P3:** per‑role mixed formats beat a single global format at equal total bytes by **≥1 MMLU point / ≥5 % PPL**, mirroring Gemma 4's own audio‑encoder {2,4,8} result generalizing to the LLM.

### Pillar 4 — Ride the capacity–precision duality
- **Build:** use **AltUp** to widen the representation cheaply, then **spend the incoherence dividend on fewer bits.** Co‑tune `(K, b)` toward the equal‑marginal‑returns point of §4.2.
- **Prediction P4:** an **AltUp‑K× widened** model quantizes to **~½·log₂K fewer bits** at equal quality vs a same‑active‑param baseline — *and* the activation kurtosis drops measurably with `K`. This is the single cleanest test of the whole thesis: **width and bits are on one frontier.** *Falsified if* widening doesn't lower kurtosis or doesn't buy bits.

**The composite legendary target:** a ~1 B‑active model that, at **≈2.5–3 effective bits and no online rotation**, holds within **2–3 %** of its own FP16 quality *and* within a chosen margin of a much larger reference — i.e., pushing the whole (size × precision) product down the frontier while staying on the quality ceiling.

### 6.5 Mock‑run results — mechanisms tested on synthetic tensors (`research/experiments/`)

Before touching a GPU, each pillar's *mechanism* was tested on synthetic tensors engineered to
reproduce the documented pathologies (massive‑activation channels, sink tokens, super‑weights).
Full write‑up in `research/experiments/RESULTS.md`; harness in `mock_run.py` (pure numpy, seed
`20260714`). Headlines:

- **Pillar 4 validated — the riskiest claim held.** A correct linear redundant‑embedding test
  (exact fp reconstruction) measured **−0.60 bits saved per 2× width**, vs the predicted −0.50.
  Width and bits are on one frontier.
- **Pillar 2 validated, including the trained‑ordering proxy.** Importance‑ordered precision beat
  uniform 2.3× at equal average bits; a *noisy* MatFormer‑style ordering (correlation 0.89 with
  the true Hessian) still beat uniform 1.8× — supporting "trained ordering ≈ free Hessian."
- **Pillar 1 confirmed but scope‑corrected.** Hadamard rotation cut activation incoherence 8.4×
  and kurtosis 156× and won at **W4A4 (1.23×)** — but a sub‑hypothesis ("edge grows as bits drop")
  was **falsified**: at W3A3/W2A2 plain per‑token quant matched or beat it, because signal‑bearing
  outlier channels are accidentally preserved by per‑token scaling. **Rotation is a 4‑bit‑activation
  enabler; sub‑4‑bit activations are a QAT problem** (consistent with ParetoQ). Scope narrowed
  accordingly.
- **New constraint G discovered** (see §4.4): rotation and sparse super‑weight protection are
  basis‑incompatible; split super‑weights *before* rotating. With that fix the combined RANGER stack
  wins at W2A4 (0.927 vs AWQ 0.941 vs QuaRot 0.995).
- **VQ shaping gain confirmed** (+9–10 dB over scalar at equal bits/dim), supporting Pillar 3's
  FFN→VQ format choice.

These are mechanism tests on synthetic data, not end‑to‑end LLM validation; the QAT‑dependent parts
of P1/P4 still need Phase 1 on a real ≤1.5 B model. But the load‑bearing math survived contact with
numbers — and where it didn't (the two items above), the theory was corrected rather than the test.

**E6 — the P1 × P4 compound test (`sweep_width_bits.py`, `sweep.png`).** The decisive follow‑up:
do rotation (P1) and width (P4) *compound* against the 4‑bit activation floor, or overlap? Answer:
**partial — complementary across error *types*, redundant within the outlier type.**
- **Outlier axis → redundant.** Both rotation and width are "spread energy across the basis," so
  both crush activation kurtosis (raw 58.7 → rotation 1.6, → width ~5). At W?A4, rotation alone buys
  +5.3 dB, width alone +9.3 dB, and **both together +9.4 dB — rotation adds only +0.1 dB on top of
  width.** Once one spreader has run, the other has almost nothing left to do on outliers.
- **Averaging axis → width is unique.** On a clean (outlier‑free) signal where rotation does nothing,
  width still buys **~+2.3 dB per 2× (~0.4 bit/2×)** from redundancy‑averaging of the residual error.
- **Design rule (now in §4.3 coupling 2):** use **one** spreader for outliers — rotation is far
  cheaper than 16× width — and spend width on averaging + capacity, not as a second outlier fix.
  This *corrects* the earlier framing of AltUp⊕rotation as a clean synergy: it is largely redundant
  on the axis it was claimed to help, and synergistic only on the axes (averaging, capacity) that
  rotation never touched.

---

## 6. Experiment plan — melt exactly as many GPUs as necessary (single 24–48 GB is enough)

**Base models (fully open, small):** SmolLM2‑1.7B (2502.02737), Qwen3‑1.7B (2505.09388), and Gemma‑3n‑E2B where license permits — so results transfer to the Gemma line.

**Toolchain:** `llm-compressor` / GPTQ / AutoAWQ for PTQ baselines; QuaRot + SpinQuant reference code for rotations; QuIP#/AQLM/QTIP or `vptq` for VQ; a small custom QAT loop (STE) for the incoherence‑native and ternary pieces; **LLMC (2405.06001)** as the unified evaluation harness so every point is measured identically. Report **WikiText‑2 PPL, C4 PPL, MMLU, GSM8K**, plus **on‑device decode tok/s** via llama.cpp.

**Phased schedule (each phase gates the next):**

| Phase | Question | Compute | Kill/greenlight signal |
|---|---|---|---|
| **0. Instrumentation** | measure kurtosis + incoherence + KronQ Hessian per layer on all baselines | hours, 1 GPU | reproduce known massive‑activation layers → tool trust |
| **1. Pillar 4 (cheapest, most diagnostic)** | does AltUp width lower kurtosis and buy bits? | ~1–2 GPU‑days QAT fine‑tune at K∈{1,2,4} | P4 kurtosis‑vs‑K trend; if flat → rethink |
| **2. Pillar 1** | incoherence‑native → W4A4/W2A8 with no online rotation? | ~2–4 GPU‑days QAT | P1 threshold vs vanilla + QuaRot baselines |
| **3. Pillar 2** | nested precision ≥ estimated mixed‑precision at equal bits? | reuse a MatFormer‑style run; ~2 GPU‑days | P2 head‑to‑head on the Pareto plot |
| **4. Pillar 3** | per‑role formats beat global at equal bytes? | days, PTQ+light QAT | P3 ≥1 MMLU pt |
| **5. Composite** | all four together on the (size×precision) frontier | ~1 GPU‑week | beat the best single‑method point on accuracy‑per‑bit‑per‑joule |

**Ablation discipline:** every claim is a *difference at equal budget* (equal bits, or equal bytes, or equal active‑params), plotted on a Pareto frontier, with FP16 and the strongest published single method (SpinQuant W4A4, QuIP# 2‑bit, QAT‑int4) as the two anchors. No cherry‑picking a single bit‑width.

---

## 7. Bottlenecks, framed simply — where I want your help

You said: *if you hit bottlenecks, frame them in simple terms.* Here are the six real ones, in plain language, each with the workaround I'd try first and the open question I'd want to reason through with you.

1. **The spin and the compass want the same knob (rotation ⟂ RoPE).**
   *Plain:* To hide the loud outliers we "spin" the numbers (a rotation). But the model's sense of word *position* (RoPE) is also a spin on the same numbers. Spin everything and you scramble word order.
   *Workaround:* Gemma 4 already only spins ¼ of the position dims (p‑RoPE), leaving ¾ free to clean. Rotate only that free ¾.
   *Open question for you:* is `p=0.25` the right split, or do we want to *learn* which dims are "position dims" vs "cleanable dims" jointly with the rotation? That's a small optimization I can set up if you think it's worth the compute.

2. **1.58‑bit is a diet you must be born on.**
   *Plain:* Ternary (−1/0/+1) models are amazing but you can't shrink a normal model down to ternary after the fact — you have to train it ternary from scratch, which is expensive.
   *Workaround:* only make the *outer shell* and the *drafter* ternary (cheap, and the drafter's mistakes get caught), keep an inner core at higher bits, and bolt on a tiny full‑precision "correction lane" (HGF).
   *Open question:* how thin can the correction lane be before quality falls off? I suspect rank ≈ 8–16 is enough; worth a sweep.

3. **The phone book problem (VQ lookup).**
   *Plain:* The best compression looks each weight up in a shared "phone book" (codebook). On a phone's NPU, doing millions of random look‑ups is slow even though the file is tiny.
   *Workaround:* use a *trellis* (QTIP) that *computes* the value instead of looking it up, or a cleverly‑tuned plain scalar scheme (GSQ) that nearly matches the phone‑book quality but runs on any hardware.
   *Open question:* which one your target hardware likes is an empirical call — do we have a specific edge chip in mind (Blackwell‑class GPU vs a phone NPU vs a ternary ASIC)? The answer changes pillar 3's format choices.

4. **Experts are fragile (MoE).**
   *Plain:* In a Mixture‑of‑Experts, each expert only saw a slice of the data, so it's more delicate — round it too hard and it breaks.
   *Workaround:* spend a few extra bits on the traffic‑cop (router) and any always‑on expert; quantize the rest normally.
   *Open question:* do we even want MoE for the *edge* target? MoE helps quality‑per‑active‑FLOP but hurts quality‑per‑*byte* (you still store all experts). For phones, dense + PLE may dominate; for a Blackwell edge box, MoE wins. Depends on your deployment.

5. **Measuring "delicate" costs almost as much as fixing it (the Hessian).**
   *Plain:* To know which weights are sensitive we need curvature info (the Hessian), which is expensive to compute.
   *Workaround:* KronQ's Kronecker trick makes it cheap; *and* MatFormer may hand us the sensitivity ordering for free (the inner slice is the sensitive part, by training design), so we might skip the measurement for the coarse split.
   *Open question:* is the trained MatFormer ordering actually a good proxy for the Hessian ordering? That's directly testable in Phase 3 and is, honestly, the most interesting single experiment here.

6. **Free rotation isn't obviously free (the folding assumption).**
   *Plain:* My whole pillar‑1 pitch is that we can "bake in" the outlier‑cleaning rotation so it costs nothing at run time. That's true for rotations that sit *next to* a matmul (you multiply them into the weights ahead of time). It's *not* automatically true for the rotations that sit inside attention.
   *Workaround:* combine source‑suppression (QK‑Norm, gated attention) so the in‑attention rotation is small/unnecessary, and only fold the ones adjacent to linears.
   *Open question:* the residual‑stream rotation between blocks — can it be fully folded, or does the per‑layer PLE injection break the fold? I think PLE (which adds a vector mid‑stream) may force a small online step. Worth checking early; if it breaks, pillar 1's "zero cost" becomes "small cost" and the P1 prediction needs softening.

---

## 8. Primary sources (all verified on the Hugging Face papers index)

**Architecture**
- Gemma 4 Technical Report — arXiv **2607.02770** (Jul 2026)
- Gemma 3 — 2503.19786 · Gemma 2 — 2408.00118
- MatFormer — 2310.07707 · AltUp (Alternating Updates) — 2301.13310 · LAuReL — 2411.07501
- p‑RoPE / "Round and round we go" — Barbero et al. 2025 (ICLR) · "Do transformers need three projections?" — Kayyam et al. 2606.04032
- Qwen3 — 2505.09388 · Phi‑4‑mini — 2503.01743 · SmolLM2 — 2502.02737 · LFM2 — 2511.23404
- Mamba — 2312.00752 · Mamba‑2‑Hybrid empirical — 2406.07887 · Zamba — 2405.16712 · Nemotron‑H SSM pruning — 2504.11409 · Apriel‑H1 — 2511.02651
- MobileLLM/PhoneLM — 2411.05046 · BlueLM‑V‑3B — 2411.10640 · BlueLM‑2.5‑3B — 2507.05934

**Quantization — rotations / incoherence**
- QuaRot — 2404.00456 · SpinQuant — 2405.16406 · QuIP# — 2402.04396 · KurTail — 2503.01483 · DuQuant — 2406.01721 · ParoQuant — 2511.10645 · PolarQuant (KV) — 2502.00527 · PolarQuant (Gaussian/Hadamard) — 2603.29078 · RotateKV — 2501.16383 · ResQ — 2412.14363

**Quantization — vector / codebook / trellis**
- AQLM (Additive) — 2401.06118 · VPTQ — 2409.17066 · QTIP — 2406.11235 · GPTVQ — 2402.15319 · PV‑Tuning — 2405.14852 · Grouped‑Lattice‑VQ — 2510.20984 · SSVQ — 2503.08668 · GSQ (scalar matches VQ) — 2604.18556

**Quantization — extreme low‑bit / formats / QAT**
- BitNet b1.58 — 2402.17764 · bitnet.cpp — 2410.16144 · ParetoQ — 2502.02631 · HGF (ternary + low‑rank) — 2602.05269 · VitaLLM (ternary HW) — 2604.27396 · LUT ternary HW — 2604.25183
- MXFP/NVFP: Adaptive Block‑Scaled / IF4 — 2603.28765 · ARCQuant (NVFP4) — 2601.07475 · ZeroQuant‑FP — 2307.09782
- QAT scaling: Scaling Law for QAT — 2505.14302 · Compute‑Optimal QAT — 2509.22935 · k‑bit inference scaling laws (Dettmers) — 2212.09720

**Quantization — Hessian / activation‑aware / calibration‑free**
- GPTQ‑lineage / KronQ — 2607.07964 · AWQ — 2306.00978 · SmoothQuant (via AWQ refs) · GWQ — 2411.00850 · OWQ — 2306.02272 · SINQ — 2509.22944 · LLM.int8() — 2208.07339 · TurboBoA — 2602.04929 · GPTAQ/residual — 2604.07955

**The outlier bridge (architecture ↔ quantization)**
- Massive Activations — 2402.17762 · Super Weight — 2411.07191 · Unified View of Attention & Residual Sinks (→W4A4) — 2601.22966 · Massive Emergence Layer — 2605.08504 · Attention Sinks Induce Gradient Sinks — 2603.17771 · Active‑Dormant Heads — 2410.13835 · Quantizable Transformers (gated attention→INT8) — 2306.12929 · S2D Selective Spectral Decay — 2602.14432 · Intriguing Properties of Quantization at Scale — 2305.19268 · MambaQuant — 2501.13484

**Tooling / benchmarks**
- LLMC toolkit — 2405.06001 · PTQ benchmark taxonomy — 2502.13178

---

*Nothing in §5–§6 has been trained yet; those sections are a hypothesis and a plan. Everything in §1–§4 and §8 is sourced to the papers above. The next step is Phase 0–1 of §6, which fits on a single modern GPU.*
