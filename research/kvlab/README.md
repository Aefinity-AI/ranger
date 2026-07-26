# Experiment B — the KV-cache quantization lab

**Claims under test** (report §2.5/§2.6 — including the one coupling nobody has published):

- **B-1 · sink exemption** — attention sinks park massive KV on the first few tokens
  (2601.22966); keeping **≤4 sink tokens** in fp should rescue most of low-bit KV damage.
- **B-2 · local/global split** (unclaimed territory) — on models interleaving
  sliding-window and global attention (Gemma-2/3 style), local-layer KV has bounded,
  stationary statistics → it should tolerate 2-bit at **≲half the per-layer damage** of
  global-layer KV. A **placebo even/odd split** runs on every model: if the placebo shows
  the same asymmetry, the effect is depth, not locality, and the claim fails honestly.
- **B-3 · RoPE tax** (probe) — keys should quantize measurably worse **after** RoPE than
  before, and worse than values (no RoPE). This is the empirical motivation for the
  p-RoPE ⊕ rotation coupling.

## Run it

```bash
cd research/kvlab
source ../phase1/.venv/bin/activate

python kv_lab.py --model HuggingFaceTB/SmolLM2-135M   # B-1, B-3, K/V asym, placebo
python kv_lab.py --model google/gemma-3-1b-pt         # + the real B-2 split (gated license)
python kv_lab.py --selftest                           # network-free rig validation
```

## How it works (and why to trust it)

`FakeQuantKVCache` subclasses transformers' `DynamicCache` and quantize-dequantizes every
incoming K/V chunk **at cache-update time** — the real streaming path, operating on
post-RoPE keys, with per-layer bit policies and absolute-position sink exemption. PPL is
computed by chunked evaluation *through* the cache.

**The anchor:** the fp-cache arm must equal the plain full-forward PPL exactly. The
selftest asserts this (observed: 526.3914 vs 526.3914) — if the rig ever diverges from
the model's native math, no quantized number downstream would be trustworthy.

**Fair split design:** group sizes differ (Gemma is 5:1 local:global), so mixed 2/4-bit
assignments would have unequal average bits between arms. Instead each arm quantizes
**only one group to 2-bit (rest fp)** and compares **damage per quantized layer**.

## Arms & outputs

| Arm | Question |
|---|---|
| bits sweep KV{8,4,3,2} | where does the KV cliff sit? |
| sink ∈ {0,1,4,16} at KV2/KV3 | B-1 — % of damage rescued per exempted token |
| K2/V4 vs K4/V2 | which of K/V is fragile? (theory: K, via RoPE) |
| local-only@2 vs global-only@2 (+ placebo) | B-2 — per-layer group fragility |
| RoPE-tax probe | pre-RoPE vs post-RoPE key quant error, V as control |

Output: console verdicts + `kv_results.json`.

## Notes

- Selftest runs on a tiny random model: rig equality, logit-perturbation ordering,
  sink-exemption identity, per-layer policy differentiation, probe execution. Random
  weights carry no trained KV structure, so PPL *direction* and the real RoPE tax are
  only meaningful on real weights.
- The cache stores dequantized fp tensors — this measures the **quality** effect of KV
  quantization; memory savings are the (well-understood) kernel-side implementation.
- SmolLM2/Qwen3 have no sliding-window interleave — they run B-1/B-3 + placebo only.
  B-2's treatment arm needs Gemma-3-1B (or Gemma-2-2B), license-gated on HF.
