# Experiment D — the super-weight hunt

**The claim under test** (Super Weight, arXiv 2411.07191 + report §3.9): a literal
**handful of individual weights** — not 0.5% ≈ hundreds of thousands — create the
massive-activation pathway, and protecting just those few during low-bit quantization
recovers most of blanket outlier protection's benefit at **~1000× less fp storage**.
If true, Phase-2's `--superweight-pct` cost collapses to near zero.

## Run it

```bash
cd research/superweights
source ../phase1/.venv/bin/activate

python hunt.py --model HuggingFaceTB/SmolLM2-135M    # CPU-fine
python hunt.py --selftest                            # network-free, planted ground truth
```

## The three-part protocol

1. **HUNT** — iterative spike tracing through every `down_proj`: if output channel *j*
   spikes massively at a token where input channel *i* spikes, then `y_j ≈ W[j,i]·x_i`
   is dominated by one weight. Find it, zero it, re-run, repeat. The **spike-decay
   elbow** across rounds shows how many true super weights the model has (papers report
   1–6 per model). The `dominance` column (≈1.0 = one weight explains the whole spike)
   is the confidence signal per find.
2. **ABLATE** — zeroing the found weights should **crater** PPL; zeroing the same number
   of random weights should do nothing. This is the causal confirmation.
3. **PROTECT** — W3 RTN with four arms at equal quant settings:
   `none` / `sw_top3` (3 fp weights) / `sw_all` (~10) / `blanket_0.5pct` (~hundreds of
   thousands). The report prints each arm's **% of blanket's recovery** — the falsifiable
   number. All-found ≥ ~70–80% of blanket would confirm the claim.

Output: console tables + `superweights.json` (coordinates, values, dominance, all PPLs).

`--selftest` plants a super weight at known coordinates in a tiny random model and
asserts the hunt finds exactly it, ablation distinguishes it from random controls, and
every protection arm runs — validation with ground truth, no network.

## Notes

- Detection assumes the paper's finding that super weights live in `down_proj`; the scan
  covers all such layers and ranks globally, so depth is discovered, not assumed.
- Real models' spikes are ~1000× rms; the selftest's planted spike is modest (random
  weights don't build the amplification chain trained models do) — the selftest validates
  the *machinery*, real weights provide the *phenomenon*.
- Runs comfortably on CPU at 135M (PPL evals dominate the runtime).
