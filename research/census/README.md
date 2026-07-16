# Experiment A — cross-family outlier census

**The claim under test** (the spine of RANGER, report §4.1): activation-outlier structure
is caused by architecture choices, and a 5-minute kurtosis measurement predicts PTQ damage
across model families **better than parameter count does**. Secondary prediction: QK-Norm
families show materially lower peak kurtosis than no-QK-Norm families.

This is the highest evidence-per-FLOP experiment on the roadmap: inference-only, CPU-viable,
and it can *kill the theory* — if kurtosis and W3 damage don't correlate across families,
the causal chain is wrong.

## Run it

```bash
cd research/census
source ../phase1/.venv/bin/activate

python census.py                 # default ungated roster (~5 models, fp32)
python plot_census.py            # -> census.png
```

Fault-isolated and resumable: results accumulate in `census.json` after every model,
errors (gated repo, OOM) are recorded and skipped, rerun to fill gaps. `--force` re-measures.

Default roster (all ungated): SmolLM2-135M, SmolLM2-360M, Qwen3-0.6B-Base (QK-Norm),
Pythia-410M (LayerNorm control), OLMo-2-1B (QK-Norm). If you've accepted the licenses on HF,
strengthen the sample:

```bash
python census.py --models meta-llama/Llama-3.2-1B google/gemma-3-1b-pt
```

Gemma-3-1B matters most as an add-on — it's the only sandwich-norm + sliding-window family
in the roster.

Architecture flags (QK-Norm, sandwich norms, sliding window, norm class) are **detected at
runtime from the loaded modules**, not hardcoded — the census can't be biased by our
expectations of what each family contains.

## Reading the result

`census.json → _analysis`:

| Field | Theory predicts |
|---|---|
| `spearman_kurt_vs_w3damage` | **strongly positive** (the causal chain) |
| `spearman_params_vs_w3damage` | weaker than kurtosis |
| `spearman_weightkurt_vs_w3damage` | weaker than activation kurtosis (control) |
| `mean_peak_kurt_qk_norm` vs `_no_qk_norm` | QK-Norm lower |

With n≈5–7 families, treat correlations as directional, not conclusive — the point is
whether the *ordering* matches the theory, and whether any family is a flagrant
counterexample.

`--selftest` runs the whole pipeline on tiny random models with no network — it validates
the harness, **not** the theory (untrained weights have no real massive activations).

## Memory notes

Models load fp32 for measurement fidelity: 1B ≈ 4 GB RAM. Everything is inference-only —
no optimizer state, so even CPU-only machines can run the full roster (slowly).
