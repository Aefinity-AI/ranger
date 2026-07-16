# RANGER Phase-1 — run the theory against a real model on your PC

This is the bridge from the synthetic mechanism tests (which all passed / were
corrected) to a **real open model**. Everything here is CPU-safe: the 135M model
runs on a laptop in a couple of minutes. Scale up to 360M / 1.7B when you have a GPU.

---

## 1. Pull it onto your machine

In your home shell (where you run Claude Code):

```bash
# clone the repo and switch to the research branch
git clone https://github.com/Aefinity-AI/ranger.git
cd ranger
git checkout claude/edge-quantization-hybrid-research-9clrdh

# open Claude Code right here so it has the full context (theory + experiments)
claude          # (or: code .  /  cursor .  — whatever you use)

cd research/phase1
```

If you already cloned it, just `git fetch origin` then
`git checkout claude/edge-quantization-hybrid-research-9clrdh && git pull`.

## 2. Set up the environment

```bash
bash setup.sh                 # makes .venv, installs deps, runs a smoke test
source .venv/bin/activate
```

CPU is fine to start. For GPU, install the CUDA torch wheel instead of the CPU one
(see the comment in `setup.sh`), everything else is unchanged.

## 3. Get the model + run the first two experiments

The models are **Apache-2.0 and ungated** — no login or license click needed. The
first run auto-downloads the weights (135M ≈ 270 MB) to your HF cache.

```bash
# Experiment 0 — does a REAL model have the outlier pathology RANGER targets?
python measure_outliers.py --model HuggingFaceTB/SmolLM2-135M

# Experiments 1+2 — PTQ bit-sweep, Pillar 2 (ordered precision), Pillar 1 (rotation)
python ptq_eval.py --model HuggingFaceTB/SmolLM2-135M
```

Swap the model any time:
`--model HuggingFaceTB/SmolLM2-1.7B` or `--model Qwen/Qwen3-1.7B-Base`.

## What each script tests (maps to the theory)

| Script | RANGER link | Question |
|---|---|---|
| `measure_outliers.py` | premise of §4 | Does the real residual stream have massive-activation channels / a kurtosis-spike emergence layer? |
| `ptq_eval.py` A | ParetoQ cliff | Where does PPL break as bits drop 4→3→2? |
| `ptq_eval.py` B | **Pillar 2** | Does importance-ordered mixed precision beat uniform at equal avg bits? |
| `ptq_eval.py` C | **Pillar 1** | Does Hadamard rotation cut weight-quant error on real activations? |

Outputs: `outliers.json`, `ptq_results.json` (plus console tables).

## What's deliberately NOT here yet (Phase 2 — needs a GPU)

- **Full activation-rotation inference (QuaRot/SpinQuant end-to-end).** The mechanism
  check in `ptq_eval.py` C is layer-local; the full online-Hadamard forward is a bigger
  rewrite. Use the maintained implementations:
  - `pip install git+https://github.com/Dao-AILab/fast-hadamard-transform` (CUDA kernel)
  - `llm-compressor` (SpinQuant/GPTQ/AWQ production pipelines)
  - AMD Quark's QuaRot tutorial for a turnkey rotate→quantize flow.
- **QAT loop** (the only route below ~3 bits, per ParetoQ). Scaffold plan:
  1. wrap each `nn.Linear` weight in a fake-quant (STE) at the target bits,
  2. add the incoherence-native pieces (QK-norm is already in these models; add the
     kurtosis/spectral-decay reg from S2D as an aux loss on the hidden states),
  3. fine-tune ~100–500 steps on a slice of FineWeb-Edu / SlimPajama,
  4. re-run `measure_outliers.py` and `ptq_eval.py` to see the outliers shrink and the
     low-bit PPL recover.
- **AltUp width sweep on a real model** (Pillar 4) — needs a width-elastic checkpoint or
  a from-scratch tiny run; the synthetic result (`../experiments/sweep.png`) stands in for now.

## The one honest caveat

Phase-1 is **post-training** (PTQ) and **weight-side**. It can confirm the outlier premise,
the bit cliff, Pillar 2, and Pillar 1's mechanism — but the headline sub-4-bit and
capacity claims are **training-time** effects that only Phase-2 QAT can settle. Start here,
read the numbers, then decide whether to spin up the GPU phase.
