#!/usr/bin/env bash
# Phase 1 model acquisition — public repos, no token required.
# Primary: Qwen3-0.6B (modern arch: GQA + QK-norm, the design RANGER targets).
# Iteration model: SmolLM2-135M (fits fully in RAM on the 6 GB dev VM).
set -euo pipefail
VENV=/home/killboxincorporated/ranger-venv
export HF_HUB_ENABLE_HF_TRANSFER=0

"$VENV/bin/hf" download HuggingFaceTB/SmolLM2-135M \
  --include "*.safetensors" --include "*.json" --include "tokenizer*" --include "merges.txt"
"$VENV/bin/hf" download Qwen/Qwen3-0.6B \
  --include "*.safetensors" --include "*.json" --include "tokenizer*" --include "merges.txt"
echo "MODELS DOWNLOADED"
