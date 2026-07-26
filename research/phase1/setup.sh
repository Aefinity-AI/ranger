#!/usr/bin/env bash
# RANGER Phase-1 setup — creates a venv, installs deps, runs a smoke test.
# Usage:  bash setup.sh   (from research/phase1/)
set -e

PYTHON=${PYTHON:-python3}
echo "==> creating virtualenv .venv"
$PYTHON -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> upgrading pip"
pip install --upgrade pip >/dev/null

# CPU torch by default; if you have CUDA, replace with the matching cu12x wheel
# (see https://pytorch.org/get-started/locally/).
if python -c "import torch" 2>/dev/null; then
  echo "==> torch already present"
else
  echo "==> installing CPU torch (for GPU, install the CUDA wheel instead)"
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> installing the rest"
pip install transformers datasets accelerate safetensors numpy matplotlib sentencepiece

echo "==> smoke test: FWHT + quantizer round-trip"
python - <<'PY'
import torch
from common import hadamard, quantize_weight
x = torch.randn(4, 8)
assert torch.allclose(hadamard(hadamard(x)), x, atol=1e-5), "FWHT not involutive"
W = torch.randn(16, 16)
assert (W - quantize_weight(W, 4)).norm() < W.norm(), "quantizer sanity"
print("OK — environment ready")
PY

echo ""
echo "Next:"
echo "  source .venv/bin/activate"
echo "  python measure_outliers.py --model HuggingFaceTB/SmolLM2-135M   # ~1-2 min on CPU"
echo "  python ptq_eval.py       --model HuggingFaceTB/SmolLM2-135M"
