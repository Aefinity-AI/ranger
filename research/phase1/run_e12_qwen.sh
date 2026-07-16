#!/usr/bin/env bash
# E12 — Qwen3-0.6B cross-model replication (escalated to REQUIRED by E9's
# null at 135M). Memory discipline: one arm per e9 invocation (--no-copy),
# fresh model load each time, resume-safe JSON accumulates across runs.
set -euo pipefail
cd "$(dirname "$0")"
source ~/ranger-venv/bin/activate
export HF_HUB_DISABLE_XET=1
M=Qwen/Qwen3-0.6B

echo "== step 1: weight census =="
if [ ! -s phase1_results_Qwen3-0_6B.json ]; then
  python analyze_real_weights.py $M > e12_weight_census.log 2>&1
else
  echo "  (already done, skipping)"
fi

echo "== step 2: extract recurring residual channels =="
CHANS=$(python extract_census_channels.py phase1_results_Qwen3-0_6B.json 1024 2>e12_channels.log)
echo "census channels: $CHANS" | tee -a e12_channels.log

echo "== step 3: activation census =="
if [ ! -s e8_activation_census_Qwen3-0_6B.json ] || \
   ! python -c "import json,sys; d=json.load(open('e8_activation_census_Qwen3-0_6B.json')); sys.exit(0 if 'verdicts' in d else 1)" 2>/dev/null; then
  python e8_activation_census.py --model $M --tokens 4096 \
    --census-channels "$CHANS" > e8_qwen_run.log 2>&1
else
  echo "  (already done, skipping)"
fi

echo "== step 4: ppl arms (one mutating arm per invocation) =="
python e9_holdout_w4.py --model $M --no-copy --arms ""  > e9_qwen_bf16.log 2>&1
python e9_holdout_w4.py --model $M --no-copy --arms 2   > e9_qwen_arm2.log 2>&1
python e9_holdout_w4.py --model $M --no-copy --arms 4 --holdout-ks 64 \
  > e9_qwen_arm4.log 2>&1
python e9_holdout_w4.py --model $M --no-copy --arms 5 \
  --e8-json e8_activation_census_Qwen3-0_6B.json > e9_qwen_arm5.log 2>&1

echo "== E12 done =="
