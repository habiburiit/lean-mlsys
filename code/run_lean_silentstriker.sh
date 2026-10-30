#!/usr/bin/env bash
# run_lean_silentstriker.sh <GPU> <tag> <HF base> [cov]  -  SilentStriker (stealthy, MMLU objective, 10 flips/iteration, 5 iterations)
#   against the weight-level code, control (lam0_1ep) vs LEAN (lam10_sum), seeds 0 and 1, then the six-task evaluation
#   (seed 0 flip list).  Existing scripts unchanged.  Log: sensreg_results/llm/ss_<tag>.log
# usage:  nohup bash run_lean_silentstriker.sh 2 qwen2_0.5b Qwen/Qwen2-0.5B 0.005 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; BASE=$3; COV=${4:-0.005}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=sensreg_results/llm; R=results_llm; mkdir -p $O $R; LOG=$O/ss_$TAG.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=$(python lean_llm_export.py --base $BASE --ckpt ckpt_llm/$TAG/$ck) || { echo "export failed for $ck"; continue; }
  for S in 0 1; do
    echo "=== SilentStriker $TAG $ck cov $COV seed $S  $(date) ==="
    python -u lean_llm_silentstriker_w.py --model $MP --cov $COV --objective mmlu --n-iter 5 --k-top 10 --seed $S --tag ${TAG}_${ck}_s$S --out $R/ss
  done
  echo "--- six tasks before/after SilentStriker (seed 0) $ck ---"
  python -u lean_llm_harness.py --model $MP --flips $R/ss_${TAG}_${ck}_s0.json --tag ss_${TAG}_${ck} --out $R/harness
done
echo "DONE $TAG $(date)"
} >> $LOG 2>&1
