#!/usr/bin/env bash
# run_lean_silentstriker_rank.sh <GPU> <tag> <HF base> [cov]  -  SilentStriker (MMLU objective) vs the weight-level code ranked by the REAL sign-flip
#   damage |g| x step (--score flip), placed by loss sensitivity (--rank loss) and half loss / half answer-letter (--rank both); control vs LEAN lam10.
#   SilentStriker is deterministic, one run each.  Log: sensreg_results/llm/ssrank_<tag>.log
# usage:  nohup bash run_lean_silentstriker_rank.sh 2 qwen2_0.5b Qwen/Qwen2-0.5B 0.005 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; BASE=$3; COV=${4:-0.005}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=sensreg_results/llm; R=results_llm; mkdir -p $O $R; LOG=$O/ssflip_$TAG.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=$(python lean_llm_export.py --base $BASE --ckpt ckpt_llm/$TAG/$ck) || { echo "export failed for $ck"; continue; }
  for RK in loss both; do
    echo "=== SilentStriker $TAG $ck cov $COV rank $RK  $(date) ==="
    python -u lean_llm_silentstriker_w.py --model $MP --cov $COV --rank $RK --score flip --objective mmlu --n-iter 5 --k-top 10 --seed 0 --tag ${TAG}_${ck}_${RK}_flip --out $R/ss
  done
done
echo "DONE $TAG $(date)"
} >> $LOG 2>&1
