#!/usr/bin/env bash
# run_lean_attnbreaker_w.sh <GPU>  -  AttentionBreaker against a WEIGHT-level code (top 0 / 0.5 / 1 / 2% of weights by damage),
#   Qwen2-0.5B control lam0_1ep vs LEAN lam3_sum / lam10_sum.  Uses lean_llm_attnbreaker_w.py (new); nothing existing modified.
# usage:  nohup bash run_lean_attnbreaker_w.sh 2 > /dev/null 2>&1 &    tail -f sensreg_results/llm/attnbreaker_w_qwen2_0.5b.log
export CUDA_VISIBLE_DEVICES=${1:-0}
BASE=${BASE:-Qwen/Qwen2-0.5B}; D=ckpt_llm/qwen2_0.5b; O=sensreg_results/llm; R=results_llm; mkdir -p $O $R
LOG=$O/attnbreaker_w_qwen2_0.5b.log; : > $LOG
{
echo "=== AttentionBreaker vs weight-level code, Qwen2-0.5B  $(date) ==="
for COV in 0.005 0.01 0.02; do
  for ck in lam0_1ep lam3_sum lam10_sum; do
    M=$(python lean_llm_export.py --base $BASE --ckpt $D/$ck) || { echo "export failed for $ck"; continue; }
    echo "--- cov $COV  $ck ---"
    python -u lean_llm_attnbreaker_w.py --model $M --cov $COV --n-questions 200 --seed 0 --out $R/lean_abw_${ck}_cov$COV
  done
done
echo "DONE attnbreaker weight-level $(date)"
} >> $LOG 2>&1
