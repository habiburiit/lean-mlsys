#!/usr/bin/env bash
# run_lean_attnbreaker_sweep.sh <GPU>  -  code-budget sweep under AttentionBreaker (GenBFA) on Qwen2-0.5B:
#   code on 5% / 2% / 1% of the layers (probe ranking), control lam0_1ep vs LEAN lam3_sum / lam10_sum.
#   Finds the budget where the plain model breaks and checks whether LEAN still holds.  Coded setting only
#   (the no-code rows exist from run_lean_attnbreaker.sh).  SEPARATE file; ecc_llm_attnbreaker.py unchanged.
# usage:  nohup bash run_lean_attnbreaker_sweep.sh 2 > /dev/null 2>&1 &    tail -f sensreg_results/llm/attnbreaker_sweep_qwen2_0.5b.log
export CUDA_VISIBLE_DEVICES=${1:-0}
BASE=${BASE:-Qwen/Qwen2-0.5B}; D=ckpt_llm/qwen2_0.5b; O=sensreg_results/llm; R=results_llm; mkdir -p $O $R
LOG=$O/attnbreaker_sweep_qwen2_0.5b.log; : > $LOG
NQ=${NQ:-200}
{
echo "=== AttentionBreaker code-budget sweep, Qwen2-0.5B  $(date) ==="
for SENS in 0.05 0.02 0.01; do
  for ck in lam0_1ep lam3_sum lam10_sum; do
    M=$(python lean_llm_export.py --base $BASE --ckpt $D/$ck) || { echo "export failed for $ck"; continue; }
    echo "--- sens $SENS  $ck ---"
    python -u ecc_llm_attnbreaker.py --model $M --attack genbfa --sens-frac $SENS --sens-proxy probe --interleave 1 --n-questions $NQ --seed 0 --out $R/lean_ab_${ck}_code${SENS}_genbfa
  done
done
echo "DONE sweep $(date)"
} >> $LOG 2>&1
