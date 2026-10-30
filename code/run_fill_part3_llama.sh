#!/usr/bin/env bash
# run_fill_part3_llama.sh <GPU>  -  Llama-3.2-1B, 0.1% code, AttentionBreaker fixed budgets 20/50/100 flips, control and LEAN lam10
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=sensreg_results/fill_part3_llama.log; mkdir -p sensreg_results results_llm; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=ckpt_llm/llama32_1b/$ck; [ -f $MP/config.json ] || { echo "missing $MP"; continue; }
  for N in 20 50 100; do
    python -u lean_llm_showcase.py --model $MP --cov 0.001 --seed 0 --fixed $N --tag llama32_1b_${ck}_fixed$N --out results_llm/showcase
  done
done
echo "DONE part3 $(date)"
} >> $LOG 2>&1
