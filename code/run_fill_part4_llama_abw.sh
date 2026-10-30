#!/usr/bin/env bash
# run_fill_part4_llama_abw.sh <GPU>  -  Table 22 Llama-3.2-1B cells: AttentionBreaker with the 0.1% code, control and LEAN lam10,
#   seeds 0 and 1; reports flips, ppl and MMLU after the search, and ppl at fixed 10/20/50/100 flips (lean_llm_attnbreaker_w.py).
#   Log: sensreg_results/fill_part4_llama_abw.log      watch: tail -f sensreg_results/fill_part4_llama_abw.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=sensreg_results/fill_part4_llama_abw.log; mkdir -p sensreg_results results_llm; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=$(python lean_llm_export.py --base meta-llama/Llama-3.2-1B --ckpt ckpt_llm/llama32_1b/$ck) || { echo "export failed for $ck"; continue; }
  for S in 0 1; do
    echo "=== llama32_1b $ck seed $S cov 0.001 $(date) ==="
    python -u lean_llm_attnbreaker_w.py --model $MP --cov 0.001 --n-questions 200 --seed $S --out results_llm/lean_abw_llama32_1b_${ck}_cov0.001_s$S
  done
done
echo "DONE part4 $(date)"
} >> $LOG 2>&1
