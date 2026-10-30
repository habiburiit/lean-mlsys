#!/usr/bin/env bash
# run_lean_showcase.sh <GPU>  -  generations before/after AttentionBreaker for Qwen2-0.5B control (lam0_1ep) and LEAN (lam10_sum),
#   0.5% weight-level code, seed 0 (same attack as the paper numbers).  Log: sensreg_results/llm/showcase_qwen2_0.5b.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=ckpt_llm/qwen2_0.5b; O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/showcase_qwen2_0.5b.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  FX=0; [ "$ck" = "lam10_sum" ] && FX=${FIXED:-100}      # LEAN: top-100 flips through the code, no search (fast)
  MP=$(python lean_llm_export.py --base Qwen/Qwen2-0.5B --ckpt $C/$ck) || { echo "export failed for $ck"; continue; }
  echo "--- $ck ($MP) ---"
  python -u lean_llm_showcase.py --model $MP --cov 0.005 --seed 0 --tag qwen2_0.5b_$ck --fixed $FX --out results_llm/showcase
done
echo "DONE $(date)"
} >> $LOG 2>&1
