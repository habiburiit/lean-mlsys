#!/usr/bin/env bash
# run_lean_owt.sh <GPU>  -  OpenWebText held-out perplexity for every control / LEAN LLM checkpoint in the paper.
#   Log: sensreg_results/llm/owt.log ;  results: grep RESULT_OWT sensreg_results/llm/owt.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=sensreg_results/llm; mkdir -p $O; LOG=$O/owt.log; : > $LOG
{
# tag  base  control  lean
while read TAG BASE C L; do
  for CK in $C $L; do
    [ -d ckpt_llm/$TAG/$CK ] || { echo "skip $TAG/$CK (missing)"; continue; }
    echo "=== $TAG $CK $(date) ==="
    python -u lean_llm_ppl_owt.py --model ckpt_llm/$TAG/$CK --base $BASE --tag ${TAG}_${CK}
  done
done <<LIST
qwen2_0.5b   Qwen/Qwen2-0.5B           lam0_1ep  lam10_sum
qwen2_1.5b   Qwen/Qwen2-1.5B           lam0_1ep  lam10_fd2
llama32_1b   meta-llama/Llama-3.2-1B   lam0_1ep  lam10_sum
qwen25_7b    Qwen/Qwen2.5-7B           lam0_1ep  lam10_sum
llama31_8b   meta-llama/Llama-3.1-8B   lam0_1ep  lam10_sum
LIST
echo "DONE owt $(date)"
} >> $LOG 2>&1
