#!/usr/bin/env bash
# run_lean_llm_full.sh <GPU> <tag> <HF base> [cov]  -  for one LLM: control (lam0_1ep) and LEAN (lam10_sum):
#   1) AttentionBreaker search with the weight-level code, flip list + text samples saved  (lean_llm_showcase.py)
#   2) LM-Eval-Harness on ARC-E/C, BoolQ, HellaSwag, PIQA, OpenBookQA, SocialIQA before/after those flips (lean_llm_harness.py)
#   Log: sensreg_results/llm/full_<tag>.log      RESULT lines: grep RESULT sensreg_results/llm/full_<tag>.log
# usage:  nohup bash run_lean_llm_full.sh 2 qwen2_1.5b Qwen/Qwen2-1.5B 0.005 > /dev/null 2>&1 &
#         nohup bash run_lean_llm_full.sh 0 llama32_1b meta-llama/Llama-3.2-1B 0.001 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; BASE=$3; COV=${4:-0.005}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_$TAG.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=$(python lean_llm_export.py --base $BASE --ckpt ckpt_llm/$TAG/$ck) || { echo "export failed for $ck"; continue; }
  echo "=== $TAG $ck  cov $COV  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}.json ] || python -u lean_llm_showcase.py --model $MP --cov $COV --seed 0 --tag ${TAG}_${ck} --out results_llm/showcase
  python -u lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}.json --tag ${TAG}_${ck} --out results_llm/harness
done
echo "DONE $TAG $(date)"
} >> $LOG 2>&1
