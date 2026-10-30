#!/usr/bin/env bash
# run_lean_qwen7b_attack.sh <GPU>  -  after run_lean_qwen7b_train.sh: for control and LEAN, AttentionBreaker with the weight-level code
#   (lean_llm_showcase.py) and the six LM-eval tasks before/after (lean_llm_harness.py), both through the bf16 shim so a 7B model
#   fits one 48 GB GPU.  Log: sensreg_results/llm/full_qwen25_7b.log ; results: grep RESULT sensreg_results/llm/full_qwen25_7b.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=${BASE:-Qwen/Qwen2.5-7B}; TAG=${TAG:-qwen25_7b}; COV=${COV:-0.005}
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_$TAG.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=ckpt_llm/$TAG/$ck; [ -f $MP/config.json ] || MP=$(python lean_llm_bf16.py lean_llm_export.py --base $BASE --ckpt ckpt_llm/$TAG/$ck) || { echo "export failed for $ck"; continue; }
  echo "=== $TAG $ck  cov $COV  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}.json ] || python -u lean_llm_bf16.py lean_llm_showcase.py --model $MP --cov $COV --seed 0 --tag ${TAG}_${ck} --out results_llm/showcase
  python -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}.json --tag ${TAG}_${ck} --out results_llm/harness --bs 8
done
echo "DONE $TAG $(date)"
} >> $LOG 2>&1
