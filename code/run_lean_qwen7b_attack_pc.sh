#!/usr/bin/env bash
# run_lean_qwen7b_attack_pc.sh <GPU>  -  same as run_lean_qwen7b_attack.sh but with PER-CHANNEL INT8 (the format that removes the
#   outlier-weight fragility seen on Qwen2 under per-tensor INT8).  Control and LEAN, AttentionBreaker + six tasks, bf16 shim.
#   Log: sensreg_results/llm/full_<tag>_pc.log ; results: grep RESULT sensreg_results/llm/full_<tag>_pc.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=${BASE:-Qwen/Qwen2.5-7B}; TAG=${TAG:-qwen25_7b}; COV=${COV:-0.005}
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_${TAG}_pc.log; : > $LOG
{
for ck in lam0_1ep lam10_sum; do
  MP=ckpt_llm/$TAG/$ck; [ -f $MP/config.json ] || { echo "missing $MP"; continue; }
  echo "=== $TAG $ck  per-channel  cov $COV  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}_pc.json ] || python -u lean_llm_bf16.py lean_llm_showcase.py --model $MP --cov $COV --seed 0 --per-channel --tag ${TAG}_${ck}_pc --out results_llm/showcase
  python -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}_pc.json --tag ${TAG}_${ck}_pc --out results_llm/harness --bs 8 --per-channel
done
echo "DONE ${TAG}_pc $(date)"
} >> $LOG 2>&1
