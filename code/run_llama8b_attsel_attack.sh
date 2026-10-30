#!/usr/bin/env bash
# run_llama8b_attsel_attack.sh <GPU> [ckpt]  -  Table 11 follow-up (give lam0_1ep or lam10_sum as 2nd arg to run one model per GPU in parallel): AttentionBreaker on Llama-3.1-8B control (lam0_1ep) and LEAN (lam10_sum)
#   with the 0.5% code placed by the ATTACKER'S OWN sign-flip-damage ranking (lean_llm_showcase_codesel.py --code-score attacker)
#   instead of |g|x|w|; everything else as the Table 11 rows (per-tensor INT8, seed 0, loss threshold 7.0, blind attacker).
#   Then the six-task harness plus MMLU before/after.  Attack only, no retraining.
#   Log: sensreg_results/llm/full_llama31_8b_attsel[_<ckpt>].log     results: grep -h "^RESULT" sensreg_results/llm/full_llama31_8b_attsel*.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
TAG=llama31_8b; COV=0.005; CKS=${2:-lam0_1ep lam10_sum}
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_${TAG}_attsel${2:+_$2}.log; : > $LOG
{
for ck in $CKS; do
  MP=ckpt_llm/$TAG/$ck; [ -f $MP/config.json ] || { echo "missing $MP"; continue; }
  echo "=== $TAG $ck  per-tensor INT8  cov $COV  code placed by attacker ranking  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}_attsel.json ] || $PY -u lean_llm_bf16.py lean_llm_showcase_codesel.py --model $MP --cov $COV --seed 0 --code-score attacker --tag ${TAG}_${ck}_attsel --out results_llm/showcase
  echo "--- harness: six tasks + MMLU, before and after  $(date) ---"
  $PY -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}_attsel.json --tag ${TAG}_${ck}_attsel --out results_llm/harness --bs 8 --tasks arc_easy,arc_challenge,boolq,hellaswag,piqa,openbookqa,mmlu
done
echo "DONE ${TAG}_attsel $(date)"
} >> $LOG 2>&1
