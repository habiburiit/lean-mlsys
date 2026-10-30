#!/usr/bin/env bash
# run_llama8b_flipcheck.sh <GPU>  -  diagnose the 2-flip s1 break of the LEAN Llama-3.1-8B model under the attacker-ranking code:
#   applies the saved flips (results_llm/showcase_llama31_8b_lam10_sum_attacker_s1.json) to the LEAN model and to the control,
#   prints the flipped weights, their rank/coverage under both placements, and the attack loss before/after.  Minutes.
#   Log: sensreg_results/llm/flipcheck_llama31_8b_s1.log     watch: tail -f sensreg_results/llm/flipcheck_llama31_8b_s1.log
export CUDA_VISIBLE_DEVICES=${1:-0}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
F=results_llm/showcase_llama31_8b_lam10_sum_attacker_s1.json; LOG=sensreg_results/llm/flipcheck_llama31_8b_s1.log; : > $LOG
{
for ck in lam10_sum lam0_1ep; do
  echo "=== flipcheck $ck  flips from $F  $(date) ==="
  $PY -u lean_llm_bf16.py lean_llm_flipcheck.py --model ckpt_llm/llama31_8b/$ck --flips $F --cov 0.005 --seed 1
done
echo "DONE flipcheck $(date)"
} >> $LOG 2>&1
