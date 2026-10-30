#!/usr/bin/env bash
# run_qwen7b_codesel_attack.sh <GPU>  -  per-channel attack on the three Qwen2.5-7B models with the code placed by the ATTACKER'S score
#   (lean_llm_showcase_codesel.py --code-score attacker).  Attack only, no retraining.  Log: sensreg_results/llm/full_<TAG>_pcsel.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-qwen25_7b}; COV=${COV:-0.005}
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_${TAG}_pcsel.log; : > $LOG
{
for ck in lam0_1ep lam10_sum lam10_pcstep; do
  MP=ckpt_llm/$TAG/$ck; [ -f $MP/config.json ] || { echo "missing $MP"; continue; }
  echo "=== $TAG $ck  per-channel  cov $COV  code placed by attacker score  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}_pcsel.json ] || python -u lean_llm_bf16.py lean_llm_showcase_codesel.py --model $MP --cov $COV --seed 0 --per-channel --code-score attacker --tag ${TAG}_${ck}_pcsel --out results_llm/showcase
  python -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}_pcsel.json --tag ${TAG}_${ck}_pcsel --out results_llm/harness --bs 8 --per-channel
done
echo "DONE ${TAG}_pcsel $(date)"
} >> $LOG 2>&1
