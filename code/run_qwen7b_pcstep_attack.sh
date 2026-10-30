#!/usr/bin/env bash
# run_qwen7b_pcstep_attack.sh <GPU>  -  per-channel attack of the per-channel-trained LEAN model (ckpt lam10_pcstep) only;
#   control and lam10_sum results already exist in full_<TAG>_pc.log.   Log: sensreg_results/llm/full_<TAG>_pcstep.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${TAG:-qwen25_7b}; COV=${COV:-0.005}; ck=lam10_pcstep
O=sensreg_results/llm; mkdir -p $O results_llm; LOG=$O/full_${TAG}_pcstep.log; : > $LOG
{
  MP=ckpt_llm/$TAG/$ck; [ -f $MP/config.json ] || { echo "missing $MP (run run_qwen7b_pcstep_train.sh first)"; exit 1; }
  echo "=== $TAG $ck  per-channel  cov $COV  $(date) ==="
  [ -f results_llm/showcase_${TAG}_${ck}_pc.json ] || python -u lean_llm_bf16.py lean_llm_showcase.py --model $MP --cov $COV --seed 0 --per-channel --tag ${TAG}_${ck}_pc --out results_llm/showcase
  python -u lean_llm_bf16.py lean_llm_harness.py --model $MP --flips results_llm/showcase_${TAG}_${ck}_pc.json --tag ${TAG}_${ck}_pc --out results_llm/harness --bs 8 --per-channel
  echo "DONE ${TAG}_pcstep $(date)"
} >> $LOG 2>&1
