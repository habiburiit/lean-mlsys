#!/usr/bin/env bash
# run_lean_attnbreaker_w2.sh <GPU>  -  AttentionBreaker vs weight-level code on the other two LEAN LLMs
#   (Llama-3.2-1B lam3/lam10, Qwen2-1.5B lam3) plus a second attack seed on Qwen2-0.5B.  Code on 0.5% and 1% of weights.
#   Checkpoint folder names are taken from ckpt_llm/<model>/: control = lam0_1ep (or lam0), LEAN = lam3_sum/lam10_sum (or lam3/lam10).
# usage:  nohup bash run_lean_attnbreaker_w2.sh 2 > /dev/null 2>&1 &    tail -f sensreg_results/llm/attnbreaker_w2.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/llm; R=results_llm; mkdir -p $O $R; LOG=$O/attnbreaker_w2.log; : > $LOG
pick() { for c in "$@"; do [ -d "$c" ] && { echo $c; return; }; done; }
{
echo "=== AttentionBreaker vs weight-level code, Llama-3.2-1B / Qwen2-1.5B / Qwen2-0.5B seed 1  $(date) ==="
for spec in "llama32_1b meta-llama/Llama-3.2-1B" "qwen2_1.5b Qwen/Qwen2-1.5B"; do
  set -- $spec; D=ckpt_llm/$1; BASE=$2; echo "### $1  ($(ls $D | tr '\n' ' '))"
  C=$(pick $D/lam0_1ep $D/lam0); L3=$(pick $D/lam3_sum $D/lam3); L10=$(pick $D/lam10_sum $D/lam10)
  for ck in $C $L3 $L10; do
    [ -z "$ck" ] && continue
    M=$(python lean_llm_export.py --base $BASE --ckpt $ck) || { echo "export failed for $ck"; continue; }
    for COV in 0.005 0.01; do
      echo "--- $1 cov $COV  $(basename $ck) ---"
      python -u lean_llm_attnbreaker_w.py --model $M --cov $COV --n-questions 200 --seed 0 --out $R/lean_abw_$1_$(basename $ck)_cov$COV
    done
  done
done
echo "### qwen2_0.5b seed 1"
for ck in lam0_1ep lam10_sum; do
  M=$(python lean_llm_export.py --base Qwen/Qwen2-0.5B --ckpt ckpt_llm/qwen2_0.5b/$ck)
  echo "--- qwen2_0.5b cov 0.005 $ck seed 1 ---"
  python -u lean_llm_attnbreaker_w.py --model $M --cov 0.005 --n-questions 200 --seed 1 --out $R/lean_abw_qwen2_0.5b_${ck}_cov0.005_s1
done
echo "DONE attnbreaker w2 $(date)"
} >> $LOG 2>&1
