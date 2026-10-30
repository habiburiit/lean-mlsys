#!/usr/bin/env bash
# run_lean_attnbreaker.sh <GPU>  -  AttentionBreaker (GenBFA, ecc_llm_attnbreaker.py unchanged) on the LEAN LLM checkpoints.
#   Qwen2-0.5B: control lam0_1ep vs LEAN lam3_sum / lam10_sum.  Two settings each:
#     (a) no code: GenBFA minimal flip set + MMLU at fixed 10/20/50 flips   -> does the reshaped tail alone cost the attacker flips?
#     (b) sensitivity-aware code on --sens-frac of the layers (probe ranking) -> the paper's protected row
#   SEPARATE file.  Results: results_llm/lean_ab_<ckpt>_<setting>*, log sensreg_results/llm/attnbreaker_qwen2_0.5b.log
# usage:  nohup bash run_lean_attnbreaker.sh 2 > /dev/null 2>&1 &     tail -f sensreg_results/llm/attnbreaker_qwen2_0.5b.log
export CUDA_VISIBLE_DEVICES=${1:-0}
BASE=${BASE:-Qwen/Qwen2-0.5B}; D=ckpt_llm/qwen2_0.5b; O=sensreg_results/llm; R=results_llm; mkdir -p $O $R
LOG=$O/attnbreaker_qwen2_0.5b.log; : > $LOG
SENS=${SENS:-0.10}            # fraction of layers under the code in setting (b)
NQ=${NQ:-200}                 # MMLU questions
{
echo "=== LEAN vs AttentionBreaker, Qwen2-0.5B  $(date) ==="
for ck in lam0_1ep lam3_sum lam10_sum; do
  M=$(python lean_llm_export.py --base $BASE --ckpt $D/$ck) || { echo "export failed for $ck"; continue; }
  echo "--- $ck  model dir $M ---"
  echo "[a] genbfa, no code"
  python -u ecc_llm_attnbreaker.py --model $M --attack genbfa --sens-frac 0 --n-questions $NQ --seed 0 --out $R/lean_ab_${ck}_nocode_genbfa
  for F in 10 20 50; do
    echo "[a] fixed $F flips, no code"
    python -u ecc_llm_attnbreaker.py --model $M --attack fixed --fixed-flips $F --sens-frac 0 --n-questions $NQ --seed 0 --out $R/lean_ab_${ck}_nocode_fixed$F
  done
  echo "[b] genbfa, code on $SENS of layers (probe ranking)"
  python -u ecc_llm_attnbreaker.py --model $M --attack genbfa --sens-frac $SENS --sens-proxy probe --interleave 1 --n-questions $NQ --seed 0 --out $R/lean_ab_${ck}_code${SENS}_genbfa
done
echo "DONE attnbreaker $(date)"
} >> $LOG 2>&1
