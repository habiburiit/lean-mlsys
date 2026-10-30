#!/usr/bin/env bash
# run_lean_llm_more.sh <GPU> <tag> <HF model>  -  LEAN fine-tunes of a second/third LLM with the Qwen2-0.5B recipe that worked
#   (1 epoch, 1M WikiText-2 tokens, lr 2e-5, --reg both, --tail-frac 0.001, lam 3 and 10 -> ckpt_llm/<tag>/lam{3,10}_sum),
#   then AttentionBreaker against the 0.5% weight-level code (lean_llm_attnbreaker_w.py) on control lam0_1ep and both LEAN models.
#   Existing scripts unchanged.  Logs: sensreg_results/llm/more_<tag>.log
# usage:  nohup bash run_lean_llm_more.sh 0 qwen2_1.5b Qwen/Qwen2-1.5B > /dev/null 2>&1 &
#         nohup bash run_lean_llm_more.sh 1 llama32_1b meta-llama/Llama-3.2-1B > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; MODEL=$3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C=ckpt_llm/$TAG; O=sensreg_results/llm; R=results_llm; mkdir -p $C $O $R; LOG=$O/more_$TAG.log; : > $LOG
BS=${BS:-4}; [ "$TAG" = "llama32_1b" ] && BS=${BS_LLAMA:-2}
SEQ=${SEQ:-512}
TRAIN="--epochs 1 --lr 2e-5 --tail-every 2 --max-train-tokens 1000000 --tail-frac 0.001 --bs $BS --seq-len $SEQ"
TR=${TR:-"python -u lean_llm_train_fd.py train"}   # lean_sensreg_llm.py unchanged; finite-difference HVP (fits 44 GB); set TR to another launcher if wanted
{
echo "=== LEAN $TAG ($MODEL)  $(date) ==="
[ -f $C/lam0_1ep/config.json ] || $TR --model $MODEL $TRAIN --reg none --lam 0 --out $C/lam0_1ep
for L in 3 10; do
  [ -f $C/lam${L}_sum/config.json ] || $TR --model $MODEL $TRAIN --reg both --lam $L --out $C/lam${L}_sum
done
echo "--- shares (hist vs lam0_1ep) ---"
for L in 3 10; do python -u lean_sensreg_llm.py hist --model $C/lam0_1ep --ckpt2 $C/lam${L}_sum --out $O/hist_${TAG}_lam${L}_sum; done
for ck in lam0_1ep lam3_sum lam10_sum; do
  M=$(python lean_llm_export.py --base $MODEL --ckpt $C/$ck) || { echo "export failed for $ck"; continue; }
  echo "--- $TAG cov 0.005  $ck ---"
  python -u lean_llm_attnbreaker_w.py --model $M --cov 0.005 --n-questions 200 --seed 0 --out $R/lean_abw_${TAG}_${ck}_cov0.005
done
echo "DONE $TAG $(date)"
} >> $LOG 2>&1
