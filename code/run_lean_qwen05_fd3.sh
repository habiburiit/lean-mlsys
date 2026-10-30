#!/usr/bin/env bash
# run_lean_qwen05_fd3.sh <GPU> <lam>  -  Qwen2-0.5B LEAN with the ANSWER-SENSITIVITY tail (lean_llm_train_fd3.py), then
#   SilentStriker (MMLU objective, code placed by --rank both) and AttentionBreaker (0.5% code) against it, then the six tasks.
#   Checkpoint: ckpt_llm/qwen2_0.5b/lam<lam>_fd3     Log: sensreg_results/llm/fd3_qwen2_0.5b_lam<lam>.log
# usage:  nohup bash run_lean_qwen05_fd3.sh 3 10 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; LAM=$2; TAG=qwen2_0.5b; BASE=Qwen/Qwen2-0.5B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export ANS_W=${ANS_W:-1.0}; export ANS_N=${ANS_N:-2}
CK=ckpt_llm/$TAG/lam${LAM}_fd3; O=sensreg_results/llm; R=results_llm; mkdir -p $O $R; LOG=$O/fd3_${TAG}_lam${LAM}.log; : > $LOG
{
echo "=== $TAG lam $LAM, answer-sensitivity tail (ANS_W=$ANS_W)  $(date) ==="
if [ ! -f $CK/config.json ]; then
  python -u lean_llm_train_fd3.py train --model $BASE --epochs 1 --lr 2e-5 --tail-every 2 --max-train-tokens 1000000 --tail-frac 0.001 \
      --bs 4 --seq-len 512 --reg both --lam $LAM --out $CK
fi
[ -f $CK/config.json ] || { echo "training failed"; exit 1; }
MP=$(python lean_llm_export.py --base $BASE --ckpt $CK) || { echo "export failed"; exit 1; }
for RK in both mmlu loss; do
  echo "--- SilentStriker, code rank $RK ---"
  python -u lean_llm_silentstriker_w.py --model $MP --cov 0.005 --rank $RK --objective mmlu --n-iter 5 --k-top 10 --seed 0 --tag ${TAG}_lam${LAM}_fd3_${RK} --out $R/ss
done
echo "--- AttentionBreaker, 0.5% code ---"
python -u lean_llm_attnbreaker_w.py --model $MP --cov 0.005 --n-questions 200 --seed 0 --out $R/lean_abw_${TAG}_lam${LAM}_fd3_cov0.005
echo "--- six tasks before/after SilentStriker (rank both) ---"
python -u lean_llm_harness.py --model $MP --flips $R/ss_${TAG}_lam${LAM}_fd3_both.json --tag ss_${TAG}_lam${LAM}_fd3 --out $R/harness
echo "DONE $(date)"
} >> $LOG 2>&1
