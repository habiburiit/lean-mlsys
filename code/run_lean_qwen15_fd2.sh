#!/usr/bin/env bash
# run_lean_qwen15_fd2.sh <GPU> <lam>  -  Qwen2-1.5B LEAN retrain with the STABILIZED launcher (lean_llm_train_fd2.py), then
#   AttentionBreaker (0.5% weight-level code, seeds 0 and 1) and the six-task evaluation for seed 0.  One lambda per GPU.
#   Checkpoint: ckpt_llm/qwen2_1.5b/lam<lam>_fd2     Log: sensreg_results/llm/fd2_qwen2_1.5b_lam<lam>.log
# usage:  BS=1 nohup bash run_lean_qwen15_fd2.sh 2 3  > /dev/null 2>&1 &
#         BS=1 nohup bash run_lean_qwen15_fd2.sh 3 10 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; LAM=$2; TAG=qwen2_1.5b; BASE=Qwen/Qwen2-1.5B
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BS=${BS:-1}; SEQ=${SEQ:-512}; CK=ckpt_llm/$TAG/lam${LAM}_fd2
O=sensreg_results/llm; R=results_llm; mkdir -p $O $R ckpt_llm/$TAG; LOG=$O/fd2_${TAG}_lam${LAM}.log; : > $LOG
{
echo "=== $TAG lam $LAM, stabilized fd2 launcher  $(date) ==="
if [ ! -f $CK/config.json ]; then
  python -u lean_llm_train_fd2.py train --model $BASE --epochs 1 --lr 2e-5 --tail-every 2 --max-train-tokens 1000000 --tail-frac 0.001 \
      --bs $BS --seq-len $SEQ --reg both --lam $LAM --out $CK
fi
[ -f $CK/config.json ] || { echo "training failed, no checkpoint"; exit 1; }
MP=$(python lean_llm_export.py --base $BASE --ckpt $CK) || { echo "export failed"; exit 1; }
for S in 0 1; do
  echo "--- attack seed $S ---"
  python -u lean_llm_attnbreaker_w.py --model $MP --cov 0.005 --n-questions 200 --seed $S --out $R/lean_abw_${TAG}_lam${LAM}_fd2_s${S}_cov0.005
done
echo "--- flip list + six tasks (seed 0) ---"
python -u lean_llm_showcase.py --model $MP --cov 0.005 --seed 0 --tag ${TAG}_lam${LAM}_fd2 --out $R/showcase
python -u lean_llm_harness.py --model $MP --flips $R/showcase_${TAG}_lam${LAM}_fd2.json --tag ${TAG}_lam${LAM}_fd2 --out $R/harness
echo "DONE $TAG lam $LAM $(date)"
} >> $LOG 2>&1
