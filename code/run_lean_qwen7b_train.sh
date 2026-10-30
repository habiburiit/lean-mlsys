#!/usr/bin/env bash
# run_lean_qwen7b_train.sh <GPU list, e.g. 0,1,2,3>  -  Qwen2.5-7B sharded over the listed GPUs (lean_llm_train_big.py):
#   control lam0_1ep (same recipe, lam 0) then LEAN lam10_sum; WikiText-2, 1 epoch, 1M train tokens, bs 1, tail every 2 steps.
#   Log: sensreg_results/llm/train_qwen25_7b.log   (tail -f).  About 1 h per model on 4 x 48 GB.
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=${BASE:-Qwen/Qwen2.5-7B}; TAG=${TAG:-qwen25_7b}
O=sensreg_results/llm; mkdir -p $O ckpt_llm/$TAG; LOG=$O/train_$TAG.log; : > $LOG
{
echo "=== $TAG control lam0 $(date) ==="
[ -d ckpt_llm/$TAG/lam0_1ep ] || python -u lean_llm_train_big.py train --model $BASE --epochs 1 --lr 1e-5 --bs 1 --reg none --lam 0 \
    --max-train-tokens 1000000 --out ckpt_llm/$TAG/lam0_1ep
echo "=== $TAG LEAN lam10 $(date) ==="
[ -d ckpt_llm/$TAG/lam10_sum ] || python -u lean_llm_train_big.py train --model $BASE --epochs 1 --lr 1e-5 --bs 1 --reg both --lam 10 \
    --tail-frac 0.001 --tail-every 2 --max-train-tokens 1000000 --out ckpt_llm/$TAG/lam10_sum
echo "DONE train $TAG $(date)"
} >> $LOG 2>&1
