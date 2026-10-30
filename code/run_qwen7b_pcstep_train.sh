#!/usr/bin/env bash
# run_qwen7b_pcstep_train.sh  -  LEAN fine-tune with the PER-CHANNEL sensitivity step (lean_llm_train_pcstep.py), 4 GPUs.
#   The control (lam0_1ep) is reused from run_lean_qwen7b_train.sh; only the LEAN model is retrained.
#   Log: sensreg_results/llm/train_<TAG>_lam10_pcstep.log      then attack with:  COV=0.005 bash run_qwen7b_pcstep_attack.sh <GPU>
export CUDA_VISIBLE_DEVICES=${GPUS:-0,1,2,3}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
BASE=${BASE:-Qwen/Qwen2.5-7B}; TAG=${TAG:-qwen25_7b}; TOK=${TOK:-1000000}
O=sensreg_results/llm; mkdir -p $O ckpt_llm/$TAG
OUT=ckpt_llm/$TAG/lam10_pcstep
[ -f $OUT/config.json ] && { echo "exists: $OUT"; exit 0; }
python -u lean_llm_train_pcstep.py train --model $BASE --epochs 1 --lr 1e-5 --bs 1 --reg both --lam 10 --tail-frac 0.001 --tail-every 2 \
    --max-train-tokens $TOK --out $OUT > $O/train_${TAG}_lam10_pcstep.log 2>&1
echo "DONE train ${TAG} lam10_pcstep $(date)" | tee -a $O/train_${TAG}_lam10_pcstep.log
