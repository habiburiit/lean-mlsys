#!/usr/bin/env bash
# Qwen2-1.5B: control (lambda 0) then LEAN lambda 3, then hist + INT4 PTQ, then the flip attack with and without the 1% code.
# usage: bash run_lean_llm_qwen15.sh <GPU>        watch: tail -f res_llm/qwen2_1.5b/*.log
export CUDA_VISIBLE_DEVICES=$1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
T=qwen2_1.5b; M=Qwen/Qwen2-1.5B; C=ckpt_llm/$T; O=res_llm/$T; mkdir -p $C $O
TR="--epochs 1 --lr 2e-5 --bs 2 --max-train-tokens 1000000"
[ -f $C/lam0_1ep/config.json ] || python -u lean_sensreg_llm.py train --model $M $TR --reg none --lam 0 --out $C/lam0_1ep > $O/train_lam0_1ep.log 2>&1
[ -f $C/lam3/config.json ]     || python -u lean_sensreg_llm.py train --model $M $TR --reg both --lam 3 --tail-frac 0.001 --tail-every 2 --out $C/lam3 > $O/train_lam3.log 2>&1
python -u lean_sensreg_llm.py hist --model $C/lam0_1ep --ckpt2 $C/lam3 --out $O/hist_lam3 > $O/eval_lam3.log 2>&1
python -u lean_sensreg_llm.py ptq --model $C/lam0_1ep --ckpts $C/lam0_1ep $C/lam3 --labels lam0 lam3 --bits 8,6,4 --per-channel --out $O/ptq_pc3 >> $O/eval_lam3.log 2>&1
bash run_lean_llm_flip.sh $1 $T $C/lam0_1ep $C/lam3 lam3
echo "DONE $T" >> $O/eval_lam3.log
