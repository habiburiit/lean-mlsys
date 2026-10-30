#!/usr/bin/env bash
# Progressive sign-bit flips on INT8 LLMs: control vs LEAN, no code vs 1% code.
# usage: bash run_lean_llm_flip.sh <GPU> <tag> <control ckpt> <lean ckpt> <lean label>
#   bash run_lean_llm_flip.sh 2 llama32_1b ckpt_llm/llama32_1b/lam0_1ep ckpt_llm/llama32_1b/lam3 lam3
#   bash run_lean_llm_flip.sh 3 qwen2_0.5b ckpt_llm/qwen2_0.5b/lam0_1ep ckpt_llm/qwen2_0.5b/lam10_sum lam10
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; C0=$3; C1=$4; L1=$5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=res_llm/$TAG/flip_code; mkdir -p $O; LOG=$O/run.log; : > $LOG
python -u lean_llm_flip.py --model $C0 --label lam0 --protect none  --out $O >> $LOG 2>&1
python -u lean_llm_flip.py --model $C0 --label lam0 --protect selcode --cov 0.01 --out $O >> $LOG 2>&1
python -u lean_llm_flip.py --model $C1 --label $L1  --protect none  --out $O >> $LOG 2>&1
python -u lean_llm_flip.py --model $C1 --label $L1  --protect selcode --cov 0.01 --out $O >> $LOG 2>&1
echo "DONE $TAG" >> $LOG
