#!/usr/bin/env bash
# same as run_lean_llm_flip.sh but with the memory-lean attack (for >1B models)
# usage: bash run_lean_llm_flip2.sh <GPU> <tag> <control ckpt> <lean ckpt> <lean label>
export CUDA_VISIBLE_DEVICES=$1; TAG=$2; C0=$3; C1=$4; L1=$5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=res_llm/$TAG/flip_code; mkdir -p $O; LOG=$O/run.log; : > $LOG
python -u lean_llm_flip2.py --model $C0 --label lam0 --protect none  --out $O >> $LOG 2>&1
python -u lean_llm_flip2.py --model $C0 --label lam0 --protect selcode --cov 0.01 --out $O >> $LOG 2>&1
python -u lean_llm_flip2.py --model $C1 --label $L1  --protect none  --out $O >> $LOG 2>&1
python -u lean_llm_flip2.py --model $C1 --label $L1  --protect selcode --cov 0.01 --out $O >> $LOG 2>&1
echo "DONE $TAG" >> $LOG
