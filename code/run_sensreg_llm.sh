#!/usr/bin/env bash
# run_sensreg_llm.sh v2  -  All_final, ANON.  usage: bash run_sensreg_llm.sh <GPU> <HF model> <tag>
#   bash run_sensreg_llm.sh 0 Qwen/Qwen2-0.5B qwen2_0.5b
#   bash run_sensreg_llm.sh 1 meta-llama/Llama-3.2-1B llama32_1b
# lam0 = fine-tuned WITHOUT penalty (the fair control); lam10/lam30 = with penalty.
set -e -o pipefail
export CUDA_VISIBLE_DEVICES=$1; MODEL=$2; TAG=$3
LAMS="0 10 30"; O=res_llm/$TAG; C=ckpt_llm/$TAG; mkdir -p $O $C
TRAIN="--epochs 2 --lr 2e-5 --tail-every 2 --max-train-tokens 1000000"
for L in $LAMS; do
  CK=$C/lam$L
  [ "$L" = "0" ] && REG="--reg none" || REG="--reg both"
  if [ ! -f $CK/config.json ]; then
    python -u lean_sensreg_llm.py train --model $MODEL $TRAIN $REG --lam $L --out $CK 2>&1 | tee $O/train_lam$L.log
  fi
done
# fair comparison: baseline = lam0 (fine-tuned, no penalty); pretrained model kept as reference
CKPTS="$C/lam0 $C/lam10 $C/lam30 $MODEL"; LABELS="lam0 lam10 lam30 pretrained"
for L in 10 30; do
  python -u lean_sensreg_llm.py hist --model $C/lam0 --ckpt2 $C/lam$L --out $O/hist_lam$L 2>&1 | tee $O/hist_lam$L.log
done
python -u lean_sensreg_llm.py plot --model $C/lam0 --ckpts $CKPTS --labels $LABELS --out $O/qhist_log 2>&1 | tee $O/qhist_log.log
python -u lean_sensreg_llm.py ptq  --model $C/lam0 --ckpts $CKPTS --labels $LABELS --bits 8,6,5,4 --per-channel --out $O/ptq_pc 2>&1 | tee $O/ptq_pc.log
python -u lean_sensreg_llm.py ptq  --model $C/lam0 --ckpts $CKPTS --labels $LABELS --bits 8,6 --out $O/ptq 2>&1 | tee $O/ptq.log
python -u lean_sensreg_llm.py flip --model $C/lam0 --ckpts $CKPTS --labels $LABELS --nflips 1,2,3,5,10,20,50 --out $O/flip 2>&1 | tee $O/flip.log
echo DONE $TAG
