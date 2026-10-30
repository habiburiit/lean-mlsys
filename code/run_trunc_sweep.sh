#!/usr/bin/env bash
# run_trunc_sweep.sh <dataset> <arch> [GPU]   -  trains missing lambdas (10..100), then lambda x bits sweep
set -e -o pipefail
DS=$1; ARCH=$2; export CUDA_VISIBLE_DEVICES=${3:-0}
DIR=artifacts/models/$DS/$ARCH; BASE=$DIR/model_float32.pth; O=sensreg_results/${DS}_${ARCH}; mkdir -p $O
CKPTS="$BASE"; LABELS="baseline"
for L in 10 20 30 40 60 100; do
  CK=$DIR/model_float32_sensreg_lam$L.pth
  [ -f $CK ] || python -u lean_sensreg.py train --dataset $DS --arch $ARCH --init $BASE --reg both --lam $L --lr 0.002 --epochs 10 --out $CK 2>&1 | tee $O/train_lam$L.log
  CKPTS="$CKPTS $CK"; LABELS="$LABELS lam$L"
done
python -u lean_trunc_sweep.py --dataset $DS --arch $ARCH --ckpts $CKPTS --labels $LABELS --bits 8,6,4,3,2,1 --mode requant --out $O/trunc_sweep_requant 2>&1 | tee $O/trunc_sweep.log
echo DONE
