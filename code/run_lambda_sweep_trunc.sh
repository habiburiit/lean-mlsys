#!/usr/bin/env bash
# run_lambda_sweep_trunc.sh  -  All_final, ANON
# 1) train the extra lambdas (15, 20, 40) for one config, 2) hist for each, 3) per-bin truncation
#    (HIGH 8 bits, MID/LOW truncated) for baseline + all lambdas: accuracy vs bits/weight.
# usage: bash run_lambda_sweep_trunc.sh <dataset> <arch> [GPU]      e.g. bash run_lambda_sweep_trunc.sh cifar10 resnet18 0
set -e -o pipefail
DS=$1; ARCH=$2; export CUDA_VISIBLE_DEVICES=${3:-0}
DIR=artifacts/models/$DS/$ARCH; BASE=$DIR/model_float32.pth; O=sensreg_results/${DS}_${ARCH}; mkdir -p $O
CKPTS="$BASE"; LABELS="baseline"
for L in 10 15 20 30 40; do
  CK=$DIR/model_float32_sensreg_lam$L.pth
  if [ ! -f $CK ]; then
    python -u lean_sensreg.py train --dataset $DS --arch $ARCH --init $BASE --reg both --lam $L --lr 0.002 --epochs 10 --out $CK 2>&1 | tee $O/train_lam$L.log
  fi
  [ -f $O/hist_lam$L/summary.json ] || python -u lean_sensreg.py hist --dataset $DS --arch $ARCH --ckpt $BASE --ckpt2 $CK --out $O/hist_lam$L 2>&1 | tee $O/hist_lam$L.log
  CKPTS="$CKPTS $CK"; LABELS="$LABELS lam$L"
done
python -u lean_sensreg.py knob --dirs $O/hist_lam10 $O/hist_lam15 $O/hist_lam20 $O/hist_lam30 $O/hist_lam40 --lams 10,15,20,30,40 --out $O/knob5 2>&1 | tee $O/knob5.log
python -u lean_trunc.py --dataset $DS --arch $ARCH --ckpts $CKPTS --labels $LABELS --bits-cfg 8/8,8/6,8/4,6/4,4/4,6/2,4/2 --out $O/trunc 2>&1 | tee $O/trunc.log
python -u lean_sensreg.py figure --root sensreg_results --configs ${DS}_${ARCH} --lams 10 15 20 30 40 --out $O/fig_bins_accuracy_5lam.png
echo DONE
