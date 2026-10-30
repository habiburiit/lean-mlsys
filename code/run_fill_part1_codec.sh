#!/usr/bin/env bash
# run_fill_part1_codec.sh <GPU>  -  real BCH codec, CIFAR-10 ResNet-18, key-aware attacker at 5 and 20 overloaded codewords (65 and 260 errors)
export CUDA_VISIBLE_DEVICES=${1:-0}
LOG=sensreg_results/fill_part1_codec.log; mkdir -p sensreg_results; : > $LOG
{
D=artifacts/models/cifar10/resnet18
for ck in model_float32:base model_float32_sensreg_lam40:lam40; do
  F=${ck%%:*}; LAB=cifar10_resnet18_${ck##*:}_fill
  python -u lean_pbs_codec.py --dataset cifar10 --arch resnet18 --ckpt $D/$F.pth --cov 0.01 --attack cwaware --attacker-knows-key 1 --cw-counts 5,20 --label $LAB
done
echo "DONE part1 $(date)"
} >> $LOG 2>&1
