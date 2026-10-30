#!/usr/bin/env bash
# T-BFA on the four CIFAR INT8 models: baseline vs LEAN (lam30), no code vs 1% code.
# usage: bash run_lean_tbfa.sh <GPU>        logs: sensreg_results/tbfa/<dataset>_<arch>.log
export CUDA_VISIBLE_DEVICES=$1; LAM=${LAM:-30}
mkdir -p sensreg_results/tbfa
for cfg in "cifar10 resnet18" "cifar10 resnet50" "cifar100 resnet18" "cifar100 resnet50"; do
  set -- $cfg; D=$1; A=$2; M=artifacts/models/$D/$A; LOG=sensreg_results/tbfa/${D}_${A}.log
  : > $LOG
  for P in none selcode; do
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32.pth --label base --protect $P --cov 0.01 >> $LOG 2>&1
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32_sensreg_lam$LAM.pth --label lam$LAM --protect $P --cov 0.01 >> $LOG 2>&1
  done
  echo "DONE $D $A" >> $LOG
done
echo ALL DONE
