#!/usr/bin/env bash
# T-BFA with the 1% code chosen by the class-max ranking, CIFAR-100 (and CIFAR-10 for completeness).
# usage: bash run_lean_tbfa_classmax.sh <GPU>     logs: sensreg_results/tbfa/<dataset>_<arch>_classmax.log
export CUDA_VISIBLE_DEVICES=$1; LAM=${LAM:-30}
for cfg in "cifar100 resnet18" "cifar100 resnet50" "cifar10 resnet18" "cifar10 resnet50"; do
  set -- $cfg; D=$1; A=$2; M=artifacts/models/$D/$A; LOG=sensreg_results/tbfa/${D}_${A}_classmax.log
  : > $LOG
  python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32.pth --label base --protect selcode --cov 0.01 --rank classmax >> $LOG 2>&1
  python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32_sensreg_lam$LAM.pth --label lam$LAM --protect selcode --cov 0.01 --rank classmax >> $LOG 2>&1
  echo "DONE $D $A" >> $LOG
done
echo ALL DONE
