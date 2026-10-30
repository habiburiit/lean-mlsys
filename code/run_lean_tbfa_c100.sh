#!/usr/bin/env bash
# rerun of the two CIFAR-100 models only.  usage: bash run_lean_tbfa_c100.sh <GPU>
export CUDA_VISIBLE_DEVICES=$1; LAM=${LAM:-30}
for cfg in "cifar100 resnet18" "cifar100 resnet50"; do
  set -- $cfg; D=$1; A=$2; M=artifacts/models/$D/$A; LOG=sensreg_results/tbfa/${D}_${A}.log
  : > $LOG
  for P in none selcode; do
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32.pth --label base --protect $P --cov 0.01 >> $LOG 2>&1
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32_sensreg_lam$LAM.pth --label lam$LAM --protect $P --cov 0.01 >> $LOG 2>&1
  done
  echo "DONE $D $A" >> $LOG
done
echo ALL DONE
