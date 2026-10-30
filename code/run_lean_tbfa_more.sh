#!/usr/bin/env bash
# T-BFA on VGG-16 (CIFAR-10/100) and the three ImageNet models: baseline vs LEAN, no code vs 1% code.
# usage: bash run_lean_tbfa_more.sh <GPU>        logs: sensreg_results/tbfa/<dataset>_<arch>.log
export CUDA_VISIBLE_DEVICES=$1
IM=${IM:-./imagenet-val}
mkdir -p sensreg_results/tbfa
# dataset arch lambda
for cfg in "cifar10 vgg16 40" "cifar100 vgg16 40" "imagenet resnet18 30" "imagenet resnet50 40" "imagenet vgg16 100"; do
  set -- $cfg; D=$1; A=$2; L=$3; M=artifacts/models/$D/$A; LOG=sensreg_results/tbfa/${D}_${A}.log
  EXTRA=""; [ "$D" = "imagenet" ] && EXTRA="--imagenet-root $IM --full-val --n-src 25 --n-eval-src 25 --n-other 256 --pairs 0:1,100:200,500:900"
  : > $LOG
  for P in none selcode; do
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32.pth --label base --protect $P --cov 0.01 $EXTRA >> $LOG 2>&1
    python -u lean_tbfa.py --dataset $D --arch $A --ckpt $M/model_float32_sensreg_lam$L.pth --label lam$L --protect $P --cov 0.01 $EXTRA >> $LOG 2>&1
  done
  echo "DONE $D $A" >> $LOG
done
echo ALL DONE
