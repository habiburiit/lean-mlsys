#!/usr/bin/env bash
# run_lean_tbfa_all.sh <GPU>  -  all-to-one T-BFA on the full ImageNet held-out split (lean_tbfa_all.py), 1% code (paper protocol),
#   ResNet-18 / ResNet-50 base vs LEAN, 3 target classes, 30 flips.  Existing scripts unchanged.  Log: sensreg_results/tbfa_all.log
# usage:  nohup bash run_lean_tbfa_all.sh 0 > /dev/null 2>&1 &     tail -f sensreg_results/tbfa_all.log
export CUDA_VISIBLE_DEVICES=$1; LOG=sensreg_results/tbfa_all.log; mkdir -p sensreg_results; : > $LOG
{
echo "=== all-to-one T-BFA, ImageNet held-out, cov 0.01  $(date) ==="
for A in resnet18 resnet50; do
  D=artifacts/models/imagenet/$A
  for CK in model_float32.pth model_float32_sensreg_lam40.pth; do
    [ -f $D/$CK ] || { echo "missing $D/$CK"; continue; }
    python -u lean_tbfa_all.py --dataset imagenet --arch $A --imagenet-root ./imagenet-val --ckpt $D/$CK --cov 0.01 \
        --targets 1,100,500 --n-flips 30 --report 10,20,30 --label ${A}_${CK%.pth}
  done
done
echo "DONE $(date)"
} >> $LOG 2>&1
