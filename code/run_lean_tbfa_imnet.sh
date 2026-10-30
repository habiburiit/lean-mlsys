#!/usr/bin/env bash
# run_lean_tbfa_imnet.sh <GPU>  -  A5: T-BFA on the ImageNet HELD-OUT split (lean_tbfa.py default loader = the seed-0
#   10k held-out images), ResNet-18 (lam30) and ResNet-50 (lam40), control vs LEAN, no code and 1% code, 3 class pairs.
#   Existing lean_tbfa.py unchanged.
# usage:  nohup bash run_lean_tbfa_imnet.sh 1 > /dev/null 2>&1 &    tail -f sensreg_results/tbfa/imnet_heldout.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/tbfa; mkdir -p $O; LOG=$O/imnet_heldout.log; : > $LOG
IM=${IMAGENET_ROOT:-./imagenet-val}
{
echo "=== T-BFA ImageNet held-out  $(date) ==="
for spec in "resnet18 30" "resnet50 40"; do
  set -- $spec; A=$1; L=$2; D=artifacts/models/imagenet/$A
  for ck in model_float32:base model_float32_sensreg_lam$L:lam$L; do
    F=${ck%%:*}; LAB=${ck##*:}
    for P in none selcode; do
      echo "--- $A $LAB protect $P ---"
      python -u lean_tbfa.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $D/$F.pth --label ${A}_$LAB --protect $P --cov 0.01 --max-flips 30 --report 10,20,30
    done
  done
done
echo "DONE tbfa imnet $(date)"
} >> $LOG 2>&1
