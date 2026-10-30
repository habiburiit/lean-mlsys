#!/usr/bin/env bash
# run_fill_part2_tbfa.sh <GPU>  -  all-to-one T-BFA, ImageNet ResNet-18 held-out, 1% code by loss, control (lam0) and LEAN (lam30), reported at 4,8,10,12,20,30 flips
export CUDA_VISIBLE_DEVICES=${1:-0}
LOG=sensreg_results/fill_part2_tbfa.log; mkdir -p sensreg_results; : > $LOG
IM=${IMAGENET_ROOT:-./imagenet-val}
{
D=artifacts/models/imagenet/resnet18
for CK in model_float32.pth model_float32_sensreg_lam30.pth; do
  python -u lean_tbfa_all.py --dataset imagenet --arch resnet18 --imagenet-root $IM --ckpt $D/$CK --cov 0.01 --rank ce \
      --targets 1,100,500 --n-flips 30 --report 4,8,10,12,20,30 --label resnet18_${CK%.pth}_fill
done
echo "DONE part2 $(date)"
} >> $LOG 2>&1
