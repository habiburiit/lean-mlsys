#!/usr/bin/env bash
# run_lean_tbfa_all_logit.sh <GPU>  -  all-to-one T-BFA on ImageNet ResNet-50 held-out, 1% code chosen by LOGIT sensitivity
#   (targeted-aware; --rank logit), base vs LEAN, 10 flips.  Log: sensreg_results/tbfa_all_r50logit.log
export CUDA_VISIBLE_DEVICES=$1; LOG=sensreg_results/tbfa_all_r50logit.log; mkdir -p sensreg_results; : > $LOG
{
echo "=== all-to-one T-BFA, ImageNet held-out, cov 0.01, rank logit  $(date) ==="
D=artifacts/models/imagenet/resnet50
for CK in model_float32.pth model_float32_sensreg_lam40.pth; do
  python -u lean_tbfa_all.py --dataset imagenet --arch resnet50 --imagenet-root ./imagenet-val --ckpt $D/$CK --cov 0.01 --rank logit \
      --targets 1,100,500 --n-flips 10 --report 1,2,3,4,5,6,8,10 --label resnet50_${CK%.pth}_logit
done
echo "DONE $(date)"
} >> $LOG 2>&1
