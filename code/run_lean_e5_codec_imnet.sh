#!/usr/bin/env bash
# run_lean_e5_codec_imnet.sh <GPU>  -  E5: real BCH codec on ImageNet ResNet-50 against the key-aware attacker at larger tiers
#   (2, 3, 5% of the weights), normal vs LEAN lam40.  Log: sensreg_results/ecc/e5_codec_imnet.log
export CUDA_VISIBLE_DEVICES=$1; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/resnet50; O=sensreg_results/ecc; mkdir -p $O; LOG=$O/e5_codec_imnet.log; : > $LOG
{
for COV in 0.02 0.03 0.05; do
  for M in model_float32:base model_float32_sensreg_lam40:lam40; do
    F=${M%%:*}; LAB=${M##*:}
    echo "=== codec imagenet resnet50 $LAB cov $COV  $(date) ==="
    python -u lean_pbs_codec.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $D/$F.pth --cov $COV --attack cwaware --attacker-knows-key 1 --cw-counts 1,2,5,10,20 --label e5_imnet_r50_${LAB}_cov$COV
  done
done
echo "DONE e5 $(date)"
} >> $LOG 2>&1
