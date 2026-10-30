#!/usr/bin/env bash
# run_lean_e1_pc.sh <GPU>  -  E1: where the flip step cannot shrink (per-channel INT8), which term carries the protection?
#   Baseline, LEAN (both), range-only and tail-only checkpoints of ResNet-18 CIFAR-10 under the paper's PBS protocol
#   (1% code, coded weights unflippable) with PER-CHANNEL INT8 scales (lean_pbs_codec_pc.py) and, for reference, per-tensor.
#   Log: sensreg_results/cifar10_resnet18/e1_pc.log
export CUDA_VISIBLE_DEVICES=$1
CK=artifacts/models/cifar10/resnet18; O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/e1_pc.log; : > $LOG
{
for M in model_float32:baseline model_float32_sensreg_lam40:lean_both ablate_clip_lam40:clip_only ablate_tail_lam40:tail_only ablate_tail_lam400:tail_lam400; do
  F=${M%%:*}; LAB=${M##*:}
  [ -f $CK/$F.pth ] || { echo "missing $CK/$F.pth"; continue; }
  for Q in pc pt; do
    SCRIPT=lean_pbs_codec_pc.py; [ "$Q" = pt ] && SCRIPT=lean_pbs_codec.py
    echo "=== $LAB  INT8 $Q  $(date) ==="
    python -u $SCRIPT --dataset cifar10 --arch resnet18 --ckpt $CK/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 25,50,100,200,300,500 --label e1_${LAB}_${Q}
  done
done
echo "DONE e1 $(date)"
} >> $LOG 2>&1
