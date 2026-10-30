#!/usr/bin/env bash
# run_lean_group.sh <GPU>  -  MLSys item 1: group-wise tiers with total bit accounting, ImageNet ResNet-50.
#   totals 3.5 and 4.0 bits/weight (payload + 2-bit group tags + scales + parity), lambda 40 and lambda 0, seeds 0 1 2,
#   then PBS in the deployed format (BCH on the 8-bit tier + t=19 n=1023 sign-plane code) on the seed-0 models.
# usage:  nohup bash run_lean_group.sh 0 > /dev/null 2>&1 &      tail -f sensreg_results/group/imagenet_resnet50.log
# INIT_LEAN: the LEAN INT8 ResNet-50 checkpoint used for the lambda=40 rows of the storage table (set it below);
# INIT_BASE: init of the lambda=0 rows (pretrained, as in run_lean_mp9.sh).
export CUDA_VISIBLE_DEVICES=${1:-0}
IM=${IM:-./imagenet-val}
INIT_LEAN=${INIT_LEAN:-artifacts/models/imagenet/resnet50/lean_lam40.pth}
INIT_BASE=${INIT_BASE:-pretrained}
G=${G:-32}
D=artifacts/models/imagenet/resnet50/group; O=sensreg_results/group; mkdir -p $D $O; LOG=$O/imagenet_resnet50.log; : > $LOG
TR="--dataset imagenet --arch resnet50 --imagenet-root $IM --lr 0.0005 --bs 32 --qclip 0.99 --pen-frac 0.03 --group $G"
{
for T in 3.5 4.0; do
  for SD in 0 1 2; do
    for L in 40 0; do
      [ $L = 40 ] && INIT=$INIT_LEAN || INIT=$INIT_BASE
      M=$D/r50_g${G}_t${T}_lam${L}_s$SD.pth
      echo "=== total $T  lam $L  seed $SD  init $INIT ==="
      [ -f $M ] || python -u lean_train_qat_group.py $TR --init $INIT --total-bits $T --lam $L --seed $SD --out $M
    done
  done
done
for T in 3.5 4.0; do
  M=$D/r50_g${G}_t${T}_lam40_s0.pth
  echo "=== PBS deployed format, total $T, lam 40, seed 0 ==="
  python -u lean_pbs_secded.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $M --secded on --interleave on --t 19 --n 1023 --flips 10,25,50,100,200,300,500
done
echo "DONE group"
} >> $LOG 2>&1
