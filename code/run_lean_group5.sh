#!/usr/bin/env bash
# run_lean_group5.sh <GPU>  -  group-wise tiers with total bit accounting on the STORAGE-TABLE recipe (mp5 quantizer).
#   ImageNet ResNet-50, totals 3.5 / 4.0 / 4.5 bits per weight (payload + tags + scales + parity), lambda 0 (init
#   pretrained) and lambda 40 (init the LEAN INT8 checkpoint model_float32_sensreg_lam40.pth), seeds 0 1 2.
# usage:  nohup bash run_lean_group5.sh 0 > /dev/null 2>&1 &      tail -f sensreg_results/group/imagenet_resnet50_mp5.log
export CUDA_VISIBLE_DEVICES=${1:-0}
IM=${IM:-./imagenet-val}
INIT_LEAN=${INIT_LEAN:-artifacts/models/imagenet/resnet50/model_float32_sensreg_lam40.pth}
INIT_BASE=${INIT_BASE:-pretrained}
G=${G:-32}
D=artifacts/models/imagenet/resnet50/group; O=sensreg_results/group; mkdir -p $D $O; LOG=$O/imagenet_resnet50_mp5.log; : > $LOG
TR="--dataset imagenet --arch resnet50 --imagenet-root $IM --lr 0.0005 --bs 32 --group $G"
{
for T in 4.0 3.5 4.5; do
  for SD in 0 1 2; do
    for L in 0 40; do
      [ $L = 40 ] && INIT=$INIT_LEAN || INIT=$INIT_BASE
      M=$D/r50_mp5_g${G}_t${T}_lam${L}_s$SD.pth
      echo "=== total $T  lam $L  seed $SD  init $INIT ==="
      [ -f $M ] || python -u lean_train_qat_group5.py $TR --init $INIT --total-bits $T --lam $L --seed $SD --out $M
    done
  done
done
echo "DONE group5"
} >> $LOG 2>&1
