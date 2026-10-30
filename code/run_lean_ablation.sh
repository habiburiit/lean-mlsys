#!/usr/bin/env bash
# run_lean_ablation.sh  -  P_tail-only and P_clip-only ablation + true flip damage, ResNet-18 CIFAR-10.
# usage:  nohup bash run_lean_ablation.sh <GPU> > /dev/null 2>&1 &   then   tail -f sensreg_results/cifar10_resnet18/ablation.log
export CUDA_VISIBLE_DEVICES=$1
CK=artifacts/models/cifar10/resnet18; BASE=$CK/model_float32.pth; LEAN=$CK/model_float32_sensreg_lam40.pth
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/ablation.log; : > $LOG
{
echo "=== 1/5 P_tail only, lam 40 ==="
[ -f $CK/ablate_tail_lam40.pth ] || python -u lean_sensreg.py train --dataset cifar10 --arch resnet18 --init $BASE --reg tail --lam 40 --lr 0.002 --bs 128 --epochs 10 --out $CK/ablate_tail_lam40.pth
echo "=== 2/5 P_clip only, lam 40 ==="
[ -f $CK/ablate_clip_lam40.pth ] || python -u lean_sensreg.py train --dataset cifar10 --arch resnet18 --init $BASE --reg clip --lam 40 --lr 0.002 --bs 128 --epochs 10 --out $CK/ablate_clip_lam40.pth
echo "=== 3/5 hist: tail only ==="
python -u lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt $BASE --ckpt2 $CK/ablate_tail_lam40.pth --out $O/hist_ablate_tail | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "=== 4/5 hist: clip only ==="
python -u lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt $BASE --ckpt2 $CK/ablate_clip_lam40.pth --out $O/hist_ablate_clip | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "=== 5/5 true flip damage: baseline, LEAN(both), tail only, clip only ==="
python -u lean_truedamage.py --dataset cifar10 --arch resnet18 --ckpts $BASE $LEAN $CK/ablate_tail_lam40.pth $CK/ablate_clip_lam40.pth \
   --labels baseline lam40 tail_only clip_only --topk 1000 --nrand 1000 --out $O/truedamage
echo "DONE ablation"
} >> $LOG 2>&1
