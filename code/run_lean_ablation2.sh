#!/usr/bin/env bash
# run_lean_ablation2.sh  -  can P_tail alone do the job if pushed harder?  ResNet-18 CIFAR-10, lam 100 and 400, no clip term.
# usage:  nohup bash run_lean_ablation2.sh <GPU> > /dev/null 2>&1 &   then   tail -f sensreg_results/cifar10_resnet18/ablation2.log
export CUDA_VISIBLE_DEVICES=$1
CK=artifacts/models/cifar10/resnet18; BASE=$CK/model_float32.pth
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/ablation2.log; : > $LOG
{
for L in 100 400; do
  echo "=== P_tail only, lam $L ==="
  [ -f $CK/ablate_tail_lam$L.pth ] || python -u lean_sensreg.py train --dataset cifar10 --arch resnet18 --init $BASE --reg tail --lam $L --lr 0.002 --bs 128 --epochs 10 --out $CK/ablate_tail_lam$L.pth
  python -u lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt $BASE --ckpt2 $CK/ablate_tail_lam$L.pth --out $O/hist_ablate_tail_lam$L | grep -A3 "Bin occupancy\|tail_mass_top1pct2\|clean_acc2"
done
echo "=== true flip damage ==="
python -u lean_truedamage.py --dataset cifar10 --arch resnet18 --ckpts $BASE $CK/ablate_tail_lam100.pth $CK/ablate_tail_lam400.pth \
   --labels baseline tail_lam100 tail_lam400 --topk 1000 --nrand 1000 --out $O/truedamage2
echo "DONE ablation2"
} >> $LOG 2>&1
