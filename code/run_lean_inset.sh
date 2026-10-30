#!/usr/bin/env bash
# run_lean_inset.sh  -  LEAN in the last 10 epochs of normal training (ResNet-18 CIFAR-10), then compare to the control.
# usage:  nohup bash run_lean_inset.sh <GPU> > /dev/null 2>&1 &    then   tail -f sensreg_results/cifar10_resnet18/inset.log
export CUDA_VISIBLE_DEVICES=$1
O=artifacts/models/cifar10/resnet18/inset; R=sensreg_results/cifar10_resnet18; mkdir -p $O $R; LOG=$R/inset.log; : > $LOG
{
python -u lean_train_inset.py --dataset cifar10 --arch resnet18 --epochs 200 --last 10 --lam 40 --lr 0.1 --bs 128 --out $O
echo "=== hist: control vs inset (thresholds from control) ==="
python -u lean_sensreg.py hist --dataset cifar10 --arch resnet18 --ckpt $O/control_e200.pth --ckpt2 $O/inset_lam40_e200.pth --out $R/hist_inset | grep -A3 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "DONE inset"
} >> $LOG 2>&1
