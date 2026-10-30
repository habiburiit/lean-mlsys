#!/usr/bin/env bash
# run_lean_inset_attack.sh  -  PBS (1% BCH code) and TBFA on the last-epochs models of ResNet-18 CIFAR-10,
# same settings as the paper's protection and TBFA tables. Live log, no filtering in the pipe.
# usage:  nohup bash run_lean_inset_attack.sh <GPU> > /dev/null 2>&1 &     tail -f sensreg_results/cifar10_resnet18/inset_attack.log
export CUDA_VISIBLE_DEVICES=$1
M=artifacts/models/cifar10/resnet18/inset; R=sensreg_results/cifar10_resnet18; LOG=$R/inset_attack.log; : > $LOG
{
for TAG in inset_heldlr:heldlr.pth inset_ctrl:control_e200.pth inset_last10:inset_lam40_e200.pth; do
  LAB=${TAG%%:*}; F=$M/${TAG##*:}
  echo; echo "=== PBS  $LAB  ($F) ==="
  python -u lean_pbs_ckpt.py --ckpt $F --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label $LAB
  echo; echo "=== TBFA $LAB  ($F) ==="
  python -u lean_tbfa.py --dataset cifar10 --arch resnet18 --ckpt $F --label $LAB --protect selcode --cov 0.01
done
echo "DONE inset_attack"
} >> $LOG 2>&1
