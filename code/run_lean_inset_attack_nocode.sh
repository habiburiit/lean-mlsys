#!/usr/bin/env bash
# run_lean_inset_attack_nocode.sh  -  PBS WITHOUT a code on the three last-epochs models (fills the "no code" columns of the protection table).
# usage:  nohup bash run_lean_inset_attack_nocode.sh <GPU> > /dev/null 2>&1 &     tail -f sensreg_results/cifar10_resnet18/inset_attack_nocode.log
export CUDA_VISIBLE_DEVICES=$1
M=artifacts/models/cifar10/resnet18/inset; R=sensreg_results/cifar10_resnet18; LOG=$R/inset_attack_nocode.log; : > $LOG
{
for TAG in inset_ctrl:control_e200.pth inset_last10:inset_lam40_e200.pth inset_heldlr:heldlr.pth; do
  LAB=${TAG%%:*}; F=$M/${TAG##*:}
  echo; echo "=== PBS no code  $LAB  ($F) ==="
  python -u lean_pbs_ckpt.py --ckpt $F --dataset CIFAR10 --arch resnet18 --bits 8 --protect none --flips 10,25,50,100 --label $LAB
done
echo "DONE inset_attack_nocode"
} >> $LOG 2>&1
