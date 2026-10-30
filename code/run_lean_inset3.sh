#!/usr/bin/env bash
# run_lean_inset3.sh  -  last-epochs variants heldlr_cos / heldlr20 (ResNet-18 CIFAR-10): train, hist, PBS with the 1% code.
# usage:  nohup bash run_lean_inset3.sh <GPU> > /dev/null 2>&1 &     tail -f sensreg_results/cifar10_resnet18/inset3.log
export CUDA_VISIBLE_DEVICES=$1
O=artifacts/models/cifar10/resnet18/inset; R=sensreg_results/cifar10_resnet18; LOG=$R/inset3.log; : > $LOG
H="python -u lean_hist.py --dataset cifar10 --arch resnet18"
P="python -u lean_pbs_ckpt.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,400,500"
{
for V in heldlr_cos heldlr20_ctrl heldlr20; do
  echo "=== train $V ==="
  [ -f $O/$V.pth ] || python -u lean_train_inset3.py --variant $V --out $O
done
echo; echo "=== hist ==="
echo "--- heldlr_cos vs heldlr_ctrl ---";  $H --ckpt $O/heldlr_ctrl.pth --ckpt2 $O/heldlr_cos.pth --out $R/hist_inset_heldlr_cos
echo "--- heldlr20 vs heldlr20_ctrl ---"; $H --ckpt $O/heldlr20_ctrl.pth --ckpt2 $O/heldlr20.pth --out $R/hist_inset_heldlr20
echo; echo "=== PBS with 1% code ==="
echo "--- heldlr_cos ---"; $P --ckpt $O/heldlr_cos.pth --label inset_heldlr_cos
echo "--- heldlr20 ---";   $P --ckpt $O/heldlr20.pth --label inset_heldlr20
echo "DONE inset3"
} >> $LOG 2>&1
