#!/usr/bin/env bash
# run_lean_inset_hist.sh  -  the four HIGH/MID/LOW comparisons of the last-epochs experiment, via lean_hist.py.
# usage:  bash run_lean_inset_hist.sh <GPU> <arch>      e.g.  bash run_lean_inset_hist.sh 0 vgg16_bn
export CUDA_VISIBLE_DEVICES=$1; ARCH=$2
O=artifacts/models/cifar10/$ARCH/inset; R=sensreg_results/cifar10_$ARCH; LOG=$R/inset_hist.log; : > $LOG
H="python -u lean_hist.py --dataset cifar10 --arch $ARCH"
{
echo "--- inset vs control_e200 ---";   $H --ckpt $O/control_e200.pth --ckpt2 $O/inset_lam40_e200.pth --out $R/hist_inset
echo "--- heldlr vs heldlr_ctrl ---";   $H --ckpt $O/heldlr_ctrl.pth --ckpt2 $O/heldlr.pth --out $R/hist_inset_heldlr
echo "--- long20 vs long20_ctrl ---";   $H --ckpt $O/long20_ctrl.pth --ckpt2 $O/long20.pth --out $R/hist_inset_long20
echo "--- lam100 vs control_e200 ---";  $H --ckpt $O/control_e200.pth --ckpt2 $O/lam100.pth --out $R/hist_inset_lam100
echo "DONE hist $ARCH"
} 2>&1 | grep --line-buffered -v pynvml | tee $LOG
