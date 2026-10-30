#!/usr/bin/env bash
# run_lean_inset2.sh  -  ablations of LEAN in the last epochs (held LR, 20 epochs, lam 100), from the saved epoch-190 checkpoint.
# usage:  nohup bash run_lean_inset2.sh <GPU> > /dev/null 2>&1 &    then   tail -f sensreg_results/cifar10_resnet18/inset2.log
export CUDA_VISIBLE_DEVICES=$1
O=artifacts/models/cifar10/resnet18/inset; R=sensreg_results/cifar10_resnet18; LOG=$R/inset2.log; : > $LOG
H="python -u lean_sensreg.py hist --dataset cifar10 --arch resnet18"
{
for V in heldlr_ctrl heldlr long20_ctrl long20 lam100; do
  echo "=== train $V ==="
  [ -f $O/$V.pth ] || python -u lean_train_inset2.py --variant $V --out $O
done
echo; echo "=== hist (thresholds from each variant's own control) ==="
echo "--- heldlr vs heldlr_ctrl ---";   $H --ckpt $O/heldlr_ctrl.pth --ckpt2 $O/heldlr.pth --out $R/hist_inset_heldlr | grep -A3 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "--- long20 vs long20_ctrl ---";   $H --ckpt $O/long20_ctrl.pth --ckpt2 $O/long20.pth --out $R/hist_inset_long20 | grep -A3 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "--- lam100 vs control_e200 ---";  $H --ckpt $O/control_e200.pth --ckpt2 $O/lam100.pth --out $R/hist_inset_lam100 | grep -A3 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
echo "--- heldlr vs control_e200 (same thresholds as before) ---"; $H --ckpt $O/control_e200.pth --ckpt2 $O/heldlr.pth --out $R/hist_inset_heldlr_c200 | grep -A3 "Bin occupancy\|ratio_high"
echo "DONE inset2"
} >> $LOG 2>&1
