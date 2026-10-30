#!/usr/bin/env bash
# run_lean_inset_arch.sh  -  LEAN in the last 10 epochs of training + the three ablations + hists, any CIFAR dataset/arch.
# usage:  nohup bash run_lean_inset_arch.sh <GPU> <dataset> <arch> > /dev/null 2>&1 &
#         e.g.  bash run_lean_inset_arch.sh 0 cifar100 resnet18        bash run_lean_inset_arch.sh 1 cifar10 resnet50
#         tail -f sensreg_results/<dataset>_<arch>/inset.log
export CUDA_VISIBLE_DEVICES=$1; DS=$2; ARCH=$3
O=artifacts/models/$DS/$ARCH/inset; R=sensreg_results/${DS}_$ARCH; mkdir -p $O $R; LOG=$R/inset.log; : > $LOG
H="python -u lean_hist.py --dataset $DS --arch $ARCH"
{
echo "=== main: 190 plain epochs, then control / LEAN for the last 10 ($DS $ARCH) ==="
python -u lean_train_inset.py --dataset $DS --arch $ARCH --epochs 200 --last 10 --lam 40 --lr 0.1 --bs 128 --out $O
for V in heldlr_ctrl heldlr long20_ctrl long20 lam100; do
  echo "=== train $V ==="
  [ -f $O/$V.pth ] || python -u lean_train_inset2.py --dataset $DS --arch $ARCH --variant $V --out $O
done
echo; echo "=== hist ==="
echo "--- inset vs control_e200 ---";   $H --ckpt $O/control_e200.pth --ckpt2 $O/inset_lam40_e200.pth --out $R/hist_inset
echo "--- heldlr vs heldlr_ctrl ---";   $H --ckpt $O/heldlr_ctrl.pth --ckpt2 $O/heldlr.pth --out $R/hist_inset_heldlr
echo "--- long20 vs long20_ctrl ---";   $H --ckpt $O/long20_ctrl.pth --ckpt2 $O/long20.pth --out $R/hist_inset_long20
echo "--- lam100 vs control_e200 ---";  $H --ckpt $O/control_e200.pth --ckpt2 $O/lam100.pth --out $R/hist_inset_lam100
echo "DONE inset $DS $ARCH"
} 2>&1 | grep --line-buffered -v pynvml >> $LOG
