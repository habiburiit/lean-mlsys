#!/usr/bin/env bash
# run_lean_nvfp4_sweep.sh <GPU>  -  lambda sweep for LEAN in NVFP4 on ResNet-18 CIFAR-10, tail term only (clip-w 0),
#   lam 400 / 2000 / 10000, each followed by the sensitivity distribution against the NVFP4 control (lam0.pth from run_lean_nvfp4.sh).
# usage:  nohup bash run_lean_nvfp4_sweep.sh 2 > /dev/null 2>&1 &      tail -f sensreg_results/nvfp4/sweep_cifar10_resnet18.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/sweep_cifar10_resnet18.log; : > $LOG
D=artifacts/models/cifar10/resnet18/nvfp4; mkdir -p $D; B=artifacts/models/cifar10/resnet18/model_float32.pth
{
[ -f $D/lam0.pth ] || python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init $B --lam 0 --out $D/lam0.pth
for L in 400 2000 10000; do
  echo "=== nvfp4 cifar10 resnet18 lam $L (tail only) ==="
  [ -f $D/tail_lam$L.pth ] || python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init $B --lam $L --clip-w 0 --out $D/tail_lam$L.pth
  echo "--- hist lam $L ---"
  python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --ckpt $D/lam0.pth --ckpt2 $D/tail_lam$L.pth --out $O/hist_cifar10_resnet18_tail_lam$L
done
echo "DONE nvfp4 sweep"
} >> $LOG 2>&1
