#!/usr/bin/env bash
# run_lean_nvfp4c_sweep.sh <GPU>  -  LEAN in NVFP4 v3 (block-scale lever) on ResNet-18 CIFAR-10,
#   lam 10 / 100 / 1000 with block-w 1, histogram after each against the NVFP4 control lam0.pth (from run_lean_nvfp4.sh).
# usage:  nohup bash run_lean_nvfp4c_sweep.sh 2 > /dev/null 2>&1 &      tail -f sensreg_results/nvfp4/sweep_v3_cifar10_resnet18.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/sweep_v3_cifar10_resnet18.log; : > $LOG
D=artifacts/models/cifar10/resnet18/nvfp4; mkdir -p $D; B=artifacts/models/cifar10/resnet18/model_float32.pth
{
[ -f $D/lam0.pth ] || python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init $B --lam 0 --out $D/lam0.pth
for L in 10 100 1000; do
  echo "=== nvfp4 v3 cifar10 resnet18 lam $L block-w 1 ==="
  [ -f $D/v3_lam$L.pth ] || python -u lean_train_qat_nvfp4c.py --dataset cifar10 --arch resnet18 --init $B --lam $L --block-w 1 --out $D/v3_lam$L.pth
  echo "--- hist v3 lam $L ---"
  python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --split test --ckpt $D/lam0.pth --ckpt2 $D/v3_lam$L.pth --out $O/hist_cifar10_resnet18_v3_lam$L
done
echo "DONE nvfp4 v3 sweep"
} >> $LOG 2>&1
