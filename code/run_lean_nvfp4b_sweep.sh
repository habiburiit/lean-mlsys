#!/usr/bin/env bash
# run_lean_nvfp4b_sweep.sh <GPU>  -  LEAN in NVFP4 v2 (live flip step + block-scale term) on ResNet-18 CIFAR-10,
#   lam 30 / 300 / 3000 with block-w 1, histogram after each against the NVFP4 control lam0.pth (from run_lean_nvfp4.sh).
# usage:  nohup bash run_lean_nvfp4b_sweep.sh 2 > /dev/null 2>&1 &      tail -f sensreg_results/nvfp4/sweep_v2_cifar10_resnet18.log
export CUDA_VISIBLE_DEVICES=${1:-0}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/sweep_v2_cifar10_resnet18.log; : > $LOG
D=artifacts/models/cifar10/resnet18/nvfp4; mkdir -p $D; B=artifacts/models/cifar10/resnet18/model_float32.pth
{
[ -f $D/lam0.pth ] || python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init $B --lam 0 --out $D/lam0.pth
for L in 30 300 3000; do
  echo "=== nvfp4 v2 cifar10 resnet18 lam $L block-w 1 ==="
  [ -f $D/v2_lam$L.pth ] || python -u lean_train_qat_nvfp4b.py --dataset cifar10 --arch resnet18 --init $B --lam $L --block-w 1 --out $D/v2_lam$L.pth
  echo "--- hist v2 lam $L ---"
  python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --ckpt $D/lam0.pth --ckpt2 $D/v2_lam$L.pth --out $O/hist_cifar10_resnet18_v2_lam$L
done
echo "DONE nvfp4 v2 sweep"
} >> $LOG 2>&1
