#!/usr/bin/env bash
# run_lean_nvfp4.sh <GPU>  -  LEAN in NVFP4: control (lam 0) and LEAN (lam 40) QAT fine-tunes plus the sensitivity
#   distribution, first ResNet-18 CIFAR-10 (minutes), then ResNet-50 ImageNet (hours).
# usage:  nohup bash run_lean_nvfp4.sh 2 > /dev/null 2>&1 &      tail -f sensreg_results/nvfp4/nvfp4.log
export CUDA_VISIBLE_DEVICES=${1:-0}
IM=${IM:-./imagenet-val}
O=sensreg_results/nvfp4; mkdir -p $O; LOG=$O/nvfp4.log; : > $LOG
{
# ---- CIFAR-10 ResNet-18 ----
D=artifacts/models/cifar10/resnet18/nvfp4; mkdir -p $D; B=artifacts/models/cifar10/resnet18/model_float32.pth
for L in 0 40; do
  echo "=== nvfp4 cifar10 resnet18 lam $L ==="
  [ -f $D/lam$L.pth ] || python -u lean_train_qat_nvfp4.py --dataset cifar10 --arch resnet18 --init $B --lam $L --out $D/lam$L.pth
done
echo "--- hist cifar10 resnet18 ---"
python -u lean_hist_nvfp4.py --dataset cifar10 --arch resnet18 --ckpt $D/lam0.pth --ckpt2 $D/lam40.pth --out $O/hist_cifar10_resnet18
# ---- ImageNet ResNet-50 ----
D=artifacts/models/imagenet/resnet50/nvfp4; mkdir -p $D
for L in 0 40; do
  echo "=== nvfp4 imagenet resnet50 lam $L ==="
  [ -f $D/lam$L.pth ] || python -u lean_train_qat_nvfp4.py --dataset imagenet --arch resnet50 --imagenet-root $IM --init pretrained --lr 0.0005 --bs 32 --lam $L --out $D/lam$L.pth
done
echo "--- hist imagenet resnet50 ---"
python -u lean_hist_nvfp4.py --dataset imagenet --arch resnet50 --imagenet-root $IM --ckpt $D/lam0.pth --ckpt2 $D/lam40.pth --out $O/hist_imagenet_resnet50
echo "DONE nvfp4"
} >> $LOG 2>&1
