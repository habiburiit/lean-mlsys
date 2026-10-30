#!/usr/bin/env bash
# run_lean_inset3b.sh  -  heldlr_cos on VGG-16 CIFAR-10 (train + hist) and no-code PBS on the ResNet-18 heldlr_cos model.
# usage:  nohup bash run_lean_inset3b.sh <GPU> > /dev/null 2>&1 &     tail -f sensreg_results/cifar10_vgg16_bn/inset3.log
export CUDA_VISIBLE_DEVICES=$1
OV=artifacts/models/cifar10/vgg16_bn/inset; RV=sensreg_results/cifar10_vgg16_bn; LOG=$RV/inset3.log; : > $LOG
OR=artifacts/models/cifar10/resnet18/inset
{
echo "=== train heldlr_cos vgg16_bn ==="
[ -f $OV/heldlr_cos.pth ] || python -u lean_train_inset3.py --arch vgg16_bn --variant heldlr_cos --out $OV
echo "--- heldlr_cos vs heldlr_ctrl (vgg16_bn) ---"
python -u lean_hist.py --dataset cifar10 --arch vgg16_bn --ckpt $OV/heldlr_ctrl.pth --ckpt2 $OV/heldlr_cos.pth --out $RV/hist_inset_heldlr_cos
echo; echo "=== PBS no code, resnet18 heldlr_cos ==="
python -u lean_pbs_ckpt.py --ckpt $OR/heldlr_cos.pth --dataset CIFAR10 --arch resnet18 --bits 8 --protect none --flips 10,25,50,100 --label inset_heldlr_cos
echo "DONE inset3b"
} >> $LOG 2>&1
