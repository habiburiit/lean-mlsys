#!/usr/bin/env bash
# run_lean_review.sh  -  the reviewer experiments, ResNet-18 CIFAR-10 (plus VGG heldlr20). Three independent jobs, one GPU each.
# usage:  nohup bash run_lean_review.sh A <GPU> > /dev/null 2>&1 &      gradient-norm baseline, lam 1 / 10 / 100  (hist + PBS with code)
#         nohup bash run_lean_review.sh B <GPU> > /dev/null 2>&1 &      LEAN lam 40 seeds 1,2  and  tail fraction 0.5% / 2%  (hist + PBS with code)
#         nohup bash run_lean_review.sh C <GPU> > /dev/null 2>&1 &      PBS on the ablation models (code and no code)  +  VGG-16 heldlr20
#         nohup bash run_lean_review.sh D <GPU> > /dev/null 2>&1 &      gradient-norm lam 0.01: no-code PBS + real-flip damage
#         logs: sensreg_results/cifar10_resnet18/review_<A|B|C>.log   (live)
JOB=$1; export CUDA_VISIBLE_DEVICES=$2
CK=artifacts/models/cifar10/resnet18; BASE=$CK/model_float32.pth
R=sensreg_results/cifar10_resnet18; mkdir -p $R; LOG=$R/review_$JOB.log; : > $LOG
H="python -u lean_hist.py --dataset cifar10 --arch resnet18"
P="python -u lean_pbs_ckpt.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500"
PN="python -u lean_pbs_ckpt.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect none --flips 10,25,50,100"
T="python -u lean_sensreg.py train --dataset cifar10 --arch resnet18 --init $BASE --reg both --lr 0.002 --bs 128 --epochs 10"
{
case $JOB in
A|A2)
  [ $JOB = A ] && LS="1 10 100" || LS="0.01 0.03 0.1 0.3"
  for L in $LS; do
    echo "=== gradnorm lam $L ==="
    [ -f $CK/gradnorm_lam$L.pth ] || python -u lean_train_gradnorm.py --dataset cifar10 --arch resnet18 --init $BASE --lam $L --out $CK/gradnorm_lam$L.pth
    echo "--- hist gradnorm lam $L ---"; $H --ckpt $BASE --ckpt2 $CK/gradnorm_lam$L.pth --out $R/hist_gradnorm_lam$L
  done
  for L in $LS; do echo "--- PBS code gradnorm lam $L ---"; $P --ckpt $CK/gradnorm_lam$L.pth --label gradnorm_lam$L; done
  echo "DONE review $JOB";;
B)
  for SD in 1 2; do
    echo "=== LEAN lam 40 seed $SD ==="
    [ -f $CK/lean_lam40_seed$SD.pth ] || $T --lam 40 --seed $SD --out $CK/lean_lam40_seed$SD.pth
    echo "--- hist seed $SD ---"; $H --ckpt $BASE --ckpt2 $CK/lean_lam40_seed$SD.pth --out $R/hist_lam40_seed$SD
  done
  for K in 0.005 0.02; do
    echo "=== LEAN lam 40 tail-frac $K ==="
    [ -f $CK/lean_lam40_k$K.pth ] || $T --lam 40 --tail-frac $K --out $CK/lean_lam40_k$K.pth
    echo "--- hist tail-frac $K ---"; $H --ckpt $BASE --ckpt2 $CK/lean_lam40_k$K.pth --out $R/hist_lam40_k$K
  done
  for SD in 1 2; do echo "--- PBS code seed $SD ---"; $P --ckpt $CK/lean_lam40_seed$SD.pth --label lam40_seed$SD; done
  for K in 0.005 0.02; do echo "--- PBS code tail-frac $K ---"; $P --ckpt $CK/lean_lam40_k$K.pth --label lam40_k$K; done
  echo "DONE review B";;
C)
  for TAG in clip_only:ablate_clip_lam40.pth tail_only:ablate_tail_lam40.pth tail_lam400:ablate_tail_lam400.pth; do
    LAB=${TAG%%:*}; F=$CK/${TAG##*:}
    echo "--- PBS code $LAB ---";    $P  --ckpt $F --label $LAB
    echo "--- PBS no code $LAB ---"; $PN --ckpt $F --label $LAB
  done
  OV=artifacts/models/cifar10/vgg16_bn/inset; RV=sensreg_results/cifar10_vgg16_bn
  for V in heldlr20_ctrl heldlr20; do
    echo "=== train $V vgg16_bn ==="
    [ -f $OV/$V.pth ] || python -u lean_train_inset3.py --arch vgg16_bn --variant $V --out $OV
  done
  echo "--- heldlr20 vs heldlr20_ctrl (vgg16_bn) ---"
  python -u lean_hist.py --dataset cifar10 --arch vgg16_bn --ckpt $OV/heldlr20_ctrl.pth --ckpt2 $OV/heldlr20.pth --out $RV/hist_inset_heldlr20
  echo "DONE review C";;
D)
  echo "--- PBS no code gradnorm lam 0.01 ---"; $PN --ckpt $CK/gradnorm_lam0.01.pth --label gradnorm_lam0.01
  echo "--- true damage gradnorm lam 0.01 ---"
  python -u lean_truedamage.py --dataset cifar10 --arch resnet18 --ckpts $BASE $CK/gradnorm_lam0.01.pth --labels baseline gradnorm --topk 1000 --nrand 1000 --out $R/truedamage_gradnorm
  echo "DONE review D";;
esac
} >> $LOG 2>&1
