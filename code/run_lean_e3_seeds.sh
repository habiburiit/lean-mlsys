#!/usr/bin/env bash
# run_lean_e3_seeds.sh <GPU>  -  E3: three training seeds for WCAT and for the range-only ablation, ResNet-18 CIFAR-10,
#   then the paper's PBS protocol with the 1% code (coded weights unflippable, lean_pbs_codec.py --exclude-coded 1).
#   Existing scripts unchanged.  Log: sensreg_results/cifar10_resnet18/e3_seeds.log
export CUDA_VISIBLE_DEVICES=$1
CK=artifacts/models/cifar10/resnet18; B=$CK/model_float32.pth; O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/e3_seeds.log; : > $LOG
{
for S in 0 1 2; do
  echo "=== WCAT seed $S  $(date) ==="
  [ -f $CK/wcat_s$S.pth ] || python -u lean_train_pc_wcat.py --method wcat --dataset cifar10 --arch resnet18 --init $B --epochs 10 --seed $S --out $CK/wcat_s$S.pth
  echo "=== range-only lam 40 seed $S  $(date) ==="
  [ -f $CK/ablate_clip_lam40_s$S.pth ] || python -u lean_sensreg.py train --dataset cifar10 --arch resnet18 --init $B --reg clip --lam 40 --lr 0.002 --bs 128 --epochs 10 --seed $S --out $CK/ablate_clip_lam40_s$S.pth
done
for S in 0 1 2; do
  for M in wcat_s$S:wcat ablate_clip_lam40_s$S:clip_only; do
    F=${M%%:*}; LAB=${M##*:}
    echo "=== PBS 1% code  $LAB seed $S  $(date) ==="
    python -u lean_pbs_codec.py --dataset cifar10 --arch resnet18 --ckpt $CK/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label e3_${LAB}_s$S
  done
done
echo "DONE e3 $(date)"
} >> $LOG 2>&1
