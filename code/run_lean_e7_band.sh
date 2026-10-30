#!/usr/bin/env bash
# run_lean_e7_band.sh <GPU>  -  E7: code-aware tail (ranks 1% to 5%), lam 40, three seeds, ResNet-18 CIFAR-10, then pbs_four with the
#   1% code (md5-checked swap, baseline restored at exit).  Log: sensreg_results/cifar10_resnet18/e7_band.log
export CUDA_VISIBLE_DEVICES=$1; SRC=artifacts/models/cifar10/resnet18; T=$SRC/model_float32.pth; BAK=$SRC/model_float32.pth.BASELINE_BAK
O=sensreg_results/cifar10_resnet18; LOG=$O/e7_band.log; : > $LOG
restore () { [ -f $BAK ] && cp $BAK $T && echo "[restored baseline]" >> $LOG; }; trap restore EXIT
{
[ -f $BAK ] && cp $BAK $T; cp $T $BAK
for S in 0 1 2; do
  [ -f $SRC/lean_band_lam40_s$S.pth ] || python -u lean_train_band.py --dataset cifar10 --arch resnet18 --init $BAK --reg both --lam 40 --tail-frac 0.05 --code-frac 0.01 --lr 0.002 --bs 128 --epochs 10 --seed $S --out $SRC/lean_band_lam40_s$S.pth
done
for S in 0 1 2; do
  cp $SRC/lean_band_lam40_s$S.pth $T; [ "$(md5sum $T | cut -c1-32)" = "$(md5sum $SRC/lean_band_lam40_s$S.pth | cut -c1-32)" ] || { echo "md5 mismatch"; break; }
  echo "=== code-aware band lam40 seed $S ==="; python -u pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label e7_band_s$S
done
echo DONE
} >> $LOG 2>&1
