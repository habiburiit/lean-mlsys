#!/usr/bin/env bash
# run_lean_e8_flipaware.sh <GPU>  -  E8: LEAN lam40 + flip-aware fine-tune (lean_train_flipaware.py), three seeds, ResNet-18 CIFAR-10,
#   then pbs_four with the 1% code (md5-checked swap, baseline restored at exit).  Log: sensreg_results/cifar10_resnet18/e8_flipaware.log
export CUDA_VISIBLE_DEVICES=$1; export FLIPS_PER_STEP=${FLIPS_PER_STEP:-200}; export FLIP_TARGETED=${FLIP_TARGETED:-0.5}
SRC=artifacts/models/cifar10/resnet18; T=$SRC/model_float32.pth; BAK=$SRC/model_float32.pth.BASELINE_BAK
O=sensreg_results/cifar10_resnet18; LOG=$O/e8_flipaware.log; : > $LOG
restore () { [ -f $BAK ] && cp $BAK $T && echo "[restored baseline]" >> $LOG; }; trap restore EXIT
{
[ -f $BAK ] && cp $BAK $T; cp $T $BAK
for S in 0 1 2; do
  [ -f $SRC/lean_flipaware_lam40_s$S.pth ] || python -u lean_train_flipaware.py --dataset cifar10 --arch resnet18 --init $BAK --reg both --lam 40 --lr 0.002 --bs 128 --epochs 10 --seed $S --out $SRC/lean_flipaware_lam40_s$S.pth
done
for S in 0 1 2; do
  cp $SRC/lean_flipaware_lam40_s$S.pth $T; [ "$(md5sum $T | cut -c1-32)" = "$(md5sum $SRC/lean_flipaware_lam40_s$S.pth | cut -c1-32)" ] || { echo "md5 mismatch"; break; }
  echo "=== LEAN lam40 + flip-aware seed $S ==="; python -u pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label e8_flipaware_s$S
done
echo DONE
} >> $LOG 2>&1
