#!/usr/bin/env bash
# run_lean_e3_pbsfour.sh <GPU>  -  E3 with the PAPER's attack (pbs_four.py, protocol of Table 7). pbs_four.py always loads
#   artifacts/models/cifar10/resnet18/model_float32.pth, so each checkpoint is copied to that path in turn (md5-verified) and the
#   baseline is restored at the end.  Three seeds of WCAT (seed 2 diverged, seed 3 replaces it) and of range-only, plus every
#   LEAN lam40 seed checkpoint found.  Log: sensreg_results/cifar10_resnet18/e3_pbsfour.log
export CUDA_VISIBLE_DEVICES=$1
SRC=artifacts/models/cifar10/resnet18; TARGET=$SRC/model_float32.pth; BAK=$SRC/model_float32.pth.BASELINE_BAK
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/e3_pbsfour.log; : > $LOG
restore () { if [ -f $BAK ]; then cp $BAK $TARGET; echo "[restored baseline $(md5sum $TARGET | cut -c1-8)]" >> $LOG; fi; }
trap restore EXIT
{
if [ -f $BAK ]; then echo "backup already exists (earlier run interrupted?): restoring it first"; cp $BAK $TARGET; fi
cp $TARGET $BAK; echo "baseline md5 $(md5sum $TARGET | cut -c1-8) backed up to $BAK"
[ -f $SRC/wcat_s3.pth ] || python -u lean_train_pc_wcat.py --method wcat --dataset cifar10 --arch resnet18 --init $BAK --epochs 10 --seed 3 --out $SRC/wcat_s3.pth
LIST="wcat_s0 wcat_s1 wcat_s3 ablate_clip_lam40_s0 ablate_clip_lam40_s1 ablate_clip_lam40_s2"
for f in $(ls $SRC | grep -E "sensreg_lam40.*(seed|_s)[0-9]+\.pth$" | sed 's/\.pth$//'); do LIST="$LIST $f"; done
echo "checkpoints: $LIST"
for F in $LIST; do
  [ -f $SRC/$F.pth ] || { echo "missing $SRC/$F.pth"; continue; }
  cp $SRC/$F.pth $TARGET
  A=$(md5sum $SRC/$F.pth | cut -c1-32); B=$(md5sum $TARGET | cut -c1-32)
  [ "$A" = "$B" ] || { echo "md5 mismatch for $F, aborting"; break; }
  echo; echo "=== $F  (attacking $TARGET = $F, md5 ${A:0:8})  $(date) ==="
  python -u pbs_four.py --dataset CIFAR10 --arch resnet18 --bits 8 --protect selcode --covs 0.01 --flips 100,200,300,500 --label e3_$F
done
echo "DONE e3_pbsfour $(date)"
} >> $LOG 2>&1
