#!/usr/bin/env bash
# run_lean_e10_adapt.sh <GPU>  -  E10 (v2, ADAPTIVE attacker: it knows the code and never flips protected weights): attack-in-the-loop tier placement (lean_pbs_adapt.py, pbs_four protocol, 1% code) on the
#   paper's LEAN lam40 seed checkpoints, then WCAT seeds 0/1/3 and the baseline for a fair comparison.  Same md5-checked swap of
#   artifacts/models/cifar10/resnet18/model_float32.pth as E3/E8; baseline restored at exit.  ONE job at a time (shared file).
#   Log: sensreg_results/cifar10_resnet18/e10_adaptive.log      ROUNDS (default 2) and FLIPS (default 300,500) via environment.
export CUDA_VISIBLE_DEVICES=$1; ROUNDS=${ROUNDS:-2}; FLIPS=${FLIPS:-300,500}
SRC=artifacts/models/cifar10/resnet18; T=$SRC/model_float32.pth; BAK=$SRC/model_float32.pth.BASELINE_BAK
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/e10_adaptive.log; : > $LOG
restore () { [ -f $BAK ] && cp $BAK $T && echo "[restored baseline $(md5sum $T | cut -c1-8)]" >> $LOG; }; trap restore EXIT
{
[ -f $BAK ] && { echo "backup exists (earlier run interrupted?): restoring it first"; cp $BAK $T; }
cp $T $BAK; echo "baseline md5 $(md5sum $T | cut -c1-8) backed up"
LIST=""
# LEAN lam40 checkpoints: anything with lam40 in the name that is not one of the variants
for f in $(ls $SRC | grep "lam40" | grep -vE "wcat|ablate|band|flipaware|tail2|e3c|logit|_t5|bins" | grep -E "\.pth$" | sed 's/\.pth$//'); do LIST="$LIST $f"; done
LIST="$LIST wcat_s0 wcat_s1 wcat_s3 model_float32.pth.BASELINE_BAK"
echo "checkpoints:$LIST   rounds=$ROUNDS flips=$FLIPS"
for F in $LIST; do
  P=$SRC/$F.pth; [ "$F" = "model_float32.pth.BASELINE_BAK" ] && P=$BAK
  [ -f $P ] || { echo "missing $P"; continue; }
  cp $P $T; [ "$(md5sum $P | cut -c1-32)" = "$(md5sum $T | cut -c1-32)" ] || { echo "md5 mismatch for $F, aborting"; break; }
  L=$F; [ "$F" = "model_float32.pth.BASELINE_BAK" ] && L=baseline
  echo; echo "=== $L  $(date) ==="
  python -u lean_pbs_adapt.py --dataset CIFAR10 --arch resnet18 --bits 8 --covs 0.01 --flips $FLIPS --adapt-flips 500 --rounds $ROUNDS --label e10a_$L --attacker ${ATTACKER:-adaptive}
done
echo "DONE e10_adapt $(date)"
} >> $LOG 2>&1
