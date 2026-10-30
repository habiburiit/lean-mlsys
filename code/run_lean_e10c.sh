#!/usr/bin/env bash
# run_lean_e10c.sh <GPU>  -  E10c: attacker-algorithm-aware placement under the paper's BLIND attacker.
#   Placement rounds: the defender runs the attacker's own algorithm (PBS, blind to the code) on ITS data and moves the chain into the code.
#   Evaluation: the blind attacker then runs with DIFFERENT probe data (reshuffled test loader).  Printed per checkpoint:
#   "before placement" rows (clean-ranked code) and "after placement" rows, both at 300 and 500 flips, same attacker data.
#   Checkpoints: LEAN lam40 seeds 0/1/2, WCAT seeds 0/1/3, baseline.  pbs_four protocol, md5-checked swap, baseline restored at exit.
#   Log: sensreg_results/cifar10_resnet18/e10c.log
export CUDA_VISIBLE_DEVICES=$1; ROUNDS=${ROUNDS:-2}; FLIPS=${FLIPS:-300,500}; EVSEED=${EVSEED:-1}
SRC=artifacts/models/cifar10/resnet18; T=$SRC/model_float32.pth; BAK=$SRC/model_float32.pth.BASELINE_BAK
O=sensreg_results/cifar10_resnet18; mkdir -p $O; LOG=$O/e10c.log; : > $LOG
restore () { [ -f $BAK ] && cp $BAK $T && echo "[restored baseline $(md5sum $T | cut -c1-8)]" >> $LOG; }; trap restore EXIT
{
[ -f $BAK ] && { echo "backup exists (earlier run interrupted?): restoring it first"; cp $BAK $T; }
cp $T $BAK; echo "baseline md5 $(md5sum $T | cut -c1-8) backed up"
LIST="model_float32_sensreg_lam40 lean_lam40_seed1 lean_lam40_seed2 wcat_s0 wcat_s1 wcat_s3 model_float32.pth.BASELINE_BAK"
echo "checkpoints: $LIST   rounds=$ROUNDS flips=$FLIPS eval-shuffle-seed=$EVSEED attacker=blind"
for F in $LIST; do
  P=$SRC/$F.pth; [ "$F" = "model_float32.pth.BASELINE_BAK" ] && P=$BAK
  [ -f $P ] || { echo "missing $P"; continue; }
  cp $P $T; [ "$(md5sum $P | cut -c1-32)" = "$(md5sum $T | cut -c1-32)" ] || { echo "md5 mismatch for $F, aborting"; break; }
  L=$F; [ "$F" = "model_float32.pth.BASELINE_BAK" ] && L=baseline
  echo; echo "=== $L  $(date) ==="
  python -u lean_pbs_adapt.py --dataset CIFAR10 --arch resnet18 --bits 8 --covs 0.01 --flips $FLIPS --adapt-flips 500 --rounds $ROUNDS \
      --attacker fixed --eval-shuffle-seed $EVSEED --label e10c_$L
done
echo "DONE e10c $(date)"
} >> $LOG 2>&1
