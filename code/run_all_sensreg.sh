#!/usr/bin/env bash
# run_all_sensreg.sh  -  run in All_final with ANON active
# Produces every figure for the "controllable skewing" experiment in ./sensreg_results/
set -e
ARCH=resnet18
DIR=artifacts/models/cifar10/$ARCH
BASE=$DIR/model_float32.pth
LAMS="1 3 10 30"
OUT=sensreg_results
mkdir -p $OUT

echo "=== 1. train sweep (skips lambdas already trained) ==="
for L in $LAMS; do
  CK=$DIR/model_float32_sensreg_lam$L.pth
  if [ ! -f $CK ]; then
    python lean_sensreg.py train --arch $ARCH --init $BASE --reg both --lam $L --lr 0.002 --epochs 10 --out $CK | tee $OUT/train_lam$L.log
  else
    echo "have $CK"
  fi
done

CKPTS="$BASE"; LABELS="baseline"; DIRS=""; LAMLIST=""
for L in $LAMS; do
  CKPTS="$CKPTS $DIR/model_float32_sensreg_lam$L.pth"; LABELS="$LABELS lam$L"
  DIRS="$DIRS $OUT/hist_lam$L"; LAMLIST="$LAMLIST,$L"
done
LAMLIST=${LAMLIST#,}

echo "=== 2. HIGH/MID/LOW bins per lambda (fixed baseline thresholds) ==="
for L in $LAMS; do
  python lean_sensreg.py hist --arch $ARCH --ckpt $BASE --ckpt2 $DIR/model_float32_sensreg_lam$L.pth --out $OUT/hist_lam$L | tee $OUT/hist_lam$L.log
done

echo "=== 3. quantized-sensitivity histograms, all models on one axis ==="
python lean_sensreg.py plot --arch $ARCH --ckpts $CKPTS --labels $LABELS --nbins 20 --scale linear --out $OUT/qhist_linear | tee $OUT/qhist_linear.log
python lean_sensreg.py plot --arch $ARCH --ckpts $CKPTS --labels $LABELS --nbins 20 --scale log    --out $OUT/qhist_log    | tee $OUT/qhist_log.log

echo "=== 4. control knob: lambda -> HIGH bin %, tail mass, accuracy ==="
python lean_sensreg.py knob --dirs $DIRS --lams $LAMLIST --out $OUT/knob | tee $OUT/knob.log

echo "=== 5. quantization loss (PTQ INT8/6/5/4) ==="
python lean_sensreg.py ptq --arch $ARCH --ckpts $CKPTS --labels $LABELS --out $OUT/ptq | tee $OUT/ptq.log

echo "=== 6. bit-flip robustness without ECC (random INT8 sign-bit flips) ==="
python lean_sensreg.py flip --arch $ARCH --ckpts $CKPTS --labels $LABELS --out $OUT/flip | tee $OUT/flip.log

echo
echo "DONE. Figures:"
find $OUT -name '*.png' | sort
echo
echo "Step 7 (the selective code coverage, your existing pipeline) is manual; see the note."
