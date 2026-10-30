#!/usr/bin/env bash
# run_c4_cifar_pc.sh <GPU>  -  C4 on ResNet-18 CIFAR-10 under PER-CHANNEL INT8, existing checkpoints only (no training):
#   baseline, LEAN lam40 (both terms), P_tail only (lam40, lam400), P_clip only (lam40).
#   1) HIGH/MID with the per-channel step (lean_hist_pc.py)  2) real single-flip damage, worst 100 (lean_truedamage_pc.py)
#   3) PBS with the 1% code, coded weights unflippable (lean_pbs_codec_pc.py).   Log: sensreg_results/c4/cifar_pc.log
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
export CUDA_VISIBLE_DEVICES=${1:-0}
CK=artifacts/models/cifar10/resnet18; B=$CK/model_float32.pth; L=$CK/model_float32_sensreg_lam40.pth
T=$CK/ablate_tail_lam40.pth; T4=$CK/ablate_tail_lam400.pth; C=$CK/ablate_clip_lam40.pth
O=sensreg_results/c4; mkdir -p $O; LOG=$O/cifar_pc.log; : > $LOG
{
echo "=== C4 CIFAR-10 ResNet-18, per-channel INT8  $(date) ==="; ls -1 $B $L $T $T4 $C
echo "--- 1) HIGH/MID, per-channel step ---"
for X in $L:lam40 $T:tail40 $T4:tail400 $C:clip40; do
  echo "## ${X##*:}"; $PY -u lean_hist_pc.py --dataset cifar10 --arch resnet18 --ckpt $B --ckpt2 ${X%%:*} --out $O/hist_pc_${X##*:} | grep -A4 "Bin occupancy\|tail_mass\|clean_acc\|ratio_high"
done
echo "--- 2) true single-flip damage, per-channel ---"
$PY -u lean_truedamage_pc.py --dataset cifar10 --arch resnet18 --ckpts $B $L $T $T4 $C --labels baseline lam40 tail_only tail_lam400 clip_only --topk 1000 --nrand 1000 --out $O/truedamage_pc_cifar
echo "--- 3) PBS 1% code, per-channel ---"
for X in $B:base $L:lam40 $T:tail40 $T4:tail400 $C:clip40; do
  $PY -u lean_pbs_codec_pc.py --dataset cifar10 --arch resnet18 --ckpt ${X%%:*} --cov 0.01 --attack pbs --exclude-coded 1 --flips 100,200,300,500 --label c10_${X##*:}_pc_code1x
done
echo "DONE c4_cifar_pc $(date)"
} >> $LOG 2>&1
