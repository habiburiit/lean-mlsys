#!/usr/bin/env bash
# run_lean_archs_range.sh <GPU_A> <GPU_B> <arch> [lams]  -  transformers: the RANGE term alone (--reg clip, no tail term), which
#   shrinks each layer's INT8 scale so every flip moves the weight less.  Trains clip-only models at the given lams on GPU_A
#   (10 epochs, lr 5e-4, bs 32, same recipe as the other backbones), measures HIGH/MID shares, then PBS with the 1% code
#   (coded weights unflippable) under per-channel INT8 (deployment default) for base, the existing lam5 both-term model and
#   the clip models, and per-tensor for the clip models; PBS runs split over GPU_A and GPU_B.
#   Logs: sensreg_results/archs/range_<arch>.log (training, shares) and pbs_range_<arch>_<label>.log (one per PBS run)
#   usage:  nohup bash run_lean_archs_range.sh 0 1 vit_b_16 "5 20" > /dev/null 2>&1 &
#           nohup bash run_lean_archs_range.sh 2 3 swin_t "1 5" > /dev/null 2>&1 &
#   status: grep -h "^ep \|^RESULT\|^DONE\|HIGH" sensreg_results/archs/range_<arch>.log sensreg_results/archs/pbs_range_<arch>_*.log
GA=$1; GB=$2; A=$3; LAMS=${4:-"5 20"}
PY=$(command -v python 2>/dev/null); [ -z "$PY" ] && PY=python
O=sensreg_results/archs; mkdir -p $O; LOG=$O/range_$A.log; : > $LOG; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/$A; B=$D/model_float32.pth
pbs() { # gpu ckpt label pc
  local SUF=pt; [ "$4" = "1" ] && SUF=pc; local L=$O/pbs_range_${A}_$3_$SUF.log; : > $L
  { echo "--- $A $3 PBS 1% code, $SUF INT8  $(date) ---"
    if [ "$4" = "1" ]; then CUDA_VISIBLE_DEVICES=$1 $PY -u lean_archs.py run lean_pbs_codec_pc.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $2 --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${A}_$3_${SUF}_code1x
    else CUDA_VISIBLE_DEVICES=$1 $PY -u lean_archs.py run lean_pbs_codec.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $2 --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${A}_$3_${SUF}_code1x; fi
    echo "DONE $A $3 $SUF $(date)"; } >> $L 2>&1
}
{
echo "=== $A range-only term, lams: $LAMS  $(date) ==="
[ -f $B ] || CUDA_VISIBLE_DEVICES=$GA $PY -u lean_archs.py base --arch $A
CK=""
for LAM in $LAMS; do
  L=$D/model_float32_sensreg_clip$LAM.pth
  echo "--- train clip-only lam $LAM  $(date) ---"
  [ -f $L ] || CUDA_VISIBLE_DEVICES=$GA $PY -u lean_archs.py train --arch $A --dataset imagenet --imagenet-root $IM --init $B --reg clip --lam $LAM --lr 0.0005 --bs 32 --epochs 10 --out $L
  [ -f $L ] && CK="$CK $L"
done
echo "--- shares ---"
CUDA_VISIBLE_DEVICES=$GA $PY -u lean_archs.py share --arch $A --dataset imagenet --imagenet-root $IM --base $B --ckpt $CK
echo "--- PBS  $(date) ---"
set -- $CK; C1=$1; C2=$2; L1=clip$(echo $LAMS | cut -d' ' -f1); L2=clip$(echo $LAMS | cut -d' ' -f2)
BOTH=$(ls $D/model_float32_sensreg_lam*.pth 2>/dev/null | head -1)
( pbs $GA $B base 1; [ -n "$C1" ] && pbs $GA $C1 $L1 1; [ -n "$C1" ] && pbs $GA $C1 $L1 0 ) &
( [ -n "$BOTH" ] && pbs $GB $BOTH both$(basename $BOTH .pth | sed 's/.*lam//') 1; [ -n "$C2" ] && pbs $GB $C2 $L2 1; [ -n "$C2" ] && pbs $GB $C2 $L2 0 ) &
wait
echo "DONE range $A $(date)"
} >> $LOG 2>&1
