#!/usr/bin/env bash
# run_lean_archs_dw.sh <GPU> <arch> <lam>  -  PBS with the 1% tail code PLUS all depthwise weights coded (lean_pbs_codec_dw.py),
#   base vs LEAN, per-tensor INT8 and then per-channel INT8 (LEAN_PC=1).  Log: sensreg_results/archs/pbs_dw_<arch>.log
#   usage: nohup bash run_lean_archs_dw.sh 0 mobilenet_v2 5 > /dev/null 2>&1 &     nohup bash run_lean_archs_dw.sh 1 efficientnet_b0 5 > /dev/null 2>&1 &
export CUDA_VISIBLE_DEVICES=$1; A=$2; LAM=$3
O=sensreg_results/archs; mkdir -p $O; LOG=$O/pbs_dw_$A.log; : > $LOG; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/$A
{
for PC in 0 1; do
for ck in model_float32:base model_float32_sensreg_lam$LAM:lam$LAM; do
  F=${ck%%:*}; LAB=${A}_${ck##*:}; SUF=dw; [ "$PC" = "1" ] && SUF=dw_pc
  [ -f $D/$F.pth ] || { echo "missing $D/$F.pth"; continue; }
  echo "--- $LAB PBS, 1% code + depthwise coded, PC=$PC  $(date) ---"
  LEAN_PC=$PC python -u lean_archs.py run lean_pbs_codec_dw.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${LAB}_${SUF}_code1x
done
done
echo "DONE pbs_dw $A $(date)"
} >> $LOG 2>&1
