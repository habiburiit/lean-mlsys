#!/usr/bin/env bash
# run_lean_archs_pbs.sh <GPU> <arch> <lam>  -  the paper's PBS protocol (coded weights unflippable, 1% code) on an extra backbone,
#   base vs LEAN, ImageNet held-out.  Uses lean_pbs_codec.py --exclude-coded 1 through lean_archs.py.
# usage:  nohup bash run_lean_archs_pbs.sh 0 mobilenet_v2 5 > /dev/null 2>&1 &   tail -f sensreg_results/archs/pbs_mobilenet_v2.log
export CUDA_VISIBLE_DEVICES=$1; A=$2; LAM=$3
O=sensreg_results/archs; mkdir -p $O; LOG=$O/pbs_$A.log; : > $LOG; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/$A
{
for ck in model_float32:base model_float32_sensreg_lam$LAM:lam$LAM; do
  F=${ck%%:*}; LAB=${A}_${ck##*:}
  echo "--- $LAB PBS, 1% code, coded weights unflippable ---"
  python -u lean_archs.py run lean_pbs_codec.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${LAB}_code1x
done
echo "DONE pbs $A $(date)"
} >> $LOG 2>&1
