#!/usr/bin/env bash
# run_lean_archs_pc.sh <GPU> <arch> <lam>  -  paper PBS protocol (1% code, coded weights unflippable) with PER-CHANNEL INT8
#   (lean_pbs_codec_pc.py), base vs LEAN.  Log: sensreg_results/archs/pbs_pc_<arch>.log
export CUDA_VISIBLE_DEVICES=$1; A=$2; LAM=$3
O=sensreg_results/archs; mkdir -p $O; LOG=$O/pbs_pc_$A.log; : > $LOG; IM=${IMAGENET_ROOT:-./imagenet-val}
D=artifacts/models/imagenet/$A
{
for ck in model_float32:base model_float32_sensreg_lam$LAM:lam$LAM; do
  F=${ck%%:*}; LAB=${A}_${ck##*:}
  echo "--- $LAB PBS, 1% code, per-channel INT8 ---"
  python -u lean_archs.py run lean_pbs_codec_pc.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $D/$F.pth --cov 0.01 --attack pbs --exclude-coded 1 --flips 10,25,50,100,200,300 --label ${LAB}_pc_code1x
done
echo "DONE pbs_pc $A $(date)"
} >> $LOG 2>&1
