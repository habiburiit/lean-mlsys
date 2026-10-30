#!/usr/bin/env bash
# run_lean_archs.sh <GPU> [archs...]  -  LEAN on extra ImageNet backbones, end to end:
#   base checkpoint -> LEAN fine-tune (lam 40, 10 epochs, lr 5e-4, bs 32) -> HIGH/MID shares vs the baseline ->
#   PBS with the real decoder (lean_pbs_codec.py, no code and 1% code) -> T-BFA held-out (lean_tbfa.py, 1% code).
#   Default archs: mobilenet_v2 vit_b_16 swin_t.  Everything through lean_archs.py; existing scripts unchanged.
# usage (one model per GPU, in parallel):
#   nohup bash run_lean_archs.sh 0 mobilenet_v2 > /dev/null 2>&1 &
#   nohup bash run_lean_archs.sh 1 vit_b_16 > /dev/null 2>&1 &
#   nohup bash run_lean_archs.sh 2 swin_t > /dev/null 2>&1 &
#   tail -f sensreg_results/archs/archs_vit_b_16.log
export CUDA_VISIBLE_DEVICES=${1:-0}; shift
ARCHS=${@:-mobilenet_v2 vit_b_16 swin_t}
O=sensreg_results/archs; mkdir -p $O; LOG=$O/archs_$(echo $ARCHS | tr " " _).log; : > $LOG
IM=${IMAGENET_ROOT:-./imagenet-val}; LAM=${LAM:-40}
{
echo "=== LEAN on extra backbones: $ARCHS  $(date) ==="
for A in $ARCHS; do
  D=artifacts/models/imagenet/$A; B=$D/model_float32.pth; L=$D/model_float32_sensreg_lam$LAM.pth
  echo "##### $A"
  [ -f $B ] || python -u lean_archs.py base --arch $A
  echo "--- train lam $LAM ---"
  [ -f $L ] || python -u lean_archs.py train --arch $A --dataset imagenet --imagenet-root $IM --init $B --reg both --lam $LAM --lr 0.0005 --bs 32 --epochs 10 --out $L
  echo "--- shares ---"
  python -u lean_archs.py share --arch $A --dataset imagenet --imagenet-root $IM --base $B --ckpt $L
  for ck in $B:base $L:lam$LAM; do
    F=${ck%%:*}; LAB=${A}_${ck##*:}
    echo "--- $LAB PBS no code ---"
    python -u lean_archs.py run lean_pbs_codec.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $F --cov 0 --attack pbs --flips 10,25,50,100 --label ${LAB}_nocode
    echo "--- $LAB PBS 1% code (real decoder) ---"
    python -u lean_archs.py run lean_pbs_codec.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $F --cov 0.01 --attack pbs --flips 10,25,50,100,200,300 --label ${LAB}_code1
    echo "--- $LAB T-BFA 1% code ---"
    python -u lean_archs.py run lean_tbfa.py --dataset imagenet --arch $A --imagenet-root $IM --ckpt $F --label $LAB --protect selcode --cov 0.01 --max-flips 30 --report 10,20,30
  done
done
echo "DONE archs $(date)"
} >> $LOG 2>&1
