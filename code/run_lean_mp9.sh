#!/usr/bin/env bash
# run_lean_mp9.sh <dataset> <arch> <gpu>   -  LEAN low-bit resilient mode on one model, end to end, logs in sensreg_results/mp9/
#   1. baseline (CIFAR: trained from scratch if missing; ImageNet: torchvision pretrained)
#   2. one fine-tune at 3.29 bits: lean_train_qat_mp9.py (lambda 100, qclip 0.99, penalty on top 3%, 1% tier)
#   3. PBS in deployed format: BCH on the 1% tier only, then + interleaved BCH t=19 n=1023 on the sign plane
# e.g.  bash run_lean_mp9.sh cifar100 resnet34 1
set -o pipefail
DS=$1; ARCH=$2; export CUDA_VISIBLE_DEVICES=${3:-0}
IM=${IM:-./imagenet-val}
D=artifacts/models/$DS/$ARCH; O=sensreg_results/mp9; mkdir -p $D $O
FL=10,25,50,100,200,300,500
if [ "$DS" = "imagenet" ]; then
  B=pretrained; TR="--lr 0.0005 --bs 32 --imagenet-root $IM"; AT="--imagenet-root $IM"
else
  B=$D/model_float32.pth; TR=""; AT=""
  if [ ! -f $B ]; then
    echo "[$DS/$ARCH] no baseline, training from scratch (60 epochs)"
    python -u lean_vgg.py base --dataset $DS --arch $ARCH --epochs 60 --out $B > $O/${DS}_${ARCH}_base.log 2>&1
  fi
fi
M=$D/model_mp9_b329_lam100_q99_p3.pth
[ -f $M ] || python -u lean_train_qat_mp9.py --dataset $DS --arch $ARCH --init $B --budget 3.29 --lam 100 --qclip 0.99 --pen-frac 0.03 $TR --out $M > $O/${DS}_${ARCH}_train.log 2>&1
python -u lean_pbs_secded.py --dataset $DS --arch $ARCH --ckpt $M --secded off $AT --flips $FL > $O/${DS}_${ARCH}_pbs_bch_only.log 2>&1
python -u lean_pbs_secded.py --dataset $DS --arch $ARCH --ckpt $M --secded on --interleave on --t 19 --n 1023 $AT --flips $FL > $O/${DS}_${ARCH}_pbs_t19.log 2>&1
echo "[$DS/$ARCH] DONE"; tail -n 2 $O/${DS}_${ARCH}_pbs_bch_only.log $O/${DS}_${ARCH}_pbs_t19.log
