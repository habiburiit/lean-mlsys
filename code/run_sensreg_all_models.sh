#!/usr/bin/env bash
# run_sensreg_all_models.sh  -  run in All_final with ANON active
# Six configs: {cifar10, cifar100, imagenet} x {resnet18, resnet50}
# ImageNet: only imagenet-val exists locally, so the baseline is the torchvision
# pretrained model, fine-tuning uses 40k of the 50k val images and evaluation the
# held-out 10k (fixed split, seed 0).  Say this in the email.
set -e
OUT=sensreg_results
IMNET=./imagenet-val
LAMS="10 30"
GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=$GPU

run_cfg () {   # dataset arch epochs lr bs
  DS=$1; ARCH=$2; EP=$3; LR=$4; BS=$5
  TAG=${DS}_${ARCH}; O=$OUT/$TAG; mkdir -p $O
  echo; echo "################ $TAG ################"
  if [ "$DS" = "imagenet" ]; then
    BASE=pretrained; CKDIR=artifacts/models/imagenet/$ARCH; mkdir -p $CKDIR
    EXTRA="--imagenet-root $IMNET"
  else
    CKDIR=artifacts/models/$DS/$ARCH; BASE=$CKDIR/model_float32.pth; EXTRA=""
    if [ ! -f $BASE ]; then echo "MISSING $BASE  (skipping $TAG)"; return; fi
  fi
  CKPTS="$BASE"; LABELS="baseline"; DIRS=""; LAMLIST=""
  for L in $LAMS; do
    CK=$CKDIR/model_float32_sensreg_lam$L.pth
    if [ ! -f $CK ]; then
      python lean_sensreg.py train --dataset $DS --arch $ARCH $EXTRA --init $BASE --reg both \
        --lam $L --lr $LR --bs $BS --epochs $EP --out $CK | tee $O/train_lam$L.log
    fi
    python lean_sensreg.py hist --dataset $DS --arch $ARCH $EXTRA --ckpt $BASE --ckpt2 $CK --out $O/hist_lam$L | tee $O/hist_lam$L.log
    CKPTS="$CKPTS $CK"; LABELS="$LABELS lam$L"; DIRS="$DIRS $O/hist_lam$L"; LAMLIST="$LAMLIST,$L"
  done
  LAMLIST=${LAMLIST#,}
  python lean_sensreg.py plot --dataset $DS --arch $ARCH $EXTRA --ckpts $CKPTS --labels $LABELS --nbins 20 --scale log --out $O/qhist_log | tee $O/qhist_log.log
  python lean_sensreg.py knob --dirs $DIRS --lams $LAMLIST --out $O/knob | tee $O/knob.log
  python lean_sensreg.py ptq  --dataset $DS --arch $ARCH $EXTRA --ckpts $CKPTS --labels $LABELS --bits 8,6,5,4 --out $O/ptq | tee $O/ptq.log
  python lean_sensreg.py flip --dataset $DS --arch $ARCH $EXTRA --mode targeted --nflips 1,3,10,30,100 --ckpts $CKPTS --labels $LABELS --out $O/flip_targeted | tee $O/flip_targeted.log
}

#        dataset   arch      epochs lr     bs
run_cfg  cifar10   resnet18  10     0.002  128
run_cfg  cifar10   resnet50  10     0.002  128
run_cfg  cifar100  resnet18  10     0.002  128
run_cfg  cifar100  resnet50  10     0.002  128
run_cfg  imagenet  resnet18  2      0.0005 64
run_cfg  imagenet  resnet50  2      0.0005 32

echo; echo "################ cross-model summary ################"
python - <<'EOF'
import json, os, glob
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rows=[]
for d in sorted(glob.glob("sensreg_results/*_resnet*")):
    tag=os.path.basename(d)
    try:
        h={L: json.load(open(f"{d}/hist_lam{L}/summary.json")) for L in (10,30)}
        p=json.load(open(f"{d}/ptq/ptq.json")); f=json.load(open(f"{d}/flip_targeted/flip.json"))
    except Exception as e:
        print("skip", tag, e); continue
    b=h[10]
    rows.append(dict(tag=tag, acc=b["clean_acc"], acc10=h[10]["clean_acc2"], acc30=h[30]["clean_acc2"],
        high0=b["bins_pct"]["baseline"]["HIGH"], high10=h[10]["bins_pct"]["regularized"]["HIGH"], high30=h[30]["bins_pct"]["regularized"]["HIGH"],
        mid0=b["bins_pct"]["baseline"]["MID"], mid10=h[10]["bins_pct"]["regularized"]["MID"], mid30=h[30]["bins_pct"]["regularized"]["MID"],
        int4_0=p["baseline"]["fp32"]-p["baseline"]["int4"], int4_10=p["lam10"]["fp32"]-p["lam10"]["int4"], int4_30=p["lam30"]["fp32"]-p["lam30"]["int4"],
        f100_0=f["baseline"]["100"], f100_10=f["lam10"]["100"], f100_30=f["lam30"]["100"]))
print(f"{'config':20s}{'acc b/10/30':>20s}{'HIGH% b/10/30':>22s}{'MID% b/10/30':>20s}{'INT4 loss b/10/30':>22s}{'acc@100flips b/10/30':>24s}")
for r in rows:
    print(f"{r['tag']:20s}{r['acc']:6.2f}/{r['acc10']:5.2f}/{r['acc30']:5.2f} "
          f"{r['high0']:6.3f}/{r['high10']:5.3f}/{r['high30']:5.3f}   "
          f"{r['mid0']:5.2f}/{r['mid10']:5.2f}/{r['mid30']:5.2f}   "
          f"{r['int4_0']:5.2f}/{r['int4_10']:5.2f}/{r['int4_30']:5.2f}    "
          f"{r['f100_0']:5.1f}/{r['f100_10']:5.1f}/{r['f100_30']:5.1f}")
json.dump(rows, open("sensreg_results/cross_model_summary.json","w"), indent=2)
if rows:
    fig,ax=plt.subplots(1,3,figsize=(12,3.4)); x=np.arange(len(rows)); w=0.27
    tags=[r["tag"].replace("_resnet","\nR") for r in rows]
    for i,(k,ttl,yl) in enumerate([("high","HIGH bin (% of weights)","%"),("int4","INT4 PTQ accuracy loss","points"),("f100","accuracy after 100 targeted flips","%")]):
        for j,(suf,lab,c) in enumerate([("0","baseline","#898781"),("10","lam10","#2a78d6"),("30","lam30","#7c3aed")]):
            key={"high":"high","int4":"int4_","f100":"f100_"}[k]+suf if k!="high" else "high"+suf
            ax[i].bar(x+(j-1)*w,[r[key] for r in rows],w,label=lab,color=c)
        ax[i].set_xticks(x); ax[i].set_xticklabels(tags,fontsize=7); ax[i].set_title(ttl,fontsize=9); ax[i].set_ylabel(yl)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("sensreg_results/cross_model_summary.png",dpi=200)
    print("[saved] sensreg_results/cross_model_summary.png")
EOF
python lean_sensreg.py figure --root $OUT --out $OUT/fig_bins_accuracy.png
echo DONE
