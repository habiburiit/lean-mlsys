#!/usr/bin/env python
"""
lean_overhead.py  -  training overhead of the LEAN term (item 8): wall time per epoch and peak GPU memory of
lean_sensreg.py train, one epoch, plain fine-tune (--reg none, lam 0) against LEAN (--reg both, lam 40), same data,
batch size and learning rate.  Runs lean_sensreg.py UNCHANGED through runpy with a patched argv.  SEPARATE file.

usage:  CUDA_VISIBLE_DEVICES=1 python -u lean_overhead.py --imagenet-root ./imagenet-val 2>&1 | tee sensreg_results/overhead.log
        (add --only cifar10:resnet18 to run one config)
"""
import argparse, json, os, runpy, sys, time, torch
try:
    import lean_vgg  # registers vgg16_bn for lean_sensreg
except Exception:
    pass

CFG = [  # dataset, arch, lr, bs
    ("cifar10", "resnet18", 0.002, 128), ("cifar10", "resnet50", 0.002, 128), ("cifar10", "vgg16_bn", 0.002, 128),
    ("imagenet", "resnet50", 0.0005, 32)]


def run(dataset, arch, lr, bs, lam, reg, a):
    out = f"{a.tmp}/overhead_{dataset}_{arch}_lam{lam}.pth"
    argv = ["lean_sensreg.py", "train", "--dataset", dataset, "--arch", arch, "--init", f"artifacts/models/{dataset}/{arch}/model_float32.pth",
            "--reg", reg, "--lam", str(lam), "--lr", str(lr), "--bs", str(bs), "--epochs", "1", "--out", out,
            "--imagenet-root", a.imagenet_root, "--workers", str(a.workers)]
    sys.argv = argv; torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    # call cmd_train of the IMPORTED lean_sensreg (so lean_vgg's VGG-aware build_model applies), parsing the same argv
    import lean_sensreg as S
    ns = argparse.Namespace(arch=arch, dataset=dataset, imagenet_root=a.imagenet_root, data="./data", workers=a.workers, seed=0,
                            init=argv[argv.index("--init") + 1], out=out, epochs=1, bs=bs, lr=lr, reg=reg, lam=lam,
                            tail_frac=0.01, tail_every=4, clip_q=0.99, clip_w=100.0)
    S.cmd_train(ns)
    torch.cuda.synchronize(); dt = time.time() - t0; mem = torch.cuda.max_memory_allocated() / 2**30
    try: os.remove(out)
    except Exception: pass
    return dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lam", type=float, default=40); ap.add_argument("--only", default="")
    ap.add_argument("--tmp", default="sensreg_results/overhead_tmp"); ap.add_argument("--out", default="sensreg_results/overhead.json")
    a = ap.parse_args(); os.makedirs(a.tmp, exist_ok=True); rows = []
    print(f"{'model':22s}{'plain s/epoch':>15s}{'LEAN s/epoch':>14s}{'time x':>8s}{'plain GB':>10s}{'LEAN GB':>9s}{'mem x':>7s}", flush=True)
    for ds, arch, lr, bs in CFG:
        if a.only and a.only != f"{ds}:{arch}": continue
        try:
            t0, m0 = run(ds, arch, lr, bs, 0, "none", a)
            t1, m1 = run(ds, arch, lr, bs, a.lam, "both", a)
        except SystemExit:
            print(f"{ds} {arch}: training script exited early", flush=True); continue
        except Exception as e:
            print(f"{ds} {arch}: skipped: {e}", flush=True); continue
        rows.append(dict(dataset=ds, arch=arch, plain_s=t0, lean_s=t1, plain_gb=m0, lean_gb=m1))
        print(f"{ds+' '+arch:22s}{t0:15.0f}{t1:14.0f}{t1/t0:8.2f}{m0:10.2f}{m1:9.2f}{m1/m0:7.2f}", flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
