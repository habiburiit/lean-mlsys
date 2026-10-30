#!/usr/bin/env python
"""
lean_llm_flip2.py  -  memory-lean version of lean_llm_flip.py for models above ~1B weights.
Same attack, same output.  Differences: INT8 codes kept as int16 per layer, one scalar scale per layer,
candidate ranking done layer by layer (no full-size delta / gain tensors), gradient freed per layer.
Peak memory ~ model + grads + 2 bytes/weight + 2 bits/weight instead of ~7 full fp32 copies.

Usage (same flags as lean_llm_flip.py):
  python lean_llm_flip2.py --model ckpt_llm/qwen2_1.5b/lam0_1ep --label lam0 --protect selcode --cov 0.01 --out res_llm/qwen2_1.5b/flip_code
"""
import argparse, json, os, time
import numpy as np, torch
import lean_sensreg_llm as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--label", default=None)
    ap.add_argument("--protect", choices=["none", "selcode"], default="none")
    ap.add_argument("--cov", type=float, default=0.01)
    ap.add_argument("--nflips", default="10,20,50,100")
    ap.add_argument("--seq-len", type=int, default=512); ap.add_argument("--max-eval-tokens", type=int, default=40000)
    ap.add_argument("--probe-blocks", type=int, default=16)
    ap.add_argument("--attack-blocks", type=int, default=2)
    ap.add_argument("--topk", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="res_llm/flip_code")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"; lab = a.label or os.path.basename(a.model.rstrip("/"))
    nflips = [int(x) for x in a.nflips.split(",")]

    model, tok = L.load_model(a.model, dev); params = L.weight_params(model); plist = [p for _, p in params]
    tb = L.wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens)
    xa = tb[:a.attack_blocks].to(dev)
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes); n_total = int(offs[-1])

    # defender: coverage by damage ranking on probe blocks; stored as one bool tensor per layer
    prot = [torch.zeros(n, dtype=torch.bool, device=dev) for n in sizes]
    if a.protect == "selcode":
        s_all = L.score_model(model, tok, params, dev, a.seq_len, a.probe_blocks, "first")
        k = max(1, int(a.cov * n_total)); idx = np.sort(np.argpartition(-s_all, k)[:k]); del s_all
        for j in range(len(plist)):
            sel = idx[(idx >= offs[j]) & (idx < offs[j+1])] - offs[j]
            if sel.size: prot[j][torch.from_numpy(sel).to(dev)] = True
        del idx
    n_prot = sum(int(p.sum()) for p in prot); cov = 100.0 * n_prot / n_total

    # deploy INT8 per tensor; keep codes as int16 (values -128..127) and one scale per layer
    store = L.quantize_(params, 8)
    q = [qq.flatten().to(torch.int16) for _, _, qq, _ in store]; sc = [float(s) for _, _, _, s in store]
    del store; torch.cuda.empty_cache()
    clean = L.perplexity(model, tb, dev)
    print(f"[{lab}] INT8 clean ppl={clean:.2f}  protect={a.protect}  coverage={cov:.2f}%  ({n_total/1e6:.0f}M weights)  [lean memory]", flush=True)

    def attack_loss():
        return model(input_ids=xa, labels=xa).loss

    def flipped_code(j, i):
        v = int(q[j][i]); return v - 128 if v >= 0 else v + 128

    def set_one(j, i, code):
        with torch.no_grad(): plist[j].view(-1)[i] = float(code) * sc[j]

    model.eval(); flipped = [torch.zeros(n, dtype=torch.bool, device=dev) for n in sizes]
    res = {}; t0 = time.time()
    for step in range(1, max(nflips) + 1):
        model.zero_grad(set_to_none=True); attack_loss().backward()
        # candidate ranking layer by layer: gain = g * (sign-flip weight change)
        cand = []   # (gain, layer, index)
        with torch.no_grad():
            for j, p in enumerate(plist):
                g = p.grad.view(-1)
                delta = torch.where(q[j] >= 0, -128.0, 128.0) * sc[j]
                gain = g * delta; del delta
                gain[prot[j]] = -float("inf"); gain[flipped[j]] = -float("inf")
                kk = min(a.topk, gain.numel()); top = torch.topk(gain, kk)
                cand += [(float(v), j, int(i)) for v, i in zip(top.values, top.indices)]
                del gain; p.grad = None
        cand.sort(reverse=True); cand = cand[:a.topk]
        best, best_l = None, -1.0
        with torch.no_grad():
            for _, j, i in cand:
                set_one(j, i, flipped_code(j, i)); l = attack_loss().item(); set_one(j, i, int(q[j][i]))
                if l > best_l: best_l, best = l, (j, i)
        j, i = best; q[j][i] = flipped_code(j, i); flipped[j][i] = True; set_one(j, i, int(q[j][i]))
        if step in nflips:
            ppl = L.perplexity(model, tb, dev); res[step] = ppl
            print(f"   [{lab}] {a.protect:5s} flips={step:4d}  attack loss={best_l:.3f}  ppl={ppl:10.2f}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'model':12s}{'protect':8s}{'cov':>7s}{'clean':>9s}" + "".join(f"{'n='+str(n):>11s}" for n in nflips))
    print(f"{lab:12s}{a.protect:8s}{cov:6.2f}%{clean:9.2f}" + "".join(f"{res[n]:11.2f}" for n in nflips))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"label": lab, "protect": a.protect, "cov_pct": cov, "clean": clean, "ppl": {str(k): v for k, v in res.items()}},
              open(os.path.join(a.out, f"{lab}_{a.protect}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
