#!/usr/bin/env python
"""
lean_llm_flip.py  -  progressive sign-bit flip attack on an INT8 language model, with and without a
sensitivity-aware code over the top --cov of weights.  SEPARATE file; imports lean_sensreg_llm (unchanged).

Attack (adaptive, PBS-style): every step compute the gradient of the LM loss on the attack blocks, rank the
uncovered weights by first-order gain of a sign flip, evaluate the top-k candidates by the real loss, commit the
most damaging one.  Perplexity on held-out WikiText-2 test blocks is reported at --nflips.

Protection: the top --cov of weights by sign-flip damage (|g| * INT8 sign step), computed by the defender on
probe blocks of THIS model, are covered by the code; a flip on a covered weight is corrected, so the attacker
only searches the rest.  --protect none runs the same attack with nothing covered.

Usage:
  python lean_llm_flip.py --model ckpt_llm/llama32_1b/lam0_1ep --label lam0 --protect none
  python lean_llm_flip.py --model ckpt_llm/llama32_1b/lam0_1ep --label lam0 --protect selcode --cov 0.01
  python lean_llm_flip.py --model ckpt_llm/llama32_1b/lam3     --label lam3 --protect selcode --cov 0.01
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
    ap.add_argument("--probe-blocks", type=int, default=16, help="defender's probe blocks for the coverage ranking")
    ap.add_argument("--attack-blocks", type=int, default=2, help="attacker's blocks (from the test split, first ones)")
    ap.add_argument("--topk", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="res_llm/flip_code")
    a = ap.parse_args(); torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda"; lab = a.label or os.path.basename(a.model.rstrip("/"))
    nflips = [int(x) for x in a.nflips.split(",")]

    model, tok = L.load_model(a.model, dev); params = L.weight_params(model); plist = [p for _, p in params]
    tb = L.wikitext_blocks(tok, "test", a.seq_len, a.max_eval_tokens)
    xa = tb[:a.attack_blocks].to(dev)                         # attacker's data
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes); n_total = int(offs[-1])

    # defender: coverage from the damage ranking on probe blocks (same scorer as the hist command)
    prot = torch.zeros(n_total, dtype=torch.bool, device=dev)
    if a.protect == "selcode":
        s_all = L.score_model(model, tok, params, dev, a.seq_len, a.probe_blocks, "first")
        k = max(1, int(a.cov * n_total)); idx = np.argpartition(-s_all, k)[:k]
        prot[torch.from_numpy(idx).to(dev)] = True
    cov = 100.0 * float(prot.sum()) / n_total

    # deploy INT8 (per-tensor, as in the flip command)
    store = L.quantize_(params, 8)
    q_all = torch.cat([q.flatten() for _, _, q, _ in store])
    sc_all = torch.cat([torch.full((q.numel(),), float(s), device=dev) for _, _, q, s in store])
    clean = L.perplexity(model, tb, dev)
    print(f"[{lab}] INT8 clean ppl={clean:.2f}  protect={a.protect}  coverage={cov:.2f}%  ({n_total/1e6:.0f}M weights)", flush=True)

    def write(q_flat):
        with torch.no_grad():
            for i, p in enumerate(plist):
                p.copy_((q_flat[offs[i]:offs[i+1]] * sc_all[offs[i]:offs[i+1]]).view_as(p))

    def attack_loss():
        return model(input_ids=xa, labels=xa).loss

    def flip(q_flat, idx):
        qf = q_flat.clone(); v = qf[idx]; qf[idx] = torch.where(v >= 0, v - 128, v + 128); return qf

    def flip_val(q_flat, idx):
        v = float(q_flat[idx]); return v - 128.0 if v >= 0 else v + 128.0

    def set_one(idx, qval):                                     # write one weight only (fast candidate test)
        j = int(np.searchsorted(offs, idx, side="right") - 1)
        with torch.no_grad(): plist[j].view(-1)[idx - offs[j]] = qval * float(sc_all[idx])

    model.eval(); q_cur = q_all.clone(); flipped = torch.zeros(n_total, dtype=torch.bool, device=dev)
    res = {}; t0 = time.time()
    for step in range(1, max(nflips) + 1):
        model.zero_grad(set_to_none=True); attack_loss().backward()
        g = torch.cat([p.grad.flatten() for p in plist]); model.zero_grad(set_to_none=True)
        delta = torch.where(q_cur >= 0, -128.0, 128.0) * sc_all
        gain = g * delta                                           # first-order INcrease of the loss
        gain[prot] = -float("inf"); gain[flipped] = -float("inf")
        cand = torch.topk(gain, a.topk).indices
        best_i, best_l = None, -1.0
        with torch.no_grad():
            for i in cand.tolist():
                set_one(i, flip_val(q_cur, i)); l = attack_loss().item(); set_one(i, float(q_cur[i]))
                if l > best_l: best_l, best_i = l, i
        q_cur[best_i] = flip_val(q_cur, best_i); flipped[best_i] = True; set_one(best_i, float(q_cur[best_i]))
        del g, gain, delta
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
