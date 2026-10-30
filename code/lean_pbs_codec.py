#!/usr/bin/env python
"""
lean_pbs_codec.py  -  PBS with the REAL BCH codec inside the attack loop (reviewer item 2), per-tensor INT8 models.
SEPARATE file; imports lean_sensreg, ecc_engine and ecc_llm_mmlu (to_ub/from_ub) unchanged.

Difference to the paper's PBS tables: there the coded weights are unflippable.  Here every bit of every weight is flippable;
the coded weights (top --cov by deployed damage) are grouped into BCH codewords (G weights x K=--plane-bits planes,
t=--strong-t, keyed interleaving of the member order when --interleave 1), and after every flip the touched codewords are
DECODED with ecc_engine.XLayerECC.planes_decode, exactly as the LLM path does.  The attacker is adaptive: candidate and
committed losses are measured on the corrected model, so flips the code removes count for nothing.  Parity is modeled as
intact (hidden by STDM in LOW weights; its own corruption is the paper's separate STDM ablation).

Two attackers:
  --attack pbs       progressive bit search over all bits (50 candidates per step, real loss after correction)
  --attack cwaware   codeword-aware: knows the layout (--attacker-knows-key 1) or assumes adjacent grouping (0), ranks
                     codewords by summed damage, overloads each with t+1 bit errors (the minimum that defeats BCH), greedy
                     accumulation, reports accuracy against the TOTAL number of flips spent.

usage:
  python -u lean_pbs_codec.py --dataset cifar10 --arch resnet18 --ckpt <ckpt> --cov 0.01 --attack pbs --label c10_base
  python -u lean_pbs_codec.py ... --attack cwaware --attacker-knows-key 1 --cw-counts 1,2,5,10,20,40
"""
import argparse, time, numpy as np, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass
import ecc_engine as ENG
import ecc_llm_mmlu as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10"); ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--imagenet-root", default="./imagenet-val"); ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--cov", type=float, default=0.01); ap.add_argument("--probe-batches", type=int, default=8)
    ap.add_argument("--plane-bits", type=int, default=4); ap.add_argument("--strong-t", type=int, default=12); ap.add_argument("--light-t", type=int, default=2)
    ap.add_argument("--bch-m", type=int, default=7); ap.add_argument("--interleave", type=int, default=1); ap.add_argument("--attacker-knows-key", type=int, default=1)
    ap.add_argument("--attack", choices=["pbs", "cwaware"], default="pbs")
    ap.add_argument("--flips", default="10,25,50,100,200,300,500"); ap.add_argument("--topk", type=int, default=50); ap.add_argument("--attack-batches", type=int, default=2)
    ap.add_argument("--cw-counts", default="1,2,5,10,20,40"); ap.add_argument("--cw-probe", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-coded", type=int, default=0, help="1 = the paper's PBS protocol: the attacker knows the coded set and never flips it (decoder still runs on everything else)")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model); plist = [p for _, p in params]; model.eval()
    sizes = [p.numel() for p in plist]; offs = np.cumsum([0] + sizes); N = int(offs[-1])

    # ---- per-tensor INT8 (two's complement codes as int16 on the GPU), flat view ----
    with torch.no_grad():
        sc = torch.cat([torch.full((p.numel(),), p.abs().max().item() / 127.0 + 1e-12, device=dev) for p in plist])
        q = torch.cat([torch.clamp(torch.round(p.flatten() / sc[offs[i]:offs[i + 1]]), -128, 127).to(torch.int16) for i, p in enumerate(plist)])
    def write_all(qf):
        with torch.no_grad():
            for i, p in enumerate(plist): p.copy_((qf[offs[i]:offs[i + 1]].float() * sc[offs[i]:offs[i + 1]]).view_as(p))
    write_all(q); clean = S.evaluate(model, vl, dev); q_clean = q.clone()

    ab = []
    for x, y in vl:
        ab.append((x.to(dev), y.to(dev)))
        if len(ab) >= a.attack_batches: break
    def loss_now():
        with torch.no_grad(): return sum(F.cross_entropy(model(x), y).item() for x, y in ab) / len(ab)
    def grads():
        model.zero_grad(set_to_none=True); loss = sum(F.cross_entropy(model(x), y) for x, y in ab) / len(ab)
        return torch.cat([g.flatten() for g in torch.autograd.grad(loss, plist)]).detach()

    # ---- coded set: top --cov by damage on probe batches ----
    dsum = None; nb = 0
    for x, y in vl:
        if nb >= a.probe_batches: break
        x, y = x.to(dev), y.to(dev); model.zero_grad(set_to_none=True)
        g = torch.cat([gi.flatten() for gi in torch.autograd.grad(F.cross_entropy(model(x), y), plist)])
        d = g.abs() * 128.0 * sc; dsum = d if dsum is None else dsum + d; nb += 1
    k = max(1, int(a.cov * N)); prot_idx = torch.topk(dsum, k).indices.cpu().numpy(); prot_idx.sort()
    coded_mask = torch.zeros(N, dtype=torch.bool, device=dev); coded_mask[torch.from_numpy(prot_idx).to(dev)] = True
    order = rng.permutation(k) if a.interleave else np.arange(k)         # encoded order of the coded weights (keyed)
    enc_idx = prot_idx[order]                                              # position j in the code -> flat weight index
    pos_of = np.empty(N, dtype=np.int64); pos_of.fill(-1); pos_of[enc_idx] = np.arange(k)
    code = ENG.XLayerECC(strong_t=a.strong_t, light_t=a.light_t, bch_m=a.bch_m, delta=1.0, host_key=1234)
    code.host_bits = 5; code.plane_bits = a.plane_bits; code.bit_width = 8; code.embed_mode = "stdm"; code.stdm_L = 1
    wb_clean = M.to_ub(q_clean[enc_idx].cpu().numpy().astype(np.int8)).reshape(-1)
    ecc, n0, G = code.planes_encode(wb_clean, a.plane_bits, W=8); nk = code.nk; t = code.strong_t; ncw = int(np.ceil(n0 / G))
    print(f"[{a.label or a.ckpt}] INT8 clean {clean:.2f}  N {N}  coded {k} weights ({100*k/N:.2f}%) in {ncw} codewords of G={G} x K={a.plane_bits} planes, "
          f"BCH t={t}, parity {len(ecc)} bits ({len(ecc)/N:.3f} b/w), interleave {a.interleave}", flush=True)

    def correct(qf, touched_pos):
        """decode the codewords that contain encoded positions touched_pos; returns corrected flat codes and #codewords beyond t"""
        if len(touched_pos) == 0: return qf, 0
        cws = np.unique(np.asarray(touched_pos) // G)
        wb = M.to_ub(qf[enc_idx].cpu().numpy().astype(np.int8)).reshape(-1)
        pad = (-n0) % G; wf = np.concatenate([wb, np.zeros(pad, np.uint8)]) if pad else wb
        sub_w = np.concatenate([wf[c * G:(c + 1) * G] for c in cws]); sub_p = np.concatenate([ecc[c * nk:(c + 1) * nk] for c in cws])
        fixed = code.planes_decode(sub_w, sub_p, a.plane_bits, len(sub_w), G, W=8)
        out = wf.copy()
        for j, c in enumerate(cws): out[c * G:(c + 1) * G] = fixed[j * G:(j + 1) * G]
        out = out[:n0]; qc = qf.clone()
        qc[enc_idx] = torch.from_numpy(M.from_ub(out).astype(np.int16)).to(dev)
        # a codeword is "beyond t" if after decoding it still differs from clean in a covered plane
        beyond = 0
        for c in cws:
            m = enc_idx[c * G:min((c + 1) * G, n0)]
            if (qc[m] != q_clean[m]).any(): beyond += 1
        return qc, beyond

    def flip_bit(v, b):
        nv = v ^ (1 << b); return nv - 256 if nv > 127 else nv

    flips_list = [int(x) for x in a.flips.split(",")]; t0 = time.time(); results = {}
    q_att = q_clean.clone()                     # what the attacker has flipped (before correction)
    touched = set()                             # encoded positions with a flip in them

    def deployed(qf_att, touched_set):
        qc, beyond = correct(qf_att, list(touched_set)); write_all(qc); return qc, beyond

    if a.attack == "pbs":
        n_done = 0; n_in_code = 0
        for target in flips_list:
            while n_done < target:
                qc, _ = deployed(q_att, touched); g = grads(); cands = []
                for b in range(8):
                    nq = q_att ^ (1 << b); nq = torch.where(nq > 127, nq - 256, nq)
                    score = g * (nq - q_att).float() * sc
                    if a.exclude_coded: score = score.masked_fill(coded_mask, float("-inf"))
                    kk = min(a.topk, N); v, ix = torch.topk(score, kk); cands += [(float(v[j]), int(ix[j]), b) for j in range(kk)]
                cands.sort(key=lambda z: -z[0]); cands = cands[:a.topk]; best = None
                for _, ix, b in cands:                                  # adaptive: loss after correction
                    old = int(q_att[ix]); q_att[ix] = flip_bit(old, b)
                    tp = set(touched); p = pos_of[ix]
                    if p >= 0: tp.add(int(p))
                    deployed(q_att, tp); L = loss_now()
                    if best is None or L > best[0]: best = (L, ix, b)
                    q_att[ix] = old
                L, ix, b = best; q_att[ix] = flip_bit(int(q_att[ix]), b); p = pos_of[ix]
                if p >= 0: touched.add(int(p)); n_in_code += 1
                n_done += 1
            qc, beyond = deployed(q_att, touched); acc = S.evaluate(model, vl, dev); results[target] = acc
            print(f"[{a.label}] flips {target:4d}  acc {acc:6.2f}  loss {loss_now():.3f}  flips in coded weights {n_in_code}  codewords beyond t {beyond}  {time.time()-t0:.0f}s", flush=True)
        print(f"RESULT {a.label} attack=pbs cov={a.cov} clean={clean:.2f} " + " ".join(f"{k_}:{v:.2f}" for k_, v in results.items()), flush=True)
    else:
        # codeword-aware: rank codewords by summed damage of their members, overload the best with t+1 bit errors each
        counts = sorted(int(x) for x in a.cw_counts.split(","))
        knows = bool(a.attacker_knows_key) or not a.interleave
        if knows:
            members = {c: enc_idx[c * G:min((c + 1) * G, n0)] for c in range(ncw)}
        else:                                                           # attacker assumes adjacent grouping of the sorted coded set
            members = {c: prot_idx[c * G:min((c + 1) * G, n0)] for c in range(ncw)}
        dcpu = dsum.cpu().numpy(); cw_rank = sorted(members, key=lambda c: -dcpu[members[c]].sum())[:a.cw_probe]
        def overload(c):
            """t+1 bit errors in codeword c: sign bits of the G members first, then plane 6 of the most damaging ones"""
            m = members[c][np.argsort(-dcpu[members[c]])]; errs = []; need = t + 1; plane = 7
            while need > 0 and plane >= 8 - a.plane_bits:
                take = min(need, len(m)); errs += [(int(i), plane) for i in m[:take]]; need -= take; plane -= 1
            return errs
        measured = []
        for c in cw_rank:                                               # single-codeword damage, measured after correction
            errs = overload(c); qa = q_clean.clone()
            for i, b in errs: qa[i] = flip_bit(int(qa[i]), b)
            deployed(qa, {int(pos_of[i]) for i, _ in errs if pos_of[i] >= 0}); measured.append((loss_now(), c, errs))
        measured.sort(key=lambda z: -z[0]); kept = []; cur = []; cur_loss = -1; spent = 0
        print(f"[{a.label}] cwaware: attacker {'KNOWS' if knows else 'does NOT know'} the layout; {len(measured)} codewords measured, "
              f"best single-codeword loss {measured[0][0]:.3f} with {len(measured[0][2])} errors", flush=True)
        for L1, c, errs in measured:
            if len(kept) >= counts[-1]: break
            trial = cur + errs; qa = q_clean.clone()
            for i, b in trial: qa[i] = flip_bit(int(qa[i]), b)
            qc, beyond = deployed(qa, {int(pos_of[i]) for i, _ in trial if pos_of[i] >= 0}); L = loss_now()
            if L > cur_loss:
                kept.append(c); cur = trial; cur_loss = L
                if len(kept) in counts:
                    acc = S.evaluate(model, vl, dev); results[len(cur)] = acc
                    print(f"[{a.label}] codewords {len(kept):3d}  bit errors {len(cur):4d}  acc {acc:6.2f}  loss {L:.3f}  codewords beyond t {beyond}  {time.time()-t0:.0f}s", flush=True)
        print(f"RESULT {a.label} attack=cwaware knows={int(knows)} cov={a.cov} clean={clean:.2f} " + " ".join(f"{k_}:{v:.2f}" for k_, v in results.items()), flush=True)
    write_all(q_clean)


if __name__ == "__main__":
    main()
