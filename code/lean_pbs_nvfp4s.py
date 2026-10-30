#!/usr/bin/env python
"""
lean_pbs_nvfp4s.py  -  lean_pbs_nvfp4.py plus the SIGN-PLANE code of the paper's low-bit security mode (--sign-cost).
Derived copy; lean_pbs_nvfp4.py itself is unchanged.  With --sign-cost T+1 every sign-bit flip the attacker commits costs
T+1 flips of its budget (the long interleaved BCH over the sign plane, t=T, 1023 sign bits per codeword spread over random
addresses: an adaptive attacker has to overload one codeword, t+1 flips, to land one sign flip).  0 = no sign-plane code.
Also --sign-cost -1 = sign bits unflippable.  Everything else as in lean_pbs_nvfp4.py.
SEPARATE file; imports lean_sensreg and lean_nvfp4 unchanged.  Same protocol as the paper's INT8 PBS: at every step
the attacker ranks every flippable bit by the first-order loss increase g_i * dw (gradient on --attack-batches batches of
128 held-out images), evaluates the real loss of the --topk best candidates, commits the single bit that raises the loss
most, and repeats; accuracy on the full held-out set is reported at every count in --flips.

The NVFP4 model is the saved float checkpoint re-quantized with lean_nvfp4.codes_nvfp4 (E2M1 codes, E4M3 block scales).
Flippable bits:
  weight bits   4 per weight (sign, e1, e0, m): dw = (value(code ^ bit) - value(code)) x block scale
  scale bits    8 per block (E4M3: sign, 4 exponent, 3 mantissa) when --scale-bits on: the block's 16 weights are
                rescaled by the new scale; first-order dL = sum_block g_j w_j x (new/old - 1); a sign flip of the scale
                negates the block.  NVFP4 keeps the scales in memory next to the weights, so they are attack surface.
Protection (--protect selcode): the top --cov fraction of weights by deployed damage |g_i| x (largest single-bit change)
are covered by the code and treated as unflippable (all 4 bits), as the INT8 experiments treat the BCH-covered weights;
--protect-scales on also covers every block scale (8 bits per 16 weights, 0.5 bits/weight, i.e. the scale plane is
small enough to code entirely).  --protect none attacks everything.

usage:
  python -u lean_pbs_nvfp4.py --dataset cifar10 --arch resnet18 --ckpt artifacts/models/cifar10/resnet18/nvfp4/lam0.pth \
      --protect selcode --cov 0.01 --flips 10,25,50,100,200,300,500 --label control
"""
import argparse, time, math, torch, torch.nn.functional as F
import lean_sensreg as S
try:
    import lean_vgg
except Exception:
    pass
import lean_nvfp4 as Q

E2M1 = Q.E2M1


def e4m3_bits_to_val(bits8):
    """E4M3 decode of an 8-bit integer tensor (bias 7, no inf; 0x7F/0xFF treated as max 448)"""
    s = (bits8 >> 7) & 1; e = (bits8 >> 3) & 0xF; m = bits8 & 7
    val = torch.where(e == 0, (m.float() / 8.0) * 2.0 ** -6, (1.0 + m.float() / 8.0) * 2.0 ** (e.float() - 7))
    val = torch.where((e == 15) & (m == 7), torch.full_like(val, 448.0), val)
    return torch.where(s == 1, -val, val)


def e4m3_encode(val):
    """nearest E4M3 bit pattern for positive values (clamped to [2^-6, 448])"""
    v = val.clamp(min=2.0 ** -6, max=448.0)
    e = torch.floor(torch.log2(v)).clamp(-6, 8); m = torch.round((v / 2.0 ** e - 1.0) * 8.0).clamp(0, 7)
    return ((e + 7).long() << 3) | m.long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10")
    ap.add_argument("--arch", default="resnet18"); ap.add_argument("--imagenet-root", default="./imagenet-val")
    ap.add_argument("--data", default="./data"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--label", default="")
    ap.add_argument("--protect", choices=["none", "selcode"], default="selcode"); ap.add_argument("--cov", type=float, default=0.01)
    ap.add_argument("--protect-scales", choices=["on", "off"], default="on"); ap.add_argument("--scale-bits", choices=["on", "off"], default="on")
    ap.add_argument("--flips", default="10,25,50,100,200,300,500"); ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--attack-batches", type=int, default=2); ap.add_argument("--probe-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sign-cost", type=int, default=0, help="budget cost of one sign-bit flip (t+1 of the sign-plane code); 0 = off, -1 = sign bits unflippable")
    a = ap.parse_args(); S._DS = a.dataset; S.IMAGENET_ROOT = a.imagenet_root
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(a.seed)
    _, vl = S.cifar_loaders(a.data, 128, a.workers)
    model = S.build_model(a.arch).to(dev); S.load_state_flex(model, a.ckpt); params = S.weight_params(model)
    plist = [p for _, p in params]; flips_list = [int(x) for x in a.flips.split(",")]

    # ---- deployed NVFP4 state: codes, block scale bits, and the dequantized weights written into the model ----
    st = []                                                   # per tensor: dict(code, sbits, s_t, nblk, pad)
    with torch.no_grad():
        for p in plist:
            code, eff, s_t, s_b = Q.codes_nvfp4(p.detach())
            sbits = e4m3_encode(s_b)                          # block scale as E4M3 bit patterns
            blocks, pad = Q._blocks(p.detach())
            st.append(dict(code=code, sbits=sbits, s_t=float(s_t), pad=pad, shape=p.shape))
    vals = E2M1.to(dev)

    def dequant(i):
        d = st[i]; sc = e4m3_bits_to_val(d["sbits"]) * d["s_t"]          # (nblk,)
        eff = sc.view(-1, 1).expand(-1, Q.BLOCK).reshape(-1)
        if d["pad"]: eff = eff[:-d["pad"]]
        c = d["code"].reshape(-1); w = torch.where(c >= 8, -1.0, 1.0) * vals[c % 8] * eff
        return w.view(d["shape"])

    def write_all():
        with torch.no_grad():
            for i, p in enumerate(plist): p.copy_(dequant(i))
    write_all(); model.eval()
    clean = S.evaluate(model, vl, dev)
    n_total = sum(p.numel() for p in plist)

    # ---- attack batches (held-out) and protection set ----
    ab = []
    for x, y in vl:
        ab.append((x.to(dev), y.to(dev)))
        if len(ab) >= a.attack_batches: break

    def grads():
        model.zero_grad(set_to_none=True); loss = 0.0
        for x, y in ab:
            loss = loss + F.cross_entropy(model(x), y)
        g = torch.autograd.grad(loss / len(ab), plist); return [gi.detach() for gi in g]

    def attack_loss():
        with torch.no_grad():
            return sum(F.cross_entropy(model(x), y).item() for x, y in ab) / len(ab)

    prot = [torch.zeros(p.numel(), dtype=torch.bool, device=dev) for p in plist]
    if a.protect == "selcode":
        dsum = None; nb = 0
        for x, y in vl:                                       # damage on probe batches, like the paper's bins
            if nb >= a.probe_batches: break
            x, y = x.to(dev), y.to(dev); model.zero_grad(set_to_none=True)
            g = torch.autograd.grad(F.cross_entropy(model(x), y), plist)
            d = torch.cat([(gi.abs() * Q.flip_step_nvfp4(p)).flatten() for gi, p in zip(g, plist)])
            dsum = d if dsum is None else dsum + d; nb += 1
        k = max(1, int(a.cov * n_total)); top = torch.topk(dsum, k).indices
        flat = torch.zeros(n_total, dtype=torch.bool, device=dev); flat[top] = True
        off = 0
        for i, p in enumerate(plist): prot[i] = flat[off:off + p.numel()]; off += p.numel()
    ftab = Q._flip_table(dev)
    print(f"[{a.label or a.ckpt}] NVFP4 clean acc {clean:.2f}  sign-cost {a.sign_cost}  N {n_total}  protect {a.protect} cov {a.cov if a.protect != 'none' else 0}  "
          f"scales protected {a.protect_scales if a.protect != 'none' else 'off'}  scale bits attackable {a.scale_bits}", flush=True)

    # ---- PBS ----
    results = {}; n_done = 0; t0 = time.time(); cur = attack_loss(); n_scale_flips = 0; n_sign_flips = 0
    for target in flips_list:
        while n_done < target:
            g = grads(); cands = []                            # (score, kind, tensor, index, bit)
            for i, p in enumerate(plist):
                d = st[i]; c = d["code"].reshape(-1); gi = g[i].reshape(-1)
                sc = e4m3_bits_to_val(d["sbits"]) * d["s_t"]; eff = sc.view(-1, 1).expand(-1, Q.BLOCK).reshape(-1)
                if d["pad"]: eff = eff[:-d["pad"]]
                cur_v = torch.where(c >= 8, -1.0, 1.0) * vals[c % 8]
                for b in range(4):
                    if b == 3 and a.sign_cost < 0: continue                      # sign plane unflippable
                    nc = c ^ (1 << b); new_v = torch.where(nc >= 8, -1.0, 1.0) * vals[nc % 8]
                    score = gi * (new_v - cur_v) * eff
                    score = score.masked_fill(prot[i], float("-inf"))
                    kk = min(a.topk, score.numel()); v, ix = torch.topk(score, kk)
                    cands += [(float(v[j]), "w", i, int(ix[j]), b) for j in range(kk)]
                if a.scale_bits == "on" and not (a.protect == "selcode" and a.protect_scales == "on"):
                    gw = (gi * dequant(i).reshape(-1)); pad = d["pad"]
                    if pad: gw = torch.cat([gw, gw.new_zeros(pad)])
                    gwb = gw.view(-1, Q.BLOCK).sum(1)                          # sum_block g_j w_j
                    for b in range(8):
                        nb_ = d["sbits"] ^ (1 << b); ratio = e4m3_bits_to_val(nb_) / e4m3_bits_to_val(d["sbits"])
                        score = gwb * (ratio - 1.0)
                        kk = min(a.topk, score.numel()); v, ix = torch.topk(score, kk)
                        cands += [(float(v[j]), "s", i, int(ix[j]), b) for j in range(kk)]
            cands.sort(key=lambda t: -t[0]); cands = cands[:a.topk]
            best = None
            for sc_, kind, i, ix, b in cands:                  # real-loss evaluation of the candidates
                d = st[i]
                if kind == "w":
                    old = int(d["code"].reshape(-1)[ix]); d["code"].reshape(-1)[ix] = old ^ (1 << b)
                else:
                    old = int(d["sbits"][ix]); d["sbits"][ix] = old ^ (1 << b)
                with torch.no_grad(): plist[i].copy_(dequant(i))
                L = attack_loss()
                if best is None or L > best[0]: best = (L, kind, i, ix, b)
                if kind == "w": d["code"].reshape(-1)[ix] = old
                else: d["sbits"][ix] = old
                with torch.no_grad(): plist[i].copy_(dequant(i))
            L, kind, i, ix, b = best; d = st[i]                  # commit
            if kind == "w": d["code"].reshape(-1)[ix] ^= (1 << b)
            else: d["sbits"][ix] ^= (1 << b); n_scale_flips += 1
            with torch.no_grad(): plist[i].copy_(dequant(i))
            cost = a.sign_cost if (kind == "w" and b == 3 and a.sign_cost > 0) else 1
            n_done += cost; cur = L; n_sign_flips += int(kind == "w" and b == 3)
        acc = S.evaluate(model, vl, dev); results[target] = acc
        print(f"[{a.label}] flips {target:4d}  acc {acc:6.2f}  attack loss {cur:.3f}  scale-bit flips {n_scale_flips}  sign flips {n_sign_flips}  budget used {n_done}  {time.time()-t0:.0f}s", flush=True)
    print(f"RESULT {a.label} protect={a.protect} cov={a.cov} signcost={a.sign_cost} clean={clean:.2f} " + " ".join(f"{k}:{v:.2f}" for k, v in results.items()), flush=True)


if __name__ == "__main__":
    main()
