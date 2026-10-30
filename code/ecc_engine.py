#!/usr/bin/env python3
# =============================================================================
#  xlayer_ecc.py    CROSS-LAYER BCH + STDM parity eviction  (YOUR design)
# -----------------------------------------------------------------------------
#  THE SCHEME (what you described from the start):
#    1. Pick SENSITIVE layers to protect with BCH.  Their weights are the BCH
#       MESSAGE (kept at FULL precision  we never overwrite their own bits).
#    2. Compute each sensitive layer's BCH parity and EVICT it: store those
#       parity bits via STDM (spread-transform dither modulation) inside the
#       large NON-sensitive HOST layers, which have plenty of capacity.
#    3. At read time: the sensitive layer's (error-corrupted) weights + the
#       clean parity recovered from the host layers  ->  BCH-correct the
#       sensitive layer back to exact values.
#
#  WHY THIS WORKS (where earlier attempts failed):
#    * Parity lives in a DIFFERENT layer than the data it protects -> no fight
#      for the same bits, no circular dependency.  STDM carries an INDEPENDENT
#      payload into a host with huge capacity (parity bits << host weights),
#      which is exactly the regime STDM excels in.
#    * The protected layer keeps full precision -> ZERO embedding distortion on
#      the weights that matter most, unlike VAP/E3-CODE which distort them.
#
#  METHODS COMPARED on the accuracy-vs-BER curve:
#    noecc   : no protection (lower bound)
#    vap     : Value-Aware Parity in-place (prior art)
#    xlayer  : PROPOSED  BCH on sensitive layers, parity evicted to hosts (STDM)
#
#  USAGE
#    engine self-test (no GPU):     python xlayer_ecc.py --selftest
#    quick smoke test (GPU):        python xlayer_ecc.py --epochs 1
#    FULL RUN:                      python xlayer_ecc.py --arch resnet18 --dataset cifar10 --epochs 200
#    re-run sweep on checkpoint:    python xlayer_ecc.py --skip-train --ckpt resnet18_cifar10.pt
#
#  DEPS: torch torchvision numpy galois matplotlib
# =============================================================================
import argparse, json, copy, os, time
import numpy as np
import galois


# --- module-level worker for multiprocessed BCH decode (each proc builds its own
#     galois.BCH; avoids pickling the object across processes) ------------------
_WORKER_BCH = None
def _decode_chunk(args):
    global _WORKER_BCH
    S, k, full_chunk = args
    if _WORKER_BCH is None:
        _WORKER_BCH = galois.BCH(63, 39)
    dec, _ne = _WORKER_BCH.decode(full_chunk, errors=True)
    return np.array(dec, np.uint8)[:, S:]



# =============================================================================
#  1.  ENGINE: BCH message/parity + STDM host eviction
# =============================================================================


class XLayerECC:
    def __init__(self, shorten=7, delta=0.02, host_key=12345,
                 strong_t=4, light_t=2, bch_m=6):
        import bchlib
        self._bchlib = bchlib
        self.bch_m = bch_m
        self.strong_t = strong_t
        self.light_t = light_t
        self.bch = bchlib.BCH(t=strong_t, m=bch_m)            # strong code (sensitive layers)
        self.bch_light = bchlib.BCH(t=light_t, m=bch_m)       # light code (insensitive layers)
        self.t = self.bch.t
        # data bytes per codeword = floor((n - ecc_bits)/8), whole bytes only
        self.KB = (self.bch.n - self.bch.ecc_bits) // 8       # strong-code data bytes
        self.KB_light = (self.bch_light.n - self.bch_light.ecc_bits) // 8
        assert self.KB >= 1, f"strong_t={strong_t},m={bch_m} leaves <1 data byte; lower t or raise m"
        assert self.KB_light >= 1, f"light_t={light_t},m={bch_m} leaves <1 data byte; lower t or raise m"
        self.ecc_bytes = self.bch.ecc_bytes
        self.ecc_bytes_light = self.bch_light.ecc_bytes
        self.nk = self.ecc_bytes * 8
        self.nk_light = self.ecc_bytes_light * 8
        self.n = self.KB*8 + self.nk
        self.k = self.KB * 8
        self.S = shorten
        self.delta = delta
        self.host_key = host_key
        self.n_proc = 1

    def _encode_ecc_light(self, data_bytes):
        N = data_bytes.shape[0]
        ecc = np.zeros((N, self.ecc_bytes_light), np.uint8)
        for i in range(N):
            ecc[i] = np.frombuffer(bytes(self.bch_light.encode(bytearray(data_bytes[i].tobytes()))), np.uint8)
        return np.unpackbits(ecc, axis=1)

    def layer_parity_light(self, int8_layer):
        flat = int8_layer.reshape(-1).astype(np.uint8)
        pad = (-len(flat)) % self.KB_light
        f = np.concatenate([flat, np.zeros(pad, np.uint8)]) if pad else flat
        msgs = f.reshape(-1, self.KB_light)
        return self._encode_ecc_light(msgs).reshape(-1), len(flat)

    def correct_layer_light(self, int8_corrupt, parity_clean, n_orig):
        flat = int8_corrupt.reshape(-1).astype(np.uint8)
        pad = (-len(flat)) % self.KB_light
        f = np.concatenate([flat, np.zeros(pad, np.uint8)]) if pad else flat
        msgs = f.reshape(-1, self.KB_light)
        eb = np.packbits(parity_clean.reshape(-1, self.nk_light), axis=1)
        out = msgs.copy()
        for i in range(msgs.shape[0]):
            d = bytearray(msgs[i].tobytes()); e = bytearray(eb[i].tobytes())
            self.bch_light.decode(d, e); self.bch_light.correct(d, e)
            out[i] = np.frombuffer(bytes(d), np.uint8)
        return out.reshape(-1)[:n_orig]

    # --- per-codeword ecc via bchlib (vectorized over a layer) ---------------
    def _encode_ecc(self, data_bytes):
        """data_bytes:(N,4) uint8 -> ecc bits:(N,24)."""
        N = data_bytes.shape[0]
        ecc = np.zeros((N, self.ecc_bytes), np.uint8)
        for i in range(N):
            ecc[i] = np.frombuffer(bytes(self.bch.encode(bytearray(data_bytes[i].tobytes()))), np.uint8)
        return np.unpackbits(ecc, axis=1)                     # (N,24)

    # ===== BIT-PLANE-SELECTIVE PROTECTION (the winning scheme) ===============
    #  Protect only the top-K bits of each int8 weight (the bits whose flips cause
    #  large value changes). Pack K-bit symbols into 32-bit BCH messages, t=4.
    #  Returns ecc bits for STDM eviction. K and grouping are fixed by self.Kbits.
    def planes_encode(self, int8_layer, K, W=8):
        """Protect top-K bits of each W-bit weight. Packs G=(KB*8)//K weights' K-bit
        symbols into each KB-byte BCH message. W is the weight bit-width (8/6/4)."""
        KB = self.KB
        Mbits = KB * 8                                        # data bits per codeword (byte packing)
        G = Mbits // K                                        # weights per codeword
        w = int8_layer.reshape(-1).astype(np.uint8)
        pad = (-len(w)) % G
        wf = np.concatenate([w, np.zeros(pad, np.uint8)]) if pad else w
        topk = (wf >> (W - K)) & ((1 << K) - 1)               # top-K of the W stored bits
        # build message BIT array: G symbols x K bits -> Mbits, then to KB bytes
        sym = topk.reshape(-1, G)                             # (Ncw, G) values 0..2^K-1
        symbits = ((sym[:, :, None] >> np.arange(K)) & 1).astype(np.uint8)  # (Ncw,G,K) LSB-first
        msgbits = symbits.reshape(sym.shape[0], G*K)          # (Ncw, G*K)
        if G*K < Mbits:                                       # pad to whole KB bytes
            msgbits = np.concatenate([msgbits, np.zeros((sym.shape[0], Mbits-G*K), np.uint8)], axis=1)
        msg_bytes = np.packbits(msgbits.reshape(-1, KB, 8), axis=2).reshape(-1, KB)
        ecc = np.zeros((msg_bytes.shape[0], self.ecc_bytes), np.uint8)
        for i in range(msg_bytes.shape[0]):
            ecc[i] = np.frombuffer(bytes(self.bch.encode(bytearray(msg_bytes[i].tobytes()))), np.uint8)
        return np.unpackbits(ecc, axis=1).reshape(-1), len(w), G

    def planes_decode(self, int8_corrupt, ecc_bits, K, n_orig, G, W=8):
        """Recover top-K bits via BCH; splice back into W-bit weights (low bits untouched)."""
        KB = self.KB; Mbits = KB*8
        w = int8_corrupt.reshape(-1).astype(np.uint8)
        pad = (-len(w)) % G
        wf = np.concatenate([w, np.zeros(pad, np.uint8)]) if pad else w
        topk = (wf >> (W - K)) & ((1 << K) - 1)
        sym = topk.reshape(-1, G)
        symbits = ((sym[:, :, None] >> np.arange(K)) & 1).astype(np.uint8)
        msgbits = symbits.reshape(sym.shape[0], G*K)
        if G*K < Mbits:
            msgbits = np.concatenate([msgbits, np.zeros((sym.shape[0], Mbits-G*K), np.uint8)], axis=1)
        msg_bytes = np.packbits(msgbits.reshape(-1, KB, 8), axis=2).reshape(-1, KB)
        eccb = np.packbits(ecc_bits.reshape(-1, self.nk), axis=1)
        Ncw = msg_bytes.shape[0]
        for i in range(Ncw):
            d = bytearray(msg_bytes[i].tobytes()); e = bytearray(eccb[i].tobytes())
            self.bch.decode(d, e); self.bch.correct(d, e)
            msg_bytes[i] = np.frombuffer(bytes(d), np.uint8)
        # unpack corrected message bits -> symbols -> splice into weight top-K bits
        fixedbits = np.unpackbits(msg_bytes, axis=1)[:, :G*K].reshape(Ncw, G, K)
        symvals = (fixedbits * (1 << np.arange(K))).sum(axis=2).astype(np.uint8)  # (Ncw,G)
        out = wf.copy()
        flat_sym = symvals.reshape(-1)
        lowmask = (1 << (W-K)) - 1
        out = ((out & lowmask) | (flat_sym << (W-K))).astype(np.uint8)
        return out[:n_orig].astype(np.int8).astype(np.uint8)


    def parity(self, mbits):
        """mbits:(N,32) message bits -> (N,24) parity bits (for STDM eviction)."""
        data_bytes = np.packbits(mbits.reshape(-1, self.KB, 8), axis=2).reshape(-1, self.KB)
        return self._encode_ecc(data_bytes)

    def correct_with_ecc(self, data_bytes_corrupt, ecc_bits_clean):
        """data_bytes_corrupt:(N,4), ecc_bits_clean:(N,24) -> corrected data_bytes:(N,4)."""
        N = data_bytes_corrupt.shape[0]
        ecc_bytes = np.packbits(ecc_bits_clean.reshape(-1, self.nk), axis=1)  # (N,3)
        out = data_bytes_corrupt.copy()
        for i in range(N):
            d = bytearray(out[i].tobytes()); e = bytearray(ecc_bytes[i].tobytes())
            self.bch.decode(d, e); self.bch.correct(d, e)
            out[i] = np.frombuffer(bytes(d), np.uint8)
        return out

    # --- STDM eviction into a host float layer -------------------------------
    def stdm_store(self, host_f32, payload_bits, key, delta=None):
        delta = delta or self.delta
        w = host_f32.flatten().astype(np.float64).copy()
        nb = len(payload_bits); ch = len(w) // nb
        assert ch >= 1, "host layer too small to hold this payload"
        r = np.random.default_rng(key)
        for b in range(nb):
            sl = slice(b*ch, (b+1)*ch)
            u = r.standard_normal(ch); u /= np.linalg.norm(u)
            s = u @ w[sl]; d = (delta/2)*int(payload_bits[b])
            q = np.round((s - d)/delta)*delta + d
            w[sl] = w[sl] + (q - s)*u
        return w.reshape(host_f32.shape).astype(np.float32)

    def stdm_load(self, host_f32, nb, key, delta=None):
        delta = delta or self.delta
        w = host_f32.flatten().astype(np.float64)
        ch = len(w) // nb; out = np.zeros(nb, np.uint8)
        r = np.random.default_rng(key)
        for b in range(nb):
            sl = slice(b*ch, (b+1)*ch)
            u = r.standard_normal(ch); u /= np.linalg.norm(u)
            s = u @ w[sl]
            c0 = np.round(s/delta)*delta; c1 = np.round((s-delta/2)/delta)*delta + delta/2
            out[b] = 0 if abs(s-c0) <= abs(s-c1) else 1
        return out

    # --- slice variants: embed payload into host weights [start:start+nb] ----
    #  one host weight carries one parity bit; VECTORIZED (no per-bit Python loop).
    def stdm_store_slice(self, host_f32, payload_bits, start, key, delta=None):
        delta = delta or self.delta
        shape = host_f32.shape
        w = host_f32.flatten().astype(np.float64)
        nb = len(payload_bits)
        idx = np.arange(start, start + nb)
        s = w[idx]
        bits = payload_bits.astype(np.float64)
        d = (delta/2.0) * bits                                # dither per bit
        q = np.round((s - d)/delta)*delta + d                 # QIM quantize, all at once
        w[idx] = q
        return w.reshape(shape).astype(np.float32)

    def stdm_load_slice(self, host_f32, start, nb, key, delta=None):
        delta = delta or self.delta
        w = host_f32.flatten().astype(np.float64)
        idx = np.arange(start, start + nb)
        s = w[idx]
        c0 = np.round(s/delta)*delta
        c1 = np.round((s - delta/2)/delta)*delta + delta/2
        return (np.abs(s - c0) > np.abs(s - c1)).astype(np.uint8)   # bit=1 if closer to c1

    # --- MULTI-BIT STDM: pack B parity bits into EACH host weight ------------
    #  Each host weight holds B bits by encoding them as an integer level 0..2^B-1
    #  via uniform quantization with step delta. Insensitive layers tolerate the
    #  resulting small perturbation, so they become dense parity storage.
    def stdm_store_multibit(self, host_f32, payload_bits, B, delta=None):
        """Pack payload_bits (1-D) into host weights, B bits per weight.
        Returns modified host (float32). Capacity = host.size * B bits."""
        delta = delta or self.delta
        shape = host_f32.shape
        w = host_f32.flatten().astype(np.float64)
        payload_bits = np.asarray(payload_bits).reshape(-1).astype(np.uint8)
        nb = len(payload_bits)
        nslots = (nb + B - 1) // B
        assert nslots <= w.size, "host too small even at B bits/weight"
        # group bits into B-bit integer levels
        pad = nslots*B - nb
        pb = np.concatenate([payload_bits, np.zeros(pad, np.uint8)]).reshape(nslots, B)
        levels = (pb * (1 << np.arange(B))).sum(axis=1)        # int 0..2^B-1 per slot
        s = w[:nslots]
        # QIM: quantize weight to a lattice whose offset encodes the level
        # base grid step = delta*2^B; within a cell, sublattice offset = level*delta
        big = delta * (2**B)
        q = np.round((s - levels*delta)/big)*big + levels*delta
        w[:nslots] = q
        return w.reshape(shape).astype(np.float32), nslots

    def stdm_load_multibit(self, host_f32, nb, B, delta=None):
        delta = delta or self.delta
        w = host_f32.flatten().astype(np.float64)
        nslots = (nb + B - 1) // B
        big = delta * (2**B)
        s = w[:nslots]
        # recover level = round((s mod big)/delta)
        frac = s - np.round(s/big)*big                          # in (-big/2, big/2]
        lvl = np.round(frac/delta).astype(np.int64) % (2**B)
        bits = ((lvl[:, None] >> np.arange(B)) & 1).astype(np.uint8).reshape(-1)
        return bits[:nb]

    # ====================================================================
    #  TRUE STDM = QIM on a spread-transform projection, seeded by host_key
    #  (Chen & Wornell 2001). Each B-bit symbol is carried by the PROJECTION
    #  of an L-weight block onto a pseudo-random unit vector v (from the key),
    #  with a pseudo-random dither d. Embedding nudges the projection to the
    #  lattice and spreads the change back across all L weights along v, so no
    #  single weight moves much. The key is REQUIRED to read or alter parity.
    # ====================================================================
    def _stdm_vectors(self, nslots, L, B, key=None):
        """Deterministic pseudo-random projection vectors + dithers from a key."""
        rng = np.random.default_rng(self.host_key if key is None else key)
        V = rng.standard_normal((nslots, L))
        V /= np.linalg.norm(V, axis=1, keepdims=True)           # unit projection vectors
        d = rng.random(nslots) * (self.delta * (2**B))          # per-block dither in [0,big)
        return V, d

    def stdm_store_projected(self, host_f32, payload_bits, B, L=4, delta=None):
        """Seeded STDM embed. Carries B bits per L-weight block via the projection
        onto a key-seeded unit vector v, with key-seeded dither d. Capacity =
        (host.size // L) blocks * B bits."""
        delta = delta or self.delta
        shape = host_f32.shape
        w = host_f32.flatten().astype(np.float64)
        payload_bits = np.asarray(payload_bits).reshape(-1).astype(np.uint8)
        nb = len(payload_bits)
        nslots = (nb + B - 1)//B
        need = nslots * L
        assert need <= w.size, "host too small for projected STDM (raise host_bits or lower L)"
        pad = nslots*B - nb
        pb = np.concatenate([payload_bits, np.zeros(pad, np.uint8)]).reshape(nslots, B)
        levels = (pb * (1 << np.arange(B))).sum(axis=1)          # 0..2^B-1
        big = delta * (2**B)
        blocks = w[:need].reshape(nslots, L)
        V, d = self._stdm_vectors(nslots, L, B)
        proj = np.einsum('ij,ij->i', blocks, V)                 # scalar projection per block
        target = np.round((proj - levels*delta - d)/big)*big + levels*delta + d   # QIM w/ dither
        delta_proj = target - proj                              # how much the projection must move
        blocks = blocks + delta_proj[:, None] * V               # spread change back along v
        w[:need] = blocks.reshape(-1)
        return w.reshape(shape).astype(np.float32), nslots

    def stdm_load_projected(self, host_f32, nb, B, L=4, delta=None):
        """Seeded STDM read. Needs the SAME host_key to regenerate v and d."""
        delta = delta or self.delta
        w = host_f32.flatten().astype(np.float64)
        nslots = (nb + B - 1)//B
        need = nslots * L
        big = delta * (2**B)
        blocks = w[:need].reshape(nslots, L)
        rkey = getattr(self, 'recover_key', None)            # wrong-key attack if set
        V, d = self._stdm_vectors(nslots, L, B, key=rkey)
        proj = np.einsum('ij,ij->i', blocks, V)
        frac = (proj - d) - np.round((proj - d)/big)*big
        lvl = np.round(frac/delta).astype(np.int64) % (2**B)
        bits = ((lvl[:, None] >> np.arange(B)) & 1).astype(np.uint8).reshape(-1)
        return bits[:nb]

    # --- dispatcher: pick embedding by self.embed_mode ('qim' or 'stdm') -----
    def embed_store(self, host_f32, payload_bits, B):
        if getattr(self, 'embed_mode', 'qim') == 'stdm':
            return self.stdm_store_projected(host_f32, payload_bits, B,
                                             L=getattr(self, 'stdm_L', 4))
        return self.stdm_store_multibit(host_f32, payload_bits, B)

    def embed_load(self, host_f32, nb, B):
        if getattr(self, 'embed_mode', 'qim') == 'stdm':
            return self.stdm_load_projected(host_f32, nb, B,
                                            L=getattr(self, 'stdm_L', 4))
        return self.stdm_load_multibit(host_f32, nb, B)




    # --- compute a sensitive layer's parity payload --------------------------
    def layer_parity(self, int8_layer):
        flat = int8_layer.reshape(-1).astype(np.uint8)
        pad = (-len(flat)) % self.KB
        f = np.concatenate([flat, np.zeros(pad, np.uint8)]) if pad else flat
        msgs = f.reshape(-1, self.KB)                          # (Ncw, 4) data bytes
        return self._encode_ecc(msgs).reshape(-1), len(flat)   # (payload_bits,), n_orig

    def correct_layer(self, int8_layer_corrupt, parity_clean, n_orig):
        flat = int8_layer_corrupt.reshape(-1).astype(np.uint8)
        pad = (-len(flat)) % self.KB
        f = np.concatenate([flat, np.zeros(pad, np.uint8)]) if pad else flat
        msgs = f.reshape(-1, self.KB)                          # (Ncw, 4) corrupted data bytes
        fixed = self.correct_with_ecc(msgs, parity_clean.reshape(-1, self.nk))
        return fixed.reshape(-1)[:n_orig]

    # --- VAP baseline (bchlib): store ecc adjacent to data; both take channel BER,
    #     then BCH-correct.  Models in-place parity protection.
    def vap_protect_recover(self, int8_w, ber, rng):
        flat = int8_w.reshape(-1).astype(np.uint8)
        pad = (-len(flat)) % self.KB
        f = np.concatenate([flat, np.zeros(pad, np.uint8)]) if pad else flat
        msgs = f.reshape(-1, self.KB)                          # (N,4) data
        ecc_bits = self._encode_ecc(msgs)                      # (N,24) clean parity
        ecc_bytes = np.packbits(ecc_bits, axis=1)              # (N,3)
        # corrupt BOTH data and ecc (in-place parity is also exposed to errors)
        dcor = self._corrupt_arr(msgs, ber, rng)
        ecor = self._corrupt_arr(ecc_bytes, ber, rng)
        out = dcor.copy()
        for i in range(msgs.shape[0]):
            d = bytearray(dcor[i].tobytes()); e = bytearray(ecor[i].tobytes())
            self.bch.decode(d, e); self.bch.correct(d, e)
            out[i] = np.frombuffer(bytes(d), np.uint8)
        return out.reshape(-1)[:len(flat)]

    @staticmethod
    def _corrupt_arr(arr, ber, rng):
        bits = np.unpackbits(arr.reshape(-1, 1), axis=1)
        bits ^= (rng.random(bits.shape) < ber).astype(np.uint8)
        return np.packbits(bits, axis=1).reshape(arr.shape)

    # --- self-test -----------------------------------------------------------
    def selftest(self, seed=0):
        rng = np.random.default_rng(seed)
        small = rng.integers(0, 256, size=64*64, dtype=np.uint8)        # protected layer
        host = rng.standard_normal((512, 512)).astype(np.float32)*0.05  # host layer
        payload, n0 = self.layer_parity(small)
        host_w = self.stdm_store(host, payload, key=self.host_key)
        print(f"[selftest] protected {len(small)} wts -> {len(payload)} parity bits, "
              f"host cap {host.size} (fits: {len(payload)<=host.size})")
        print(f"           host distortion mean|?w|={np.mean(np.abs(host-host_w)):.5f}")
        print("  BER on protected layer | xlayer byte-recovery | raw no-ECC")
        for ber in [0.005, 0.01, 0.02, 0.04, 0.06]:
            corrupt = self._corrupt_bytes(small, ber, rng)
            pclean = self.stdm_load(host_w, len(payload), key=self.host_key)
            fixed = self.correct_layer(corrupt, pclean, n0)
            print(f"    {ber:5.1%}                {np.mean(fixed==small):.4f}            "
                  f"{np.mean(corrupt==small):.4f}")

    @staticmethod
    def _corrupt_bytes(b, ber, rng):
        flat = b.reshape(-1)
        bits = np.unpackbits(flat.reshape(-1, 1), axis=1)
        bits ^= (rng.random(bits.shape) < ber).astype(np.uint8)
        return np.packbits(bits, axis=1).reshape(-1)[:flat.size]