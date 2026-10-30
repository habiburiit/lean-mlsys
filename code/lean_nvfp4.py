#!/usr/bin/env python
"""
lean_nvfp4.py  -  the NVFP4 storage format for LEAN, shared by lean_train_qat_nvfp4.py, lean_hist_nvfp4.py and the
NVFP4 attack.  SEPARATE file; imports nothing from the LEAN/the selective code scripts.

NVFP4 (NVIDIA Blackwell, 2025): every weight is a 4-bit float E2M1 (sign, 2 exponent bits, 1 mantissa bit), values
  code 0..7 -> 0, 0.5, 1, 1.5, 2, 3, 4, 6   (times the sign).
Every block of 16 consecutive weights shares an FP8 E4M3 scale, and the tensor has one FP32 scale so that the block
scales fit the E4M3 range.  Stored cost = 4 + 8/16 = 4.5 bits per weight (plus one FP32 per tensor).

This file gives
  fq_nvfp4(w)          fake-quantized weights with a straight-through gradient (for QAT), block scales recomputed from w
  codes_nvfp4(w)       the 4-bit codes (sign bit | 3-bit magnitude index), per-weight effective scale, block scales
  flip_step_nvfp4(w)   per-weight largest single-bit change of its OWN NVFP4 code, in weight units: max over the 4 bits
                       of |value(code ^ bit) - value(code)| x scale.  This is the Delta_i of the LEAN damage d_i = |g_i| Delta_i
                       in this format (for a sign flip of a nonzero weight it is 2|w|; for the code 0 it is the exponent flip).
Blocks run along the flattened tensor (output-channel major, 16 consecutive input weights); the last block is padded.
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_MID = (E2M1[1:] + E2M1[:-1]) / 2                                   # decision thresholds between grid points
BLOCK = 16
E4M3_MAX = 448.0


def _e4m3(s):
    """round a positive tensor to the FP8 E4M3 grid (4 exponent bits, 3 mantissa bits, max 448, min normal 2^-6)"""
    s = s.clamp(min=2.0 ** -6, max=E4M3_MAX)
    e = torch.floor(torch.log2(s))
    m = torch.round((s / 2.0 ** e - 1.0) * 8.0) / 8.0
    return ((1.0 + m) * 2.0 ** e).clamp(max=E4M3_MAX)


def _blocks(w):
    flat = w.reshape(-1); pad = (-flat.numel()) % BLOCK
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.view(-1, BLOCK), pad


def scales_nvfp4(w):
    """per-weight effective scale (detached), same shape as w, plus the tensor scale and the E4M3 block scales"""
    wd = w.detach(); blocks, pad = _blocks(wd)
    amax = wd.abs().max().clamp(min=1e-12)
    s_t = amax / (6.0 * E4M3_MAX)                                 # FP32 tensor scale so that block scales <= 448
    s_b = _e4m3((blocks.abs().amax(dim=1) / 6.0) / s_t)          # E4M3 block scales
    eff = (s_b * s_t).view(-1, 1).expand(-1, BLOCK).reshape(-1)
    if pad: eff = eff[:-pad]
    return eff.view_as(wd), s_t, s_b


def _mag_index(x_abs_scaled):
    """index 0..7 of the nearest E2M1 magnitude for |w|/scale"""
    return torch.bucketize(x_abs_scaled.clamp(max=6.0), _MID.to(x_abs_scaled.device))


class _FQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, eff):
        idx = _mag_index(w.abs() / eff)
        return torch.sign(w) * E2M1.to(w.device)[idx] * eff
    @staticmethod
    def backward(ctx, g):
        return g, None                                             # straight-through


def fq_nvfp4(w):
    eff, _, _ = scales_nvfp4(w)
    return _FQ.apply(w, eff)


def codes_nvfp4(w):
    """4-bit codes as int (bit3 = sign, bits 2..0 = magnitude index), the effective scale, the block scales"""
    eff, s_t, s_b = scales_nvfp4(w); wd = w.detach()
    idx = _mag_index(wd.abs() / eff)
    code = idx + 8 * (wd < 0).to(idx.dtype)
    return code, eff, s_t, s_b


def _flip_table(device):
    """for each of the 16 codes, the largest |value change| over its 4 single-bit flips, in units of the scale"""
    vals = torch.cat([E2M1, -E2M1]).to(device)                    # code c -> signed value (bit3 = sign)
    t = torch.zeros(16, device=device)
    for c in range(16):
        t[c] = max(abs(float(vals[c ^ (1 << b)] - vals[c])) for b in range(4))
    return t


def flip_step_nvfp4(w):
    """Delta_i: largest single-bit change of each weight's own NVFP4 code, in weight units (detached)"""
    code, eff, _, _ = codes_nvfp4(w)
    return _flip_table(w.device)[code.long()] * eff


def bits_per_weight(w):
    n = w.numel(); nb = (n + BLOCK - 1) // BLOCK
    return (4.0 * n + 8.0 * nb + 32.0) / n
