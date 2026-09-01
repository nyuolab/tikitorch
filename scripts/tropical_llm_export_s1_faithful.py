"""tropical_llm_export_s1_faithful.py — export the KERNEL-FAITHFUL tropical LLM
(results/lm_shakespeare/tropical_llm_d64_faithful.pt, trained WITH MAD-LN + pow2 gamma + dyadic weights)
into the int32 monolithic buffer the forge_train_step_s1 FORWARD kernel reads (mode 0,
-DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1 -DS1_N_LAYERS=2 at D=64/V=64).

This is the std-LN export adapted for the FAITHFUL model: the LN keys are MADLN .g/.b (not std-LN
.weight/.bias) and GCOMP defaults to 1.0 (the model trained with the kernel's mean-abs-deviation norm,
so there is NO std/MAD compensation to apply — the std-LN export's GCOMP~0.7 fudge is GONE here).

The buffer regions + encodings (see forge_s1_config.hpp / forge_train_step_s1.cpp):
  EMBED   [V*D]   int16 values  (S1_OFF_EMBED)        : embed[v][d] = round(emb_float * SA)
  POS     [T*D]   int16 values  (S1_OFF_POS)          : pos[t][d]   = round(pos_float * SA)
  Wq/Wk/Wv/Wo/W_up/W_down  int8 dyadic CODES          : code = s1_encode(dyad(w_float * SW))
  LN1/LN2/LN_F gamma  int8 dyadic CODES               : code = s1_encode(dyad(gamma_float * GCOMP * SGAM))
  LN1/LN2/LN_F beta   int16 values                    : beta = round(beta_float * SGAM)   (LN-output scale)
  W_HEAD  int8 dyadic CODES (HEAD_MODE=1)             : code = s1_encode(dyad(head_w_float * SH))
  CE_LUT / ATTN_LUT   int32 Q16 exp2 (FT_CE_EXP2_Q16)
  TOKENS / TARGETS    int32 token ids (filled per-window by the g++ harness, left 0 here)

FIXED-POINT SCALES — the central deploy decision. The kernel runs entirely in int16 activations and the
weight codes can only hold NON-NEGATIVE dyadic exponents (5-bit field), so the trained float weights
(exp -19..+1) must be scaled UP by SW to be representable. The kernel LayerNorm normalizes by MEAN-ABS-
DEVIATION (not std) with a 2-bit mantissa, so (a) the LN gamma is pre-multiplied by GCOMP (~0.7-0.8, the
std/MAD compensation: MAD≈std/1.25, so MAD-LN output is ~1.4x too large) and (b) the LN output is carried
at scale SGAM via gamma's power-of-2 shift. Residual consistency wants SGAM·SW² ≈ SA (two cascaded GEMMs
between an LN and the residual add). These scales are chosen by the sweep in this module / the g++ harness;
the encoder math (dyad / s1_encode) is BIT-IDENTICAL to the kernel (forge_train_step_s1.cpp dyad/wdecode,
forge_train_step_s1_tb.cpp ref_s1_encode, forge_train_step_s1_f2.c h_dyad).

HONEST DEPLOY VERDICT (FAITHFUL model, measured 2026-06-28 on the REAL kernel forward, 63 val windows,
best fixed-point SA=64 SW=2 SGAM=32 SH=8 GCOMP=1.0 logit_temp~448):
  deployable on-kernel bpc = 4.71  vs  bigram 3.47  vs  this-model float HARD 2.52.  It does NOT beat bigram.
The MAD-LN-trained "faithful" model FIXED mismatch (1) of the std-LN export (the LN no longer mismatches —
that was the std-LN 4.05's biggest piece) but it STILL deploys above bigram, because TWO kernel quantizers
remain that this model did NOT train through. Decomposed by float ablation of THIS exact checkpoint (exact
scales, isolating each effect; see /tmp/emu_decomp.py):
  baseline (the model: round-log2 dyad weights, full-prec int16 acts)   bpc 2.519  acc 0.457
  + kernel dyad on weights (MSB + bump-at-1.5x2^k, NOT train's round-log2)  2.766  (+0.25)
  + activation pow2-quant q(x) in EVERY GEMM (the kernel does q(x)·q(W))    3.572  (+0.81 — DOMINANT)
  + both kernel-dyad weights AND activation pow2-quant                       3.408  (float floor ~bigram)
  + int rounding (round-to-0 kills sub-1 acts) + LN 2-bit mantissa + scale
    discretization (SW must be ~32-64 for weight fidelity; weights down to
    ~0.06 vanish at the low SW the residual-consistency SGAM·SW=SA wants) ->  ~4.71 (MEASURED kernel)
ROOT CAUSE: the faithful model's training quantizer (tropical_llm_kernelfaithful.py) snaps WEIGHTS to pow2
via dyad()=round(log2) and quantizes activations only to INT (fq = round(x*256)/256) — it does NOT snap
activations to pow2 and does NOT use the kernel's MSB-bump dyad. The kernel's GEMM is q(x)·q(W) with the
MSB-bump dyad on BOTH operands. So "faithful" was faithful to MAD-LN + dyadic-weights but NOT to the
kernel's activation-pow2-quant — the dominant +0.81 gap. The scale space was swept flat at ~4.7 (a real
sweep of SA/SW/SGAM/SH/GCOMP/temp; high SW for weight fidelity, low SA to avoid residual sat16) — it is NOT
a fixed-point miss. The correct path to sub-bigram deploy is to TRAIN THROUGH the kernel's exact quantizer
(pow2 activations + MSB-bump dyad + MAD-LN), as s1_charlm.cpp does.

Usage:
  python scripts/tropical_llm_export_s1_faithful.py --out /tmp/s1_faithful.npy --SA 64 --SW 2 --SGAM 32 --SH 8 --GCOMP 1.0
Writes an int32 numpy array of length S1_BUFLEN (=709264 at D64/L2/V64/POS=1) plus a sidecar .json with the
dims, offsets and scales so the harness can self-describe.
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# ---- s1 dims (forge_s1_config.hpp at D=64/L2/V64; overridden by --offsets for the kernel-v2 deploy) ----
D, H, DH, FFN, T, V, NL = 64, 2, 32, 256, 64, 64, 2
VREAL = 39
LUT_N = 256
KV2 = False        # kernel-v2: pack 4-term CSD byte-codes into the int32 weight slot
OFFSETS = None     # if set (from --offsets), use these exact kernel offsets instead of compute_offsets()


def enc_w(W_float, scale):
    """weight/gamma/head encoder: 4-term CSD packed int32 (kernel-v2) or single dyadic byte (legacy)."""
    return enc_codes_csd(W_float, scale, WCSD_K) if KV2 else enc_codes(W_float, scale)


def al16(x):
    return ((x + 15) // 16) * 16


def compute_offsets():
    o = {}
    o['EMBED'] = 0
    o['WQ'] = o['EMBED'] + al16(V * D)
    o['WK'] = o['WQ'] + al16(NL * D * D)
    o['WV'] = o['WK'] + al16(NL * D * D)
    o['WO'] = o['WV'] + al16(NL * D * D)
    o['W_UP'] = o['WO'] + al16(NL * D * D)
    o['W_DOWN'] = o['W_UP'] + al16(NL * FFN * D)
    o['LN1_G'] = o['W_DOWN'] + al16(NL * D * FFN)
    o['LN1_B'] = o['LN1_G'] + al16(NL * D)
    o['LN2_G'] = o['LN1_B'] + al16(NL * D)
    o['LN2_B'] = o['LN2_G'] + al16(NL * D)
    # LN2_B is a [N_L][D] layer-major beta region (N_L*D words). LNF_G MUST clear the FULL N_L*D extent
    # (matches the fixed forge_s1_config.hpp S1_OFF_LNF_G = S1_OFF_LN2_B + S1_AL16(S1_N_LAYERS*S1_D)).
    o['LNF_G'] = o['LN2_B'] + al16(NL * D)
    o['LNF_B'] = o['LNF_G'] + al16(D)
    o['W_HEAD'] = o['LNF_B'] + al16(D)
    o['CE_LUT'] = o['W_HEAD'] + al16(V * D)
    o['ATTN_LUT'] = o['CE_LUT'] + al16(LUT_N)
    o['TOKENS'] = o['ATTN_LUT'] + al16(LUT_N)
    o['TARGETS'] = o['TOKENS'] + al16(T)
    return o


# Layout-derived constants — fetched from the COMPILED config (forge_s1_config.hpp at the deploy defines)
# so the export NEVER drifts from the kernel's offset map. See _config_consts() below.
BUFLEN = 709264   # D64/L2/V64/POS=1 with the LN2_B/LNF_G overlap FIX (was 709200 with the bug)
OFF_POS = 688784
OFF_LOGITS = 194688


# ---- encoders, BIT-IDENTICAL to the kernel dyad()/wdecode() + ref_s1_encode (TB) / h_dyad (host) ----
def dyad_exp_nz(v):
    """forge_fwd_wide dyad(): magnitude masked to int16, priority-MSB + bump-below-MSB."""
    a = abs(int(v)) & 0xFFFF
    nz = 1 if a != 0 else 0
    msb = 0
    for b in range(15, 0, -1):
        if (a >> b) & 1:
            msb = b
            break
    if msb >= 1 and ((a >> (msb - 1)) & 1):
        msb += 1
    sgn = 1 if v < 0 else 0
    return sgn, msb, nz


def s1_encode_value_to_code(wi):
    """code byte = sign|nz|exp from an integer value wi (== s1_encode(dyad(wi)))."""
    sgn, e, nz = dyad_exp_nz(wi)
    ee = min(e, 31)
    c = ee & 0x1F
    if nz:
        c |= (1 << 5)
    if sgn:
        c |= (1 << 7)
    return int(np.int8(np.uint8(c)))


def i16(v):
    return int(np.clip(round(float(v)), -32768, 32767))


def enc_codes(W_float, scale):
    """encode a float weight matrix to dyadic byte codes at the given scale (code = dyad(round(w*scale)))."""
    flat = np.asarray(W_float, dtype=np.float64).reshape(-1)
    return np.array([s1_encode_value_to_code(round(v * scale)) for v in flat], dtype=np.int32)


WCSD_K = 4   # weight CSD terms (kernel-v2); set from --offsets/--wcsd_k

def csd_terms_packed(value, K):
    """greedy K-term CSD of integer 'value' — exponent = round(log2(|resid|)) PER TERM, EXACTLY matching
    scripts/tropical_llm_kernelsim.py dyad_scaled (torch.round(log2), NOT the MSB-bump dyad_exp_nz which
    differs for ~23->5 and cost +0.25 bpc). Pack K byte-codes (sign|nz|exp) LE into one int32 (kernel wmac4)."""
    resid = int(round(value)); packed = 0
    for k in range(K):
        if resid == 0:
            b = 0                              # nz=0 term
        else:
            a = abs(resid)
            ee = int(round(math.log2(a)))      # round-to-nearest-pow2 (== sim)
            if ee < 0: ee = 0
            if ee > 31: ee = 31
            sgn = 1 if resid < 0 else 0
            term = (-(1 << ee)) if sgn else (1 << ee)
            b = (ee & 0x1F) | (1 << 5) | (sgn << 7)
            resid -= term
        packed |= (int(b) & 0xFF) << (8 * k)
    return np.int32(np.uint32(packed))

def enc_codes_csd(W_float, scale, K=4):
    flat = np.asarray(W_float, dtype=np.float64).reshape(-1)
    return np.array([csd_terms_packed(round(v * scale), K) for v in flat], dtype=np.int32)


def enc_vals(W_float, scale):
    flat = np.asarray(W_float, dtype=np.float64).reshape(-1)
    return np.array([i16(v * scale) for v in flat], dtype=np.int32)


def build_buffer(ckpt_path, SA, SW, SGAM, SH, GCOMP):
    ck = torch.load(ckpt_path, map_location='cpu')
    sd = ck['model']
    o = OFFSETS if OFFSETS is not None else compute_offsets()
    buf = np.zeros(BUFLEN, dtype=np.int32)

    # EMBED [V*D] int16 (rows VREAL..V-1 left 0; padded vocab never a target)
    emb = sd['emb.weight'].numpy()
    embed = np.zeros((V, D), dtype=np.int32)
    for v in range(VREAL):
        embed[v] = enc_vals(emb[v], SA)
    buf[o['EMBED']:o['EMBED'] + V * D] = embed.reshape(-1)

    # POS [T*D] int16
    pos = sd['pos'].numpy()[0]
    buf[OFF_POS:OFF_POS + T * D] = enc_vals(pos, SA)

    # GEMM weight codes, layer-major, ROW-MAJOR [O][I]. SW may be a single int OR a per-matrix dict
    # {q,k,v,o,up,dn: scale} — the tiny residual-branch matrices (v/o/dn) want a larger scale than q/k/up.
    proj_map = {'WQ': 'q', 'WK': 'k', 'WV': 'v', 'WO': 'o', 'W_UP': 'up', 'W_DOWN': 'dn'}
    sw_of = (lambda nm: SW) if not isinstance(SW, dict) else (lambda nm: SW[nm])
    for region, nm in proj_map.items():
        chunks = []
        for L in range(NL):
            w = sd[f'blocks.{L}.{nm}.w'].numpy()  # [O][I]
            chunks.append(enc_w(w, sw_of(nm)))
        codes = np.concatenate(chunks)
        buf[o[region]:o[region] + codes.size] = codes

    # LN gammas (codes, gamma*GCOMP*SGAM) + betas (int16 values at LN-output scale SGAM), layer-major.
    # FAITHFUL model uses MADLN with keys .g / .b (NOT std-LN .weight/.bias). GCOMP=1.0 here: the model
    # TRAINED with the kernel's mean-abs-deviation norm + pow2 gamma (dyad(g)), so there is NO std/MAD
    # compensation to apply — the gamma already IS the MAD-LN gamma. (For the std-LN export GCOMP~0.7-0.8
    # fudged the std/MAD ratio; that mismatch is gone here.)
    for region_g, region_b, ln in [('LN1_G', 'LN1_B', 'ln1'), ('LN2_G', 'LN2_B', 'ln2')]:
        gc, bc = [], []
        for L in range(NL):
            g = sd[f'blocks.{L}.{ln}.g'].numpy()
            b = sd[f'blocks.{L}.{ln}.b'].numpy()
            gc.append(enc_w(g * GCOMP, SGAM))
            bc.append(enc_vals(b, SGAM))
        buf[o[region_g]:o[region_g] + NL * D] = np.concatenate(gc)
        buf[o[region_b]:o[region_b] + NL * D] = np.concatenate(bc)

    # LN_F (single) — MADLN keys lnf.g / lnf.b
    g = sd['lnf.g'].numpy()
    b = sd['lnf.b'].numpy()
    buf[o['LNF_G']:o['LNF_G'] + D] = enc_w(g * GCOMP, SGAM)
    buf[o['LNF_B']:o['LNF_B'] + D] = enc_vals(b, SGAM)

    # W_HEAD codes [V][D] (HEAD_MODE=1 dyadic-GEMM head; padded rows 0)
    hw = np.zeros((V, D), dtype=np.float64)
    hw[:VREAL] = sd['head.w'].numpy()
    buf[o['W_HEAD']:o['W_HEAD'] + V * D] = enc_w(hw, SH)

    # CE / ATTN LUTs (Q16 exp2 — FT_CE_EXP2_Q16). Parse from forge_ce_lut.hpp.
    lut = load_ce_lut()
    buf[o['CE_LUT']:o['CE_LUT'] + LUT_N] = lut
    buf[o['ATTN_LUT']:o['ATTN_LUT'] + LUT_N] = lut

    # TOKENS / TARGETS left zero; the g++ harness writes them per window.
    return buf, o


def load_ce_lut():
    txt = (REPO / 'tikitorch' / 'hardware' / 'hls' / 'forge_kernels' / 'forge_ce_lut.hpp').read_text()
    import re
    m = re.search(r'FT_CE_EXP2_Q16\[256\]\s*=\s*\{(.*?)\}', txt, re.S)
    nums = [int(x) for x in re.findall(r'-?\d+', m.group(1))]
    assert len(nums) == 256, len(nums)
    return np.array(nums, dtype=np.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(REPO / 'results' / 'lm_shakespeare' / 'tropical_llm_d64_faithful.pt'))
    ap.add_argument('--out', default='/tmp/s1_buf.npy')
    # defaults = the best deploy fixed-point found by the FAITHFUL-model scale sweep (deployable bpc 4.71 on
    # 63 val windows, real kernel; logit softmax temperature ~448). VERIFIED FLAT at ~4.7-4.9 across the whole
    # SA/SGAM/SW/SH/GCOMP grid (low-SA-SW2 .. high-SW64; residual-consistency SGAM·SW≈SA). 4.71 does NOT beat
    # bigram (3.47): the MAD-LN mismatch is GONE (that was the std-LN 4.05's biggest piece) but the kernel's
    # ACTIVATION-POW2-QUANT (q(x) in every GEMM, +0.81) + MSB-bump dyad (+0.25) remain — the faithful model
    # trained pow2 WEIGHTS + int (not pow2) acts + round-log2 (not MSB-bump) dyad. See module-docstring VERDICT.
    ap.add_argument('--SA', type=int, default=64)
    ap.add_argument('--SW', type=int, default=2, help='single GEMM weight scale (overridden by per-matrix below)')
    ap.add_argument('--SW_PM', type=str, default='', help='per-matrix scales "q:4,k:4,v:8,o:8,up:6,dn:6"')
    ap.add_argument('--SGAM', type=int, default=32)
    ap.add_argument('--SH', type=int, default=8)
    ap.add_argument('--GCOMP', type=float, default=1.0)  # FAITHFUL: trained w/ MAD-LN, no std/MAD compensation
    ap.add_argument('--offsets', default='', help='kernel offsets JSON (offsets_dump) -> exact dims/BUFLEN/offsets')
    ap.add_argument('--kv2', type=int, default=0, help='1 = pack 4-term CSD weights (kernel-v2)')
    ap.add_argument('--wcsd_k', type=int, default=4, help='weight CSD terms')
    a = ap.parse_args()
    global D, H, DH, FFN, T, V, NL, BUFLEN, OFF_POS, OFF_LOGITS, OFFSETS, KV2, WCSD_K
    KV2 = bool(a.kv2); WCSD_K = a.wcsd_k
    if a.offsets:
        j = json.loads(Path(a.offsets).read_text())
        D, H, DH, FFN, T, V, NL = j['D'], j['H'], j['DH'], j['FFN'], j['T'], j['V'], j['NL']
        BUFLEN, OFF_POS, OFF_LOGITS = j['BUFLEN'], j['OFF_POS'], j['OFF_LOGITS']
        OFFSETS = j['offsets']
    SW = a.SW
    if a.SW_PM:
        SW = {k: int(v) for k, v in (kv.split(':') for kv in a.SW_PM.split(','))}
    buf, o = build_buffer(a.ckpt, a.SA, SW, a.SGAM, a.SH, a.GCOMP)
    np.save(a.out, buf)
    meta = dict(D=D, H=H, DH=DH, FFN=FFN, T=T, V=V, VREAL=VREAL, NL=NL, BUFLEN=BUFLEN,
                OFF_POS=OFF_POS, OFF_LOGITS=OFF_LOGITS, offsets=o,
                SA=a.SA, SW=SW, SGAM=a.SGAM, SH=a.SH, GCOMP=a.GCOMP)
    Path(a.out + '.json').write_text(json.dumps(meta, indent=2))
    print(f'[export] wrote {a.out} (len {buf.size}, nonzero {int(np.count_nonzero(buf))})')
    print(f'[export] scales SA={a.SA} SW={SW} SGAM={a.SGAM} SH={a.SH} GCOMP={a.GCOMP}')


if __name__ == '__main__':
    main()
