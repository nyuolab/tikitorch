// forge_pc_xformer_mp_golden_block.hpp — MULTI-PRECISION fork of forge_pc_xformer_golden_block.hpp.
// =====================================================================================================
// PURPOSE (Gate G2, the load-bearing experiment): isolate CASCADE-vs-FORMAT.
//
// Every coherence number we have (2.016 / 1.732 / 2.898 bpb) came from a torch char-LM trained with
// loss.backward() = AUTOGRAD. The KERNEL ships a HAND-ROLLED CASCADE backward (g_route / g_wgrad /
// g_ln_bwd / __int128 attention bwd) + an integer dyadic-Adam. The integer cascade caps ~0.20 acc.
// It is UNPROVEN whether the cascade reaches coherent at ANY precision: the wall might be the CASCADE
// itself, not the integer FORMAT.
//
// This fork adds a compile/env MODE knob MP_OPT in {int, fix16, fix24, dbl} that changes ONLY:
//   (a) the BACKWARD-MULTIPLY operand precision (the g_wgrad outer-product operands),
//   (b) the MASTER-WEIGHT representation (a high-precision shadow accumulator),
//   (c) the dyadic-Adam m/v EMA + step representation.
// The FORWARD (mp_forward == the original g_forward, structurally) and the cascade STRUCTURE
// (route / ln_bwd / attention-bwd shapes, the dyadic-Adam ALGORITHM sign(m)+pow2(|m|/v)) stay identical
// across modes. MP_OPT=int reproduces the original golden bit-for-bit (verified by the test harness).
//
//   int    = today's integer path, unchanged (the sanity / wall baseline).
//   fix16  = per-tensor pow2-scaled fixed point, 16 bits: scale=2^ceil(log2(maxabs)), step=scale/2^15,
//            round+clamp to +/-(2^15-1), dequant. Applied to master weights, Adam m/v, AND the wgrad
//            multiply operands. (Mirrors the torch precision-floor quantizer that gave fix16=2.898.)
//   fix24  = same, 24 bits (margin check).
//   dbl    = double precision for master + Adam + wgrad accumulate (the CASCADE'S FLOAT CEILING).
//
// dbl-vs-autograd(~2.0) tells us if the CASCADE underperforms backprop. fix16-vs-dbl tells the FORMAT cost.
//
// HOW THE FORWARD STAYS BYTE-IDENTICAL ACROSS MODES
// -------------------------------------------------
// The forward reads an int16 master W[] and applies dyad(W)>>x (g_wapply). To keep the forward identical
// while letting fix16/fix24/dbl actually MOVE weights with sub-integer optimizer steps, the optimizer keeps
// a high-precision SHADOW master Wf (double) per weight; after each Adam step we round Wf back to the int16
// master W[] that the forward consumes. In MP_OPT=int there is NO shadow and the integer path is byte-exact
// to the original g_adam (X_LR_SHIFT=0, stochastic-round-free). This is the minimal fractional-master
// accumulator the prompt calls for: it gives fix16/fix24/dbl the sub-integer resolution the int path lacked
// (X_LR_SHIFT=0 -> dw=m>>se can be 0 = frozen) WITHOUT touching the forward.
//
// NB on operand quantization for fix16/fix24: we mirror the torch precision-FLOOR semantics — the operands
// (go, x) are dequantized through a 16/24-bit per-tensor pow2 grid before the multiply, and the m/v EMA +
// master live on the same grid. dbl skips quantization entirely (pure double).
// =====================================================================================================
#ifndef FORGE_PC_XFORMER_MP_GOLDEN_BLOCK_HPP
#define FORGE_PC_XFORMER_MP_GOLDEN_BLOCK_HPP

// Reuse the ORIGINAL golden's forward + core ops verbatim (g_forward, g_layernorm, g_wapply, g_route,
// g_ln_bwd, g_dyadval, ...). We only OVERRIDE the wgrad-operand precision and the optimizer here, in
// new mp_* symbols, so the forward + cascade structure are provably the same code path.
#include "forge_pc_xformer_golden_block.hpp"
#include <vector>
#include <cmath>

// ---------------- MODE knob ----------------
enum MPMode { MP_INT=0, MP_FIX16=1, MP_FIX24=2, MP_DBL=3 };
static int g_mp_mode = -1;
static inline int mp_mode() {
    if (g_mp_mode < 0) {
        const char* e = getenv("MP_OPT");
        if (!e) g_mp_mode = MP_INT;
        else if (!strcmp(e,"int"))   g_mp_mode = MP_INT;
        else if (!strcmp(e,"fix16")) g_mp_mode = MP_FIX16;
        else if (!strcmp(e,"fix24")) g_mp_mode = MP_FIX24;
        else if (!strcmp(e,"dbl"))   g_mp_mode = MP_DBL;
        else g_mp_mode = MP_INT;
    }
    return g_mp_mode;
}
static inline int mp_bits() { int m=mp_mode(); return m==MP_FIX16?16 : m==MP_FIX24?24 : 0; }
static inline const char* mp_name() {
    switch(mp_mode()){case MP_INT:return "int";case MP_FIX16:return "fix16";case MP_FIX24:return "fix24";default:return "dbl";}
}

// ---------------- per-tensor pow2-scaled fixed-point quantizer (the torch precision-floor mirror) ----------------
// q(x) over a tensor: scale = 2^ceil(log2(maxabs)); step = scale / 2^(bits-1); round(x/step)*step clamped to
// +/-(2^(bits-1)-1) levels. Returns the DEQUANTIZED double. bits==0 (dbl) => identity (no quantization).
static inline double mp_q_one(double x, double step, long maxlev) {
    if (step <= 0) return x;
    long q = (long)llround(x / step);
    if (q >  maxlev) q =  maxlev;
    if (q < -maxlev) q = -maxlev;
    return (double)q * step;
}
static inline double mp_tensor_maxabs(const double* x, int n) {
    double mx = 0; for (int i=0;i<n;i++){ double a = x[i]<0?-x[i]:x[i]; if (a>mx) mx=a; } return mx;
}
static inline double mp_step_for(double maxabs, int bits) {
    if (bits <= 0 || maxabs <= 0) return 0.0;                 // dbl or all-zero tensor -> no grid
    double sc = pow(2.0, ceil(log2(maxabs)));                 // scale = 2^ceil(log2(maxabs))
    return sc / (double)((long)1 << (bits-1));                // step = scale / 2^(bits-1)
}
// quantize a tensor in place (double buffer); no-op when bits==0
static inline void mp_quant_tensor(double* x, int n, int bits) {
    if (bits <= 0) return;
    double mx = mp_tensor_maxabs(x, n); double step = mp_step_for(mx, bits);
    if (step <= 0) return; long maxlev = ((long)1 << (bits-1)) - 1;
    for (int i=0;i<n;i++) x[i] = mp_q_one(x[i], step, maxlev);
}

// =====================================================================================================
// (a) BACKWARD-MULTIPLY: a precision-aware wgrad that overrides g_wgrad's operand precision.
// int   -> EXACTLY the original g_wgrad (dyadval operands, mode 0; or whatever WGRAD_MODE asks). Byte-exact.
// fix16/24 -> quantize the (go, x) operands onto the 16/24-bit per-tensor pow2 grid, multiply, accumulate.
// dbl   -> full double operands + double accumulate.
// The accumulator gW stays i64 here (the grad buffer type is shared with the int cascade); for dbl/fixN we
// round the per-(o,i) accumulate to i64 at the end. The ACCUMULATE WIDTH for the multiply itself is double
// (no 64-bit wrap), which matters for dbl (the "wgrad accumulate" the prompt names).
// =====================================================================================================
static void mp_wgrad(const i64 *go, const i64 *x, i64 *gW, int M, int I, int O) {
    int m = mp_mode();
    if (m == MP_INT) { g_wgrad(go, x, gW, M, I, O); return; }   // byte-exact integer path
    // build per-tensor grids for go and x (over the whole M*O / M*I tensor)
    int bits = mp_bits();
    double goMax=0, xMax=0;
    for (int t=0;t<M;t++) for (int o=0;o<O;o++){ double a=(double)go[t*O+o]; a=a<0?-a:a; if(a>goMax)goMax=a; }
    for (int t=0;t<M;t++) for (int i=0;i<I;i++){ double a=(double)x[t*I+i];  a=a<0?-a:a; if(a>xMax)xMax=a; }
    double goStep = mp_step_for(goMax, bits), xStep = mp_step_for(xMax, bits);
    long maxlev = bits>0 ? (((long)1<<(bits-1))-1) : 0;
    for (int o=0;o<O;o++) for (int i=0;i<I;i++){
        double acc = 0;
        for (int t=0;t<M;t++){
            double ga = (bits>0)? mp_q_one((double)go[t*O+o], goStep, maxlev) : (double)go[t*O+o];
            double xa = (bits>0)? mp_q_one((double)x[t*I+i],  xStep,  maxlev) : (double)x[t*I+i];
            acc += ga * xa;
        }
        gW[o*I+i] += (i64)llround(acc);
    }
}

// =====================================================================================================
// mp_backward — a fork of g_backward that routes every WEIGHT-GRAD multiply through mp_wgrad. The ROUTE
// (g_route, dyadic), LN-bwd (g_ln_bwd) and attention-bwd (__int128 multiplies) are STRUCTURE — identical
// to the original (they read int16 masters / cached activations the forward produced, byte-identical for
// any mode). Only the wgrad operand precision changes. (We copy the body so the wgrad calls dispatch to
// mp_wgrad; the original g_backward is left untouched.)
// =====================================================================================================
static void mp_backward(GW &w, GA &a, GG &g, const int *tok, bool zero=true) {
    if (zero) memset(&g, 0, sizeof(GG));
    i64 gXF[X_SEQ*X_D]; g_route(a.GLOG, w.WHEAD, gXF, X_SEQ, X_D, X_VOCAB);
    mp_wgrad(a.GLOG, a.XF, g.WHEAD, X_SEQ, X_D, X_VOCAB);
    i64 gRP[X_SEQ*X_D]; memset(gRP,0,sizeof(gRP));
    g_ln_bwd(gXF, a.RP, w.LNFG, a.lnfs, gRP, g.LNFG, g.LNFB, X_SEQ);
    i64 gM[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
    i64 gHD[X_SEQ*X_FFN]; g_route(gRP, w.WDN, gHD, X_SEQ, X_FFN, X_D);
    for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;
    mp_wgrad(gRP, a.HD, g.WDN, X_SEQ, X_FFN, X_D);
    i64 gX2[X_SEQ*X_D]; g_route(gHD, w.WUP, gX2, X_SEQ, X_D, X_FFN);
    mp_wgrad(gHD, a.X2, g.WUP, X_SEQ, X_D, X_FFN);
    g_ln_bwd(gX2, a.M, w.LN2G, a.ln2s, gM, g.LN2G, g.LN2B, X_SEQ);
    i64 gR[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gR[i]=gM[i];
    i64 gATTN[X_SEQ*X_D]; g_route(gM, w.WO, gATTN, X_SEQ, X_D, X_D);
    mp_wgrad(gM, a.ATTN, g.WO, X_SEQ, X_D, X_D);
    i64 gQ[X_SEQ*X_D], gK[X_SEQ*X_D], gV[X_SEQ*X_D];
    memset(gQ,0,sizeof(gQ)); memset(gK,0,sizeof(gK)); memset(gV,0,sizeof(gV));
    for(int h=0;h<X_H;h++) for(int i=0;i<X_SEQ;i++){
        int shn=a.Pshn[h*X_SEQ+i]; i64 pn[X_SEQ];
        for(int j=0;j<=i;j++) pn[j]=g_floorshr(a.P[(h*X_SEQ+i)*X_SEQ+j]<<X_AT_PFRAC, shn);
        i64 gp[X_SEQ]; for(int j=0;j<=i;j++){ i64 s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*a.V[j*X_D+h*X_DH+d]; gp[j]=s; }
        i64 ex=0; for(int j=0;j<=i;j++) ex += (i64)(((__int128)pn[j]*gp[j]) >> X_AT_PFRAC);
        for(int j=0;j<=i;j++){
            i64 gs=(i64)(((__int128)pn[j]*(gp[j]-ex)) >> X_AT_PFRAC);
            if (g_attn_dot) { for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+= (gs*a.K[j*X_D+h*X_DH+d])>>X_ABITS; gK[j*X_D+h*X_DH+d]+= (gs*a.Q[i*X_D+h*X_DH+d])>>X_ABITS; } }
            else for(int d=0;d<X_DH;d++){ i64 df=a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; int sg=df>0?1:(df<0?-1:0);
                gQ[i*X_D+h*X_DH+d]+= -gs*sg; gK[j*X_D+h*X_DH+d]+= gs*sg; }
            for(int d=0;d<X_DH;d++) gV[j*X_D+h*X_DH+d]+= (i64)(((__int128)pn[j]*gATTN[i*X_D+h*X_DH+d]) >> X_AT_PFRAC);
        }
    }
    i64 gX1[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=0;
    i64 tmp[X_SEQ*X_D];
    g_route(gQ, w.WQ, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; mp_wgrad(gQ, a.X1, g.WQ, X_SEQ, X_D, X_D);
    g_route(gK, w.WK, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; mp_wgrad(gK, a.X1, g.WK, X_SEQ, X_D, X_D);
    g_route(gV, w.WV, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; mp_wgrad(gV, a.X1, g.WV, X_SEQ, X_D, X_D);
    g_ln_bwd(gX1, a.R, w.LN1G, a.ln1s, gR, g.LN1G, g.LN1B, X_SEQ);
    for (int i=0;i<X_SEQ*X_D;i++) g.gR[i]=gR[i];
    for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; for(int d=0;d<X_D;d++){ g.EMB[v*X_D+d]+=gR[t*X_D+d]; g.POS[t*X_D+d]+=gR[t*X_D+d]; } }
}

// =====================================================================================================
// (b,c) OPTIMIZER: dyadic-Adam with a high-precision master shadow + m/v.
// ALGORITHM is identical to g_adam's integer dyadic-Adam: m = EMA(g), v = EMA(|g|), step =
// sign(m) * 2^floor(log2(|m|/v)), master -= LR*step, master -= master*WD, clamp; forward reads round(master).
// What MP_OPT changes is ONLY the numeric representation of (master shadow Wf, m, v, step):
//   int   -> defer to the original integer g_adam EXACTLY (byte-exact baseline).
//   fix16 -> Wf, m, v all snapped to the 16-bit per-tensor pow2 grid each step (the precision floor).
//   fix24 -> same on a 24-bit grid.
//   dbl   -> pure double Wf, m, v (no grid).
// The shadow gives sub-integer steps the int path can't make (X_LR_SHIFT=0 -> dw=m>>se floors to 0 = frozen),
// then rounds Wf back to the int16 master W[] the forward consumes -> forward identical, weights actually move.
//
// LR + WD are chosen to mirror the integer dyadic-Adam's effective magnitudes (step ~ O(1) on the int16
// master, WD ~ off). Tunable via env MP_LR / MP_WD; defaults match the integer path's behaviour.
// =====================================================================================================
struct MPOpt { std::vector<double> Wf, m, v; };
static std::map<const i64*, MPOpt> g_mpopt;
static double g_mp_lr = -1, g_mp_wd = -1;
static inline void mp_read_hyper() {
    if (g_mp_lr < 0) { const char* e=getenv("MP_LR"); g_mp_lr = e?atof(e):1.0; }   // 1.0 = same O(1) step as int dw
    if (g_mp_wd < 0) { const char* e=getenv("MP_WD"); g_mp_wd = e?atof(e):0.0; }   // 0 = off (X_WD_SHIFT=14 ~ off)
}
static void mp_adam(i64 *W, const i64 *gW, i64 *m_unused, i64 *v_unused, int n) {
    if (mp_mode() == MP_INT) { g_adam(W, gW, m_unused, v_unused, n); return; }   // byte-exact integer baseline
    mp_read_hyper();
    int bits = mp_bits();
    MPOpt &s = g_mpopt[W];
    if (s.Wf.empty()) { s.Wf.assign(n,0); s.m.assign(n,0); s.v.assign(n,0); for(int i=0;i<n;i++) s.Wf[i]=(double)W[i]; }
    const double beta = 1.0 - pow(2.0, -X_ADAM_BSH);   // = 0.875, same EMA as the integer m/v (>>3)
    for (int i=0;i<n;i++){
        double g = (double)gW[i], ag = g<0?-g:g;
        s.m[i] = beta*s.m[i] + (1.0-beta)*g;            // first moment  (== integer m EMA, real-valued)
        s.v[i] = beta*s.v[i] + (1.0-beta)*ag;           // second moment (== integer v EMA on |g|)
        // dyadic-Adam step: sign(m) * 2^floor(log2(|m|/v))  — identical ALGORITHM to g_adam (pow2 |m|/v ratio)
        double am = s.m[i]<0?-s.m[i]:s.m[i];
        double ratio = am / (s.v[i] + 1e-9);
        double step  = 0.0;
        if (ratio > 0) { double e = floor(log2(ratio)); step = pow(2.0, e); }
        double sgn = s.m[i]>0?1.0:(s.m[i]<0?-1.0:0.0);
        s.Wf[i] -= g_mp_lr * sgn * step;                // master -= LR * sign(m) * 2^exp
        s.Wf[i] -= s.Wf[i] * g_mp_wd;                   // weight decay (off by default)
        if (s.Wf[i] >  32767) s.Wf[i] =  32767;         // SAME int16 master clamp as g_adam
        if (s.Wf[i] < -32767) s.Wf[i] = -32767;
    }
    // FORMAT FLOOR: snap the shadow master + Adam m/v to the bits-grid (no-op for dbl). This is where fix16
    // vs dbl is realized: the optimizer STATE is quantized, not just the operands.
    if (bits > 0) {
        mp_quant_tensor(s.Wf.data(), n, bits);
        mp_quant_tensor(s.m.data(),  n, bits);
        mp_quant_tensor(s.v.data(),  n, bits);
    }
    for (int i=0;i<n;i++){ double w=s.Wf[i]; if(w>32767)w=32767; if(w<-32767)w=-32767; W[i]=(i64)llround(w); }  // round to int16 master the forward reads
}
static void mp_step(GW &w, GG &g) {
    mp_adam(w.WQ,g.WQ,w.mWQ,w.vWQ,X_D*X_D); mp_adam(w.WK,g.WK,w.mWK,w.vWK,X_D*X_D);
    mp_adam(w.WV,g.WV,w.mWV,w.vWV,X_D*X_D); mp_adam(w.WO,g.WO,w.mWO,w.vWO,X_D*X_D);
    mp_adam(w.WUP,g.WUP,w.mUP,w.vUP,X_FFN*X_D); mp_adam(w.WDN,g.WDN,w.mDN,w.vDN,X_D*X_FFN);
    mp_adam(w.WHEAD,g.WHEAD,w.mHD,w.vHD,X_VOCAB*X_D);
    // EMB/POS: SignSGD ±1 — identical to g_step (the learnable input table, not Adam-managed). LN frozen.
#ifndef X_FREEZE_EMB
    for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]-=(g.EMB[i]>0)-(g.EMB[i]<0);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]-=(g.POS[i]>0)-(g.POS[i]<0);
#endif
}

#endif
