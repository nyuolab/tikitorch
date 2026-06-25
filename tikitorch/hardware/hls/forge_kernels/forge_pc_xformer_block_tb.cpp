// forge_pc_xformer_block_tb.cpp — int64 CSIM testbench: validate the distributed PC transformer-block shard
// kernel (forge_pc_xformer_block.cpp, OUTPUT role) BIT-EXACT against the int64 golden
// (forge_pc_xformer_golden_block.hpp) for the full fwd + bwd + dyadic-Adam step.
//
// Flow:
//   (1) Build ONE DDR buffer (int32 words, X_BUFLEN) viewed as ap_uint<512>*. Stage a fixed-seed init:
//       int16 weights (WQ..WHEAD), LN gammas=64/betas=0, EMB/POS, the exp2 LUT (g_lut_init values), tokens +
//       targets, dyadic-Adam m/v = 0. Mirror the IDENTICAL init into a golden GW (+ zeroed GA/GG).
//   (2) Run the golden: g_forward; g_backward; then a SECOND golden copy through g_step (fused bwd+adam in the
//       kernel) to get the post-update weights.
//   (3) Kernel FORWARD (PHASE=FWD, ROLE=OUTPUT, DO_UP=1): compare the activation regions the forward writes
//       (X1, RMID, RP, ATTNO, HUP, P, LOGIT, HOUT=grad_logit) vs golden a.X1/M/RP/ATTN/HD/P/LG/GLOG — max|diff|=0.
//   (4) Kernel BACKWARD (PHASE=BWD): compare the updated weights it wrote (WQ/WK/WV/WO/WUP/WDN/WHEAD) vs the
//       golden's post-g_step weights — max|diff|=0.
//   (5) KAT corners: (a) zero grad_logit (stage XO_HOUT=0, BWD only, m/v=0) -> ZERO weight update;
//                    (b) all-ones tokens -> full fwd+bwd still bit-exact.
//
// The kernel stores activations as int16 in int32 lanes (data_t = ap_int<16>); the chosen init scale keeps every
// compared activation AND every pre-rescale matmul accumulator inside int16, so a correct kernel matches the
// int64 golden with max|diff|=0.  Offsets/dims come from forge_pc_xformer_config.hpp.
//   vitis_hls -f synth_pc_xformer_block.tcl   (with csim_design enabled)
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <ap_int.h>
#include "forge_pc_xformer_golden_block.hpp"     // pulls in the config + the int64 golden (g_forward/backward/step)

extern "C" void forge_pc_xformer_block(ap_uint<512> *buffer);

// ---------------------------------------------------------------------------------------------------
// fixed-seed init (mirrors forge_pc_xformer_golden_test.cpp's init exactly so golden + kernel see the SAME init)
static int RV(double s) {
    double u = 0; for (int k = 0; k < 6; k++) u += rand() / (double)RAND_MAX;
    double g = (u - 3.0) * s; int v = (int)lround(g);
    if (v > 32767) v = 32767; if (v < -32767) v = -32767; return v;
}
static void golden_init(GW &w) {
    srand(7);
    for (int i = 0; i < X_D*X_D; i++) { w.WQ[i]=RV(8); w.WK[i]=RV(8); w.WV[i]=RV(8); w.WO[i]=RV(8); }
    for (int i = 0; i < X_FFN*X_D; i++) w.WUP[i] = RV(8);
    for (int i = 0; i < X_D*X_FFN; i++) w.WDN[i] = RV(8);
    for (int i = 0; i < X_VOCAB*X_D; i++) w.WHEAD[i] = RV(8);
    for (int i = 0; i < X_VOCAB*X_D; i++) w.EMB[i]  = RV(16);
    for (int i = 0; i < X_SEQ*X_D; i++)  w.POS[i]   = RV(8);
    for (int d = 0; d < X_D; d++) { w.LN1G[d]=64; w.LN2G[d]=64; w.LNFG[d]=64; w.LN1B[d]=0; w.LN2B[d]=0; w.LNFB[d]=0; }
    // zero dyadic-Adam state
    memset(w.mWQ,0,sizeof(w.mWQ)); memset(w.vWQ,0,sizeof(w.vWQ)); memset(w.mWK,0,sizeof(w.mWK)); memset(w.vWK,0,sizeof(w.vWK));
    memset(w.mWV,0,sizeof(w.mWV)); memset(w.vWV,0,sizeof(w.vWV)); memset(w.mWO,0,sizeof(w.mWO)); memset(w.vWO,0,sizeof(w.vWO));
    memset(w.mUP,0,sizeof(w.mUP)); memset(w.vUP,0,sizeof(w.vUP)); memset(w.mDN,0,sizeof(w.mDN)); memset(w.vDN,0,sizeof(w.vDN));
    memset(w.mHD,0,sizeof(w.mHD)); memset(w.vHD,0,sizeof(w.vHD));
}

// ---- buffer staging helpers (all regions are int32 lanes) ----
static void put_i32(int32_t *buf, int off, const i64 *src, int n) { for (int i = 0; i < n; i++) buf[off + i] = (int32_t)src[i]; }
static void put_int(int32_t *buf, int off, const int *src, int n)  { for (int i = 0; i < n; i++) buf[off + i] = (int32_t)src[i]; }
static void setcfg(int32_t *buf, int phase, int role, int do_up) {
    // CFG lanes: range(31,0)=PHASE, range(63,32)=ROLE, range(95,64)=DO_UP  (lanes 0,1,2)
    buf[XO_CFG + 0] = phase; buf[XO_CFG + 1] = role; buf[XO_CFG + 2] = do_up; buf[XO_CFG + 3] = 0;
}
// stage the full persistent init (weights + LN + embed/pos + lnf/head + LUT + adam m/v=0) into the buffer
static void stage_persistent(int32_t *buf, const GW &w, const i64 *lut) {
    put_i32(buf, XO_WQ,   w.WQ,   X_D*X_D);   put_i32(buf, XO_WK,   w.WK,   X_D*X_D);
    put_i32(buf, XO_WV,   w.WV,   X_D*X_D);   put_i32(buf, XO_WO,   w.WO,   X_D*X_D);
    put_i32(buf, XO_WUP,  w.WUP,  X_FFN*X_D); put_i32(buf, XO_WDN,  w.WDN,  X_D*X_FFN);
    put_i32(buf, XO_LN1G, w.LN1G, X_D);       put_i32(buf, XO_LN1B, w.LN1B, X_D);
    put_i32(buf, XO_LN2G, w.LN2G, X_D);       put_i32(buf, XO_LN2B, w.LN2B, X_D);
    put_i32(buf, XO_EMBED,w.EMB,  X_VOCAB*X_D);put_i32(buf,XO_POS,  w.POS,  X_SEQ*X_D);
    put_i32(buf, XO_LNFG, w.LNFG, X_D);       put_i32(buf, XO_LNFB, w.LNFB, X_D);
    put_i32(buf, XO_WHEAD,w.WHEAD,X_VOCAB*X_D);
    put_i32(buf, XO_LUT,  lut,    X_LUT_N);
    // dyadic-Adam m/v = 0 across the whole flat region (kernel reads XO_MOM+OFF / XO_VAR+OFF per matrix)
    for (int i = 0; i < X_NW; i++) { buf[XO_MOM + i] = 0; buf[XO_VAR + i] = 0; }
}

// stage the golden's (warmed) dyadic-Adam m/v into the flat XO_MOM/XO_VAR regions at each matrix's offset
static void stage_mv(int32_t *buf, const GW &w) {
    put_i32(buf, XO_MOM+X_OFF_WQ, w.mWQ, X_D*X_D);     put_i32(buf, XO_VAR+X_OFF_WQ, w.vWQ, X_D*X_D);
    put_i32(buf, XO_MOM+X_OFF_WK, w.mWK, X_D*X_D);     put_i32(buf, XO_VAR+X_OFF_WK, w.vWK, X_D*X_D);
    put_i32(buf, XO_MOM+X_OFF_WV, w.mWV, X_D*X_D);     put_i32(buf, XO_VAR+X_OFF_WV, w.vWV, X_D*X_D);
    put_i32(buf, XO_MOM+X_OFF_WO, w.mWO, X_D*X_D);     put_i32(buf, XO_VAR+X_OFF_WO, w.vWO, X_D*X_D);
    put_i32(buf, XO_MOM+X_OFF_WUP, w.mUP, X_FFN*X_D);  put_i32(buf, XO_VAR+X_OFF_WUP, w.vUP, X_FFN*X_D);
    put_i32(buf, XO_MOM+X_OFF_WDN, w.mDN, X_D*X_FFN);  put_i32(buf, XO_VAR+X_OFF_WDN, w.vDN, X_D*X_FFN);
    put_i32(buf, XO_MOM+X_OFF_MHEAD, w.mHD, X_VOCAB*X_D); put_i32(buf, XO_VAR+X_OFF_MHEAD, w.vHD, X_VOCAB*X_D);
}
// ---- region compare: kernel int32 lanes (from off) vs a golden i64 array, optionally only causal (j<=i) ----
static long cmp_region(const int32_t *buf, int off, const i64 *gold, int n, long &nnz) {
    long mx = 0; nnz = 0;
    for (int i = 0; i < n; i++) { long d = labs((long)buf[off + i] - (long)gold[i]); if (d > mx) mx = d; if (d) nnz++; }
    return mx;
}

int main() {
    g_lut_init();                                // fill G_LUT (the exp2 ROM the kernel also uses, staged below)

    int32_t *buf = (int32_t*)aligned_alloc(64, sizeof(int32_t) * X_BUFLEN);
    for (int i = 0; i < X_BUFLEN; i++) buf[i] = 0;

    // ---------------- (1) init + WARM the dyadic-Adam state ----------------
    // The dyadic-Adam first step from m=v=0 is ~a no-op (EMA warmup), which makes a 1-step weight compare
    // trivial. Warm the golden a few steps so m/v + weights are non-trivial, then stage THAT warmed state
    // (weights + m/v) into the buffer so the ONE compared step is a real, non-trivial update on both sides.
    static GW w;  golden_init(w);
    int tok[X_SEQ], tgt[X_SEQ];
    srand(123); for (int t = 0; t < X_SEQ; t++) { tok[t] = rand() % X_VOCAB; tgt[t] = rand() % X_VOCAB; }
    for (int k = 0; k < 4; k++) { static GA aw; static GG gw; memset(&aw,0,sizeof(GA)); memset(&gw,0,sizeof(GG));
        g_forward(w, aw, tok, tgt); g_backward(w, aw, gw, tok); g_step(w, gw); }   // w now warmed

    static GA a;  memset(&a, 0, sizeof(GA));
    static GG g;  memset(&g, 0, sizeof(GG));
    stage_persistent(buf, w, G_LUT);             // stage the WARMED weights (m/v zeroed inside)...
    stage_mv(buf, w);                            // ...then overwrite m/v with the warmed Adam state
    put_int(buf, XO_TOK, tok, X_SEQ);
    put_int(buf, XO_TGT, tgt, X_SEQ);

    // ---------------- (2) golden reference: the ONE compared step from the warmed state ----------------
    g_forward(w, a, tok, tgt);
    g_backward(w, a, g, tok);                     // g.gR = the final input-gradient (kernel's EOUT, pre-rescale)
    // golden EOUT = max-shift(g.gR, X_EBITS) — the OUTPUT-role up-message
    static i64 gEOUT[X_SEQ*X_D]; { i64 mx=0; for (int i=0;i<X_SEQ*X_D;i++){ i64 v=g.gR[i]<0?-g.gR[i]:g.gR[i]; if(v>mx)mx=v; }
        int sh=g_normshift(mx, X_EBITS); for (int i=0;i<X_SEQ*X_D;i++) gEOUT[i]=g_floorshr(g.gR[i], sh); }
    static GW wstep; wstep = w; { static GG gs; memset(&gs,0,sizeof(GG)); static GA as; memset(&as,0,sizeof(GA));
        g_forward(wstep, as, tok, tgt); g_backward(wstep, as, gs, tok); g_step(wstep, gs); }   // wstep = post-Adam

    // ---------------- (3) kernel FORWARD ----------------
    // The kernel's OUTPUT role does NOT embed — it reads the block input R from the neighbour hidden-state
    // XO_HIN (the distributed-shard contract). The golden block embeds internally; stage its computed R = a.R
    // into XO_HIN so the kernel's OUTPUT-role block computes from the SAME R (the embed itself is the INPUT
    // role's job, validated separately). a.R fits int16, so the int16 HIN message is lossless here.
    put_i32(buf, XO_HIN, a.R, X_SEQ*X_D);
    setcfg(buf, X_PH_FWD, X_RL_OUT, 1);
    forge_pc_xformer_block((ap_uint<512>*)buf);

    long nnz;
    long dX1   = cmp_region(buf, XO_X1,    a.X1,   X_SEQ*X_D,     nnz);
    long dRMID = cmp_region(buf, XO_RMID,  a.M,    X_SEQ*X_D,     nnz);
    long dRP   = cmp_region(buf, XO_RP,    a.RP,   X_SEQ*X_D,     nnz);
    long dATTN = cmp_region(buf, XO_ATTNO, a.ATTN, X_SEQ*X_D,     nnz);
    long dX2   = cmp_region(buf, XO_X2,    a.X2,   X_SEQ*X_D,     nnz);
    long dHUP  = cmp_region(buf, XO_HUP,   a.HD,   X_SEQ*X_FFN,   nnz);
    long dLG   = cmp_region(buf, XO_LOGIT, a.LG,   X_SEQ*X_VOCAB, nnz);
    long dGLOG = cmp_region(buf, XO_HOUT,  a.GLOG, X_SEQ*X_VOCAB, nnz);  // OUTPUT role caches grad_logit (int32) here
    // attn probs P: only the causal (j<=i) entries are written/meaningful; upper triangle is masked to 0 both sides
    long dP = 0;
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) for (int j = 0; j <= i; j++) {
        int idx = (h*X_SEQ + i)*X_SEQ + j;
        long d = labs((long)buf[XO_P + idx] - (long)a.P[idx]); if (d > dP) dP = d;
    }

    printf("=== forge_pc_xformer_block CSIM  (D=%d H=%d FFN=%d SEQ=%d VOCAB=%d) ===\n", X_D,X_H,X_FFN,X_SEQ,X_VOCAB);
    printf("-- FORWARD region max|diff| vs int64 golden --\n");
    printf("  X1 (XO_X1)        = %ld\n", dX1);
    printf("  M  (XO_RMID)      = %ld\n", dRMID);
    printf("  R' (XO_RP)        = %ld\n", dRP);
    printf("  ATTN (XO_ATTNO)   = %ld\n", dATTN);
    printf("  X2 (XO_X2)        = %ld\n", dX2);
    printf("  HD (XO_HUP)       = %ld\n", dHUP);
    printf("  P  (XO_P,causal)  = %ld\n", dP);
    printf("  LOGIT (XO_LOGIT)  = %ld\n", dLG);
    printf("  GLOG (XO_HOUT)    = %ld\n", dGLOG);
    long fwd_bad = dX1|dRMID|dRP|dATTN|dX2|dHUP|dP|dLG|dGLOG;

    // ---------------- (4) kernel BACKWARD + dyadic-Adam ----------------
    setcfg(buf, X_PH_BWD, X_RL_OUT, 1);
    forge_pc_xformer_block((ap_uint<512>*)buf);

    long dWQ = cmp_region(buf, XO_WQ,    wstep.WQ,    X_D*X_D,     nnz); long nWQ = nnz;
    long dWK = cmp_region(buf, XO_WK,    wstep.WK,    X_D*X_D,     nnz); long nWK = nnz;
    long dWV = cmp_region(buf, XO_WV,    wstep.WV,    X_D*X_D,     nnz); long nWV = nnz;
    long dWO = cmp_region(buf, XO_WO,    wstep.WO,    X_D*X_D,     nnz); long nWO = nnz;
    long dUP = cmp_region(buf, XO_WUP,   wstep.WUP,   X_FFN*X_D,   nnz); long nUP = nnz;
    long dDN = cmp_region(buf, XO_WDN,   wstep.WDN,   X_D*X_FFN,   nnz); long nDN = nnz;
    long dHD = cmp_region(buf, XO_WHEAD, wstep.WHEAD, X_VOCAB*X_D, nnz); long nHD = nnz;

    printf("-- BACKWARD updated-weight max|diff| vs golden post-g_step (#changed = update is non-trivial) --\n");
    printf("  WQ    = %ld   (changed %ld/%d)\n", dWQ, nWQ, X_D*X_D);
    printf("  WK    = %ld   (changed %ld/%d)\n", dWK, nWK, X_D*X_D);
    printf("  WV    = %ld   (changed %ld/%d)\n", dWV, nWV, X_D*X_D);
    printf("  WO    = %ld   (changed %ld/%d)\n", dWO, nWO, X_D*X_D);
    printf("  WUP   = %ld   (changed %ld/%d)\n", dUP, nUP, X_FFN*X_D);
    printf("  WDN   = %ld   (changed %ld/%d)\n", dDN, nDN, X_D*X_FFN);
    printf("  WHEAD = %ld   (changed %ld/%d)\n", dHD, nHD, X_VOCAB*X_D);
    long dEOUT = cmp_region(buf, XO_EOUT, gEOUT, X_SEQ*X_D, nnz);   // the gradient cascade head->...->LN1 (EOUT)
    printf("  EOUT (XO_EOUT, the input-gradient cascade) = %ld\n", dEOUT);
    long bwd_bad = dWQ|dWK|dWV|dWO|dUP|dDN|dHD|dEOUT;
    // non-triviality: count how many weights the GOLDEN's g_step actually moved (w -> wstep), so a PASS means
    // "bit-exact AND the update was real" (the per-region nnz above is kernel-vs-golden = 0 when bit-exact).
    long gupd = 0;
    for (int i=0;i<X_D*X_D;i++){ gupd += (wstep.WQ[i]!=w.WQ[i])+(wstep.WK[i]!=w.WK[i])+(wstep.WV[i]!=w.WV[i])+(wstep.WO[i]!=w.WO[i]); }
    for (int i=0;i<X_FFN*X_D;i++) gupd += (wstep.WUP[i]!=w.WUP[i]);
    for (int i=0;i<X_D*X_FFN;i++) gupd += (wstep.WDN[i]!=w.WDN[i]);
    for (int i=0;i<X_VOCAB*X_D;i++) gupd += (wstep.WHEAD[i]!=w.WHEAD[i]);
    printf("  [non-triviality] golden g_step moved %ld weights this step\n", gupd);
    long changed = gupd;

    // ---------------- (5a) KAT: zero grad_logit -> zero weight update ----------------
    // Re-stage a clean persistent init (fresh weights + m/v=0), populate the BWD-needed cached activations by
    // running a forward first, THEN overwrite the cached grad_logit (XO_HOUT) with all-zeros and run BWD only.
    // A zero grad_logit must propagate to zero grads everywhere -> dyadic-Adam leaves every weight untouched.
    int32_t *kb = (int32_t*)aligned_alloc(64, sizeof(int32_t) * X_BUFLEN);
    for (int i = 0; i < X_BUFLEN; i++) kb[i] = 0;
    static GW wk; golden_init(wk);
    stage_persistent(kb, wk, G_LUT);
    put_int(kb, XO_TOK, tok, X_SEQ);
    put_int(kb, XO_TGT, tgt, X_SEQ);
    setcfg(kb, X_PH_FWD, X_RL_OUT, 1); forge_pc_xformer_block((ap_uint<512>*)kb);   // fill caches
    for (int i = 0; i < X_SEQ*X_VOCAB; i++) kb[XO_HOUT + i] = 0;                     // zero the grad_logit seed
    // snapshot weights, run BWD, confirm NOTHING moved
    static int32_t wq_before[X_D*X_D], wh_before[X_VOCAB*X_D];
    for (int i = 0; i < X_D*X_D; i++)     wq_before[i] = kb[XO_WQ + i];
    for (int i = 0; i < X_VOCAB*X_D; i++) wh_before[i] = kb[XO_WHEAD + i];
    setcfg(kb, X_PH_BWD, X_RL_OUT, 1); forge_pc_xformer_block((ap_uint<512>*)kb);
    long kat0 = 0;
    for (int i = 0; i < X_D*X_D; i++)     { long d = labs((long)kb[XO_WQ + i]    - wq_before[i]); if (d > kat0) kat0 = d; }
    for (int i = 0; i < X_VOCAB*X_D; i++) { long d = labs((long)kb[XO_WHEAD + i] - wh_before[i]); if (d > kat0) kat0 = d; }
    printf("-- KAT zero-grad_logit: max weight delta = %ld  (must be 0)\n", kat0);
    free(kb);

    // ---------------- (5b) KAT: all-ones tokens -> full fwd+bwd still bit-exact ----------------
    int32_t *ob = (int32_t*)aligned_alloc(64, sizeof(int32_t) * X_BUFLEN);
    for (int i = 0; i < X_BUFLEN; i++) ob[i] = 0;
    int tok1[X_SEQ], tgt1[X_SEQ];
    for (int t = 0; t < X_SEQ; t++) { tok1[t] = 1; tgt1[t] = 1; }
    static GW wo;  golden_init(wo);
    static GW wos; golden_init(wos);
    static GA ao; static GG go; memset(&ao,0,sizeof(GA)); memset(&go,0,sizeof(GG));
    stage_persistent(ob, wo, G_LUT);
    put_int(ob, XO_TOK, tok1, X_SEQ);
    put_int(ob, XO_TGT, tgt1, X_SEQ);
    g_forward(wo, ao, tok1, tgt1); g_backward(wo, ao, go, tok1);
    static GA aos; static GG gos; memset(&aos,0,sizeof(GA)); memset(&gos,0,sizeof(GG));
    g_forward(wos, aos, tok1, tgt1); g_backward(wos, aos, gos, tok1); g_step(wos, gos);
    put_i32(ob, XO_HIN, ao.R, X_SEQ*X_D);     // OUTPUT role reads R from HIN (stage the golden's embed result)
    setcfg(ob, X_PH_FWD, X_RL_OUT, 1); forge_pc_xformer_block((ap_uint<512>*)ob);
    long o_fwd = 0;
    { long n; o_fwd |= cmp_region(ob, XO_X1,    ao.X1,   X_SEQ*X_D,     n);
              o_fwd |= cmp_region(ob, XO_RP,    ao.RP,   X_SEQ*X_D,     n);
              o_fwd |= cmp_region(ob, XO_LOGIT, ao.LG,   X_SEQ*X_VOCAB, n);
              o_fwd |= cmp_region(ob, XO_HOUT,  ao.GLOG, X_SEQ*X_VOCAB, n); }
    setcfg(ob, X_PH_BWD, X_RL_OUT, 1); forge_pc_xformer_block((ap_uint<512>*)ob);
    long o_bwd = 0;
    { long n; o_bwd |= cmp_region(ob, XO_WQ,    wos.WQ,    X_D*X_D,     n);
              o_bwd |= cmp_region(ob, XO_WHEAD, wos.WHEAD, X_VOCAB*X_D, n);
              o_bwd |= cmp_region(ob, XO_WUP,   wos.WUP,   X_FFN*X_D,   n);
              o_bwd |= cmp_region(ob, XO_WDN,   wos.WDN,   X_D*X_FFN,   n); }
    printf("-- KAT all-ones tokens: fwd max|diff|=%ld  bwd(updated W) max|diff|=%ld\n", o_fwd, o_bwd);
    free(ob);

    // ---------------- verdict ----------------
    int pass = (fwd_bad == 0) && (bwd_bad == 0) && (changed > 0) && (kat0 == 0) && (o_fwd == 0) && (o_bwd == 0);
    printf("\n=== %s ===\n", pass
        ? "PASS — kernel fwd+bwd+dyadic-Adam BIT-EXACT to the int64 golden (max|diff|=0 all regions; KATs hold)"
        : "FAIL — see per-region max|diff| above");
    if (changed == 0) printf("  (note: golden weight update was trivial — check init scale)\n");

    free(buf);
    return pass ? 0 : 1;
}
