// forge_pc_xformer_bf16_golden_block.hpp — a CLEAN BF16 / wide-float backprop-with-STE backward for the
// PC transformer block, matched to the char-LM recipe (scripts/pc_xformer_charlm.py).
// =====================================================================================================
// WHY THIS EXISTS
// ---------------
// The canonical golden ships an INTEGER, hand-rolled cascade backward (g_route / g_wgrad / g_ln_bwd /
// __int128 attention bwd) + an integer dyadic-Adam. That integer backward caps at ~0.15 acc (the spread
// saturates, the wgrad operands are dyad-quantized to a coarse pow2 grid, the LN-coupling term wraps, and
// the int rescales compound the error magnitudes). The AUTOGRAD bar on the SAME forward (the char-LM with
// DyLinear w+(q-w).detach() + qact STE + warm dot attention, trained by torch autograd) is ~3.1 bpb /
// ~0.26-0.29 acc and coherent. The recipe is NOT the problem — the hand-coded integer backward is.
//
// THE FIX (this header): replace the integer cascade with TEXTBOOK transformer backprop in float/bf16,
// faithfully reproducing what torch autograd computes through the char-LM forward.
//
// HOW THE FORWARD STAYS BYTE-IDENTICAL
// ------------------------------------
// The DEPLOYED forward (g_forward, int) is UNTOUCHED — this header includes the canonical golden block
// verbatim and never modifies its forward. The model's predictions still come from g_forward, byte-identical
// (the forge_pc_xformer_bf16_torchcheck / hash check confirms it). For the BACKWARD we run a parallel
// WIDE-FLOAT "forward-for-backward" (bf_forward) that mirrors the char-LM's float semantics EXACTLY:
//   * DyLinear  -> matmul by the DEPLOYED dyadic weight (dyad(W)*2^-WFRAC), grad to the float master (STE).
//   * qact      -> per-tensor int16 max-shift that ROUNDS to a pow2 grid but KEEPS magnitude (== char-LM
//                  QAct: round(x/2^sh)*2^sh, NOT a divide); STE backward = identity.
//   * FDyadLN   -> fine-dyadic LayerNorm in float (== char-LM FDyadLN); analytic LN Jacobian in the backward.
//   * attention -> warm dot-product softmax with the 1/sqrt(DH) scale (== F.scaled_dot_product_attention),
//                  the analytic softmax Jacobian gs = p*(gp - sum p*gp) in the backward.
//   * relu      -> 0/1 mask.
// This is "cache the WIDE pre-quantization activations in the forward for the backward" — standard recompute-
// in-higher-precision. It does NOT alter the int g_forward the kernel deploys.
//
// WHICH QUANTIZERS ARE IDENTITY IN THE BACKWARD (the STE set):
//   1. dyad weight round       -> route via deployed dyad(W); d_W lands on the float master (STE).
//   2. qact max-shift round    -> identity (char-LM QAct.backward == g).
//   3. FFN floor / block-round -> identity (subsumed by qact here).
//   4. LayerNorm fine-dyadic invq + dyadic-gamma -> the QUANT is identity; the LN mean/var coupling is the
//      REAL analytic Jacobian (not identity — that is the genuine LN derivative, computed in float).
//   5. softmax exp2-LUT + >>shn normalize -> identity; real float softmax probs + analytic softmax Jacobian.
//   6. head rescale + CE LUT softmax -> the CE grad is the REAL softmax(wide logits) - onehot.
//
// VERIFICATION DISCIPLINE:
//   (A) END-TO-END TRAINING must reach the char-LM bar — forge_pc_xformer_bf16_golden_test.cpp.
//   (B) GRADIENTS cross-checked vs TORCH AUTOGRAD on the SAME float forward (ground truth) — torch_check.py.
//
// TOGGLE: BF16_BWD (env) — 0 = OFF, the original integer cascade (g_backward + g_step), reproduces the int
// cap. 1 = wide float (double) textbook backprop + float dyadic-Adam. 2 = every backward intermediate rounded
// through bf16 (1s8e7m). Default 0.
// =====================================================================================================
#ifndef FORGE_PC_XFORMER_BF16_GOLDEN_BLOCK_HPP
#define FORGE_PC_XFORMER_BF16_GOLDEN_BLOCK_HPP

#include "forge_pc_xformer_golden_block.hpp"   // canonical int forward + int backward, UNMODIFIED
#include <vector>
#include <map>
#include <cmath>
#include <cstring>

// ---------------- precision knob ----------------
static int g_bf16_bwd = -1;
static inline int bf16_bwd() {
    if (g_bf16_bwd < 0) { const char* e = getenv("BF16_BWD"); g_bf16_bwd = e ? atoi(e) : 0; }
    return g_bf16_bwd;
}
static inline double to_bf16(double x) {                  // float32 truncated to 7 mantissa bits, round-to-even
    float f = (float)x; uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 16) & 1u; u += 0x7fffu + lsb; u &= 0xffff0000u;
    std::memcpy(&f, &u, 4); return (double)f;
}
static inline double P(double x) { return (bf16_bwd() == 2) ? to_bf16(x) : x; }

// the DEPLOYED dyadic weight = nearest signed power of two of a FLOAT master (== char-LM DyadSTE). This is the
// value the route/wgrad multiply by; STE means the round is identity, grad flows to the float master. The
// float master lives at the char-LM's natural (small) weight scale — it is the trainable quantity. The int16
// deploy format is round(master*2^WFRAC) downstream; dyad(int16)*2^-WFRAC == dyad(master), so the deployed
// forward is byte-faithful to this same dyad value.
static inline double bf_dyad_f(double wf) {                  // == DyadSTE.forward: sign * 2^round(log2|w|)
    double a = std::fabs(wf); if (a < 1e-30) return 0.0;
    double e = std::round(std::log2(a));
    return (wf < 0 ? -1.0 : 1.0) * std::ldexp(1.0, (int)e);
}

// =====================================================================================================
// WIDE-FLOAT forward-for-backward cache. Holds the float activations of the char-LM-faithful forward.
// =====================================================================================================
struct BFA {
    double R[X_SEQ*X_D];                       // embed+pos (block input)
    double X1[X_SEQ*X_D], Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D];   // post-qact Q/K/V (the matmul feeds)
    double Praw[X_H*X_SEQ*X_SEQ];              // softmax probs (sum=1 per causal row)
    double ATTN[X_SEQ*X_D], M[X_SEQ*X_D], X2[X_SEQ*X_D];
    double HDpre[X_SEQ*X_FFN]; int RMASK[X_SEQ*X_FFN];
    double HD[X_SEQ*X_FFN], RP[X_SEQ*X_D], XF[X_SEQ*X_D], LG[X_SEQ*X_VOCAB];
    // LN per-token mean + normalizer s + the dyadic gamma/beta snapshot (for the analytic LN backward)
    double ln1mean[X_SEQ], ln1s[X_SEQ], ln2mean[X_SEQ], ln2s[X_SEQ], lnfmean[X_SEQ], lnfs[X_SEQ];
    double LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D], LNFG[X_D], LNFB[X_D];
};

// char-LM qact: per-tensor max-shift, round to pow2 grid, KEEP magnitude. STE backward = identity.
static void bf_qact(double* x, int n) {
    double mx = 0; for (int i = 0; i < n; i++) { double a = std::fabs(x[i]); if (a > mx) mx = a; }
    if (mx < 1e-9) mx = 1e-9;
    double sh = std::floor(std::log2(mx)) - X_ABITS;
    double sc = std::ldexp(1.0, (int)sh);
    for (int i = 0; i < n; i++) x[i] = std::round(x[i] / sc) * sc;
}
// char-LM FDyadLN (float). y = gamma*(x-mean)/((1+mant)*2^e) + beta. Returns the normalizer s=(1+mant)*2^e.
static void bf_layernorm(const double* x, const double* gamma, const double* beta,
                         double* y, double* mean_out, double* s_out, int M) {
    for (int t = 0; t < M; t++) {
        double sum = 0; for (int d = 0; d < X_D; d++) sum += x[t*X_D+d];
        double mean = sum / X_D; double asum = 0;
        for (int d = 0; d < X_D; d++) asum += std::fabs(x[t*X_D+d]-mean);
        double scale = asum / X_D; if (scale < 1e-9) scale = 1e-9;
        double e = std::floor(std::log2(scale));
        double mant = std::round((scale/std::ldexp(1.0,(int)e) - 1.0)*(1<<X_MBITS)) / (double)(1<<X_MBITS);
        double s = (1.0 + mant) * std::ldexp(1.0,(int)e);
        mean_out[t] = mean; s_out[t] = s;
        for (int d = 0; d < X_D; d++) y[t*X_D+d] = gamma[d]*((x[t*X_D+d]-mean)/s) + beta[d];
    }
}
// FLOAT-master model: the trainable weights live here, at the char-LM's natural scale. The deployed forward
// applies dyad(master) (== char-LM DyadSTE); the int16 kernel format is round(master*2^WFRAC) downstream.
struct BFM {
    double WQ[X_D*X_D], WK[X_D*X_D], WV[X_D*X_D], WO[X_D*X_D], WUP[X_FFN*X_D], WDN[X_D*X_FFN], WHEAD[X_VOCAB*X_D];
    double EMB[X_VOCAB*X_D], POS[X_SEQ*X_D];
    double LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D], LNFG[X_D], LNFB[X_D];   // gammas frozen (as int path)
};
static void bf_matmul(const double* x, const double* Wf, double* y, int M, int I, int O) {
    for (int t = 0; t < M; t++) for (int o = 0; o < O; o++) {
        double acc = 0; for (int i = 0; i < I; i++) acc += bf_dyad_f(Wf[o*I+i]) * x[t*I+i];   // dyad of float master
        y[t*O+o] = acc;
    }
}

// WIDE-FLOAT forward-for-backward (mirrors the char-LM EXACTLY). The DEPLOYED int forward (g_forward) is
// untouched; this is the higher-precision recompute the textbook backward differentiates. Operates on the
// FLOAT-master model m (char-LM scale).
static void bf_forward(BFM &m, BFA &a, const int *tok) {
    for (int d=0; d<X_D; d++) { a.LN1G[d]=m.LN1G[d]; a.LN1B[d]=m.LN1B[d]; a.LN2G[d]=m.LN2G[d]; a.LN2B[d]=m.LN2B[d]; a.LNFG[d]=m.LNFG[d]; a.LNFB[d]=m.LNFB[d]; }
    for (int t = 0; t < X_SEQ; t++) { int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1;
        for (int d = 0; d < X_D; d++) a.R[t*X_D+d] = m.EMB[v*X_D+d] + m.POS[t*X_D+d]; }
    bf_layernorm(a.R, a.LN1G, a.LN1B, a.X1, a.ln1mean, a.ln1s, X_SEQ);
    bf_matmul(a.X1, m.WQ, a.Q, X_SEQ, X_D, X_D); bf_qact(a.Q, X_SEQ*X_D);
    bf_matmul(a.X1, m.WK, a.K, X_SEQ, X_D, X_D); bf_qact(a.K, X_SEQ*X_D);
    bf_matmul(a.X1, m.WV, a.V, X_SEQ, X_D, X_D); bf_qact(a.V, X_SEQ*X_D);
    double scale = 1.0/std::sqrt((double)X_DH);            // F.scaled_dot_product_attention scale
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) {
        double sc[X_SEQ], mx=-1e300;
        for (int j = 0; j <= i; j++) { double dot=0; for (int d=0;d<X_DH;d++) dot += a.Q[i*X_D+h*X_DH+d]*a.K[j*X_D+h*X_DH+d];
            sc[j]=dot*scale; if(sc[j]>mx)mx=sc[j]; }
        double sum=0; for (int j=0;j<=i;j++){ double e=std::exp(sc[j]-mx); a.Praw[(h*X_SEQ+i)*X_SEQ+j]=e; sum+=e; }
        for (int j=0;j<=i;j++) a.Praw[(h*X_SEQ+i)*X_SEQ+j] /= sum;
        for (int d=0; d<X_DH; d++){ double blend=0; for(int j=0;j<=i;j++) blend+=a.Praw[(h*X_SEQ+i)*X_SEQ+j]*a.V[j*X_D+h*X_DH+d]; a.ATTN[i*X_D+h*X_DH+d]=blend; }
    }
    double yq[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) yq[i]=a.ATTN[i]; bf_qact(yq, X_SEQ*X_D);  // char-LM: proj(qact(y))
    for(int i=0;i<X_SEQ*X_D;i++) a.ATTN[i]=yq[i];
    double AO[X_SEQ*X_D]; bf_matmul(a.ATTN, m.WO, AO, X_SEQ, X_D, X_D);
    for(int i=0;i<X_SEQ*X_D;i++) a.M[i]=a.R[i]+AO[i];
    bf_layernorm(a.M, a.LN2G, a.LN2B, a.X2, a.ln2mean, a.ln2s, X_SEQ);
    bf_matmul(a.X2, m.WUP, a.HDpre, X_SEQ, X_D, X_FFN);
    for(int i=0;i<X_SEQ*X_FFN;i++){ a.RMASK[i]=a.HDpre[i]>0?1:0; a.HD[i]=a.RMASK[i]?a.HDpre[i]:0; }
    bf_qact(a.HD, X_SEQ*X_FFN);                            // char-LM: fc2(qact(relu(fc1(...))))
    double FO[X_SEQ*X_D]; bf_matmul(a.HD, m.WDN, FO, X_SEQ, X_FFN, X_D);
    for(int i=0;i<X_SEQ*X_D;i++) a.RP[i]=a.M[i]+FO[i];
    bf_layernorm(a.RP, a.LNFG, a.LNFB, a.XF, a.lnfmean, a.lnfs, X_SEQ);
    bf_matmul(a.XF, m.WHEAD, a.LG, X_SEQ, X_D, X_VOCAB);
}

// float gradient accumulators
struct BFG {
    std::vector<double> WQ, WK, WV, WO, WUP, WDN, WHEAD, EMB, POS, gR;
    // debug intermediates (attention-grad localization for the torch cross-check)
    double dbg_gATTN[X_SEQ*X_D], dbg_gV[X_SEQ*X_D], dbg_gQ[X_SEQ*X_D], dbg_gK[X_SEQ*X_D];
    void alloc() {
        WQ.assign(X_D*X_D,0); WK.assign(X_D*X_D,0); WV.assign(X_D*X_D,0); WO.assign(X_D*X_D,0);
        WUP.assign(X_FFN*X_D,0); WDN.assign(X_D*X_FFN,0); WHEAD.assign(X_VOCAB*X_D,0);
        EMB.assign(X_VOCAB*X_D,0); POS.assign(X_SEQ*X_D,0); gR.assign(X_SEQ*X_D,0);
    }
    void zero() { for(auto*v:{&WQ,&WK,&WV,&WO,&WUP,&WDN,&WHEAD,&EMB,&POS}) std::fill(v->begin(),v->end(),0); }
};
// dyadic-Adam optimizer state (one m/v per float weight matrix, keyed by the master ptr)
struct BFW { std::map<double*,std::vector<double>> m,v; };

// TEXTBOOK linear backward (STE on dyad weight): dx = dyad(W)^T @ gy ; dW = gy^T @ x  (wide).
// Route uses dyad(float master) (STE on the round, grad flows to the master).
static void bf_linear_bwd(const double* gy, const double* Wf, const double* x,
                          double* dx, double* dW, int M, int I, int O) {
    for (int t = 0; t < M; t++) for (int i = 0; i < I; i++) {
        double acc = 0; for (int o = 0; o < O; o++) acc += bf_dyad_f(Wf[o*I+i]) * gy[t*O+o];
        dx[t*I+i] = P(acc);
    }
    for (int o = 0; o < O; o++) for (int i = 0; i < I; i++) {
        double acc = 0; for (int t = 0; t < M; t++) acc += gy[t*O+o] * x[t*I+i];
        dW[o*I+i] += P(acc);
    }
}
// LayerNorm backward, EXACT to torch autograd of FDyadLN. y = gamma*(x-mean)/s + beta. The normalizer
// s = (1+mant)*2^e with mant=round(...), e=floor(log2(mean|c|)) — the round/floor quantizers BLOCK the
// gradient to s, so torch treats s as a CONSTANT (stop-grad). Hence the ONLY x-coupling is the mean term:
//   d_x = (1/s)(gxhat - mean(gxhat)),   gxhat = gy*gamma.
// (Verified bit-exact vs torch autograd of FDyadLN; the RMS variance-coupling term xhat*mean(gxhat*xhat) is
//  SPURIOUS here — it belongs to an RMS/std-normalized LN, not this MAD+dyadic-quantized one. Including it
//  was the cause of the 0.8 sign-agreement on the attention/LN-coupled grads.)
static void bf_ln_bwd(const double* gy, const double* xin, const double* gamma,
                      const double* mean, const double* s, double* dx, int M) {
    (void)xin; (void)mean;
    for (int t = 0; t < M; t++) {
        double gxhat[X_D], mg=0;
        for (int d=0; d<X_D; d++){ gxhat[d]=P(gy[t*X_D+d]*gamma[d]); mg+=gxhat[d]; }
        mg/=X_D;
        for (int d=0; d<X_D; d++) dx[t*X_D+d]=P((gxhat[d]-mg)/s[t]);
    }
}

// =====================================================================================================
// BF16/wide-float TEXTBOOK BACKWARD — mirrors torch autograd through bf_forward exactly.
// =====================================================================================================
static void bf_backward(BFM &m, BFA &a, BFG &g, const int *tok, const int *tgt, bool zero = true) {
    if (zero) g.zero();
    // (0) CE: grad_logit = softmax(wide logits) - onehot, mean over tokens.
    double gLG[X_SEQ*X_VOCAB];
    for (int t=0; t<X_SEQ; t++) {
        double mx=-1e300; for(int v=0;v<X_VOCAB;v++){ double l=a.LG[t*X_VOCAB+v]; if(l>mx)mx=l; }
        double sum=0,pr[X_VOCAB]; for(int v=0;v<X_VOCAB;v++){ pr[v]=std::exp(a.LG[t*X_VOCAB+v]-mx); sum+=pr[v]; }
        for(int v=0;v<X_VOCAB;v++){ double p=pr[v]/sum, oh=(v==tgt[t])?1.0:0.0; gLG[t*X_VOCAB+v]=P((p-oh)/X_SEQ); }
    }
    double gXF[X_SEQ*X_D]; bf_linear_bwd(gLG, m.WHEAD, a.XF, gXF, g.WHEAD.data(), X_SEQ, X_D, X_VOCAB);
    double gRP[X_SEQ*X_D]; bf_ln_bwd(gXF, a.RP, a.LNFG, a.lnfmean, a.lnfs, gRP, X_SEQ);
    // (1) FFN: RP=M+FO ; FO=WDN(HD) ; HD=qact(relu(WUP(X2)))
    double gM[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];     // residual
    double gHD[X_SEQ*X_FFN]; bf_linear_bwd(gRP, m.WDN, a.HD, gHD, g.WDN.data(), X_SEQ, X_FFN, X_D);
    for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;            // relu mask (qact STE = identity)
    double gX2[X_SEQ*X_D]; bf_linear_bwd(gHD, m.WUP, a.X2, gX2, g.WUP.data(), X_SEQ, X_D, X_FFN);
    // (2) LN2 over M -> add into gM
    double gM2[X_SEQ*X_D]; bf_ln_bwd(gX2, a.M, a.LN2G, a.ln2mean, a.ln2s, gM2, X_SEQ);
    for(int i=0;i<X_SEQ*X_D;i++) gM[i]=P(gM[i]+gM2[i]);
    // (3) attention: M=R+AO ; AO=WO(ATTN) ; ATTN=qact(P@V) ; P=softmax(QK/sqrt(DH))
    double gR[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gR[i]=gM[i];     // residual
    double gATTN[X_SEQ*X_D]; bf_linear_bwd(gM, m.WO, a.ATTN, gATTN, g.WO.data(), X_SEQ, X_D, X_D);  // qact STE identity
    double gQ[X_SEQ*X_D], gK[X_SEQ*X_D], gV[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++){gQ[i]=0;gK[i]=0;gV[i]=0;}
    double scale = 1.0/std::sqrt((double)X_DH);
    for (int h=0; h<X_H; h++) for (int i=0; i<X_SEQ; i++) {
        const double* p=&a.Praw[(h*X_SEQ+i)*X_SEQ];
        double gp[X_SEQ];
        for (int j=0;j<=i;j++){ double s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*a.V[j*X_D+h*X_DH+d]; gp[j]=s;
            for(int d=0;d<X_DH;d++) gV[j*X_D+h*X_DH+d]+=P(p[j]*gATTN[i*X_D+h*X_DH+d]); }
        double ex=0; for(int j=0;j<=i;j++) ex+=p[j]*gp[j];             // sum_k p_k gp_k
        for (int j=0;j<=i;j++){ double gs=P(p[j]*(gp[j]-ex))*scale;     // analytic softmax Jacobian * 1/sqrt(DH)
            for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+=gs*a.K[j*X_D+h*X_DH+d]; gK[j*X_D+h*X_DH+d]+=gs*a.Q[i*X_D+h*X_DH+d]; } }
    }
    for(int i=0;i<X_SEQ*X_D;i++){ gQ[i]=P(gQ[i]); gK[i]=P(gK[i]); gV[i]=P(gV[i]); }
    for(int i=0;i<X_SEQ*X_D;i++){ g.dbg_gATTN[i]=gATTN[i]; g.dbg_gV[i]=gV[i]; g.dbg_gQ[i]=gQ[i]; g.dbg_gK[i]=gK[i]; }
    // route Q/K/V back to gX1 (+ wgrads). qact STE = identity on Q/K/V.
    double gX1[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=0; double tmp[X_SEQ*X_D];
    bf_linear_bwd(gQ, m.WQ, a.X1, tmp, g.WQ.data(), X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    bf_linear_bwd(gK, m.WK, a.X1, tmp, g.WK.data(), X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    bf_linear_bwd(gV, m.WV, a.X1, tmp, g.WV.data(), X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i];
    for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=P(gX1[i]);
    // (4) LN1 over R -> add into gR
    double gR1[X_SEQ*X_D]; bf_ln_bwd(gX1, a.R, a.LN1G, a.ln1mean, a.ln1s, gR1, X_SEQ);
    for(int i=0;i<X_SEQ*X_D;i++) gR[i]=P(gR[i]+gR1[i]);
    for(int i=0;i<X_SEQ*X_D;i++) g.gR[i]=gR[i];
    // (6) scatter gR -> EMB + POS
    for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1;
        for(int d=0;d<X_D;d++){ g.EMB[v*X_D+d]+=gR[t*X_D+d]; g.POS[t*X_D+d]+=gR[t*X_D+d]; } }
}

// =====================================================================================================
// FLOAT dyadic-Adam — IDENTICAL to the char-LM dyadic_adam (scripts/pc_xformer_charlm.py):
//   m = 0.875 m + 0.125 g ;  v = 0.875 v + 0.125 |g| ;  step = sign(m) * 2^floor(log2(|m|/v)) ;  p -= lr*step.
// Operates on the FLOAT master directly (char-LM scale) — no int16 round-trip in the trainer (the int16
// deploy format is round(master*2^WFRAC) downstream, dyad-equivalent). lr follows the char-LM warmup.
// =====================================================================================================
static void bf_adam_one(double *Wf, const double *gW, BFW &b, int n, double lr) {
    auto &m=b.m[Wf]; auto &v=b.v[Wf];
    if (m.empty()){ m.assign(n,0); v.assign(n,0); }
    for (int i=0;i<n;i++){
        double gv=gW[i], ag=gv<0?-gv:gv;
        m[i]=0.875*m[i]+0.125*gv; v[i]=0.875*v[i]+0.125*ag;
        double am=m[i]<0?-m[i]:m[i], ratio=am/(v[i]+1e-9);
        double step=ratio>0?std::ldexp(1.0,(int)std::floor(std::log2(ratio))):0.0;
        double sgn=m[i]>0?1.0:(m[i]<0?-1.0:0.0);
        Wf[i]-=lr*sgn*step;
    }
}
// init the FLOAT master at the char-LM's natural scale: Linear ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) (the torch
// nn.Linear default), Embedding ~ N(0,1), LN gamma=1/beta=0 (frozen). srng = simple LCG (deterministic per seed).
static unsigned g_bf_rng = 1;
static inline double bf_u(){ g_bf_rng=g_bf_rng*1103515245u+12345u; return ((g_bf_rng>>8)&0xffffff)/(double)0x1000000; } // [0,1)
static inline double bf_n(){ double u=0; for(int k=0;k<12;k++) u+=bf_u(); return u-6.0; }   // ~N(0,1)
static void bf_init(BFM &m, unsigned seed){
    g_bf_rng = seed*2654435761u + 1u;
    auto uni=[&](double*W,int n,int fan){ double a=1.0/std::sqrt((double)fan); for(int i=0;i<n;i++) W[i]=(bf_u()*2-1)*a; };
    uni(m.WQ,X_D*X_D,X_D); uni(m.WK,X_D*X_D,X_D); uni(m.WV,X_D*X_D,X_D); uni(m.WO,X_D*X_D,X_D);
    uni(m.WUP,X_FFN*X_D,X_D); uni(m.WDN,X_D*X_FFN,X_FFN); uni(m.WHEAD,X_VOCAB*X_D,X_D);
    for(int i=0;i<X_VOCAB*X_D;i++) m.EMB[i]=bf_n(); for(int i=0;i<X_SEQ*X_D;i++) m.POS[i]=bf_n();
    for(int d=0;d<X_D;d++){ m.LN1G[d]=1;m.LN2G[d]=1;m.LNFG[d]=1; m.LN1B[d]=0;m.LN2B[d]=0;m.LNFB[d]=0; }
}
static void bf_step(BFM &mw, BFG &g, BFW &b, double lr) {
    bf_adam_one(mw.WQ,g.WQ.data(),b,X_D*X_D,lr);   bf_adam_one(mw.WK,g.WK.data(),b,X_D*X_D,lr);
    bf_adam_one(mw.WV,g.WV.data(),b,X_D*X_D,lr);   bf_adam_one(mw.WO,g.WO.data(),b,X_D*X_D,lr);
    bf_adam_one(mw.WUP,g.WUP.data(),b,X_FFN*X_D,lr); bf_adam_one(mw.WDN,g.WDN.data(),b,X_D*X_FFN,lr);
    bf_adam_one(mw.WHEAD,g.WHEAD.data(),b,X_VOCAB*X_D,lr);
    bf_adam_one(mw.EMB,g.EMB.data(),b,X_VOCAB*X_D,lr); bf_adam_one(mw.POS,g.POS.data(),b,X_SEQ*X_D,lr);
}

#endif
