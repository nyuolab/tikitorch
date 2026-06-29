// forge_pc_xformer_golden_block.hpp — the full int64 block fwd+bwd+dyadic-Adam reference (one transformer
// block, OUTPUT role = with head+CE). Standalone/g++-runnable; the HLS kernel transcribes these ops.
#ifndef FORGE_PC_XFORMER_GOLDEN_BLOCK_HPP
#define FORGE_PC_XFORMER_GOLDEN_BLOCK_HPP
#include "forge_pc_xformer_golden.hpp"
#include <map>
#include <vector>   // GFOpt uses std::vector; Vitis CLANG csim needs the explicit include (g++ pulled it in transitively)
#include <cmath>

// All weights (int16 masters) + Adam m,v (int32) for one block + head.
struct GW {
    i64 WQ[X_D*X_D], WK[X_D*X_D], WV[X_D*X_D], WO[X_D*X_D], WUP[X_FFN*X_D], WDN[X_D*X_FFN], WHEAD[X_VOCAB*X_D];
    i64 LN1G[X_D], LN1B[X_D], LN2G[X_D], LN2B[X_D], LNFG[X_D], LNFB[X_D];
    i64 EMB[X_VOCAB*X_D], POS[X_SEQ*X_D];
    // dyadic-Adam state (one m,v per weight), kept per-matrix for clarity
    i64 mWQ[X_D*X_D],vWQ[X_D*X_D], mWK[X_D*X_D],vWK[X_D*X_D], mWV[X_D*X_D],vWV[X_D*X_D], mWO[X_D*X_D],vWO[X_D*X_D];
    i64 mUP[X_FFN*X_D],vUP[X_FFN*X_D], mDN[X_D*X_FFN],vDN[X_D*X_FFN], mHD[X_VOCAB*X_D],vHD[X_VOCAB*X_D];
};
// cached forward activations + the LUT
struct GA {
    i64 R[X_SEQ*X_D], X1[X_SEQ*X_D], Q[X_SEQ*X_D], K[X_SEQ*X_D], V[X_SEQ*X_D], ATTN[X_SEQ*X_D];
    i64 M[X_SEQ*X_D], X2[X_SEQ*X_D], HD[X_SEQ*X_FFN], RMASK[X_SEQ*X_FFN], RP[X_SEQ*X_D];
    i64 XF[X_SEQ*X_D], LG[X_SEQ*X_VOCAB], GLOG[X_SEQ*X_VOCAB];
    i64 P[X_H*X_SEQ*X_SEQ];                 // Q16 softmax probs (post-normalize is via shn at blend; store raw p + shn)
    int Pshn[X_H*X_SEQ];                    // per (head,query) normalize shift
    int ln1s[X_SEQ*2], ln2s[X_SEQ*2], lnfs[X_SEQ*2];
};

static i64 G_LUT[X_LUT_N];
static void g_lut_init() { for (int g = 0; g < X_LUT_N; g++) { double v = 65536.0; for (int k=0;k<g;k++) v *= 0.917004; G_LUT[g] = (i64)(v + 0.5); } } // ~2^(-g/8)
// PRECISE tropical softmax exp: exp(-d) = G_EXP[frac(u)] >> int(u), u = d*log2(e) via shift-add (x1.4375). Right
// temperature (raw gap d) + a fine mantissa LUT -> matches the float softmax (0.341 vs 0.347), DSP_MULT=0.
static i64 G_EXP[1 << X_EXPFRAC];
static void g_exp_init() { for (int f = 0; f < (1 << X_EXPFRAC); f++) G_EXP[f] = (i64)(65536.0 * pow(2.0, -(double)f/(1<<X_EXPFRAC)) + 0.5); }
// PRECISE exp with a per-row ADAPTIVE temperature: exp(-d/2^smsh). The smsh (= bitlen(spread)-SMBITS) handles any
// score scale; the shift is applied AFTER scaling into the mantissa so the sub-bucket bits SURVIVE (the coarse
// LUT lost them: gap=(max-s)>>smsh integer). Combines adaptive range + fine precision. exp = G_EXP[frac]>>int.
static inline i64 g_exp_q16(i64 d, int smsh) {
    if (d < 0) d = 0;
    i64 base = d << X_EXPFRAC;
    i64 u = (base + (base>>2) + (base>>3) + (base>>4)) >> smsh;    // (d * 1.4375 / 2^smsh) in Q(EXPFRAC)
    int intp = (int)(u >> X_EXPFRAC); if (intp >= 31) return 0;
    return G_EXP[(int)(u & ((1<<X_EXPFRAC)-1))] >> intp;
}

static int g_attn_dot=-1;   // 1 = warm DOT-product attention (#3); 0 = L1 (mult-free, capped). Set in g_forward, read in g_backward.
// ---------------- FORWARD (OUTPUT role): tokens -> logits + grad_logit, caches everything ----------------
static i64 g_forward(GW &w, GA &a, const int *tok, const int *tgt) {
    for (int t = 0; t < X_SEQ; t++) for (int d = 0; d < X_D; d++) {
        int v = tok[t]; if (v < 0) v = 0; if (v >= X_VOCAB) v = X_VOCAB - 1;
        a.R[t*X_D+d] = w.EMB[v*X_D+d] + w.POS[t*X_D+d];
    }
    g_layernorm(a.R, w.LN1G, w.LN1B, a.X1, a.ln1s, X_SEQ);
    static int g_akb=-1; if(g_akb<0){const char*e=getenv("AK_BITS");g_akb=e?atoi(e):X_ABITS;}
    g_matmul(a.X1, w.WQ, a.Q, X_SEQ, X_D, X_D); g_rescale(a.Q, X_SEQ*X_D, g_akb);   // Q/K scale sets the attention score spread
    g_matmul(a.X1, w.WK, a.K, X_SEQ, X_D, X_D); g_rescale(a.K, X_SEQ*X_D, g_akb);
    g_matmul(a.X1, w.WV, a.V, X_SEQ, X_D, X_D); g_rescale(a.V, X_SEQ*X_D, X_ABITS);
    if(g_attn_dot<0){const char*e=getenv("ATTN_DOT");g_attn_dot=e?atoi(e):0;}  // 1 = warm DOT-product attn (#3, the one multiply); 0 = L1 (mult-free, capped)
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) {
        i64 s[X_SEQ]; i64 rowmax = -(((i64)1)<<40); i64 rowmin = (((i64)1)<<40);
        for (int j = 0; j <= i; j++) { i64 sc;
            if (g_attn_dot) { i64 dot=0; for (int d=0;d<X_DH;d++) dot += a.Q[i*X_D+h*X_DH+d]*a.K[j*X_D+h*X_DH+d]; sc = dot>>X_ABITS; }  // DOT score (the allowed multiply), scaled
            else { i64 l1 = 0; for (int d = 0; d < X_DH; d++) { i64 df = a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; l1 += df<0?-df:df; } sc = -l1; }
            s[j] = sc; if (s[j]>rowmax) rowmax=s[j]; if (s[j]<rowmin) rowmin=s[j]; }
        static int g_asm=-1,g_afix=-2; if(g_asm<0){const char*e=getenv("A_SMBITS");g_asm=e?atoi(e):X_SMBITS;const char*f=getenv("A_FIXTEMP");g_afix=f?atoi(f):-1;}
        int smsh = (g_afix>=0) ? g_afix : g_normshift(rowmax-rowmin, g_asm);  // FIXED temp (preserves close-key structure) or adaptive
        i64 rowsum = 0;
        for (int j = 0; j <= i; j++) { i64 p=g_exp_q16(rowmax-s[j], smsh); a.P[(h*X_SEQ+i)*X_SEQ+j]=p; rowsum+=p; }  // PRECISE exp softmax
        int shn = g_bitlen(rowsum); a.Pshn[h*X_SEQ+i]=shn;
        for (int d = 0; d < X_DH; d++) { i64 blend=0; for (int j=0;j<=i;j++) blend += a.P[(h*X_SEQ+i)*X_SEQ+j]*a.V[j*X_D+h*X_DH+d]; a.ATTN[i*X_D+h*X_DH+d]=g_floorshr(blend,shn); }
    }
    i64 AO[X_SEQ*X_D]; g_matmul(a.ATTN, w.WO, AO, X_SEQ, X_D, X_D); g_rescale(AO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.M[i] = a.R[i] + AO[i];
    g_layernorm(a.M, w.LN2G, w.LN2B, a.X2, a.ln2s, X_SEQ);
    g_matmul(a.X2, w.WUP, a.HD, X_SEQ, X_D, X_FFN);
    for (int i = 0; i < X_SEQ*X_FFN; i++) { int pos=a.HD[i]>0; a.RMASK[i]=pos?1:0; if(!pos)a.HD[i]=0; }
    if (g_mxbs <= 0) g_rescale(a.HD, X_SEQ*X_FFN, X_ABITS);   // per-tensor crush (baseline)
    else g_block_round(a.HD, X_SEQ, X_FFN, X_ABITS, g_mxbs);   // MX: per-block round, keep WIDE (fwd matmul + wgrad share it)
    i64 FO[X_SEQ*X_D]; g_matmul(a.HD, w.WDN, FO, X_SEQ, X_FFN, X_D); g_rescale(FO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.RP[i] = a.M[i] + FO[i];
    g_layernorm(a.RP, w.LNFG, w.LNFB, a.XF, a.lnfs, X_SEQ);
    g_matmul(a.XF, w.WHEAD, a.LG, X_SEQ, X_D, X_VOCAB);
    // FIX#3 (head-logit bound, default ON ~ C_SMBITS=6): the head logits are the ONE matmul output the golden does
    // NOT max-shift. With dyadic head weights + an unbounded residual stream the logit SPREAD diverges (~2^21 @
    // d3/h4), saturating the CE softmax -> GLOG goes near-one-hot, the calibration gradient dies. A single mult-free
    // max-shift to ~HEAD_RESCALE bits (g_rescale = a per-tensor arithmetic right-shift, the SAME op used on every
    // other matmul output, DSP=0, kernel-implementable) bounds the spread so the precise-exp CE keeps a graded
    // signal. Default 6 (= C_SMBITS, matches the adaptive CE shift target). -1 disables = the old canonical forward.
    { static int hr=-2; if(hr<-1){const char*e=getenv("HEAD_RESCALE"); hr=e?atoi(e):6;} if(hr>=0) g_rescale(a.LG, X_SEQ*X_VOCAB, hr); }
    // CE: softmax via LUT -> grad_logit = softmax - onehot (Q16); also accumulate an integer surrogate loss
    i64 loss = 0;
    for (int t = 0; t < X_SEQ; t++) {
        i64 mx=-(((i64)1)<<40); i64 mn=(((i64)1)<<40); for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx)mx=lv; if(lv<mn)mn=lv; }
        static int g_csm=-1; if(g_csm<0){const char*e=getenv("C_SMBITS");g_csm=e?atoi(e):X_SMBITS;}
        int smsh = g_normshift(mx-mn, g_csm);               // adaptive CE temperature
        i64 sum=0; i64 pr[X_VOCAB];
        for(int v=0;v<X_VOCAB;v++){ pr[v]=g_exp_q16(mx-a.LG[t*X_VOCAB+v], smsh); sum+=pr[v]; }   // PRECISE exp softmax
        int shn=g_bitlen(sum);
        for(int v=0;v<X_VOCAB;v++){ i64 pq=g_floorshr(pr[v]<<X_AT_PFRAC,shn); i64 oh=(v==tgt[t])?((i64)1<<X_AT_PFRAC):0; a.GLOG[t*X_VOCAB+v]=pq-oh; }
        // surrogate loss = the target's raw gap (monotone training signal)
        int gtt=(int)(mx-a.LG[t*X_VOCAB+tgt[t]]); loss += gtt;
    }
    return loss;
}

// q(.)^T q(.) mult-free weight-grad accumulate: gW[o*I+i] += sum_t dyadval(go[t*O+o]) * dyadval(x[t*I+i])
// WGRAD_MODE (env, for ablation): 0=dyadic-both (kernel), 1=full mul (precision ceiling), 2=dyad-go only, 3=dyad-x only
static int g_wgrad_mode = -1;
static void g_wgrad(const i64 *go, const i64 *x, i64 *gW, int M, int I, int O) {
    if (g_wgrad_mode < 0) { const char*e=getenv("WGRAD_MODE"); g_wgrad_mode = e?atoi(e):0; }
    for (int o=0;o<O;o++) for (int i=0;i<I;i++) { i64 acc=0;
        for (int t=0;t<M;t++) {
            i64 a = (g_wgrad_mode==1||g_wgrad_mode==3) ? go[t*O+o] : g_dyadval(go[t*O+o]);
            i64 b = (g_wgrad_mode==1||g_wgrad_mode==2) ? x[t*I+i]  : g_dyadval(x[t*I+i]);
            acc += a*b;
        }
        gW[o*I+i]+=acc; }
}
// route gIn[t*I+i] = sum_o dyadval(W[o*I+i]) * gOut[t*O+o]   (transpose-free, W dyadic)
static void g_route(const i64 *gout, const i64 *W, i64 *gin, int M, int I, int O) {
    for (int t=0;t<M;t++) for (int i=0;i<I;i++){ i64 acc=0; for(int o=0;o<O;o++) acc += g_wapply(W[o*I+i], gout[t*O+o]); gin[t*I+i]=acc; }
}
// LN backward: FULL LayerNorm gradient (mult-free). g_x = (1/s)(g_xhat - mean(g_xhat) - x_hat*mean(g_xhat*x_hat)).
// The first two terms are mean-centering (straight-through). The third is the VARIANCE-COUPLING term (ablation:
// recovers ~0.07 acc, the dominant LN gap) — it removes the x_hat-direction component of the gradient that LN's
// forward makes invariant. Computed with the SAME mult-free dyadic dot the gamma-grad uses (dyadval*dyadval =
// signed-exponent add + shift, DSP_MULT=0). chat == x_hat (already needed for the gamma grad).
static void g_ln_bwd(const i64 *gy, const i64 *xin, const i64 *gcode, const int *stats, i64 *gin, i64 *gG, i64 *gB, int M) {
    for (int t=0;t<M;t++){
        int e=stats[2*t], nq=stats[2*t+1];
        i64 gc[X_D], chat[X_D]; i64 msum=0, vc=0;
        i64 xs=0; for(int d=0;d<X_D;d++) xs+=xin[t*X_D+d]; i64 xmean=xs>>X_LOG2_D;
        for(int d=0;d<X_D;d++){
            int gs,ge,gnz; g_dyad(gcode[d],gs,ge,gnz); i64 gch = gnz?(gy[t*X_D+d]<<ge):0; if(gnz&&gs)gch=-gch;
            gc[d]=g_applyinvq(gch,e,nq); msum+=gc[d]; gB[d]+=gy[t*X_D+d];
            chat[d]=g_applyinvq(xin[t*X_D+d]-xmean,e,nq);
            gG[d]+=g_dyadval(gy[t*X_D+d])*g_dyadval(chat[d]);
            vc += g_dyadval(gc[d])*g_dyadval(chat[d]);            // variance-coupling dot: sum_d g_xhat_hat * x_hat
        }
        i64 mean=msum>>X_LOG2_D; i64 vcm=vc>>X_LOG2_D;            // mean(g_xhat * x_hat)
        for(int d=0;d<X_D;d++) gin[t*X_D+d]+= gc[d]-mean-(i64)(g_dyadval(chat[d])*g_dyadval(vcm)); // full LN bwd
    }
}

// grad accumulators (zeroed each step)
struct GG {
    i64 WQ[X_D*X_D],WK[X_D*X_D],WV[X_D*X_D],WO[X_D*X_D],WUP[X_FFN*X_D],WDN[X_D*X_FFN],WHEAD[X_VOCAB*X_D];
    i64 LN1G[X_D],LN1B[X_D],LN2G[X_D],LN2B[X_D],LNFG[X_D],LNFB[X_D];
    i64 EMB[X_VOCAB*X_D],POS[X_SEQ*X_D];
    i64 gR[X_SEQ*X_D];      // final input-gradient (the OUTPUT/HIDDEN role's EOUT-before-rescale) — for the tb
};

// ---------------- BACKWARD (OUTPUT role) ----------------
static void g_backward(GW &w, GA &a, GG &g, const int *tok, bool zero=true) {
    if (zero) memset(&g, 0, sizeof(GG));    // zero=false accumulates grads across sequences (effective batch)
    // (0) seed: head bwd of grad_logit -> gXF ; gWHEAD ; LNF bwd -> gRP
    i64 gXF[X_SEQ*X_D]; g_route(a.GLOG, w.WHEAD, gXF, X_SEQ, X_D, X_VOCAB);
    g_wgrad(a.GLOG, a.XF, g.WHEAD, X_SEQ, X_D, X_VOCAB);
    i64 gRP[X_SEQ*X_D]; memset(gRP,0,sizeof(gRP));
    g_ln_bwd(gXF, a.RP, w.LNFG, a.lnfs, gRP, g.LNFG, g.LNFB, X_SEQ);
    // (1) FFN bwd: gM += gRP ; gFO = gRP ; gHD = route(gFO)*RMASK ; gX2 = route(gHD)
    i64 gM[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
    i64 gHD[X_SEQ*X_FFN]; g_route(gRP, w.WDN, gHD, X_SEQ, X_FFN, X_D);
    for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;
    g_wgrad(gRP, a.HD, g.WDN, X_SEQ, X_FFN, X_D);
    i64 gX2[X_SEQ*X_D]; g_route(gHD, w.WUP, gX2, X_SEQ, X_D, X_FFN);
    g_wgrad(gHD, a.X2, g.WUP, X_SEQ, X_D, X_FFN);
    // (2) LN2 bwd over input M -> add into gM
    g_ln_bwd(gX2, a.M, w.LN2G, a.ln2s, gM, g.LN2G, g.LN2B, X_SEQ);
    // (3) attention bwd: gR = gM ; gAO = gM ; gATTN = route(gAO,WO) ; gWO
    i64 gR[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gR[i]=gM[i];
    i64 gATTN[X_SEQ*X_D]; g_route(gM, w.WO, gATTN, X_SEQ, X_D, X_D);
    g_wgrad(gM, a.ATTN, g.WO, X_SEQ, X_D, X_D);
    i64 gQ[X_SEQ*X_D], gK[X_SEQ*X_D], gV[X_SEQ*X_D];
    memset(gQ,0,sizeof(gQ)); memset(gK,0,sizeof(gK)); memset(gV,0,sizeof(gV));
    for(int h=0;h<X_H;h++) for(int i=0;i<X_SEQ;i++){
        int shn=a.Pshn[h*X_SEQ+i]; i64 pn[X_SEQ];
        for(int j=0;j<=i;j++) pn[j]=g_floorshr(a.P[(h*X_SEQ+i)*X_SEQ+j]<<X_AT_PFRAC, shn);  // Q16 normalized prob
        i64 gp[X_SEQ]; for(int j=0;j<=i;j++){ i64 s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*a.V[j*X_D+h*X_DH+d]; gp[j]=s; }  // MUL
        // NOTE: real ap_int computes pn*gp at FULL width (128b RType), THEN >>PFRAC, THEN truncates to acc_t<64>
        // on assignment. Mirror that with __int128 so the golden == the HW kernel (a plain i64 product wraps at
        // 64 BEFORE the shift -> diverges for products > 2^63, which random data hits).
        i64 ex=0; for(int j=0;j<=i;j++) ex += (i64)(((__int128)pn[j]*gp[j]) >> X_AT_PFRAC);  // MUL (128b)
        for(int j=0;j<=i;j++){
            i64 gs=(i64)(((__int128)pn[j]*(gp[j]-ex)) >> X_AT_PFRAC);                // MUL (softmax bwd, 128b)
            // FIX#1 (int64-overflow at depth): the DOT-attention gs*K / gs*Q products wrap at 2^63 in HIDDEN blocks
            // (multi-block, no inter-block grad rescale): max|gs|~3.97e18, max|gs*K|~6.47e20 >> 2^63=9.22e18 ->
            // i64-wrap sign-corrupts gQ/gK (signagree 0.55/0.71 vs the correct gradient). Promote to __int128 BEFORE
            // the >>X_ABITS truncation, EXACTLY like the pn*gp / pn*(gp-ex) / pn*gATTN lines already do above. This is
            // BIT-IDENTICAL to the old plain-i64 product whenever it fits 63 bits (the single-block / L1 production
            // regime never trips it), so single-block bit-exactness is preserved. Mult-free unaffected: the L1 path
            // (else-branch, the production kernel) has no gs*K multiply at all.
            if (g_attn_dot) { for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.K[j*X_D+h*X_DH+d])>>X_ABITS); gK[j*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.Q[i*X_D+h*X_DH+d])>>X_ABITS); } }  // DOT grad (s=Q.K>>ABITS -> dQ=gs.K>>ABITS)
            else for(int d=0;d<X_DH;d++){ i64 df=a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; int sg=df>0?1:(df<0?-1:0);
                gQ[i*X_D+h*X_DH+d]+= -gs*sg; gK[j*X_D+h*X_DH+d]+= gs*sg; }                 // L1 subgradient (mult-free)
            for(int d=0;d<X_DH;d++) gV[j*X_D+h*X_DH+d]+= (i64)(((__int128)pn[j]*gATTN[i*X_D+h*X_DH+d]) >> X_AT_PFRAC); // MUL (128b)
        }
    }
    // route Q/K/V grads back to gX1 (+ wgrads); accumulate into gX1
    i64 gX1[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gX1[i]=0;
    i64 tmp[X_SEQ*X_D];
    g_route(gQ, w.WQ, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; g_wgrad(gQ, a.X1, g.WQ, X_SEQ, X_D, X_D);
    g_route(gK, w.WK, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; g_wgrad(gK, a.X1, g.WK, X_SEQ, X_D, X_D);
    g_route(gV, w.WV, tmp, X_SEQ, X_D, X_D); for(int i=0;i<X_SEQ*X_D;i++) gX1[i]+=tmp[i]; g_wgrad(gV, a.X1, g.WV, X_SEQ, X_D, X_D);
    // (4) LN1 bwd over input R -> add into gR
    g_ln_bwd(gX1, a.R, w.LN1G, a.ln1s, gR, g.LN1G, g.LN1B, X_SEQ);
    for (int i=0;i<X_SEQ*X_D;i++) g.gR[i]=gR[i];                  // expose the final input-gradient for the tb
    // (6) INPUT-role grads: scatter gR into embed + pos by token id (OUTPUT role would emit EOUT=rescale(gR))
    for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; for(int d=0;d<X_D;d++){ g.EMB[v*X_D+d]+=gR[t*X_D+d]; g.POS[t*X_D+d]+=gR[t*X_D+d]; } }
}

// ---------------- mult-free dyadic-Adam (one matrix) ----------------
static inline int g_dyad_exp(i64 v){ i64 a=v<0?-v:v; int m=0; for(int b=30;b>=1;b--) if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1)&&m<30)m++; return m; }
static int g_lrboost=-1, g_wdsh=-1, g_sround=-1; static unsigned g_srng=0x12345u;
// ISOLATION knob: FLOAT_OPT=1 runs a FLOAT dyadic-Adam (float master, continuous step) while everything else
// stays integer (int softmax, int cascade backward, int activations). If the golden then reaches ~0.30, the
// INTEGER optimizer step is confirmed as the 0.20->0.30 gap. FLR = float learning rate, FWD = float weight decay.
struct GFOpt { std::vector<double> Wf, m, v; };
static std::map<const i64*, GFOpt> g_gfopt; static int g_float_opt=-1; static double g_flr=0.5, g_fwd=0.0156;
static void g_adam(i64 *W, const i64 *gW, i64 *m, i64 *v, int n) {
    if(g_lrboost<0){ const char*e=getenv("LR_BOOST"); g_lrboost=e?atoi(e):0; const char*w=getenv("WDSH"); g_wdsh=w?atoi(w):X_WD_SHIFT; const char*s=getenv("SROUND"); g_sround=s?atoi(s):0;
        const char*fo=getenv("FLOAT_OPT"); g_float_opt=fo?atoi(fo):0; const char*fl=getenv("FLR"); if(fl)g_flr=atof(fl); const char*fw=getenv("FWD"); if(fw)g_fwd=atof(fw); }
    if(g_float_opt){
        GFOpt &s=g_gfopt[W]; if(s.Wf.empty()){ s.Wf.assign(n,0); s.m.assign(n,0); s.v.assign(n,0); for(int i=0;i<n;i++) s.Wf[i]=(double)W[i]; }
        for(int i=0;i<n;i++){ double g=(double)gW[i];
            s.m[i]=0.875*s.m[i]+0.125*g; s.v[i]=0.875*s.v[i]+0.125*fabs(g);
            double scale=pow(2.0, floor(log2(fabs(s.m[i])/(s.v[i]+1e-9)+1e-12)));   // float dyadic-Adam: sign * pow2(|m|/v)
            double sgn=s.m[i]>0?1.0:(s.m[i]<0?-1.0:0.0);
            s.Wf[i]-= g_flr*sgn*scale; s.Wf[i]-= s.Wf[i]*g_fwd;
            double w=s.Wf[i]; if(w>32767)w=32767; if(w<-32767)w=-32767; W[i]=(i64)llround(w); }
        return;
    }
    for(int i=0;i<n;i++){ i64 g=gW[i], ag=g<0?-g:g;
        m[i]=m[i]-(m[i]>>X_ADAM_BSH)+(g>>X_ADAM_BSH); v[i]=v[i]-(v[i]>>X_ADAM_BSH)+(ag>>X_ADAM_BSH);
        int se=g_dyad_exp(v[i])-g_lrboost; if(se<0)se=0; int sh=se+X_LR_SHIFT; i64 dw;
        if(!g_sround || sh<=0) dw=m[i]>>(sh>0?sh:0);
        else { // STOCHASTIC ROUNDING: dw = floor(m/2^sh) + 1 w.p. (m mod 2^sh)/2^sh -> sub-integer master resolution
            i64 q=m[i]>>sh, rem=m[i]-(q<<sh); int sb=sh<30?sh:30;
            g_srng=g_srng*1103515245u+12345u; i64 thr=(g_srng>>1)&((1LL<<sb)-1);
            dw=q+(((rem>>(sh-sb))>thr)?1:0); }
        i64 dec = (W[i]>=0) ? (W[i]>>g_wdsh) : -((-W[i])>>g_wdsh);   // round-toward-0 decay (symmetric)
        i64 nw=W[i]-dw-dec;
        if(nw>32767)nw=32767; if(nw<-32767)nw=-32767; W[i]=nw;             // int16 master clamp
    }
}
static void g_step(GW &w, GG &g) {
    g_adam(w.WQ,g.WQ,w.mWQ,w.vWQ,X_D*X_D); g_adam(w.WK,g.WK,w.mWK,w.vWK,X_D*X_D);
    g_adam(w.WV,g.WV,w.mWV,w.vWV,X_D*X_D); g_adam(w.WO,g.WO,w.mWO,w.vWO,X_D*X_D);
    g_adam(w.WUP,g.WUP,w.mUP,w.vUP,X_FFN*X_D); g_adam(w.WDN,g.WDN,w.mDN,w.vDN,X_D*X_FFN);
    g_adam(w.WHEAD,g.WHEAD,w.mHD,w.vHD,X_VOCAB*X_D);
    // LN gammas/betas: FROZEN at init. SignSGD ±1 on the gamma drifts it monotonically -> the final-LN stops
    // bounding the activations -> logits explode. The LN scale is a fixed normalizer here, not learned.
    // betas frozen at 0 too. Embed/pos: SignSGD ±1 (these are the learnable input table).
#ifndef X_FREEZE_EMB
    for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]-=(g.EMB[i]>0)-(g.EMB[i]<0);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]-=(g.POS[i]>0)-(g.POS[i]<0);
#endif
}

#endif

