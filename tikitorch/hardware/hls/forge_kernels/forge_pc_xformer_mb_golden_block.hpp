// mb_golden_block.hpp — MULTI-BLOCK cascade golden for the confirmation gate.
// =====================================================================================================
// Stacks L transformer blocks (L-1 HIDDEN + 1 OUTPUT) sharing the SAME per-block arithmetic as
// forge_pc_xformer_golden_block.hpp (and its MP fork). The single-block golden fuses embed + block +
// head + CE in g_forward; the confirmation gate needs MATCHED CAPACITY (L=3 blocks, H=4 heads), so we
// decompose:
//   INPUT  : embed/pos  -> R0
//   HIDDEN : block(Rin) -> Rout         (LN1, attn, +res, LN2, FFN, +res)         [L-1 of these]
//   OUTPUT : block(Rin) -> Rout, then final-LN + head + CE -> logits + grad_logit  [the last block]
//
// The OUTPUT block forward+backward is EXACTLY g_forward / mp_backward applied to its OWN weights+acts,
// EXCEPT the embed is pulled out (so the residual at OUTPUT's input is the previous block's output, not
// the token embed) and the EMB/POS scatter is moved to the INPUT stage. We add the HIDDEN block fwd/bwd
// (block only) reusing every op verbatim (g_layernorm, g_matmul, g_route, g_ln_bwd, the attention
// fwd/bwd, mp_wgrad). The cascade backward chains gRout(block L) -> gRin(block L) = gRout(block L-1).
//
// MP_OPT (int/fix16/fix24/dbl) + ATTN_DOT + the precise exp all flow through unchanged — the HIDDEN
// block fwd is a copy of g_forward's block portion, the HIDDEN block bwd a copy of mp_backward's body
// from the FFN seed onward. NOTHING in the canonical golden is altered.
// =====================================================================================================
#ifndef MB_GOLDEN_BLOCK_HPP
#define MB_GOLDEN_BLOCK_HPP
#include "forge_pc_xformer_mp_golden_block.hpp"

// per-block weights (no EMB/POS — those live at INPUT; LNF/WHEAD only used by OUTPUT block but kept in GW)
// We reuse GW per block (it carries EMB/POS/LNF/WHEAD slots; only OUTPUT touches LNF/WHEAD, only INPUT
// touches EMB/POS). GA per block caches that block's activations. GG per block its grads.

// ---- HIDDEN block forward: Rin[SEQ*D] -> Rout[SEQ*D], caches a.* (block portion of g_forward) ----
static void mb_forward_hidden(GW &w, GA &a, const i64 *Rin) {
    for (int i = 0; i < X_SEQ*X_D; i++) a.R[i] = Rin[i];
    g_layernorm(a.R, w.LN1G, w.LN1B, a.X1, a.ln1s, X_SEQ);
    static int g_akb=-1; if(g_akb<0){const char*e=getenv("AK_BITS");g_akb=e?atoi(e):X_ABITS;}
    g_matmul(a.X1, w.WQ, a.Q, X_SEQ, X_D, X_D); g_rescale(a.Q, X_SEQ*X_D, g_akb);
    g_matmul(a.X1, w.WK, a.K, X_SEQ, X_D, X_D); g_rescale(a.K, X_SEQ*X_D, g_akb);
    g_matmul(a.X1, w.WV, a.V, X_SEQ, X_D, X_D); g_rescale(a.V, X_SEQ*X_D, X_ABITS);
    if(g_attn_dot<0){const char*e=getenv("ATTN_DOT");g_attn_dot=e?atoi(e):0;}
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) {
        i64 s[X_SEQ]; i64 rowmax = -(((i64)1)<<40); i64 rowmin = (((i64)1)<<40);
        for (int j = 0; j <= i; j++) { i64 sc;
            if (g_attn_dot) { i64 dot=0; for (int d=0;d<X_DH;d++) dot += a.Q[i*X_D+h*X_DH+d]*a.K[j*X_D+h*X_DH+d]; sc = dot>>X_ABITS; }
            else { i64 l1 = 0; for (int d = 0; d < X_DH; d++) { i64 df = a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; l1 += df<0?-df:df; } sc = -l1; }
            s[j] = sc; if (s[j]>rowmax) rowmax=s[j]; if (s[j]<rowmin) rowmin=s[j]; }
        static int g_asm=-1,g_afix=-2; if(g_asm<0){const char*e=getenv("A_SMBITS");g_asm=e?atoi(e):X_SMBITS;const char*f=getenv("A_FIXTEMP");g_afix=f?atoi(f):-1;}
        int smsh = (g_afix>=0) ? g_afix : g_normshift(rowmax-rowmin, g_asm);
        i64 rowsum = 0;
        for (int j = 0; j <= i; j++) { i64 p=g_exp_q16(rowmax-s[j], smsh); a.P[(h*X_SEQ+i)*X_SEQ+j]=p; rowsum+=p; }
        int shn = g_bitlen(rowsum); a.Pshn[h*X_SEQ+i]=shn;
        for (int d = 0; d < X_DH; d++) { i64 blend=0; for (int j=0;j<=i;j++) blend += a.P[(h*X_SEQ+i)*X_SEQ+j]*a.V[j*X_D+h*X_DH+d]; a.ATTN[i*X_D+h*X_DH+d]=g_floorshr(blend,shn); }
    }
    i64 AO[X_SEQ*X_D]; g_matmul(a.ATTN, w.WO, AO, X_SEQ, X_D, X_D); g_rescale(AO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.M[i] = a.R[i] + AO[i];
    g_layernorm(a.M, w.LN2G, w.LN2B, a.X2, a.ln2s, X_SEQ);
    g_matmul(a.X2, w.WUP, a.HD, X_SEQ, X_D, X_FFN);
    for (int i = 0; i < X_SEQ*X_FFN; i++) { int pos=a.HD[i]>0; a.RMASK[i]=pos?1:0; if(!pos)a.HD[i]=0; }
    if (g_mxbs <= 0) g_rescale(a.HD, X_SEQ*X_FFN, X_ABITS);
    else g_block_round(a.HD, X_SEQ, X_FFN, X_ABITS, g_mxbs);
    i64 FO[X_SEQ*X_D]; g_matmul(a.HD, w.WDN, FO, X_SEQ, X_FFN, X_D); g_rescale(FO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.RP[i] = a.M[i] + FO[i];   // Rout = residual stream (NO final LN / head)
}

// ---- HIDDEN block backward: seed gRout (grad on a.RP) -> gRin (grad on a.R), accumulate weight grads ----
// Copy of mp_backward's body from the FFN seed onward (skips head/LNF seed + EMB/POS scatter).
static void mb_backward_hidden(GW &w, GA &a, GG &g, const i64 *gRout, i64 *gRin, bool zero=true) {
    if (zero) memset(&g, 0, sizeof(GG));
    i64 gRP[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gRP[i]=gRout[i];   // seed = upstream grad on this block's output residual
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
            // FIX#1 (int64-overflow): gs grows once per HIDDEN block (no inter-block grad rescale) -> max|gs|~3e18,
            // gs*K ~ 9.2e18 ~= 2^63 WRAPS in i64, sign-corrupting hidden-block gQ/gK (multi-block grad-check root cause).
            // Promote the product to __int128 before the >>X_ABITS, exactly like the pn*gp / pn*gATTN lines already do.
            if (g_attn_dot) { for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.K[j*X_D+h*X_DH+d])>>X_ABITS); gK[j*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.Q[i*X_D+h*X_DH+d])>>X_ABITS); } }
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
    for(int i=0;i<X_SEQ*X_D;i++) gRin[i]=gR[i];   // gradient on this block's INPUT residual (= prev block's gRout)
    // FIX#2 (inter-block gradient rescale): the forward max-shifts activations to X_ABITS each block, but the backward
    // gRout->gRin had NO matching down-shift, so the residual-stream gradient magnitude grew once per hidden block ->
    // the gs*K overflow above + the optimizer m/v EMA (m=g8x) overflowing i64 for ~3e18 grads. A single per-tensor
    // arithmetic right-shift (g_rescale, the SAME max-shift op as every matmul output, DSP=0, mult-free) bounds the
    // upstream gradient handed to the previous block. Default ON, bits = GRAD_RESCALE (env, default X_ABITS+something).
    // DEFAULT = X_EBITS (12) to MATCH the production kernel forge_pc_xformer_block.cpp, which already emits
    // EOUT=maxshift(gR,X_EBITS) as the cross-block error message (lines 621-634). The multi-block golden was the
    // ONLY place missing this rescale -> that omission is the overflow bug. -1 disables (to reproduce the old golden).
    { static int gr=-2; if(gr<-1){const char*e=getenv("GRAD_RESCALE"); gr=e?atoi(e):12;} if(gr>=0) g_rescale(gRin, X_SEQ*X_D, gr); }
}

// ---- OUTPUT block: forward (block + final-LN + head + CE) on a residual input Rin (no embed) ----
// = g_forward minus the embed step. We set a.R = Rin then run the rest. Reuse g_forward by pre-loading R?
// g_forward writes a.R from tokens; here R is the prev block's output. So we inline g_forward's tail.
static i64 mb_forward_output(GW &w, GA &a, const i64 *Rin, const int *tgt) {
    for (int i = 0; i < X_SEQ*X_D; i++) a.R[i] = Rin[i];
    g_layernorm(a.R, w.LN1G, w.LN1B, a.X1, a.ln1s, X_SEQ);
    static int g_akb=-1; if(g_akb<0){const char*e=getenv("AK_BITS");g_akb=e?atoi(e):X_ABITS;}
    g_matmul(a.X1, w.WQ, a.Q, X_SEQ, X_D, X_D); g_rescale(a.Q, X_SEQ*X_D, g_akb);
    g_matmul(a.X1, w.WK, a.K, X_SEQ, X_D, X_D); g_rescale(a.K, X_SEQ*X_D, g_akb);
    g_matmul(a.X1, w.WV, a.V, X_SEQ, X_D, X_D); g_rescale(a.V, X_SEQ*X_D, X_ABITS);
    if(g_attn_dot<0){const char*e=getenv("ATTN_DOT");g_attn_dot=e?atoi(e):0;}
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) {
        i64 s[X_SEQ]; i64 rowmax = -(((i64)1)<<40); i64 rowmin = (((i64)1)<<40);
        for (int j = 0; j <= i; j++) { i64 sc;
            if (g_attn_dot) { i64 dot=0; for (int d=0;d<X_DH;d++) dot += a.Q[i*X_D+h*X_DH+d]*a.K[j*X_D+h*X_DH+d]; sc = dot>>X_ABITS; }
            else { i64 l1 = 0; for (int d = 0; d < X_DH; d++) { i64 df = a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; l1 += df<0?-df:df; } sc = -l1; }
            s[j] = sc; if (s[j]>rowmax) rowmax=s[j]; if (s[j]<rowmin) rowmin=s[j]; }
        static int g_asm=-1,g_afix=-2; if(g_asm<0){const char*e=getenv("A_SMBITS");g_asm=e?atoi(e):X_SMBITS;const char*f=getenv("A_FIXTEMP");g_afix=f?atoi(f):-1;}
        int smsh = (g_afix>=0) ? g_afix : g_normshift(rowmax-rowmin, g_asm);
        i64 rowsum = 0;
        for (int j = 0; j <= i; j++) { i64 p=g_exp_q16(rowmax-s[j], smsh); a.P[(h*X_SEQ+i)*X_SEQ+j]=p; rowsum+=p; }
        int shn = g_bitlen(rowsum); a.Pshn[h*X_SEQ+i]=shn;
        for (int d = 0; d < X_DH; d++) { i64 blend=0; for (int j=0;j<=i;j++) blend += a.P[(h*X_SEQ+i)*X_SEQ+j]*a.V[j*X_D+h*X_DH+d]; a.ATTN[i*X_D+h*X_DH+d]=g_floorshr(blend,shn); }
    }
    i64 AO[X_SEQ*X_D]; g_matmul(a.ATTN, w.WO, AO, X_SEQ, X_D, X_D); g_rescale(AO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.M[i] = a.R[i] + AO[i];
    g_layernorm(a.M, w.LN2G, w.LN2B, a.X2, a.ln2s, X_SEQ);
    g_matmul(a.X2, w.WUP, a.HD, X_SEQ, X_D, X_FFN);
    for (int i = 0; i < X_SEQ*X_FFN; i++) { int pos=a.HD[i]>0; a.RMASK[i]=pos?1:0; if(!pos)a.HD[i]=0; }
    if (g_mxbs <= 0) g_rescale(a.HD, X_SEQ*X_FFN, X_ABITS);
    else g_block_round(a.HD, X_SEQ, X_FFN, X_ABITS, g_mxbs);
    i64 FO[X_SEQ*X_D]; g_matmul(a.HD, w.WDN, FO, X_SEQ, X_FFN, X_D); g_rescale(FO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.RP[i] = a.M[i] + FO[i];
    g_layernorm(a.RP, w.LNFG, w.LNFB, a.XF, a.lnfs, X_SEQ);
    g_matmul(a.XF, w.WHEAD, a.LG, X_SEQ, X_D, X_VOCAB);
    // HEAD_RESCALE (toggle, default off = canonical golden unchanged): the head logits are the ONE matmul
    // output the golden does NOT max-shift. With dyadic head weights + an unbounded residual stream the logit
    // SPREAD diverges (measured ~2^21 @ d3/h4), saturating the CE softmax -> GLOG goes near-one-hot, the
    // calibration gradient dies. A mult-free max-shift to ~HEAD_RESCALE bits (kernel's g_rescale, a shift)
    // bounds the spread so the precise-exp CE keeps a graded signal. Kernel-implementable (no new op).
    // FIX#3 (head-logit bound): DEFAULT ON in the FIX golden (was default off=-1 in the canonical). Bounds the
    // logit-cloud spread to ~HEAD_RESCALE bits so the adaptive CE shift smsh=g_normshift(max-min,C_SMBITS) stays
    // small and the learned per-token contrast (low bits) survives the precise exp. Kernel-native (a single
    // per-tensor arithmetic right-shift, the same g_rescale max-shift used on every other matmul output, DSP=0).
    { static int hr=-2; if(hr<-1){const char*e=getenv("HEAD_RESCALE"); hr=e?atoi(e):6;} if(hr>=0) g_rescale(a.LG, X_SEQ*X_VOCAB, hr); }
    i64 loss = 0;
    for (int t = 0; t < X_SEQ; t++) {
        i64 mx=-(((i64)1)<<40); i64 mn=(((i64)1)<<40); for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx)mx=lv; if(lv<mn)mn=lv; }
        static int g_csm=-1; if(g_csm<0){const char*e=getenv("C_SMBITS");g_csm=e?atoi(e):X_SMBITS;}
        int smsh = g_normshift(mx-mn, g_csm);
        i64 sum=0; i64 pr[X_VOCAB];
        for(int v=0;v<X_VOCAB;v++){ pr[v]=g_exp_q16(mx-a.LG[t*X_VOCAB+v], smsh); sum+=pr[v]; }
        int shn=g_bitlen(sum);
        for(int v=0;v<X_VOCAB;v++){ i64 pq=g_floorshr(pr[v]<<X_AT_PFRAC,shn); i64 oh=(v==tgt[t])?((i64)1<<X_AT_PFRAC):0; a.GLOG[t*X_VOCAB+v]=pq-oh; }
        int gtt=(int)(mx-a.LG[t*X_VOCAB+tgt[t]]); loss += gtt;
    }
    return loss;
}

// ---- OUTPUT block backward: head+LNF seed -> through the block -> gRin (grad on Rin). No EMB/POS scatter. ----
static void mb_backward_output(GW &w, GA &a, GG &g, i64 *gRin, bool zero=true) {
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
            // FIX#1 (int64-overflow): same __int128 promotion as the hidden block (no-op here, output gs stays bounded).
            if (g_attn_dot) { for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.K[j*X_D+h*X_DH+d])>>X_ABITS); gK[j*X_D+h*X_DH+d]+= (i64)(((__int128)gs*a.Q[i*X_D+h*X_DH+d])>>X_ABITS); } }
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
    for(int i=0;i<X_SEQ*X_D;i++) gRin[i]=gR[i];   // grad on OUTPUT block's input residual = prev block's gRout
    // FIX#2 (inter-block gradient rescale): bound the gradient handed to the previous block (see mb_backward_hidden).
    // DEFAULT = X_EBITS (12) to MATCH the production kernel forge_pc_xformer_block.cpp, which already emits
    // EOUT=maxshift(gR,X_EBITS) as the cross-block error message (lines 621-634). The multi-block golden was the
    // ONLY place missing this rescale -> that omission is the overflow bug. -1 disables (to reproduce the old golden).
    { static int gr=-2; if(gr<-1){const char*e=getenv("GRAD_RESCALE"); gr=e?atoi(e):12;} if(gr>=0) g_rescale(gRin, X_SEQ*X_D, gr); }
}

#endif
