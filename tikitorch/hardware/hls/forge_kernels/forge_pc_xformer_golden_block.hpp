// forge_pc_xformer_golden_block.hpp — the full int64 block fwd+bwd+dyadic-Adam reference (one transformer
// block, OUTPUT role = with head+CE). Standalone/g++-runnable; the HLS kernel transcribes these ops.
#ifndef FORGE_PC_XFORMER_GOLDEN_BLOCK_HPP
#define FORGE_PC_XFORMER_GOLDEN_BLOCK_HPP
#include "forge_pc_xformer_golden.hpp"

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

// ---------------- FORWARD (OUTPUT role): tokens -> logits + grad_logit, caches everything ----------------
static i64 g_forward(GW &w, GA &a, const int *tok, const int *tgt) {
    for (int t = 0; t < X_SEQ; t++) for (int d = 0; d < X_D; d++) {
        int v = tok[t]; if (v < 0) v = 0; if (v >= X_VOCAB) v = X_VOCAB - 1;
        a.R[t*X_D+d] = w.EMB[v*X_D+d] + w.POS[t*X_D+d];
    }
    g_layernorm(a.R, w.LN1G, w.LN1B, a.X1, a.ln1s, X_SEQ);
    g_matmul(a.X1, w.WQ, a.Q, X_SEQ, X_D, X_D); g_rescale(a.Q, X_SEQ*X_D, X_ABITS);
    g_matmul(a.X1, w.WK, a.K, X_SEQ, X_D, X_D); g_rescale(a.K, X_SEQ*X_D, X_ABITS);
    g_matmul(a.X1, w.WV, a.V, X_SEQ, X_D, X_D); g_rescale(a.V, X_SEQ*X_D, X_ABITS);
    for (int h = 0; h < X_H; h++) for (int i = 0; i < X_SEQ; i++) {
        i64 s[X_SEQ]; i64 rowmax = -(((i64)1)<<40);
        for (int j = 0; j <= i; j++) { i64 l1 = 0; for (int d = 0; d < X_DH; d++) { i64 df = a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; l1 += df<0?-df:df; } s[j] = -l1; if (s[j]>rowmax) rowmax=s[j]; }
        i64 rowsum = 0;
        for (int j = 0; j <= i; j++) { int gap=(int)((rowmax-s[j])>>X_AT_DELTA); if(gap<0)gap=0; if(gap>=X_LUT_N)gap=X_LUT_N-1; i64 p=G_LUT[gap]; a.P[(h*X_SEQ+i)*X_SEQ+j]=p; rowsum+=p; }
        int shn = g_bitlen(rowsum); a.Pshn[h*X_SEQ+i]=shn;
        for (int d = 0; d < X_DH; d++) { i64 blend=0; for (int j=0;j<=i;j++) blend += a.P[(h*X_SEQ+i)*X_SEQ+j]*a.V[j*X_D+h*X_DH+d]; a.ATTN[i*X_D+h*X_DH+d]=g_floorshr(blend,shn); }
    }
    i64 AO[X_SEQ*X_D]; g_matmul(a.ATTN, w.WO, AO, X_SEQ, X_D, X_D); g_rescale(AO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.M[i] = a.R[i] + AO[i];
    g_layernorm(a.M, w.LN2G, w.LN2B, a.X2, a.ln2s, X_SEQ);
    g_matmul(a.X2, w.WUP, a.HD, X_SEQ, X_D, X_FFN);
    for (int i = 0; i < X_SEQ*X_FFN; i++) { int pos=a.HD[i]>0; a.RMASK[i]=pos?1:0; if(!pos)a.HD[i]=0; }
    g_rescale(a.HD, X_SEQ*X_FFN, X_ABITS);
    i64 FO[X_SEQ*X_D]; g_matmul(a.HD, w.WDN, FO, X_SEQ, X_FFN, X_D); g_rescale(FO, X_SEQ*X_D, X_ABITS);
    for (int i = 0; i < X_SEQ*X_D; i++) a.RP[i] = a.M[i] + FO[i];
    g_layernorm(a.RP, w.LNFG, w.LNFB, a.XF, a.lnfs, X_SEQ);
    g_matmul(a.XF, w.WHEAD, a.LG, X_SEQ, X_D, X_VOCAB);
    // CE: softmax via LUT -> grad_logit = softmax - onehot (Q16); also accumulate an integer surrogate loss
    i64 loss = 0;
    for (int t = 0; t < X_SEQ; t++) {
        i64 mx=-(((i64)1)<<40); for(int v=0;v<X_VOCAB;v++) if(a.LG[t*X_VOCAB+v]>mx)mx=a.LG[t*X_VOCAB+v];
        i64 sum=0; i64 pr[X_VOCAB];
        for(int v=0;v<X_VOCAB;v++){ int gap=(int)((mx-a.LG[t*X_VOCAB+v])>>X_AT_DELTA); if(gap<0)gap=0; if(gap>=X_LUT_N)gap=X_LUT_N-1; pr[v]=G_LUT[gap]; sum+=pr[v]; }
        int shn=g_bitlen(sum);
        for(int v=0;v<X_VOCAB;v++){ i64 pq=g_floorshr(pr[v]<<X_AT_PFRAC,shn); i64 oh=(v==tgt[t])?((i64)1<<X_AT_PFRAC):0; a.GLOG[t*X_VOCAB+v]=pq-oh; }
        // surrogate loss = -log2(softmax[target]) approx via the gap of the target (for a monotone training signal)
        int gtt=(int)((mx-a.LG[t*X_VOCAB+tgt[t]])>>X_AT_DELTA); loss += gtt;
    }
    return loss;
}

// q(.)^T q(.) mult-free weight-grad accumulate: gW[o*I+i] += sum_t dyadval(go[t*O+o]) * dyadval(x[t*I+i])
static void g_wgrad(const i64 *go, const i64 *x, i64 *gW, int M, int I, int O) {
    for (int o=0;o<O;o++) for (int i=0;i<I;i++) { i64 acc=0; for (int t=0;t<M;t++) acc += g_dyadval(go[t*O+o])*g_dyadval(x[t*I+i]); gW[o*I+i]+=acc; }
}
// route gIn[t*I+i] = sum_o dyadval(W[o*I+i]) * gOut[t*O+o]   (transpose-free, W dyadic)
static void g_route(const i64 *gout, const i64 *W, i64 *gin, int M, int I, int O) {
    for (int t=0;t<M;t++) for (int i=0;i<I;i++){ i64 acc=0; for(int o=0;o<O;o++) acc += g_dyadval(W[o*I+i])*gout[t*O+o]; gin[t*I+i]=acc; }
}
// LN backward (straight-through invq + gamma-STE + mean-subtract); returns gIn; accumulates gGamma,gBeta
static void g_ln_bwd(const i64 *gy, const i64 *xin, const i64 *gcode, const int *stats, i64 *gin, i64 *gG, i64 *gB, int M) {
    for (int t=0;t<M;t++){
        int e=stats[2*t], nq=stats[2*t+1];
        i64 gc[X_D]; i64 msum=0;
        for(int d=0;d<X_D;d++){ int gs,ge,gnz; g_dyad(gcode[d],gs,ge,gnz); i64 gch = gnz?(gy[t*X_D+d]<<ge):0; if(gnz&&gs)gch=-gch; i64 g=g_applyinvq(gch,e,nq); gc[d]=g; msum+=g; gB[d]+=gy[t*X_D+d]; }
        i64 mean=msum>>X_LOG2_D;
        // recompute c=x-mean of the INPUT for the gamma grad
        i64 xs=0; for(int d=0;d<X_D;d++) xs+=xin[t*X_D+d]; i64 xmean=xs>>X_LOG2_D;
        for(int d=0;d<X_D;d++){ gin[t*X_D+d]+=gc[d]-mean; i64 chat=g_applyinvq(xin[t*X_D+d]-xmean,e,nq); gG[d]+=g_dyadval(gy[t*X_D+d])*g_dyadval(chat); }
    }
}

// grad accumulators (zeroed each step)
struct GG {
    i64 WQ[X_D*X_D],WK[X_D*X_D],WV[X_D*X_D],WO[X_D*X_D],WUP[X_FFN*X_D],WDN[X_D*X_FFN],WHEAD[X_VOCAB*X_D];
    i64 LN1G[X_D],LN1B[X_D],LN2G[X_D],LN2B[X_D],LNFG[X_D],LNFB[X_D];
    i64 EMB[X_VOCAB*X_D],POS[X_SEQ*X_D];
};

// ---------------- BACKWARD (OUTPUT role) ----------------
static void g_backward(GW &w, GA &a, GG &g, const int *tok) {
    memset(&g, 0, sizeof(GG));
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
        i64 ex=0; for(int j=0;j<=i;j++) ex += g_floorshr(pn[j]*gp[j], X_AT_PFRAC);  // MUL
        for(int j=0;j<=i;j++){
            i64 gs=g_floorshr(pn[j]*(gp[j]-ex), X_AT_PFRAC);                         // MUL (softmax bwd)
            for(int d=0;d<X_DH;d++){ i64 df=a.Q[i*X_D+h*X_DH+d]-a.K[j*X_D+h*X_DH+d]; int sg=df>0?1:(df<0?-1:0);
                gQ[i*X_D+h*X_DH+d]+= -gs*sg; gK[j*X_D+h*X_DH+d]+= gs*sg; }                 // L1 subgradient (mult-free)
            for(int d=0;d<X_DH;d++) gV[j*X_D+h*X_DH+d]+= g_floorshr(pn[j]*gATTN[i*X_D+h*X_DH+d], X_AT_PFRAC); // MUL
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
    // (6) INPUT-role grads: scatter gR into embed + pos by token id (OUTPUT role would emit EOUT=rescale(gR))
    for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; for(int d=0;d<X_D;d++){ g.EMB[v*X_D+d]+=gR[t*X_D+d]; g.POS[t*X_D+d]+=gR[t*X_D+d]; } }
}

// ---------------- mult-free dyadic-Adam (one matrix) ----------------
static inline int g_dyad_exp(i64 v){ i64 a=v<0?-v:v; int m=0; for(int b=30;b>=1;b--) if((a>>b)&1){m=b;break;} if(m>=1&&((a>>(m-1))&1)&&m<30)m++; return m; }
static void g_adam(i64 *W, const i64 *gW, i64 *m, i64 *v, int n) {
    for(int i=0;i<n;i++){ i64 g=gW[i], ag=g<0?-g:g;
        m[i]=m[i]-(m[i]>>X_ADAM_BSH)+(g>>X_ADAM_BSH); v[i]=v[i]-(v[i]>>X_ADAM_BSH)+(ag>>X_ADAM_BSH);
        int se=g_dyad_exp(v[i]); i64 step=m[i]>>se; i64 dw=step>>X_LR_SHIFT;
        i64 nw=W[i]-dw - (W[i]>>X_WD_SHIFT);            // + mult-free weight decay
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

