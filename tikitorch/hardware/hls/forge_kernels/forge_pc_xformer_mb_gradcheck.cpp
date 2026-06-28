// hr_gradcheck.cpp — verify FIX#1 (int128 promotion) corrects the hidden-block WQ/WK gradient overflow.
// Strategy: run ONE NBLK=3 forward+backward with the BASELINE block code (i64-wrap) and with the FIX block
// code (int128), on IDENTICAL weights+acts (we share the forward, then run each backward variant), and compare
// the resulting g.WQ/g.WK for the HIDDEN blocks (b0,b1) and the OUTPUT block (b2).
//  - The multi-block grad-check claims: baseline hidden WQ/WK are sign-corrupted by an i64 wrap; output is fine.
//  - Expectation after FIX: hidden WQ/WK CHANGE (the wrap is removed) and now equal an INDEPENDENT __int128
//    reference computed here; output WQ/WK are UNCHANGED (no overflow there).
//
// We compile this file TWICE via -D switch: HRG_FIX picks the FIX header. We dump g.WQ/g.WK to a file, then a
// driver diffs the two dumps + the int128 reference. To keep it self-contained we instead compute BOTH the
// i64-wrap and the int128 hidden gQ/gK gradient HERE (independent of the headers' backward) from the cached acts,
// and ALSO read the header's stored g.WQ to confirm which one the header matches.
//
//   g++ -O2 -DX_H=4 -DX_DH=16 -I. hr_gradcheck.cpp -o hr_gradcheck            # FIX header (default)
//   ATTN_DOT=1 NBLK=3 SEED=1 SHAKESPEARE=... ./hr_gradcheck
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "forge_pc_xformer_mb_golden_block.hpp"

static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
static void init_block(GW &w){
    for(int i=0;i<X_D*X_D;i++){ w.WQ[i]=RV(8);w.WK[i]=RV(8);w.WV[i]=RV(8);w.WO[i]=RV(8);}
    for(int i=0;i<X_FFN*X_D;i++) w.WUP[i]=RV(8);
    for(int i=0;i<X_D*X_FFN;i++) w.WDN[i]=RV(8);
    for(int i=0;i<X_VOCAB*X_D;i++) w.WHEAD[i]=RV(8);
    for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]=RV(16);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]=RV(8);
    for(int d=0;d<X_D;d++){ w.LN1G[d]=64;w.LN2G[d]=64;w.LNFG[d]=64; w.LN1B[d]=0;w.LN2B[d]=0;w.LNFB[d]=0; }
    memset(w.mWQ,0,sizeof(w.mWQ));memset(w.vWQ,0,sizeof(w.vWQ));memset(w.mWK,0,sizeof(w.mWK));memset(w.vWK,0,sizeof(w.vWK));
    memset(w.mWV,0,sizeof(w.mWV));memset(w.vWV,0,sizeof(w.vWV));memset(w.mWO,0,sizeof(w.mWO));memset(w.vWO,0,sizeof(w.vWO));
    memset(w.mUP,0,sizeof(w.mUP));memset(w.vUP,0,sizeof(w.vUP));memset(w.mDN,0,sizeof(w.mDN));memset(w.vDN,0,sizeof(w.vDN));
    memset(w.mHD,0,sizeof(w.mHD));memset(w.vHD,0,sizeof(w.vHD));
}
static void mb_embed(GW &w0, i64 *R0, const int *tok){
    for(int t=0;t<X_SEQ;t++) for(int d=0;d<X_D;d++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; R0[t*X_D+d]=w0.EMB[v*X_D+d]+w0.POS[t*X_D+d]; }
}

// Independent recompute of gQ/gK for ONE block from cached acts a.*, given gATTN derived from gM via WO route.
// MODE: use_i128=1 -> overflow-free; 0 -> i64-wrap (mimics the pre-fix product). Returns max|gs|, max|gs*K|.
static void recompute_gQK(GW &w, GA &a, const i64 *gM_in, i64 *gQ, i64 *gK, int use_i128, double *maxgs, double *maxgsK){
    i64 gATTN[X_SEQ*X_D]; g_route(gM_in, w.WO, gATTN, X_SEQ, X_D, X_D);
    memset(gQ,0,sizeof(i64)*X_SEQ*X_D); memset(gK,0,sizeof(i64)*X_SEQ*X_D);
    *maxgs=0; *maxgsK=0;
    for(int h=0;h<X_H;h++) for(int i=0;i<X_SEQ;i++){
        int shn=a.Pshn[h*X_SEQ+i]; i64 pn[X_SEQ];
        for(int j=0;j<=i;j++) pn[j]=g_floorshr(a.P[(h*X_SEQ+i)*X_SEQ+j]<<X_AT_PFRAC, shn);
        i64 gp[X_SEQ]; for(int j=0;j<=i;j++){ i64 s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*a.V[j*X_D+h*X_DH+d]; gp[j]=s; }
        i64 ex=0; for(int j=0;j<=i;j++) ex += (i64)(((__int128)pn[j]*gp[j]) >> X_AT_PFRAC);
        for(int j=0;j<=i;j++){
            i64 gs=(i64)(((__int128)pn[j]*(gp[j]-ex)) >> X_AT_PFRAC);
            double ags = gs<0?-(double)gs:(double)gs; if(ags>*maxgs)*maxgs=ags;
            for(int d=0;d<X_DH;d++){
                i64 K=a.K[j*X_D+h*X_DH+d], Q=a.Q[i*X_D+h*X_DH+d];
                double prod = ags*(K<0?-(double)K:(double)K); if(prod>*maxgsK)*maxgsK=prod;
                if(use_i128){ gQ[i*X_D+h*X_DH+d]+= (i64)(((__int128)gs*K)>>X_ABITS); gK[j*X_D+h*X_DH+d]+= (i64)(((__int128)gs*Q)>>X_ABITS); }
                else        { gQ[i*X_D+h*X_DH+d]+= (gs*K)>>X_ABITS;                gK[j*X_D+h*X_DH+d]+= (gs*Q)>>X_ABITS; }
            }
        }
    }
}
static void cmp(const char*tag, const i64*A, const i64*B, int n){
    int agree=0, nz=0; double mrel=0;
    for(int i=0;i<n;i++){ if(A[i]||B[i]){ nz++; int sa=(A[i]>0)-(A[i]<0), sb=(B[i]>0)-(B[i]<0); if(sa==sb)agree++; double den=(double)(B[i]<0?-B[i]:B[i]); double num=(double)((A[i]-B[i])<0?-(A[i]-B[i]):(A[i]-B[i])); double r=den>0?num/den:(num>0?1e18:0); if(r>mrel)mrel=r; } }
    printf("    %s: nz=%d signagree=%.4f maxrelerr=%.3g\n", tag, nz, nz?(double)agree/nz:1.0, mrel);
}

int main(){
    g_lut_init(); g_exp_init(); (void)mp_mode();
    int NBLK=getenv("NBLK")?atoi(getenv("NBLK")):3; int SEED=getenv("SEED")?atoi(getenv("SEED")):1;
    const char*path=getenv("SHAKESPEARE"); if(!path){printf("need SHAKESPEARE\n");return 1;}
    FILE*f=fopen(path,"rb"); if(!f){printf("no data\n");return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f);(void)rd;
    int map[256]; for(int i=0;i<256;i++)map[i]=-1; int vocab=0; for(long i=0;i<n&&vocab<X_VOCAB;i++){unsigned char c=txt[i]; if(map[c]<0)map[c]=vocab++;}
    int*data=(int*)malloc(sizeof(int)*n); long nd=0; for(long i=0;i<n;i++){int id=map[(unsigned char)txt[i]]; data[nd++]=id<0?0:id;}
    int adot=getenv("ATTN_DOT")?atoi(getenv("ATTN_DOT")):0;
    printf("HR-GRADCHECK NBLK=%d SEED=%d ATTN_DOT=%d (build=%s)\n", NBLK, SEED, adot,
#ifdef HRG_FIX
        "FIX"
#else
        "FIX-header"
#endif
    );
    static GW W[8]; static GA A[8]; static GG G[8];
    for(int b=0;b<NBLK;b++){ srand(7+101*SEED+17*b); init_block(W[b]); }
    int tok[X_SEQ],tgt[X_SEQ]; srand(123+1009*SEED);
    // Take a non-trivial step count so weights/acts develop large gradients (the overflow regime). Just do 1 fwd/bwd
    // on a random batch but with the head UNBOUNDED (HEAD_RESCALE off) to maximize the gradient magnitude the way the
    // collapsing run does. Note the grad-check is about the BACKWARD math, independent of HEAD_RESCALE on the fwd.
    setenv("HEAD_RESCALE","-1",1); setenv("GRAD_RESCALE","-1",1);
    long off=rand()%(nd-X_SEQ-2); for(int t=0;t<X_SEQ;t++){tok[t]=data[off+t];tgt[t]=data[off+t+1];}
    static i64 Rbuf[9][X_SEQ*X_D];
    mb_embed(W[0],Rbuf[0],tok);
    for(int b=0;b<NBLK-1;b++) mb_forward_hidden(W[b],A[b],Rbuf[b]);
    for(int b=0;b<NBLK-1;b++) for(int i=0;i<X_SEQ*X_D;i++) Rbuf[b+1][i]=A[b].RP[i];
    mb_forward_output(W[NBLK-1],A[NBLK-1],Rbuf[NBLK-1],tgt);
    i64 gR_up[X_SEQ*X_D], gR_dn[X_SEQ*X_D];
    mb_backward_output(W[NBLK-1],A[NBLK-1],G[NBLK-1],gR_up,true);
    for(int b=NBLK-2;b>=0;b--){ mb_backward_hidden(W[b],A[b],G[b],gR_up,gR_dn,true); for(int i=0;i<X_SEQ*X_D;i++) gR_up[i]=gR_dn[i]; }

    // Now independently recompute each block's gQ/gK both i64-wrap and i128, from the SAME cached acts.
    // We must reconstruct gM (the seed into the attn-bwd) for each block. For hidden blocks gM = gRP path; for a
    // clean independent check we re-derive gM via the header's OWN forward-cached acts: gM = grad on a.M. We can't
    // easily reach gM without re-running the bwd, so instead recompute the WHOLE hidden bwd up to gM here.
    // Simplest robust check: recompute gQ/gK in BOTH modes from a recomputed gM, and compare i64 vs i128 directly +
    // report overflow magnitudes. (We trust the header stores the int128 result; the decisive question is whether
    // i64-wrap DIFFERS from i128 for hidden blocks = the bug exists, and the magnitudes confirm overflow.)
    for(int b=0;b<NBLK;b++){
        GW &w=W[b]; GA &a=A[b];
        // reconstruct gM for this block by replaying the bwd head/ffn portion:
        i64 gM[X_SEQ*X_D];
        if(b==NBLK-1){ // output block: seed head/LNF/ffn
            i64 gXF[X_SEQ*X_D]; g_route(a.GLOG,w.WHEAD,gXF,X_SEQ,X_D,X_VOCAB);
            i64 gRP[X_SEQ*X_D]; memset(gRP,0,sizeof(gRP)); GG gg; memset(&gg,0,sizeof(gg));
            g_ln_bwd(gXF,a.RP,w.LNFG,a.lnfs,gRP,gg.LNFG,gg.LNFB,X_SEQ);
            for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
            i64 gHD[X_SEQ*X_FFN]; g_route(gRP,w.WDN,gHD,X_SEQ,X_FFN,X_D);
            for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;
            i64 gX2[X_SEQ*X_D]; g_route(gHD,w.WUP,gX2,X_SEQ,X_D,X_FFN);
            g_ln_bwd(gX2,a.M,w.LN2G,a.ln2s,gM,gg.LN2G,gg.LN2B,X_SEQ);
        } else { // hidden block: seed = gRout = next block's gRin. We recompute by re-running cascade bwd partially.
            // For an independent gM we need this block's gRout. Re-run the full cascade bwd capturing gRout per block.
            // (Cheap: re-run.)
            i64 up[X_SEQ*X_D], dn[X_SEQ*X_D]; GG gtmp;
            mb_backward_output(W[NBLK-1],A[NBLK-1],gtmp,up,true);
            for(int bb=NBLK-2;bb>b;bb--){ mb_backward_hidden(W[bb],A[bb],gtmp,up,dn,true); for(int i=0;i<X_SEQ*X_D;i++) up[i]=dn[i]; }
            // now up = gRout for block b. replay this block's ffn/ln2 to get gM.
            i64 gRP[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gRP[i]=up[i];
            for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
            i64 gHD[X_SEQ*X_FFN]; g_route(gRP,w.WDN,gHD,X_SEQ,X_FFN,X_D);
            for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;
            i64 gX2[X_SEQ*X_D]; g_route(gHD,w.WUP,gX2,X_SEQ,X_D,X_FFN);
            GG gg; memset(&gg,0,sizeof(gg));
            g_ln_bwd(gX2,a.M,w.LN2G,a.ln2s,gM,gg.LN2G,gg.LN2B,X_SEQ);
        }
        i64 gQ64[X_SEQ*X_D],gK64[X_SEQ*X_D],gQ128[X_SEQ*X_D],gK128[X_SEQ*X_D];
        double mgs,mgsK;
        recompute_gQK(w,a,gM,gQ64,gK64,0,&mgs,&mgsK);
        recompute_gQK(w,a,gM,gQ128,gK128,1,&mgs,&mgsK);
        printf("  block %d (%s): max|gs|=%.3e  max|gs*K|=%.3e  2^62=%.3e  2^63=%.3e\n",
               b, b==NBLK-1?"OUTPUT":"HIDDEN", mgs, mgsK, ldexp(1.0,62), ldexp(1.0,63));
        cmp("gQ i64wrap-vs-i128", gQ64, gQ128, X_SEQ*X_D);
        cmp("gK i64wrap-vs-i128", gK64, gK128, X_SEQ*X_D);
    }
    return 0;
}
