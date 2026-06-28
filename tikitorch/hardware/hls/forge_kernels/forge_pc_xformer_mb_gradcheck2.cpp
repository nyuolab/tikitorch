// forge_pc_xformer_mb_gradcheck2.cpp — DECISIVE multi-block grad-check for the depth fixes.
// =====================================================================================================
// Confirms the CANONICAL multi-block golden's HIDDEN-block backward (forge_pc_xformer_mb_golden_block.hpp,
// now carrying FIX#1 __int128 + FIX#2 inter-block g_rescale) produces a WQ/WK weight-gradient that is
// BIT-EXACT to an INDEPENDENT __int128 reference of the SAME backward math — and that the OLD i64-wrap
// reference DIVERGES (signagree ~0.5). This is the i64-wrap-vs-int128 confirmation the gate asks for, but
// run against the header's OWN computed gradient (not just a standalone recompute).
//
// Method: build NBLK blocks, run the shared forward, run the canonical header's cascade backward, and read
// the HIDDEN block's g.WQ / g.WK. Independently recompute that same g.WQ / g.WK two ways from the cached
// acts + the header-produced gRout: (A) __int128 attention-bwd (the correct path) and (B) i64-wrap (the old
// bug). Report signagree(header vs i128) — must be 1.0 — and signagree(header vs i64wrap) — the old bug's
// fingerprint (~0.5 on the deepest hidden block). GRAD_RESCALE drives FIX#2 (default 12 = X_EBITS).
//
//   g++ -O2 -DX_H=4 -DX_DH=16 -I. forge_pc_xformer_mb_gradcheck2.cpp -o /tmp/mbgc2
//   ATTN_DOT=1 NBLK=3 SEED=1 SHAKESPEARE=... /tmp/mbgc2
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

// Independent recompute of the FULL hidden-block backward gQ/gK -> g.WQ/g.WK, given gRout (this block's
// upstream grad) and the cached acts a.*. use_i128 selects the correct (1) vs i64-wrap (0) attention bwd.
// Mirrors mb_backward_hidden exactly EXCEPT the gs*K/gs*Q width. Returns gWQ/gWK.
static void ref_hidden_wgrad(GW &w, GA &a, const i64 *gRout, int use_i128, i64 *gWQ, i64 *gWK){
    memset(gWQ,0,sizeof(i64)*X_D*X_D); memset(gWK,0,sizeof(i64)*X_D*X_D);
    i64 gRP[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gRP[i]=gRout[i];
    i64 gM[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gM[i]=gRP[i];
    i64 gHD[X_SEQ*X_FFN]; g_route(gRP,w.WDN,gHD,X_SEQ,X_FFN,X_D);
    for(int i=0;i<X_SEQ*X_FFN;i++) if(!a.RMASK[i]) gHD[i]=0;
    i64 gX2[X_SEQ*X_D]; g_route(gHD,w.WUP,gX2,X_SEQ,X_D,X_FFN);
    GG gg; memset(&gg,0,sizeof(gg));
    g_ln_bwd(gX2,a.M,w.LN2G,a.ln2s,gM,gg.LN2G,gg.LN2B,X_SEQ);
    i64 gATTN[X_SEQ*X_D]; g_route(gM,w.WO,gATTN,X_SEQ,X_D,X_D);
    i64 gQ[X_SEQ*X_D],gK[X_SEQ*X_D]; memset(gQ,0,sizeof(gQ)); memset(gK,0,sizeof(gK));
    for(int h=0;h<X_H;h++) for(int i=0;i<X_SEQ;i++){
        int shn=a.Pshn[h*X_SEQ+i]; i64 pn[X_SEQ];
        for(int j=0;j<=i;j++) pn[j]=g_floorshr(a.P[(h*X_SEQ+i)*X_SEQ+j]<<X_AT_PFRAC, shn);
        i64 gp[X_SEQ]; for(int j=0;j<=i;j++){ i64 s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*a.V[j*X_D+h*X_DH+d]; gp[j]=s; }
        i64 ex=0; for(int j=0;j<=i;j++) ex += (i64)(((__int128)pn[j]*gp[j]) >> X_AT_PFRAC);
        for(int j=0;j<=i;j++){
            i64 gs=(i64)(((__int128)pn[j]*(gp[j]-ex)) >> X_AT_PFRAC);
            for(int d=0;d<X_DH;d++){ i64 K=a.K[j*X_D+h*X_DH+d], Q=a.Q[i*X_D+h*X_DH+d];
                if(use_i128){ gQ[i*X_D+h*X_DH+d]+= (i64)(((__int128)gs*K)>>X_ABITS); gK[j*X_D+h*X_DH+d]+= (i64)(((__int128)gs*Q)>>X_ABITS); }
                else        { gQ[i*X_D+h*X_DH+d]+= (gs*K)>>X_ABITS;                gK[j*X_D+h*X_DH+d]+= (gs*Q)>>X_ABITS; } }
        }
    }
    g_wgrad(gQ, a.X1, gWQ, X_SEQ, X_D, X_D);
    g_wgrad(gK, a.X1, gWK, X_SEQ, X_D, X_D);
}
static double signagr(const i64*A, const i64*B, int n, int*pnz){
    int agree=0,nz=0; for(int i=0;i<n;i++){ if(A[i]||B[i]){ nz++; int sa=(A[i]>0)-(A[i]<0), sb=(B[i]>0)-(B[i]<0); if(sa==sb)agree++; } } if(pnz)*pnz=nz; return nz?(double)agree/nz:1.0; }
static int exact(const i64*A,const i64*B,int n){ for(int i=0;i<n;i++) if(A[i]!=B[i]) return 0; return 1; }

int main(){
    g_lut_init(); g_exp_init(); (void)mp_mode();
    int NBLK=getenv("NBLK")?atoi(getenv("NBLK")):3, SEED=getenv("SEED")?atoi(getenv("SEED")):1;
    const char*path=getenv("SHAKESPEARE"); if(!path){printf("need SHAKESPEARE\n");return 1;}
    FILE*f=fopen(path,"rb"); if(!f){printf("no data\n");return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char*txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f);(void)rd;
    int map[256]; for(int i=0;i<256;i++)map[i]=-1; int vocab=0; for(long i=0;i<n&&vocab<X_VOCAB;i++){unsigned char c=txt[i]; if(map[c]<0)map[c]=vocab++;}
    int*data=(int*)malloc(sizeof(int)*n); long nd=0; for(long i=0;i<n;i++){int id=map[(unsigned char)txt[i]]; data[nd++]=id<0?0:id;}
    int adot=getenv("ATTN_DOT")?atoi(getenv("ATTN_DOT")):0;
    int gr=getenv("GRAD_RESCALE")?atoi(getenv("GRAD_RESCALE")):12;
    int hrr=getenv("HEAD_RESCALE")?atoi(getenv("HEAD_RESCALE")):6;
    printf("MB-GRADCHECK2 NBLK=%d SEED=%d ATTN_DOT=%d GRAD_RESCALE=%d HEAD_RESCALE=%d\n", NBLK,SEED,adot,gr,hrr);

    static GW W[8]; static GA A[8]; static GG G[8];
    for(int b=0;b<NBLK;b++){ srand(7+101*SEED+17*b); init_block(W[b]); }
    int tok[X_SEQ],tgt[X_SEQ]; srand(123+1009*SEED);
    long off=rand()%(nd-X_SEQ-2); for(int t=0;t<X_SEQ;t++){tok[t]=data[off+t];tgt[t]=data[off+t+1];}
    static i64 Rbuf[9][X_SEQ*X_D];
    mb_embed(W[0],Rbuf[0],tok);
    for(int b=0;b<NBLK-1;b++) mb_forward_hidden(W[b],A[b],Rbuf[b]);
    for(int b=0;b<NBLK-1;b++) for(int i=0;i<X_SEQ*X_D;i++) Rbuf[b+1][i]=A[b].RP[i];
    mb_forward_output(W[NBLK-1],A[NBLK-1],Rbuf[NBLK-1],tgt);

    // CASCADE backward via the CANONICAL header (FIX#1+FIX#2 active). Capture each hidden block's gRout (the
    // upstream grad it received) so the reference can replay the SAME backward independently.
    i64 gRout_of[8][X_SEQ*X_D];
    i64 up[X_SEQ*X_D], dn[X_SEQ*X_D];
    mb_backward_output(W[NBLK-1],A[NBLK-1],G[NBLK-1],up,true);
    for(int b=NBLK-2;b>=0;b--){
        for(int i=0;i<X_SEQ*X_D;i++) gRout_of[b][i]=up[i];   // this block's upstream grad (= prev block's rescaled gRin)
        mb_backward_hidden(W[b],A[b],G[b],up,dn,true);
        for(int i=0;i<X_SEQ*X_D;i++) up[i]=dn[i];
    }

    int fail=0;
    for(int b=0;b<NBLK-1;b++){   // hidden blocks only (OUTPUT block has bounded gs, separately verified)
        i64 rWQ128[X_D*X_D],rWK128[X_D*X_D],rWQ64[X_D*X_D],rWK64[X_D*X_D];
        ref_hidden_wgrad(W[b],A[b],gRout_of[b],1,rWQ128,rWK128);
        ref_hidden_wgrad(W[b],A[b],gRout_of[b],0,rWQ64, rWK64 );
        int nzq,nzk; double agr_q128=signagr(G[b].WQ,rWQ128,X_D*X_D,&nzq), agr_k128=signagr(G[b].WK,rWK128,X_D*X_D,&nzk);
        double agr_q64=signagr(G[b].WQ,rWQ64,X_D*X_D,0),  agr_k64=signagr(G[b].WK,rWK64,X_D*X_D,0);
        int ex_q=exact(G[b].WQ,rWQ128,X_D*X_D), ex_k=exact(G[b].WK,rWK128,X_D*X_D);
        // how badly does i64-wrap diverge from i128 on THIS block (the bug fingerprint)?
        double wrap_q=signagr(rWQ64,rWQ128,X_D*X_D,0), wrap_k=signagr(rWK64,rWK128,X_D*X_D,0);
        printf("  HIDDEN block %d (depth %d):\n", b, NBLK-1-b);
        printf("    header.WQ vs i128-ref: signagree=%.4f  BITEXACT=%s   (vs i64wrap-ref signagree=%.4f)\n", agr_q128, ex_q?"YES":"no", agr_q64);
        printf("    header.WK vs i128-ref: signagree=%.4f  BITEXACT=%s   (vs i64wrap-ref signagree=%.4f)\n", agr_k128, ex_k?"YES":"no", agr_k64);
        printf("    [bug fingerprint] i64wrap-ref vs i128-ref: gWQ signagree=%.4f gWK signagree=%.4f\n", wrap_q, wrap_k);
        if(!ex_q||!ex_k||agr_q128<0.9999||agr_k128<0.9999) fail=1;
    }
    printf("RESULT: hidden WQ/WK %s bit-exact to the __int128 reference (header carries FIX#1+FIX#2).\n", fail?"NOT":"ARE");
    return fail?1:0;
}
