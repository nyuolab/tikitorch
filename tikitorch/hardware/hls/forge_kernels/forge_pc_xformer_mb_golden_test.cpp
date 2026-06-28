// mb_golden_test.cpp — MULTI-BLOCK cascade confirmation-gate harness.
// Stacks NBLK blocks (NBLK-1 HIDDEN + 1 OUTPUT), trains the HAND-ROLLED CASCADE (mb_forward_hidden /
// mb_forward_output + mb_backward_* + mp_step) on TinyShakespeare next-char, reports BPB/acc per the
// MP_OPT mode. Drives WARM dot attention (ATTN_DOT=1) + precise mult-free exp (already the golden's
// default exp path). NBLK / per-seed init+order via env.
//
//   g++ -O2 -DX_H=4 -DX_DH=16 -I<kerneldir> mb_golden_test.cpp -o /tmp/mbtest
//   MP_OPT=dbl ATTN_DOT=1 NBLK=3 SEED=1 STEPS=2500 SHAKESPEARE=... /tmp/mbtest
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_mb_golden_block.hpp"

#ifndef MB_MAXBLK
#define MB_MAXBLK 8
#endif

static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
// init one block's matmul weights + LN (seed already set by caller so distinct blocks get distinct inits)
static void init_block(GW &w){
    int wf = getenv("WFRAC")?atoi(getenv("WFRAC")):0; int ws = 1<<wf;
    for(int i=0;i<X_D*X_D;i++){ w.WQ[i]=RV(8)*ws;w.WK[i]=RV(8)*ws;w.WV[i]=RV(8)*ws;w.WO[i]=RV(8)*ws;}
    for(int i=0;i<X_FFN*X_D;i++) w.WUP[i]=RV(8)*ws;
    for(int i=0;i<X_D*X_FFN;i++) w.WDN[i]=RV(8)*ws;
    for(int i=0;i<X_VOCAB*X_D;i++) w.WHEAD[i]=RV(8)*ws;
    for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]=RV(16);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]=RV(8);
    for(int d=0;d<X_D;d++){ w.LN1G[d]=64;w.LN2G[d]=64;w.LNFG[d]=64; w.LN1B[d]=0;w.LN2B[d]=0;w.LNFB[d]=0; }
    memset(w.mWQ,0,sizeof(w.mWQ));memset(w.vWQ,0,sizeof(w.vWQ));memset(w.mWK,0,sizeof(w.mWK));memset(w.vWK,0,sizeof(w.vWK));
    memset(w.mWV,0,sizeof(w.mWV));memset(w.vWV,0,sizeof(w.vWV));memset(w.mWO,0,sizeof(w.mWO));memset(w.vWO,0,sizeof(w.vWO));
    memset(w.mUP,0,sizeof(w.mUP));memset(w.vUP,0,sizeof(w.vUP));memset(w.mDN,0,sizeof(w.mDN));memset(w.vDN,0,sizeof(w.vDN));
    memset(w.mHD,0,sizeof(w.mHD));memset(w.vHD,0,sizeof(w.vHD));
}

static double batch_bpb(const GA &a, const int *tgt){
    double bits=0; const double LOG2=log(2.0);
    for(int t=0;t<X_SEQ;t++){
        i64 sm = a.GLOG[t*X_VOCAB+tgt[t]] + ((i64)1<<X_AT_PFRAC);
        double p = (double)sm/(double)((i64)1<<X_AT_PFRAC);
        if (p < 1.0/65536.0) p = 1.0/65536.0; if (p > 1.0) p = 1.0;
        bits += -log(p)/LOG2;
    }
    return bits/X_SEQ;
}

// EMB/POS live in block 0's GW. INPUT stage = embed/pos -> R0.
static void mb_embed(GW &w0, i64 *R0, const int *tok){
    for(int t=0;t<X_SEQ;t++) for(int d=0;d<X_D;d++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; R0[t*X_D+d]=w0.EMB[v*X_D+d]+w0.POS[t*X_D+d]; }
}
// scatter the final residual-input gradient (gRin of block 0) into EMB/POS of block 0 (SignSGD applied in step).
static void mb_embed_bwd(GG &g0, const i64 *gR0, const int *tok){
    for(int t=0;t<X_SEQ;t++){ int v=tok[t]; if(v<0)v=0; if(v>=X_VOCAB)v=X_VOCAB-1; for(int d=0;d<X_D;d++){ g0.EMB[v*X_D+d]+=gR0[t*X_D+d]; g0.POS[t*X_D+d]+=gR0[t*X_D+d]; } }
}

int main(){
    g_lut_init(); g_exp_init();
    (void)mp_mode();
    int NBLK = getenv("NBLK")?atoi(getenv("NBLK")):3; if(NBLK>MB_MAXBLK)NBLK=MB_MAXBLK; if(NBLK<1)NBLK=1;
    int SEED = getenv("SEED")?atoi(getenv("SEED")):0;
    const char* path = getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* f=fopen(path,"rb"); if(!f){ printf("no data at %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long ndata=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[ndata++]= id<0?0:id; }
    double chance_bpb = log2((double)X_VOCAB);
    int adot = getenv("ATTN_DOT")?atoi(getenv("ATTN_DOT")):0;
    printf("MB cascade: MODE=%s NBLK=%d H=%d DH=%d D=%d FFN=%d SEQ=%d VOCAB=%d seen=%d ATTN_DOT=%d SEED=%d chance_bpb=%.3f\n",
           mp_name(), NBLK, X_H, X_DH, X_D, X_FFN, X_SEQ, X_VOCAB, vocab, adot, SEED, chance_bpb);

    // per-block weights/acts/grads
    static GW W[MB_MAXBLK]; static GA A[MB_MAXBLK]; static GG G[MB_MAXBLK];
    // distinct init per block (seed offset by SEED + block index so seeds differ in init AND blocks differ)
    for(int b=0;b<NBLK;b++){ srand(7 + 101*SEED + 17*b); init_block(W[b]); }

    int tok[X_SEQ], tgt[X_SEQ];
    srand(123 + 1009*SEED);   // SEED varies the data ORDER too
    double accema=-1, bpbema=-1;
    const char* ae=getenv("ACCUM"); int ACCUM = ae?atoi(ae):1;
    const char* se=getenv("STEPS"); int NSTEP = se?atoi(se):2500;

    i64 (*Rbuf)[X_SEQ*X_D] = (i64(*)[X_SEQ*X_D])malloc(sizeof(i64)*X_SEQ*X_D*(NBLK+1));  // residual stream between blocks
    i64 gR_up[X_SEQ*X_D], gR_dn[X_SEQ*X_D];

    double final_bpb=chance_bpb, final_acc=1.0/X_VOCAB;
    for(int step=0; step<NSTEP; step++){
        for(int k=0;k<ACCUM;k++){
            long off = rand() % (ndata - X_SEQ - 2);
            for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
            // ---- FORWARD cascade ----
            mb_embed(W[0], Rbuf[0], tok);
            for(int b=0;b<NBLK-1;b++) mb_forward_hidden(W[b], A[b], Rbuf[b]);
            for(int b=0;b<NBLK-1;b++) for(int i=0;i<X_SEQ*X_D;i++) Rbuf[b+1][i]=A[b].RP[i];
            mb_forward_output(W[NBLK-1], A[NBLK-1], Rbuf[NBLK-1], tgt);
            // ---- BACKWARD cascade ----
            mb_backward_output(W[NBLK-1], A[NBLK-1], G[NBLK-1], gR_up, k==0);
            for(int b=NBLK-2;b>=0;b--){ mb_backward_hidden(W[b], A[b], G[b], gR_up, gR_dn, k==0); for(int i=0;i<X_SEQ*X_D;i++) gR_up[i]=gR_dn[i]; }
            mb_embed_bwd(G[0], gR_up, tok);   // gR_up now = grad on R0 (block-0 input residual)
        }
        for(int b=0;b<NBLK;b++) mp_step(W[b], G[b]);

        { int c=0; for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40); int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=A[NBLK-1].LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} } if(am==tgt[t])c++; }
          double ac=(double)c/X_SEQ; double bp=batch_bpb(A[NBLK-1],tgt);
          accema=accema<0?ac:0.99*accema+0.01*ac; bpbema=bpbema<0?bp:0.99*bpbema+0.01*bp; final_acc=accema; final_bpb=bpbema; }
        if(step%200==0||step==NSTEP-1){
            int correct=0; int spread=0;
            for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40),mn=((i64)1)<<40; int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=A[NBLK-1].LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} if(lv<mn)mn=lv; }
                if(am==tgt[t])correct++; if((int)(mx-mn)>spread)spread=(int)(mx-mn); }
            printf("  step %4d  bpb_ema=%.3f  acc_ema=%.3f  batch acc=%.3f spread=%d\n",
                   step, bpbema, accema, (double)correct/X_SEQ, spread);
        }
    }
    printf("RESULT mode=%s NBLK=%d SEED=%d ATTN_DOT=%d final_bpb=%.4f final_acc=%.4f chance_bpb=%.4f\n",
           mp_name(), NBLK, SEED, adot, final_bpb, final_acc, chance_bpb);
    free(Rbuf);
    return 0;
}
