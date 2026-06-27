// forge_pc_xformer_mp_golden_test.cpp — the Gate-G2 CASCADE multi-precision sweep harness.
// ============================================================================================
// Trains the HAND-ROLLED CASCADE (mp_forward==g_forward / mp_backward / mp_step) on TinyShakespeare
// next-char prediction and reports TRUE BPB (= mean cross-entropy nats / ln2) every ~200 steps + final,
// for MODE = MP_OPT in {int, fix16, fix24, dbl}. The forward + cascade structure are identical across
// modes; only the backward-multiply operand precision + master/Adam representation change.
//
// BPB is the real per-token CE in bits: the forward writes GLOG = softmax_Q16 - onehot, so the target's
// model probability is p = (GLOG[tgt] + 2^PFRAC)/2^PFRAC, and bpb = mean(-log2(p)). chance bpb=log2(VOCAB).
//
// REUSES the original golden_test's: data source (SHAKESPEARE), vocab build, init(), per-step cascade,
// gradient-accumulation loop, seeds/order. The ONLY swaps: g_backward->mp_backward, g_step->mp_step (so
// the optimizer precision is the MODE), plus the BPB metric. MP_OPT=int calls the original g_adam/g_wgrad
// internally, so the int run is bit-identical to forge_pc_xformer_golden_test.cpp (cross-checked below).
//
// Build: see synth_pc_xformer_mp_golden_csim.sh, or:
//   g++ -O2 -I. forge_pc_xformer_mp_golden_test.cpp -o /tmp/mptest
//   MP_OPT=int   SHAKESPEARE=~/.cache/tiur/tinyshakespeare.txt STEPS=2000 /tmp/mptest
// ============================================================================================
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_mp_golden_block.hpp"

// SAME initializer as the original golden_test (RV gaussian, srand(7), WFRAC fractional master), so the
// int run starts byte-identical to the reference.
static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
static void init(GW &w){ srand(7);
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

// true per-token cross-entropy in bits from the cached GLOG (= softmax_Q16 - onehot). Clamps p into a
// floor so a saturated softmax (p=0 in Q16) doesn't give inf; the floor = 1 Q16 LSB = 2^-16.
static double batch_bpb(const GA &a, const int *tgt){
    double bits=0; const double LOG2=log(2.0);
    for(int t=0;t<X_SEQ;t++){
        i64 sm = a.GLOG[t*X_VOCAB+tgt[t]] + ((i64)1<<X_AT_PFRAC);   // Q16 target prob
        double p = (double)sm/(double)((i64)1<<X_AT_PFRAC);
        if (p < 1.0/65536.0) p = 1.0/65536.0; if (p > 1.0) p = 1.0;
        bits += -log(p)/LOG2;
    }
    return bits/X_SEQ;
}

int main(){
    g_lut_init(); g_exp_init();
    (void)mp_mode();   // resolve MP_OPT once up front
    const char* path = getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* f=fopen(path,"rb"); if(!f){ printf("no data at %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long ndata=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[ndata++]= id<0?0:id; }
    double chance_bpb = log2((double)X_VOCAB);
    printf("MP cascade test: MODE=%s  D=%d H=%d FFN=%d SEQ=%d VOCAB=%d vocab_seen=%d data=%ld  chance_bpb=%.3f\n",
           mp_name(), X_D,X_H,X_FFN,X_SEQ,X_VOCAB,vocab,ndata,chance_bpb);

    static GW w; static GA a; static GG g; init(w);
    int tok[X_SEQ], tgt[X_SEQ];
    srand(123); double accema=-1, bpbema=-1;
    const char* ae=getenv("ACCUM"); int ACCUM = ae?atoi(ae):1;
    const char* se=getenv("STEPS"); int NSTEP = se?atoi(se):2000;

    double final_bpb=chance_bpb, final_acc=1.0/X_VOCAB;
    for(int step=0; step<NSTEP; step++){
        for(int k=0;k<ACCUM;k++){
            long off = rand() % (ndata - X_SEQ - 2);
            for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
            g_forward(w,a,tok,tgt);              // FORWARD: identical across modes (the cascade fwd)
            mp_backward(w,a,g,tok, k==0);        // BACKWARD: cascade, wgrad operand precision = MODE
        }
        mp_step(w,g);                            // OPTIMIZER: master+Adam representation = MODE

        { int c=0; for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40); int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} } if(am==tgt[t])c++; }
          double ac=(double)c/X_SEQ; double bp=batch_bpb(a,tgt);
          accema=accema<0?ac:0.99*accema+0.01*ac; bpbema=bpbema<0?bp:0.99*bpbema+0.01*bp; final_acc=accema; final_bpb=bpbema; }
        if(step%200==0||step==NSTEP-1){
            int correct=0; int spread=0;
            for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40),mn=((i64)1)<<40; int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} if(lv<mn)mn=lv; }
                if(am==tgt[t])correct++; if((int)(mx-mn)>spread)spread=(int)(mx-mn); }
            printf("  step %4d  bpb_ema=%.3f  acc_ema=%.3f (chance bpb=%.3f acc=%.3f)  batch acc=%.3f spread=%d\n",
                   step, bpbema, accema, chance_bpb, 1.0/X_VOCAB, (double)correct/X_SEQ, spread);
        }
    }
    printf("RESULT mode=%s final_bpb=%.4f final_acc=%.4f chance_bpb=%.4f acc_x_chance=%.2f\n",
           mp_name(), final_bpb, final_acc, chance_bpb, final_acc*X_VOCAB);
    return 0;
}
