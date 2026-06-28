// forge_pc_xformer_bf16_golden_test.cpp — END-TO-END training proof for the BF16/wide-float STE backward.
// BF16_BWD=1/2 : trains the FLOAT-master model (char-LM-faithful wide forward) with the textbook STE backward
//                + char-LM dyadic-Adam. Must reach the char-LM bar (~3.1 bpb @ these dims).
// BF16_BWD=0   : the original INTEGER golden (g_forward/g_backward/g_step) — sanity, reproduces the ~0.15 cap.
//
//   g++ -O2 -I<golden_dir> -I. forge_pc_xformer_bf16_golden_test.cpp -o /tmp/bftest
//   BF16_BWD=1 SEED=0 STEPS=1500 ACCUM=4 SHAKESPEARE=~/.cache/tiur/tinyshakespeare.txt /tmp/bftest
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"

// ---- int-path init (BF16_BWD=0), identical to forge_pc_xformer_golden_test.cpp ----
static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
static void init_int(GW &w, unsigned seed){ srand(seed);
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

int main(){
    g_lut_init(); g_exp_init();
    unsigned seed = getenv("SEED")?atoi(getenv("SEED")):7;
    const char* path = getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* f=fopen(path,"rb"); if(!f){ printf("no data at %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long ndata=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[ndata++]= id<0?0:id; }
    double chance=log2((double)X_VOCAB);
    int BWD = bf16_bwd();
    printf("BF16 bwd test: BF16_BWD=%d (0=int cascade,1=wide-float,2=bf16) SEED=%u D=%d H=%d FFN=%d SEQ=%d VOCAB=%d data=%ld chance_bpb=%.3f\n",
           BWD, seed, X_D,X_H,X_FFN,X_SEQ,X_VOCAB,ndata,chance);

    static GW w; static GA ia; static GG ig;             // int path
    static BFM bm; static BFA ba; static BFG bg; static BFW bw;  // float path
    int USEINT = (BWD==0);
    bg.alloc();
    if(USEINT) init_int(w,seed); else bf_init(bm,seed);

    int tok[X_SEQ], tgt[X_SEQ];
    srand(123); double accema=-1, bpbema=-1;
    int ACCUM = getenv("ACCUM")?atoi(getenv("ACCUM")):4;
    int NSTEP = getenv("STEPS")?atoi(getenv("STEPS")):1500;
    double LR = getenv("XF_LR")?atof(getenv("XF_LR")):3e-3;     // char-LM default
    int WARMUP = getenv("XF_WARMUP")?atoi(getenv("XF_WARMUP")):200;
    double final_bpb=chance, final_acc=1.0/X_VOCAB;

    for(int step=0; step<NSTEP; step++){
        double lr = LR * fmin(1.0, (step+1)/(double)(WARMUP>0?WARMUP:1));   // linear warmup (char-LM)
        for(int k=0;k<ACCUM;k++){
            long off = rand() % (ndata - X_SEQ - 2);
            for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
            if(USEINT){ g_forward(w,ia,tok,tgt); g_backward(w,ia,ig,tok, k==0); }
            else      { bf_forward(bm,ba,tok);   bf_backward(bm,ba,bg,tok,tgt, k==0); }
        }
        if(USEINT) g_step(w,ig); else bf_step(bm,bg,bw,lr);
        // metrics on the LAST sub-batch logits
        { int c=0; double bits=0; const double L2=log(2.0);
          for(int t=0;t<X_SEQ;t++){
            double mx=-1e300; int am=0;
            for(int v=0;v<X_VOCAB;v++){ double lv = USEINT?(double)ia.LG[t*X_VOCAB+v]:ba.LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} }
            if(am==tgt[t])c++;
            double sum=0,pt=0; for(int v=0;v<X_VOCAB;v++){ double lv=USEINT?(double)ia.LG[t*X_VOCAB+v]:ba.LG[t*X_VOCAB+v]; double e=exp(lv-mx); sum+=e; if(v==tgt[t])pt=e; }
            double p=pt/sum; if(p<1e-12)p=1e-12; bits+=-log(p)/L2; }
          double ac=(double)c/X_SEQ, bp=bits/X_SEQ;
          accema=accema<0?ac:0.99*accema+0.01*ac; bpbema=bpbema<0?bp:0.99*bpbema+0.01*bp; final_acc=accema; final_bpb=bpbema;
          if(step%200==0||step==NSTEP-1)
            printf("  step %4d  bpb_ema=%.3f  acc_ema=%.3f (chance bpb=%.3f acc=%.3f)  batch acc=%.3f bpb=%.3f lr=%.1e\n",
                   step, bpbema, accema, chance, 1.0/X_VOCAB, ac, bp, lr);
        }
    }
    printf("RESULT BF16_BWD=%d seed=%u final_bpb=%.4f final_acc=%.4f acc_x_chance=%.2f  (char-LM bar ~3.1 bpb)\n",
           BWD, seed, final_bpb, final_acc, final_acc*X_VOCAB);
    // ---- coherent-text sample from the trained float model (BF16 path) — the unfakeable proof ----
    if(!USEINT && getenv("SAMPLE")){
        char inv[X_VOCAB]; for(int i=0;i<256;i++) if(map[i]>=0) inv[map[i]]=(char)i;
        int ctx[X_SEQ]; for(int t=0;t<X_SEQ;t++) ctx[t]=map['\n']>=0?map['\n']:0;
        int NS = getenv("SAMPLE")?atoi(getenv("SAMPLE")):300; if(NS<=0)NS=300;
        printf("\n===== SAMPLE (%d chars) =====\n", NS);
        unsigned rng=12345;
        for(int s=0; s<NS; s++){
            bf_forward(bm,ba,ctx);
            int last=X_SEQ-1; double mx=-1e300; for(int v=0;v<X_VOCAB;v++){ double l=ba.LG[last*X_VOCAB+v]; if(l>mx)mx=l; }
            double sum=0,pr[X_VOCAB]; for(int v=0;v<X_VOCAB;v++){ pr[v]=exp(ba.LG[last*X_VOCAB+v]-mx); sum+=pr[v]; }
            rng=rng*1103515245u+12345u; double r=((rng>>8)&0xffffff)/(double)0x1000000*sum; int nxt=0; double acc=0;
            for(int v=0;v<X_VOCAB;v++){ acc+=pr[v]; if(acc>=r){ nxt=v; break; } }
            putchar(inv[nxt]);
            for(int t=0;t<X_SEQ-1;t++) ctx[t]=ctx[t+1]; ctx[X_SEQ-1]=nxt;
        }
        printf("\n");
    }
    return 0;
}
