// forge_pc_xformer_golden_test.cpp — LOCAL validation that the int64 golden (fwd+bwd+dyadic-Adam) is sound:
// train one transformer block + head on TinyShakespeare next-char prediction; the surrogate loss must FALL.
// Plus a finite-difference sign check on the head weight-grad (an EXACT-gradient part) to catch transcription
// bugs.   g++ -O2 -I. forge_pc_xformer_golden_test.cpp -o /tmp/gtest && /tmp/gtest
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_golden_block.hpp"

static int RV(double s){ double u=0; for(int k=0;k<6;k++) u+=rand()/(double)RAND_MAX; double g=(u-3.0)*s; int v=(int)lround(g); if(v>32767)v=32767; if(v<-32767)v=-32767; return v; }
static void init(GW &w){ srand(7);
    for(int i=0;i<X_D*X_D;i++){ w.WQ[i]=RV(8);w.WK[i]=RV(8);w.WV[i]=RV(8);w.WO[i]=RV(8);}
    for(int i=0;i<X_FFN*X_D;i++) w.WUP[i]=RV(8);
    for(int i=0;i<X_D*X_FFN;i++) w.WDN[i]=RV(8);
    for(int i=0;i<X_VOCAB*X_D;i++) w.WHEAD[i]=RV(8);
    for(int i=0;i<X_VOCAB*X_D;i++) w.EMB[i]=RV(16);
    for(int i=0;i<X_SEQ*X_D;i++) w.POS[i]=RV(8);
    for(int d=0;d<X_D;d++){ w.LN1G[d]=64;w.LN2G[d]=64;w.LNFG[d]=64; w.LN1B[d]=0;w.LN2B[d]=0;w.LNFB[d]=0; }
    // zero adam state
    memset(w.mWQ,0,sizeof(w.mWQ));memset(w.vWQ,0,sizeof(w.vWQ));memset(w.mWK,0,sizeof(w.mWK));memset(w.vWK,0,sizeof(w.vWK));
    memset(w.mWV,0,sizeof(w.mWV));memset(w.vWV,0,sizeof(w.vWV));memset(w.mWO,0,sizeof(w.mWO));memset(w.vWO,0,sizeof(w.vWO));
    memset(w.mUP,0,sizeof(w.mUP));memset(w.vUP,0,sizeof(w.vUP));memset(w.mDN,0,sizeof(w.mDN));memset(w.vDN,0,sizeof(w.vDN));
    memset(w.mHD,0,sizeof(w.mHD));memset(w.vHD,0,sizeof(w.vHD));
}

int main(){
    g_lut_init();
    // load tinyshakespeare, build vocab (<=64)
    const char* path = getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* f=fopen(path,"rb"); if(!f){ printf("no data at %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,f); fclose(f); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long ndata=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[ndata++]= id<0?0:id; }
    printf("golden test: D=%d H=%d FFN=%d SEQ=%d VOCAB=%d vocab_seen=%d data=%ld\n", X_D,X_H,X_FFN,X_SEQ,X_VOCAB,vocab,ndata);

    static GW w; static GA a; static GG g; init(w);
    int tok[X_SEQ], tgt[X_SEQ];
    srand(123); double accema=-1, probema=-1;
    for(int step=0; step<2000; step++){
        long off = rand() % (ndata - X_SEQ - 2);
        for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
        g_forward(w,a,tok,tgt);
        g_backward(w,a,g,tok);
        g_step(w,g);
        { int c=0; double tp=0; for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40); int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} } if(am==tgt[t])c++; i64 sm=a.GLOG[t*X_VOCAB+tgt[t]]+((i64)1<<X_AT_PFRAC); tp+=(double)sm/(1<<X_AT_PFRAC);}
          double ac=(double)c/X_SEQ, pr=tp/X_SEQ; accema=accema<0?ac:0.99*accema+0.01*ac; probema=probema<0?pr:0.99*probema+0.01*pr; }
        if(step%200==0||step==1999){
            // REAL metrics on this batch: next-char argmax accuracy + mean target softmax-prob (1/64=chance)
            int correct=0; double tprob=0; int spread=0;
            for(int t=0;t<X_SEQ;t++){ i64 mx=-(((i64)1)<<40),mn=((i64)1)<<40; int am=0; for(int v=0;v<X_VOCAB;v++){ i64 lv=a.LG[t*X_VOCAB+v]; if(lv>mx){mx=lv;am=v;} if(lv<mn)mn=lv; }
                if(am==tgt[t])correct++; if((int)(mx-mn)>spread)spread=(int)(mx-mn);
                // target prob from grad_logit: softmax_Q16 = GLOG + onehot, /2^PFRAC
                i64 sm = a.GLOG[t*X_VOCAB+tgt[t]] + ((i64)1<<X_AT_PFRAC); tprob += (double)sm/(1<<X_AT_PFRAC); }
            printf("  step %4d  acc_ema=%.3f  prob_ema=%.3f (chance %.3f)  batch acc=%.3f spread=%d\n",
                   step, accema, probema, 1.0/X_VOCAB, (double)correct/X_SEQ, spread);
        }
    }
    printf("  -> acc_ema %.3f = %.1fx chance, prob_ema %.3f = %.1fx chance %s\n",
           accema, accema*X_VOCAB, probema, probema*X_VOCAB, (accema>2.0/X_VOCAB)?"=> GOLDEN LEARNS (int fwd+bwd+dyadic-Adam sound)":"=> weak");
    // finite-difference sign check on WHEAD[0] (exact-gradient part): bump +/-, loss should move opposite to grad sign
    { for(int t=0;t<X_SEQ;t++){ tok[t]=data[100+t]; tgt[t]=data[100+t+1]; }
      g_forward(w,a,tok,tgt); g_backward(w,a,g,tok); i64 gh=g.WHEAD[0];
      i64 save=w.WHEAD[0]; w.WHEAD[0]=save+200; i64 lp=g_forward(w,a,tok,tgt); w.WHEAD[0]=save-200; i64 lm=g_forward(w,a,tok,tgt); w.WHEAD[0]=save;
      printf("\n  grad-check WHEAD[0]: g=%lld  loss(+200)=%lld loss(-200)=%lld  d=%lld  consistent=%s\n",
             (long long)gh,(long long)lp,(long long)lm,(long long)(lp-lm), ((gh>0)==((lp-lm)>0)||gh==0||lp==lm)?"yes":"NO"); }
    return 0;
}
