// forge_pc_xformer_bf16_dump.cpp — dump for ONE batch what torch_check.py needs to autograd the SAME
// wide-float forward (bf_forward on the FLOAT master) and compare to bf_backward.  -> /tmp/bf16bwd/dump.txt
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"

static void dvec(FILE*f,const char*nm,const double*x,int n){ fprintf(f,"%s %d",nm,n); for(int i=0;i<n;i++) fprintf(f," %.12g",x[i]); fprintf(f,"\n"); }
static void dyad_dump(FILE*f,const char*nm,const double*Wf,int n){ fprintf(f,"%s %d",nm,n); for(int i=0;i<n;i++) fprintf(f," %.12g",bf_dyad_f(Wf[i])); fprintf(f,"\n"); }

int main(){
    g_lut_init(); g_exp_init(); setenv("BF16_BWD","1",1); (void)bf16_bwd();
    const char* path=getenv("SHAKESPEARE"); char def[256]; if(!path){ snprintf(def,256,"%s/.cache/tiur/tinyshakespeare.txt",getenv("HOME")); path=def; }
    FILE* fp=fopen(path,"rb"); fseek(fp,0,SEEK_END); long n=ftell(fp); fseek(fp,0,SEEK_SET); char* txt=(char*)malloc(n); long rd=fread(txt,1,n,fp); fclose(fp); (void)rd;
    int map[256]; for(int i=0;i<256;i++) map[i]=-1; int vocab=0;
    for(long i=0;i<n && vocab<X_VOCAB;i++){ unsigned char c=txt[i]; if(map[c]<0) map[c]=vocab++; }
    int* data=(int*)malloc(sizeof(int)*n); long ndata=0; for(long i=0;i<n;i++){ int id=map[(unsigned char)txt[i]]; data[ndata++]= id<0?0:id; }

    static BFM m; static BFA a; static BFG bg; static BFW bw;
    bf_init(m, getenv("SEED")?atoi(getenv("SEED")):0); bg.alloc();
    int tok[X_SEQ], tgt[X_SEQ];
    int WARM = getenv("WARMSTEPS")?atoi(getenv("WARMSTEPS")):0;
    double LR = getenv("XF_LR")?atof(getenv("XF_LR")):3e-3;
    srand(123);
    for(int s=0;s<WARM;s++){ long o=rand()%(ndata-X_SEQ-2); for(int t=0;t<X_SEQ;t++){tok[t]=data[o+t];tgt[t]=data[o+t+1];}
        bf_forward(m,a,tok); bf_backward(m,a,bg,tok,tgt,true); bf_step(m,bg,bw,LR); }
    long off=100; for(int t=0;t<X_SEQ;t++){ tok[t]=data[off+t]; tgt[t]=data[off+t+1]; }
    bf_forward(m,a,tok); bf_backward(m,a,bg,tok,tgt,true);

    FILE*f=fopen("/tmp/bf16bwd/dump.txt","w");
    fprintf(f,"DIMS %d %d %d %d %d %d %d\n", X_D,X_H,X_DH,X_FFN,X_SEQ,X_VOCAB,X_ABITS);
    fprintf(f,"TOK %d",X_SEQ); for(int t=0;t<X_SEQ;t++) fprintf(f," %d",tok[t]); fprintf(f,"\n");
    fprintf(f,"TGT %d",X_SEQ); for(int t=0;t<X_SEQ;t++) fprintf(f," %d",tgt[t]); fprintf(f,"\n");
    dyad_dump(f,"WQd",m.WQ,X_D*X_D); dyad_dump(f,"WKd",m.WK,X_D*X_D); dyad_dump(f,"WVd",m.WV,X_D*X_D);
    dyad_dump(f,"WOd",m.WO,X_D*X_D); dyad_dump(f,"WUPd",m.WUP,X_FFN*X_D); dyad_dump(f,"WDNd",m.WDN,X_D*X_FFN);
    dyad_dump(f,"WHEADd",m.WHEAD,X_VOCAB*X_D);
    dvec(f,"LN1G",m.LN1G,X_D); dvec(f,"LN1B",m.LN1B,X_D); dvec(f,"LN2G",m.LN2G,X_D); dvec(f,"LN2B",m.LN2B,X_D); dvec(f,"LNFG",m.LNFG,X_D); dvec(f,"LNFB",m.LNFB,X_D);
    dvec(f,"EMB",m.EMB,X_VOCAB*X_D); dvec(f,"POS",m.POS,X_SEQ*X_D);
    dvec(f,"LG",a.LG,X_SEQ*X_VOCAB);
    dvec(f,"gWQ",bg.WQ.data(),X_D*X_D); dvec(f,"gWK",bg.WK.data(),X_D*X_D); dvec(f,"gWV",bg.WV.data(),X_D*X_D);
    dvec(f,"gWO",bg.WO.data(),X_D*X_D); dvec(f,"gWUP",bg.WUP.data(),X_FFN*X_D); dvec(f,"gWDN",bg.WDN.data(),X_D*X_FFN);
    dvec(f,"gWHEAD",bg.WHEAD.data(),X_VOCAB*X_D); dvec(f,"gR",bg.gR.data(),X_SEQ*X_D);
    dvec(f,"gATTNi",bg.dbg_gATTN,X_SEQ*X_D); dvec(f,"gVi",bg.dbg_gV,X_SEQ*X_D); dvec(f,"gQi",bg.dbg_gQ,X_SEQ*X_D); dvec(f,"gKi",bg.dbg_gK,X_SEQ*X_D);
    fclose(f); printf("dumped /tmp/bf16bwd/dump.txt\n"); return 0;
}
