// attn_unit.cpp — isolate + verify the attention backward (softmax Jacobian + dQ/dK/dV) against torch on a
// SOFT-attention batch (small scores -> graded probs -> nonzero Q/K grads). Reuses bf_forward/bf_backward's
// EXACT attention math by calling a standalone replica with injected small Q/K/V, then dumps grads.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"
// reuse the SAME attention fwd/bwd ops as bf_forward/bf_backward, standalone (no surrounding block).
int main(){
    setenv("BF16_BWD","1",1); (void)bf16_bwd();
    srand(getenv("SEED")?atoi(getenv("SEED")):1);
    static double Q[X_SEQ*X_D],K[X_SEQ*X_D],V[X_SEQ*X_D],Praw[X_H*X_SEQ*X_SEQ],ATTN[X_SEQ*X_D];
    // SMALL random Q/K/V so softmax is soft (graded probs) -> Q/K grads nonzero
    for(int i=0;i<X_SEQ*X_D;i++){ Q[i]=(rand()/(double)RAND_MAX-0.5)*2.0; K[i]=(rand()/(double)RAND_MAX-0.5)*2.0; V[i]=(rand()/(double)RAND_MAX-0.5)*4.0; }
    double scale=1.0/sqrt((double)X_DH);
    for(int h=0;h<X_H;h++)for(int i=0;i<X_SEQ;i++){ double sc[X_SEQ],mx=-1e300;
        for(int j=0;j<=i;j++){ double dot=0; for(int d=0;d<X_DH;d++) dot+=Q[i*X_D+h*X_DH+d]*K[j*X_D+h*X_DH+d]; sc[j]=dot*scale; if(sc[j]>mx)mx=sc[j]; }
        double sum=0; for(int j=0;j<=i;j++){ double e=exp(sc[j]-mx); Praw[(h*X_SEQ+i)*X_SEQ+j]=e; sum+=e; }
        for(int j=0;j<=i;j++) Praw[(h*X_SEQ+i)*X_SEQ+j]/=sum;
        for(int d=0;d<X_DH;d++){ double b=0; for(int j=0;j<=i;j++) b+=Praw[(h*X_SEQ+i)*X_SEQ+j]*V[j*X_D+h*X_DH+d]; ATTN[i*X_D+h*X_DH+d]=b; } }
    // upstream grad on ATTN
    static double gATTN[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++) gATTN[i]=(rand()/(double)RAND_MAX-0.5)*2.0;
    // === the EXACT bf_backward attention block ===
    static double gQ[X_SEQ*X_D],gK[X_SEQ*X_D],gV[X_SEQ*X_D]; for(int i=0;i<X_SEQ*X_D;i++){gQ[i]=0;gK[i]=0;gV[i]=0;}
    for(int h=0;h<X_H;h++)for(int i=0;i<X_SEQ;i++){ const double*p=&Praw[(h*X_SEQ+i)*X_SEQ]; double gp[X_SEQ];
        for(int j=0;j<=i;j++){ double s=0; for(int d=0;d<X_DH;d++) s+=gATTN[i*X_D+h*X_DH+d]*V[j*X_D+h*X_DH+d]; gp[j]=s;
            for(int d=0;d<X_DH;d++) gV[j*X_D+h*X_DH+d]+=p[j]*gATTN[i*X_D+h*X_DH+d]; }
        double ex=0; for(int j=0;j<=i;j++) ex+=p[j]*gp[j];
        for(int j=0;j<=i;j++){ double gs=p[j]*(gp[j]-ex)*scale;
            for(int d=0;d<X_DH;d++){ gQ[i*X_D+h*X_DH+d]+=gs*K[j*X_D+h*X_DH+d]; gK[j*X_D+h*X_DH+d]+=gs*Q[i*X_D+h*X_DH+d]; } } }
    FILE*f=fopen("/tmp/bf16bwd/attn_dump.txt","w");
    fprintf(f,"DIMS %d %d %d %d\n",X_D,X_H,X_DH,X_SEQ);
    #define DV(nm,arr,n) { fprintf(f,"%s %d",nm,n); for(int i=0;i<n;i++) fprintf(f," %.12g",arr[i]); fprintf(f,"\n"); }
    DV("Q",Q,X_SEQ*X_D) DV("K",K,X_SEQ*X_D) DV("V",V,X_SEQ*X_D) DV("gATTN",gATTN,X_SEQ*X_D)
    DV("gQ",gQ,X_SEQ*X_D) DV("gK",gK,X_SEQ*X_D) DV("gV",gV,X_SEQ*X_D)
    fclose(f); printf("attn_dump written\n"); return 0;
}
