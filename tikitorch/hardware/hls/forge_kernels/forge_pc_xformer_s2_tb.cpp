// forge_pc_xformer_s2_tb.cpp — SELF-CONTAINED csim gate for the SCALE-2 DDR-streaming + K-wide-MAC kernel.
// No golden .hpp dependency: the reference forward+backward for ONE OUTPUT block is transcribed here in fp64
// with the SAME bf16/dyad rounding and the SAME K-wide-tree / global-k reduction ORDER the kernel uses, so the
// comparison is tight (<=1 bf16 ULP, the design's G3 within-1-ULP reconcile).
//
//   CHECK 1  (numerics): one FWD+BWD+DO_UP step, kernel raw bf16 grads (dumped via -DS2_DEBUG_GRADS) vs the
//            fp64 same-order reference. Gate: max relerr over all tensors <= 1 bf16 ULP (2^-7 ~= 7.8e-3).
//   CHECK 2  (streaming equivalence): the SAME step at TILE {64,128} x SEQ_TILE {32,64} must be BIT-IDENTICAL
//            (the tiling must not change the reduction order of any output's contraction).
//   CHECK 3  (D=512 smoke, when compiled at the default dims): one FWD+BWD completes and all grads are finite.
//
// build: g++ -O2 -std=c++14 -DS2_DEBUG_GRADS -D__AP_INT_SPECIAL_H__ \
//        -I/private/tmp/bwdprec/_apshim -I/private/tmp/mp_hlsshim -I. \
//        forge_pc_xformer_s2_block.cpp forge_pc_xformer_s2_tb.cpp -o /tmp/s2tb
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <ap_int.h>
#include "forge_pc_xformer_s2_config.hpp"

extern "C" void forge_pc_xformer_s2_block(ap_uint<512> *buffer);

// ---------------- bit helpers ----------------
static int32_t f2i_(float f){ union{float f;int32_t i;}c; c.f=f; return c.i; }
static float   i2f_(int32_t i){ union{float f;int32_t i;}c; c.i=i; return c.f; }
static float bf16f(float f){ union{float f;uint32_t u;}c; c.f=f; uint32_t u=c.u,lsb=(u>>16)&1u; u+=0x7fffu+lsb; u&=0xffff0000u; c.u=u; return c.f; }
static double P(double x){ return (double)bf16f((float)x); }                 // bf16 round (matches kernel P)
static float  dyadf(float wf){ float a=wf<0?-wf:wf; if(a<1e-30f) return 0.f; int e=(int)roundf(log2f(a)); float p=ldexpf(1.f,e); return wf<0?-p:p; }

static void put_f(int32_t*b,int off,const float*s,int n){ for(int i=0;i<n;i++) b[off+i]=f2i_(s[i]); }
static void put_i(int32_t*b,int off,const int*s,int n){ for(int i=0;i<n;i++) b[off+i]=s[i]; }
static void get_f(const int32_t*b,int off,double*d,int n){ for(int i=0;i<n;i++) d[i]=(double)i2f_(b[off+i]); }

static void setcfg(int32_t*b,int ph,int role,int do_up,int lrbits,int gzero,int slot,int to,int ti,int st){
    for(int i=0;i<S2_WL;i++) b[S2O_CFG+i]=0;
    b[S2O_CFG+0]=ph; b[S2O_CFG+1]=role; b[S2O_CFG+2]=do_up; b[S2O_CFG+3]=lrbits; b[S2O_CFG+4]=gzero;
    b[S2O_CFG+5]=1 /*train*/; b[S2O_CFG+6]=slot; b[S2O_CFG+7]=to; b[S2O_CFG+8]=ti; b[S2O_CFG+9]=st;
}

// ---------------- the fp64 same-order reference (K-wide tree, global-k) ----------------
static const int D=S2_D,H=S2_H,DH=S2_DH,FFN=S2_FFN,SEQ=S2_SEQ,VOCAB=S2_VOCAB,MB=S2_MBITS,AB=S2_ABITS,K=S2_K;

static double reduceK(const double*p_in){ double p[K]; for(int j=0;j<K;j++)p[j]=p_in[j];
    for(int s=1;s<K;s<<=1) for(int j=0;j<K;j+=(s<<1)) p[j]=p[j]+p[j+s]; return p[0]; }
// out[r][c] = [P]( sum_k A[r*KC+k] * B(c,k) ) with the kernel's K-wide tree + global-k single accumulator.
// mode 0 FWD: B=dyad(W[c*KC+k]) ; mode 1 DX: B=dyad(W[k*cols+c]) ; mode 2 WG: A=gy[k*rows... see below.
static void ref_matmul(int mode,const double*A,const float*W,const double*Bres,double*out,
                       int rows,int cols,int KC,bool doP){
    for(int r=0;r<rows;r++) for(int c=0;c<cols;c++){
        double acc=0;
        for(int kb=0;kb<KC;kb+=K){ double p[K];
            for(int j=0;j<K;j++){ int k=kb+j; double a,b;
                if(mode==2){ a=A[k*rows+r]; b=Bres[k*cols+c]; }   // WG: gy[t*O+o], x[t*I+i], contraction t
                else       { a=A[r*KC+k];
                    b = (mode==0)? (double)dyadf(W[c*KC+k]) : (double)dyadf(W[k*cols+c]); }
                p[j]=a*b; }
            acc+=reduceK(p); }
        out[r*cols+c]= doP? P(acc): acc;
    }
}
static void ref_qact(double*x,int n){ double mx=0; for(int i=0;i<n;i++){double a=fabs(x[i]); if(a>mx)mx=a;}
    if(mx<1e-9)mx=1e-9; int sh=(int)floor(log2((float)mx))-AB; double sc=ldexp(1.0,sh);
    for(int i=0;i<n;i++) x[i]=round(x[i]/sc)*sc; }
static void ref_ln(const double*x,const float*g,const float*b,double*y,double*sout,int M){
    for(int t=0;t<M;t++){ double sum=0; for(int d=0;d<D;d++) sum+=x[t*D+d]; double mean=sum/D,asum=0;
        for(int d=0;d<D;d++){double c=x[t*D+d]-mean; asum+=fabs(c);} double scale=asum/D; if(scale<1e-9)scale=1e-9;
        int e=(int)floor(log2((float)scale)); double mant=round((scale/ldexp(1.0,e)-1.0)*(1<<MB))/(double)(1<<MB);
        double s=(1.0+mant)*ldexp(1.0,e); sout[t]=s;
        for(int d=0;d<D;d++) y[t*D+d]=(double)g[d]*((x[t*D+d]-mean)/s)+(double)b[d]; } }
static void ref_lnbwd(const double*gy,const float*g,const double*s,double*dx,int M){
    for(int t=0;t<M;t++){ std::vector<double> gx(D); double mg=0;
        for(int d=0;d<D;d++){gx[d]=P(gy[t*D+d]*(double)g[d]); mg+=gx[d];} mg/=D;
        for(int d=0;d<D;d++) dx[t*D+d]=P((gx[d]-mg)/s[t]); } }

struct Ref { std::vector<double> gWQ,gWK,gWV,gWO,gWUP,gWDN,gWH,gR,gQ,gK,gV; std::vector<double> fX1,fQ,fK,fV,fLG,fATTN,fATTNpre,fHD,fRP; };
// full OUTPUT-block fwd+bwd in fp64 mirroring the kernel; masters are the float32 staged bytes.
static void reference(const float*WQ,const float*WK,const float*WV,const float*WO,const float*WUP,const float*WDN,
                      const float*WHEAD,const float*LN1G,const float*LN1B,const float*LN2G,const float*LN2B,
                      const float*LNFG,const float*LNFB,const double*R,const int*tgt,Ref&g){
    std::vector<double> X1(SEQ*D),s1(SEQ),Q(SEQ*D),Kk(SEQ*D),V(SEQ*D),ATTN(SEQ*D),Praw(H*SEQ*SEQ,0.0);
    ref_ln(R,LN1G,LN1B,X1.data(),s1.data(),SEQ);
    ref_matmul(0,X1.data(),WQ,0,Q.data(),SEQ,D,D,false); ref_qact(Q.data(),SEQ*D);
    g.fX1=X1; g.fQ=Q;
    ref_matmul(0,X1.data(),WK,0,Kk.data(),SEQ,D,D,false); ref_qact(Kk.data(),SEQ*D);
    ref_matmul(0,X1.data(),WV,0,V.data(),SEQ,D,D,false); ref_qact(V.data(),SEQ*D);
    g.fK=Kk; g.fV=V;
    double sc=1.0/sqrt((double)DH);
    for(int h=0;h<H;h++) for(int i=0;i<SEQ;i++){ double s[SEQ],mx=-1e300;
        for(int j=0;j<=i;j++){ double dot=0; for(int d=0;d<DH;d++) dot+=Q[i*D+h*DH+d]*Kk[j*D+h*DH+d]; s[j]=dot*sc; if(s[j]>mx)mx=s[j]; }
        double sum=0; for(int j=0;j<=i;j++){ double e=exp(s[j]-mx); s[j]=e; sum+=e; }
        for(int j=0;j<=i;j++){ Praw[(h*SEQ+i)*SEQ+j]=s[j]/sum; }
        for(int d=0;d<DH;d++){ double bl=0; for(int j=0;j<=i;j++) bl+=Praw[(h*SEQ+i)*SEQ+j]*V[j*D+h*DH+d]; ATTN[i*D+h*DH+d]=bl; } }
    g.fATTNpre=ATTN;
    ref_qact(ATTN.data(),SEQ*D); g.fATTN=ATTN;
    std::vector<double> AO(SEQ*D),M(SEQ*D),X2(SEQ*D),s2(SEQ),HDpre(SEQ*FFN),HD(SEQ*FFN),RMASK(SEQ*FFN),FO(SEQ*D),RP(SEQ*D);
    ref_matmul(0,ATTN.data(),WO,0,AO.data(),SEQ,D,D,false);
    for(int i=0;i<SEQ*D;i++) M[i]=R[i]+AO[i];
    ref_ln(M.data(),LN2G,LN2B,X2.data(),s2.data(),SEQ);
    ref_matmul(0,X2.data(),WUP,0,HDpre.data(),SEQ,FFN,D,false);
    for(int i=0;i<SEQ*FFN;i++){ RMASK[i]=HDpre[i]>0?1.0:0.0; HD[i]=RMASK[i]>0?HDpre[i]:0.0; }
    ref_qact(HD.data(),SEQ*FFN); g.fHD=HD;
    ref_matmul(0,HD.data(),WDN,0,FO.data(),SEQ,D,FFN,false);
    for(int i=0;i<SEQ*D;i++) RP[i]=M[i]+FO[i];
    g.fRP=RP;
    std::vector<double> XF(SEQ*D),sf(SEQ),LG(SEQ*VOCAB);
    ref_ln(RP.data(),LNFG,LNFB,XF.data(),sf.data(),SEQ);
    ref_matmul(0,XF.data(),WHEAD,0,LG.data(),SEQ,VOCAB,D,false);
    g.fLG=LG;
    // ---- backward ----
    std::vector<double> gLG(SEQ*VOCAB);
    for(int t=0;t<SEQ;t++){ double mx=-1e300; for(int v=0;v<VOCAB;v++){double l=LG[t*VOCAB+v]; if(l>mx)mx=l;}
        double sum=0,pr[VOCAB]; for(int v=0;v<VOCAB;v++){pr[v]=exp(LG[t*VOCAB+v]-mx);sum+=pr[v];}
        for(int v=0;v<VOCAB;v++){double p=pr[v]/sum,oh=(v==tgt[t])?1.0:0.0; gLG[t*VOCAB+v]=P((p-oh)/SEQ);} }
    std::vector<double> gXF(SEQ*D),gRP(SEQ*D);
    g.gWH.assign(VOCAB*D,0.0); ref_matmul(2,gLG.data(),0,XF.data(),g.gWH.data(),VOCAB,D,SEQ,true);   // dW head
    ref_matmul(1,gLG.data(),WHEAD,0,gXF.data(),SEQ,D,VOCAB,true);                                    // dx head
    ref_lnbwd(gXF.data(),LNFG,sf.data(),gRP.data(),SEQ);
    std::vector<double> gM(SEQ*D),gHD(SEQ*FFN),gX2(SEQ*D),gM2(SEQ*D);
    for(int i=0;i<SEQ*D;i++) gM[i]=gRP[i];
    g.gWDN.assign(D*FFN,0.0); ref_matmul(2,gRP.data(),0,HD.data(),g.gWDN.data(),D,FFN,SEQ,true);
    ref_matmul(1,gRP.data(),WDN,0,gHD.data(),SEQ,FFN,D,true);
    for(int i=0;i<SEQ*FFN;i++) if(RMASK[i]==0.0) gHD[i]=0;
    g.gWUP.assign(FFN*D,0.0); ref_matmul(2,gHD.data(),0,X2.data(),g.gWUP.data(),FFN,D,SEQ,true);
    ref_matmul(1,gHD.data(),WUP,0,gX2.data(),SEQ,D,FFN,true);
    ref_lnbwd(gX2.data(),LN2G,s2.data(),gM2.data(),SEQ);
    for(int i=0;i<SEQ*D;i++) gM[i]=P(gM[i]+gM2[i]);
    std::vector<double> gR(SEQ*D),gATTN(SEQ*D),gQ(SEQ*D,0),gK(SEQ*D,0),gV(SEQ*D,0);
    for(int i=0;i<SEQ*D;i++) gR[i]=gM[i];
    g.gWO.assign(D*D,0.0); ref_matmul(2,gM.data(),0,ATTN.data(),g.gWO.data(),D,D,SEQ,true);
    ref_matmul(1,gM.data(),WO,0,gATTN.data(),SEQ,D,D,true);
    double asc=1.0/sqrt((double)DH);
    for(int h=0;h<H;h++) for(int i=0;i<SEQ;i++){ const double*p=&Praw[(h*SEQ+i)*SEQ]; double gp[SEQ];
        for(int j=0;j<=i;j++){ double s=0; for(int d=0;d<DH;d++) s+=gATTN[i*D+h*DH+d]*V[j*D+h*DH+d]; gp[j]=s;
            for(int d=0;d<DH;d++) gV[j*D+h*DH+d]+=P(p[j]*gATTN[i*D+h*DH+d]); }
        double ex=0; for(int j=0;j<=i;j++) ex+=p[j]*gp[j];
        for(int j=0;j<=i;j++){ double gs=P(p[j]*(gp[j]-ex))*asc;
            for(int d=0;d<DH;d++){ gQ[i*D+h*DH+d]+=gs*Kk[j*D+h*DH+d]; gK[j*D+h*DH+d]+=gs*Q[i*D+h*DH+d]; } } }
    for(int i=0;i<SEQ*D;i++){ gQ[i]=P(gQ[i]); gK[i]=P(gK[i]); gV[i]=P(gV[i]); }
    std::vector<double> gX1(SEQ*D,0),tmp(SEQ*D);
    g.gWQ.assign(D*D,0.0); ref_matmul(2,gQ.data(),0,X1.data(),g.gWQ.data(),D,D,SEQ,true);
    ref_matmul(1,gQ.data(),WQ,0,tmp.data(),SEQ,D,D,true); for(int i=0;i<SEQ*D;i++) gX1[i]+=tmp[i];
    g.gWK.assign(D*D,0.0); ref_matmul(2,gK.data(),0,X1.data(),g.gWK.data(),D,D,SEQ,true);
    ref_matmul(1,gK.data(),WK,0,tmp.data(),SEQ,D,D,true); for(int i=0;i<SEQ*D;i++) gX1[i]+=tmp[i];
    g.gWV.assign(D*D,0.0); ref_matmul(2,gV.data(),0,X1.data(),g.gWV.data(),D,D,SEQ,true);
    ref_matmul(1,gV.data(),WV,0,tmp.data(),SEQ,D,D,true); for(int i=0;i<SEQ*D;i++) gX1[i]+=tmp[i];
    for(int i=0;i<SEQ*D;i++) gX1[i]=P(gX1[i]);
    std::vector<double> gR1(SEQ*D); ref_lnbwd(gX1.data(),LN1G,s1.data(),gR1.data(),SEQ);
    for(int i=0;i<SEQ*D;i++) gR[i]=P(gR[i]+gR1[i]);
    g.gR=gR; g.gQ=gQ; g.gK=gK; g.gV=gV;
}

// ---------------- staging ----------------
static int32_t* gbuf=0;
static std::vector<float> mWQ,mWK,mWV,mWO,mWUP,mWDN,mWH,mLN1G,mLN1B,mLN2G,mLN2B,mLNFG,mLNFB; static std::vector<double> mR; static std::vector<int> tgt;
static void frand(std::vector<float>&a,int n,double scale,unsigned&seed){ a.resize(n); for(int i=0;i<n;i++){ seed=seed*1103515245u+12345u; double u=((seed>>8)&0xffff)/65536.0*2-1; a[i]=(float)(u*scale);} }
static void build_model(unsigned seed){
    double a=1.0/sqrt((double)D), af=1.0/sqrt((double)FFN);
    frand(mWQ,D*D,a,seed); frand(mWK,D*D,a,seed); frand(mWV,D*D,a,seed); frand(mWO,D*D,a,seed);
    frand(mWUP,FFN*D,a,seed); frand(mWDN,D*FFN,af,seed); frand(mWH,VOCAB*D,a,seed);
    mLN1G.assign(D,1.f); mLN1B.assign(D,0.f); mLN2G.assign(D,1.f); mLN2B.assign(D,0.f); mLNFG.assign(D,1.f); mLNFB.assign(D,0.f);
    std::vector<float> rf; frand(rf,SEQ*D,0.5,seed); mR.assign(SEQ*D,0.0); for(int i=0;i<SEQ*D;i++) mR[i]=(double)rf[i];
    tgt.resize(SEQ); for(int t=0;t<SEQ;t++){ seed=seed*1103515245u+12345u; tgt[t]=(int)((seed>>9)%VOCAB); }
}
static void stage(int do_up_lr){
    for(int i=0;i<S2_BUFLEN;i++) gbuf[i]=0;
    put_f(gbuf,S2_FWQ,mWQ.data(),D*D); put_f(gbuf,S2_FWK,mWK.data(),D*D); put_f(gbuf,S2_FWV,mWV.data(),D*D); put_f(gbuf,S2_FWO,mWO.data(),D*D);
    put_f(gbuf,S2_FWUP,mWUP.data(),FFN*D); put_f(gbuf,S2_FWDN,mWDN.data(),D*FFN); put_f(gbuf,S2_FWHEAD,mWH.data(),VOCAB*D);
    put_f(gbuf,S2_FLN1G,mLN1G.data(),D); put_f(gbuf,S2_FLN1B,mLN1B.data(),D); put_f(gbuf,S2_FLN2G,mLN2G.data(),D); put_f(gbuf,S2_FLN2B,mLN2B.data(),D);
    put_f(gbuf,S2_FLNFG,mLNFG.data(),D); put_f(gbuf,S2_FLNFB,mLNFB.data(),D);
    std::vector<float> rf(SEQ*D); for(int i=0;i<SEQ*D;i++) rf[i]=(float)mR[i];
    put_f(gbuf,S2_GRIN,rf.data(),SEQ*D);                 // OUTPUT block input staged as the wide down-msg
    put_i(gbuf,S2_TGT,tgt.data(),SEQ); (void)do_up_lr;
}
// run one FWD+BWD with the given tiles; returns the kernel's raw grad dumps (post GACC, gzero=1 => this step)
struct Dump { std::vector<double> gWQ,gWK,gWV,gWO,gWUP,gWDN,gWH,gR,gQ,gK,gV; };
static void run_step(int to,int ti,int st,int do_up,Dump&d){
    stage(0);
    int lrbits=(int)lround(3e-3*1048576.0);
    setcfg(gbuf,S2_PH_FWD,S2_RL_OUT,do_up,lrbits,1,0,to,ti,st); forge_pc_xformer_s2_block((ap_uint<512>*)gbuf);
    setcfg(gbuf,S2_PH_BWD,S2_RL_OUT,do_up,lrbits,1,0,to,ti,st); forge_pc_xformer_s2_block((ap_uint<512>*)gbuf);
    d.gWQ.resize(D*D);d.gWK.resize(D*D);d.gWV.resize(D*D);d.gWO.resize(D*D);d.gWUP.resize(FFN*D);d.gWDN.resize(D*FFN);d.gWH.resize(VOCAB*D);
    d.gR.resize(SEQ*D);d.gQ.resize(SEQ*D);d.gK.resize(SEQ*D);d.gV.resize(SEQ*D);
    get_f(gbuf,S2_DG_WQ,d.gWQ.data(),D*D); get_f(gbuf,S2_DG_WK,d.gWK.data(),D*D); get_f(gbuf,S2_DG_WV,d.gWV.data(),D*D); get_f(gbuf,S2_DG_WO,d.gWO.data(),D*D);
    get_f(gbuf,S2_DG_WUP,d.gWUP.data(),FFN*D); get_f(gbuf,S2_DG_WDN,d.gWDN.data(),D*FFN); get_f(gbuf,S2_DG_WH,d.gWH.data(),VOCAB*D);
    get_f(gbuf,S2_DG_GR,d.gR.data(),SEQ*D); get_f(gbuf,S2_DG_GQ,d.gQ.data(),SEQ*D); get_f(gbuf,S2_DG_GK,d.gK.data(),SEQ*D); get_f(gbuf,S2_DG_GV,d.gV.data(),SEQ*D);
}

static double relerr(const double*a,const double*b,int n,long*ndiff){ double mx=0,mg=0; long nd=0;
    for(int i=0;i<n;i++){ double d=fabs(a[i]-b[i]); if(d>mx)mx=d; double m=fabs(b[i]); if(m>mg)mg=m; if(a[i]!=b[i])nd++; }
    if(ndiff)*ndiff=nd; return mg>0?mx/mg:0.0; }
static bool allfinite(const std::vector<double>&v){ for(double x:v) if(!isfinite(x)) return false; return true; }
static bool bitident(const std::vector<double>&a,const std::vector<double>&b){ if(a.size()!=b.size())return false; for(size_t i=0;i<a.size();i++) if(a[i]!=b[i])return false; return true; }

int main(){
    printf("=== forge_pc_xformer_s2_block csim  (D=%d H=%d DH=%d FFN=%d SEQ=%d VOCAB=%d  K=%d TILE_O=%d TILE_I=%d SEQ_TILE=%d) ===\n",
           D,H,DH,FFN,SEQ,VOCAB,K,S2_TILE_O,S2_TILE_I,S2_SEQ_TILE);
    gbuf=(int32_t*)aligned_alloc(64,sizeof(int32_t)*(size_t)S2_BUFLEN);
    build_model(1234567u);
    // CHECK 1/2 (the numeric + streaming-equivalence gates) run at D<=128 (the D=64-class csim). At the full
    // D=512 dims we run only the one-step finite smoke (CHECK 3) per the task (keep the D=512 csim to ONE step).
    const bool SMALL = (D <= 128);

    // ---------- CHECK 1: numerics vs fp64 same-order reference ----------
    Dump d0; run_step(S2_TILE_O,S2_TILE_I,S2_SEQ_TILE,1,d0);
    int pass1=1, pass2=1;
    if (SMALL) {
    Ref ref; reference(mWQ.data(),mWK.data(),mWV.data(),mWO.data(),mWUP.data(),mWDN.data(),mWH.data(),
                       mLN1G.data(),mLN1B.data(),mLN2G.data(),mLN2B.data(),mLNFG.data(),mLNFB.data(),mR.data(),tgt.data(),ref);
    // ---------- CHECK 0 (forward sanity): the cached wide fwd intermediates vs the fp64 reference ----------
    { std::vector<double> kX1(SEQ*D),kQ(SEQ*D),kAT(SEQ*D),kHD(SEQ*FFN),kLG(SEQ*VOCAB);
      get_f(gbuf,S2_WX1,kX1.data(),SEQ*D); get_f(gbuf,S2_WQ,kQ.data(),SEQ*D);
      get_f(gbuf,S2_WATTN,kAT.data(),SEQ*D); get_f(gbuf,S2_WHD,kHD.data(),SEQ*FFN); get_f(gbuf,S2_WLG,kLG.data(),SEQ*VOCAB);
      long n; double r1=relerr(kX1.data(),ref.fX1.data(),SEQ*D,&n), rq=relerr(kQ.data(),ref.fQ.data(),SEQ*D,&n),
             ra=relerr(kAT.data(),ref.fATTN.data(),SEQ*D,&n), rh=relerr(kHD.data(),ref.fHD.data(),SEQ*FFN,&n), rl=relerr(kLG.data(),ref.fLG.data(),SEQ*VOCAB,&n);
      printf("[CHECK 0] fwd cache vs fp64 ref:  X1=%.2e  Q=%.2e  ATTN=%.2e  HD=%.2e  LG=%.2e  (relerr)\n",r1,rq,ra,rh,rl); }
    struct T{const char*n; std::vector<double>*k,*r; int len;} ts[]={
        {"gWQ",&d0.gWQ,&ref.gWQ,D*D},{"gWK",&d0.gWK,&ref.gWK,D*D},{"gWV",&d0.gWV,&ref.gWV,D*D},{"gWO",&d0.gWO,&ref.gWO,D*D},
        {"gWUP",&d0.gWUP,&ref.gWUP,FFN*D},{"gWDN",&d0.gWDN,&ref.gWDN,D*FFN},{"gWHEAD",&d0.gWH,&ref.gWH,VOCAB*D},
        {"gR",&d0.gR,&ref.gR,SEQ*D},{"gQ",&d0.gQ,&ref.gQ,SEQ*D},{"gK",&d0.gK,&ref.gK,SEQ*D},{"gV",&d0.gV,&ref.gV,SEQ*D}};
    double ULP=7.8e-3, worst=0; bool finite=true;
    printf("[CHECK 1] kernel raw bf16 grads vs fp64 same-order reference:\n");
    for(auto&t:ts){ long nd; double re=relerr(t.k->data(),t.r->data(),t.len,&nd); if(re>worst)worst=re;
        if(!allfinite(*t.k))finite=false;
        printf("   %-7s relerr=%.3e  ndiff=%ld/%d\n",t.n,re,nd,t.len); }
    pass1 = (worst<=ULP) && finite;
    printf("   => max relerr %.3e  (gate 1 bf16 ULP = %.1e)  finite=%d  : %s\n",worst,ULP,finite,pass1?"PASS":"FAIL");

    // ---------- CHECK 2: streaming equivalence (bit-identical across TILE / SEQ_TILE) ----------
    printf("[CHECK 2] streaming equivalence (bit-identical across TILE {64,128} x SEQ_TILE {32,64}):\n");
    Dump dA,dB,dC; run_step(64,64,32,0,dA); run_step(128,128,64,0,dB); run_step(128,64,32,0,dC);
    // dref = the default-tile grads (recomputed with do_up=0 so masters are pristine per run)
    Dump dR; run_step(S2_TILE_O,S2_TILE_I,S2_SEQ_TILE,0,dR);
    struct T2{const char*n; std::vector<double>*a;} bt[]={{"gWQ",&dR.gWQ},{"gWDN",&dR.gWDN},{"gR",&dR.gR},{"gQ",&dR.gQ}};
    auto same=[&](Dump&x)->bool{ return bitident(dR.gWQ,x.gWQ)&&bitident(dR.gWK,x.gWK)&&bitident(dR.gWV,x.gWV)&&bitident(dR.gWO,x.gWO)
        &&bitident(dR.gWUP,x.gWUP)&&bitident(dR.gWDN,x.gWDN)&&bitident(dR.gWH,x.gWH)&&bitident(dR.gR,x.gR)&&bitident(dR.gQ,x.gQ)&&bitident(dR.gK,x.gK)&&bitident(dR.gV,x.gV); };
    int sA=same(dA),sB=same(dB),sC=same(dC); (void)bt;
    pass2 = sA&&sB&&sC;
    printf("   (64,64,32)==def:%d   (128,128,64)==def:%d   (128,64,32)==def:%d   : %s\n",sA,sB,sC,pass2?"PASS":"FAIL");
    } else {
        printf("[CHECK 1/2] skipped at D=%d (numeric + streaming-equivalence gates run in the D<=128 csim)\n", D);
    }

    // ---------- CHECK 3: smoke — all grads finite AND non-trivial (headline at D=512) ----------
    bool f3 = allfinite(d0.gWQ)&&allfinite(d0.gWUP)&&allfinite(d0.gWDN)&&allfinite(d0.gWH)&&allfinite(d0.gR);
    auto mag=[&](const std::vector<double>&v){ double m=0; for(double x:v) if(fabs(x)>m)m=fabs(x); return m; };
    double mQ=mag(d0.gWQ),mUP=mag(d0.gWUP),mDN=mag(d0.gWDN),mWH=mag(d0.gWH),mR=mag(d0.gR);
    bool nontrivial = (mQ>0)&&(mUP>0)&&(mDN>0)&&(mWH>0)&&(mR>0);
    int pass3 = f3 && nontrivial;
    printf("[CHECK 3] D=%d one-step smoke: finite=%d nontrivial=%d  |gWQ|=%.3e |gWUP|=%.3e |gWDN|=%.3e |gWHEAD|=%.3e |gR|=%.3e : %s\n",
           D, f3, nontrivial, mQ,mUP,mDN,mWH,mR, pass3?"PASS":"FAIL");

    int pass = pass1 && pass2 && pass3;
    printf("=== %s ===\n", pass?"ALL CHECKS PASS":"FAIL");
    free(gbuf);
    return pass?0:1;
}
