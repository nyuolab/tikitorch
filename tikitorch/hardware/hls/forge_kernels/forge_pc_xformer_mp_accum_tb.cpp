// forge_pc_xformer_mp_accum_tb.cpp — verify GRADIENT ACCUMULATION (ACCUM>1). Run the BWD kick TWICE on the
// SAME input (gzero=1 then gzero=0, DO_UP=0): the persistent accumulators MP_GACC_* must equal the per-kick
// grad g after kick1 and 2g after kick2. (The per-kick grad is the already-bit-exact-verified gradcheck output.)
//   g++ -O2 -std=c++14 -DMP_DEBUG_GRADS -D__AP_INT_SPECIAL_H__ -I<apshim> -I<hlsshim> -I. \
//       forge_pc_xformer_mp_block.cpp forge_pc_xformer_mp_accum_tb.cpp -o /tmp/mpaccum
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_int.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"
#include "forge_pc_xformer_mp_config.hpp"
extern "C" void forge_pc_xformer_mp_block(ap_uint<512> *buffer);
static int32_t f2i_(float f){union{float f;int32_t i;}c;c.f=f;return c.i;}
static float   i2f_(int32_t i){union{float f;int32_t i;}c;c.i=i;return c.f;}
static void put_f(int32_t*b,int o,const double*s,int n){for(int i=0;i<n;i++)b[o+i]=f2i_((float)s[i]);}
static void put_i(int32_t*b,int o,const int*s,int n){for(int i=0;i<n;i++)b[o+i]=s[i];}
static void read_f(const int32_t*b,int o,double*d,int n){for(int i=0;i<n;i++)d[i]=(double)i2f_(b[o+i]);}
static int32_t* buf;
static void setcfg(int phase,int gzero){ buf[XO_CFG+0]=phase;buf[XO_CFG+1]=X_RL_OUT;buf[XO_CFG+2]=0;buf[XO_CFG+3]=0;buf[XO_CFG+4]=gzero; }
static void kick(int phase,int gzero){ setcfg(phase,gzero); forge_pc_xformer_mp_block((ap_uint<512>*)buf); }
static void stage_master(const BFM&m){
    put_f(buf,MP_FWQ,m.WQ,X_D*X_D); put_f(buf,MP_FWK,m.WK,X_D*X_D); put_f(buf,MP_FWV,m.WV,X_D*X_D); put_f(buf,MP_FWO,m.WO,X_D*X_D);
    put_f(buf,MP_FWUP,m.WUP,X_FFN*X_D); put_f(buf,MP_FWDN,m.WDN,X_D*X_FFN);
    put_f(buf,MP_FLN1G,m.LN1G,X_D); put_f(buf,MP_FLN1B,m.LN1B,X_D); put_f(buf,MP_FLN2G,m.LN2G,X_D); put_f(buf,MP_FLN2B,m.LN2B,X_D);
    put_f(buf,MP_FEMB,m.EMB,X_VOCAB*X_D); put_f(buf,MP_FPOS,m.POS,X_SEQ*X_D);
    put_f(buf,MP_FLNFG,m.LNFG,X_D); put_f(buf,MP_FLNFB,m.LNFB,X_D); put_f(buf,MP_FWHEAD,m.WHEAD,X_VOCAB*X_D);
}
static double cmp(int gacc, const double* g, double scale, int n){  // max| MP[gacc] - scale*g |
    static double t[X_FFN*X_D]; read_f(buf,gacc,t,n); double mx=0; for(int i=0;i<n;i++){ double d=fabs(t[i]-scale*g[i]); if(d>mx)mx=d; } return mx;
}
static double mag(const double*a,int n){ double mx=0; for(int i=0;i<n;i++){ double d=fabs(a[i]); if(d>mx)mx=d; } return mx; }
int main(){
    setenv("BF16_BWD","2",1); (void)bf16_bwd(); g_lut_init(); g_exp_init();
    buf=(int32_t*)aligned_alloc(64,sizeof(int32_t)*MP_BUFLEN);
    static BFM bm; bf_init(bm,7);
    int tok[X_SEQ], tgt[X_SEQ]; for(int t=0;t<X_SEQ;t++){ tok[t]=(t*7+3)%X_VOCAB; tgt[t]=(t*11+5)%X_VOCAB; }
    static BFA ba; bf_forward(bm,ba,tok);                       // golden fwd -> ba.R seeds the OUTPUT block input
    for(int i=0;i<MP_BUFLEN;i++) buf[i]=0;
    stage_master(bm); put_i(buf,XO_TOK,tok,X_SEQ); put_i(buf,XO_TGT,tgt,X_SEQ);
    for(int i=0;i<X_SEQ*X_D;i++) buf[MP_GRIN+i]=f2i_((float)ba.R[i]);
    for(int i=0;i<X_SEQ*X_D;i++) buf[XO_HIN+i]=0; for(int i=0;i<X_LUT_N;i++) buf[XO_LUT+i]=0;

    // KICK 1 (gzero=1): per-kick grads -> MP_DG_*; accumulators MP_GACC_* must == the per-kick grads
    kick(X_PH_FWD,1); kick(X_PH_BWD,1);
    static double gUP[X_FFN*X_D], gQ[X_D*X_D], gDN[X_D*X_FFN], gH[X_VOCAB*X_D];
    read_f(buf,MP_DG_WUP,gUP,X_FFN*X_D); read_f(buf,MP_DG_WQ,gQ,X_D*X_D);
    read_f(buf,MP_DG_WDN,gDN,X_D*X_FFN); read_f(buf,MP_DG_WH,gH,X_VOCAB*X_D);
    double a1UP=cmp(MP_GACC_WUP,gUP,1.0,X_FFN*X_D), a1Q=cmp(MP_GACC_WQ,gQ,1.0,X_D*X_D), a1H=cmp(MP_GACC_WH,gH,1.0,X_VOCAB*X_D);

    // KICK 2 (gzero=0, SAME input): accumulators must == 2 * per-kick grad
    kick(X_PH_FWD,0); kick(X_PH_BWD,0);
    double a2UP=cmp(MP_GACC_WUP,gUP,2.0,X_FFN*X_D), a2Q=cmp(MP_GACC_WQ,gQ,2.0,X_D*X_D),
           a2DN=cmp(MP_GACC_WDN,gDN,2.0,X_D*X_FFN), a2H=cmp(MP_GACC_WH,gH,2.0,X_VOCAB*X_D);

    double gm=mag(gUP,X_FFN*X_D); double tol=1e-5*fmax(gm,1e-9);
    printf("=== mp gradient-accumulation check (same input x2, DO_UP=0) ===\n");
    printf("  kick1 acc==g :  WUP=%.2e  WQ=%.2e  WHEAD=%.2e   (tol %.1e, |g|max=%.3e)\n", a1UP,a1Q,a1H, tol, gm);
    printf("  kick2 acc==2g:  WUP=%.2e  WQ=%.2e  WDN=%.2e  WHEAD=%.2e\n", a2UP,a2Q,a2DN,a2H);
    int pass = a1UP<tol && a1Q<tol && a1H<tol && a2UP<2*tol && a2Q<2*tol && a2DN<2*tol && a2H<2*tol;
    printf("%s — gradient accumulation %s\n", pass?"PASS":"FAIL", pass?"sums correctly across BWD kicks (acc1=g, acc2=2g)":"is WRONG");
    return pass?0:1;
}
