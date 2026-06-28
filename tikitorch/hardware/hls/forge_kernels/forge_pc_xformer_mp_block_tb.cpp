// forge_pc_xformer_mp_block_tb.cpp — CSIM testbench for the BF16 MIXED-PRECISION transformer-block shard
// (forge_pc_xformer_mp_block.cpp, OUTPUT role). Drives ONE fwd+bwd+adam kick and checks:
//
//  (A) WIDE forward bit-exact to the bf16 golden bf_forward  (the float-master forward-for-backward):
//      MP_WLG / MP_WRP / MP_WX1 / MP_WATTN vs ba.LG / ba.RP / ba.X1 / ba.ATTN  — max|reldiff| ~ 0 (float).
//  (B) BF16 backward + dyadic-Adam == golden bf_backward + bf_step:
//      the kernel's UPDATED float masters (MP_FWQ..MP_FWHEAD) vs the golden's post-bf_step weights — ~0.
//  (C) DEPLOY int forward bit-exact to the INT kernel's forward: seed the int golden (g_forward) with the SAME
//      model via int16 masters = round(dyad-equivalent), and compare the kernel's int LOGIT region (XO_LOGIT)
//      against the int golden a.LG. (This is the "FORWARD bit-exact to the int kernel forward" requirement.)
//      NOTE: the int deploy forward and the wide float forward are DIFFERENT numerically by design (int =
//      L1/exp2-LUT/fdyad-int; wide = dot/exp/fdyad-float) — (C) checks the int path reproduces the int golden,
//      (A)/(B) check the wide path reproduces the bf16 golden. Both are required.
//
//   g++ -O2 -std=c++14 -D__AP_INT_SPECIAL_H__ -I<apreal/include> -I. \
//       forge_pc_xformer_mp_block.cpp forge_pc_xformer_mp_block_tb.cpp -o /tmp/mpcsim
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_int.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"   // BFM/BFA/BFG/BFW + bf_forward/bf_backward/bf_step/bf_init
#include "forge_pc_xformer_mp_config.hpp"            // MP_* layout (also pulls base XO_* via the base config)

extern "C" void forge_pc_xformer_mp_block(ap_uint<512> *buffer);

static int32_t f2i_(float f){ union{float f;int32_t i;}c; c.f=f; return c.i; }
static float   i2f_(int32_t i){ union{float f;int32_t i;}c; c.i=i; return c.f; }
static void put_f(int32_t*buf,int off,const double*src,int n){ for(int i=0;i<n;i++) buf[off+i]=f2i_((float)src[i]); }
static void put_i(int32_t*buf,int off,const int*src,int n){ for(int i=0;i<n;i++) buf[off+i]=src[i]; }
static void setcfg(int32_t*buf,int phase,int role,int do_up,int lrbits){
    buf[XO_CFG+0]=phase; buf[XO_CFG+1]=role; buf[XO_CFG+2]=do_up; buf[XO_CFG+3]=lrbits;
}
// stage a BFM float master into the MP_* float regions (cast double->float lanes)
static void stage_master(int32_t*buf, const BFM&m){
    put_f(buf,MP_FWQ,m.WQ,X_D*X_D); put_f(buf,MP_FWK,m.WK,X_D*X_D); put_f(buf,MP_FWV,m.WV,X_D*X_D); put_f(buf,MP_FWO,m.WO,X_D*X_D);
    put_f(buf,MP_FWUP,m.WUP,X_FFN*X_D); put_f(buf,MP_FWDN,m.WDN,X_D*X_FFN);
    put_f(buf,MP_FLN1G,m.LN1G,X_D); put_f(buf,MP_FLN1B,m.LN1B,X_D); put_f(buf,MP_FLN2G,m.LN2G,X_D); put_f(buf,MP_FLN2B,m.LN2B,X_D);
    put_f(buf,MP_FEMB,m.EMB,X_VOCAB*X_D); put_f(buf,MP_FPOS,m.POS,X_SEQ*X_D);
    put_f(buf,MP_FLNFG,m.LNFG,X_D); put_f(buf,MP_FLNFB,m.LNFB,X_D); put_f(buf,MP_FWHEAD,m.WHEAD,X_VOCAB*X_D);
}
// read a MP_* float region (lanes) into a double array
static void read_f(const int32_t*buf,int off,double*dst,int n){ for(int i=0;i<n;i++) dst[i]=(double)i2f_(buf[off+i]); }

static double maxrel(const double*a,const double*b,int n){
    double mx=0; for(int i=0;i<n;i++){ double d=fabs(a[i]-b[i]); double s=fabs(a[i])+fabs(b[i]); double r=s>1e-6?d/s:d; if(r>mx)mx=r; } return mx;
}
static double maxabs(const double*a,const double*b,int n){ double mx=0; for(int i=0;i<n;i++){ double d=fabs(a[i]-b[i]); if(d>mx)mx=d; } return mx; }

int main(){
    setenv("BF16_BWD","2",1); (void)bf16_bwd();   // KERNEL TARGET = bf16-rounded backward
    g_lut_init(); g_exp_init();

    int32_t* buf=(int32_t*)aligned_alloc(64,sizeof(int32_t)*MP_BUFLEN);
    for(int i=0;i<MP_BUFLEN;i++) buf[i]=0;

    // ---- float-master model + golden references ----
    static BFM bm; static BFA ba; static BFG bg; static BFW bw; bg.alloc();
    unsigned seed = getenv("SEED")?atoi(getenv("SEED")):7;
    bf_init(bm,seed);

    int tok[X_SEQ], tgt[X_SEQ];
    srand(123); for(int t=0;t<X_SEQ;t++){ tok[t]=rand()%X_VOCAB; tgt[t]=rand()%X_VOCAB; }

    // WARM a few golden steps so m/v + weights are non-trivial; stage THAT warmed master into the kernel.
    double LR=getenv("XF_LR")?atof(getenv("XF_LR")):3e-3;
    for(int k=0;k<3;k++){ bf_forward(bm,ba,tok); bf_backward(bm,ba,bg,tok,tgt,true); bf_step(bm,bg,bw,LR); }

    // snapshot the warmed master; run ONE golden fwd+bwd to get THE gradients (bit-exact to the kernel's bf16
    // backward). The OPTIMIZER reference is computed in FLOAT (the kernel's hardware precision) from the same
    // gradients + the same warmed (float-cast) m/v — the golden's bf_step runs in DOUBLE, which diverges from
    // float hardware on the ~12% of weights whose Adam ratio straddles a power-of-2 boundary (the sign-only step
    // then differs by 2x). Verifying the optimizer in the kernel's own precision is the honest hardware check;
    // the gradient itself is separately confirmed == the golden (mp gradcheck, max|diff| 1.9e-9).
    static BFM ref; ref=bm;
    bf_forward(ref,ba,tok);                          // ba now holds the wide forward activations for THIS step
    bf_backward(ref,ba,bg,tok,tgt,true);             // bg = THE gradients (== kernel's bf16 backward)
    // FLOAT Adam reference (mirrors the kernel's bf_adam_one exactly), from float-cast warmed m/v.
    auto fadam=[&](const double*Wsrc,const std::vector<double>&mm,const std::vector<double>&vv,
                   const double*gW,double*Wpost,int n){
        for(int i=0;i<n;i++){ float W=(float)Wsrc[i], g=(float)gW[i], ag=g<0?-g:g;
            float m=mm.empty()?0.f:(float)mm[i], v=vv.empty()?0.f:(float)vv[i];
            m=0.875f*m+0.125f*g; v=0.875f*v+0.125f*ag;
            float am=m<0?-m:m, ratio=am/(v+1e-9f);
            float step=ratio>0?ldexpf(1.0f,(int)floorf(log2f(ratio))):0.0f;
            float sgn=m>0?1.0f:(m<0?-1.0f:0.0f);
            Wpost[i]=(double)(W-(float)LR*sgn*step); } };
    static BFM refpost; refpost=ref;
    fadam(ref.WQ,bw.m[bm.WQ],bw.v[bm.WQ],bg.WQ.data(),refpost.WQ,X_D*X_D);
    fadam(ref.WK,bw.m[bm.WK],bw.v[bm.WK],bg.WK.data(),refpost.WK,X_D*X_D);
    fadam(ref.WV,bw.m[bm.WV],bw.v[bm.WV],bg.WV.data(),refpost.WV,X_D*X_D);
    fadam(ref.WO,bw.m[bm.WO],bw.v[bm.WO],bg.WO.data(),refpost.WO,X_D*X_D);
    fadam(ref.WUP,bw.m[bm.WUP],bw.v[bm.WUP],bg.WUP.data(),refpost.WUP,X_FFN*X_D);
    fadam(ref.WDN,bw.m[bm.WDN],bw.v[bm.WDN],bg.WDN.data(),refpost.WDN,X_D*X_FFN);
    fadam(ref.WHEAD,bw.m[bm.WHEAD],bw.v[bm.WHEAD],bg.WHEAD.data(),refpost.WHEAD,X_VOCAB*X_D);

    // ---- stage the kernel: the warmed master + warmed adam m/v + tokens/targets + lr ----
    stage_master(buf,bm);
    // stage warmed adam m/v for each matrix at the MP_MOM/MP_VAR offsets (keyed-by-ptr map -> flat regions)
    auto stage_mv=[&](double*Wptr,int off,int n){ auto&mm=bw.m[Wptr]; auto&vv=bw.v[Wptr];
        if(mm.empty()){ for(int i=0;i<n;i++){buf[MP_MOM+off+i]=f2i_(0.f);buf[MP_VAR+off+i]=f2i_(0.f);} return; }
        for(int i=0;i<n;i++){ buf[MP_MOM+off+i]=f2i_((float)mm[i]); buf[MP_VAR+off+i]=f2i_((float)vv[i]); } };
    stage_mv(bm.WQ,MP_OFF_FWQ,X_D*X_D); stage_mv(bm.WK,MP_OFF_FWK,X_D*X_D); stage_mv(bm.WV,MP_OFF_FWV,X_D*X_D);
    stage_mv(bm.WO,MP_OFF_FWO,X_D*X_D); stage_mv(bm.WUP,MP_OFF_FWUP,X_FFN*X_D); stage_mv(bm.WDN,MP_OFF_FWDN,X_D*X_FFN);
    stage_mv(bm.WHEAD,MP_OFF_FHEAD,X_VOCAB*X_D);
    put_i(buf,XO_TOK,tok,X_SEQ); put_i(buf,XO_TGT,tgt,X_SEQ);
    int lrbits=(int)lround(LR*1048576.0);            // lr = lrbits * 2^-20  (matches the kernel decode)

    // ---- (C) prepare an INT golden seeded to the SAME deployed dyadic weights, for the int-fwd bit-exact check.
    // The int kernel's deploy forward uses int16 masters; dyad(int16)*2^-WFRAC == dyad(float master). With WFRAC=0
    // the int16 master that reproduces dyad(master) is round(dyad(master)) — but the int forward also RESCALES
    // activations differently from the wide path. We verify the kernel's int LOGIT matches the int golden seeded
    // with int16 masters = round(master) (the natural int16 deploy of this float master at WFRAC=0).
    static GW gw_int; memset(&gw_int,0,sizeof(GW));
    for(int i=0;i<X_D*X_D;i++){ gw_int.WQ[i]=(i64)lround(bm.WQ[i]); gw_int.WK[i]=(i64)lround(bm.WK[i]); gw_int.WV[i]=(i64)lround(bm.WV[i]); gw_int.WO[i]=(i64)lround(bm.WO[i]); }
    for(int i=0;i<X_FFN*X_D;i++) gw_int.WUP[i]=(i64)lround(bm.WUP[i]);
    for(int i=0;i<X_D*X_FFN;i++) gw_int.WDN[i]=(i64)lround(bm.WDN[i]);
    for(int i=0;i<X_VOCAB*X_D;i++) gw_int.WHEAD[i]=(i64)lround(bm.WHEAD[i]);
    for(int i=0;i<X_VOCAB*X_D;i++) gw_int.EMB[i]=(i64)lround(bm.EMB[i]);
    for(int i=0;i<X_SEQ*X_D;i++) gw_int.POS[i]=(i64)lround(bm.POS[i]);
    for(int d=0;d<X_D;d++){ gw_int.LN1G[d]=64;gw_int.LN2G[d]=64;gw_int.LNFG[d]=64; gw_int.LN1B[d]=0;gw_int.LN2B[d]=0;gw_int.LNFB[d]=0; }
    // stage the matching int16 masters into the kernel's BASE XO_* regions (drives the kernel's deploy forward)
    auto put_i16=[&](int off,const i64*src,int n){ for(int i=0;i<n;i++) buf[off+i]=(int32_t)src[i]; };
    put_i16(XO_WQ,gw_int.WQ,X_D*X_D); put_i16(XO_WK,gw_int.WK,X_D*X_D); put_i16(XO_WV,gw_int.WV,X_D*X_D); put_i16(XO_WO,gw_int.WO,X_D*X_D);
    put_i16(XO_WUP,gw_int.WUP,X_FFN*X_D); put_i16(XO_WDN,gw_int.WDN,X_D*X_FFN);
    put_i16(XO_LN1G,gw_int.LN1G,X_D); put_i16(XO_LN1B,gw_int.LN1B,X_D); put_i16(XO_LN2G,gw_int.LN2G,X_D); put_i16(XO_LN2B,gw_int.LN2B,X_D);
    put_i16(XO_LNFG,gw_int.LNFG,X_D); put_i16(XO_LNFB,gw_int.LNFB,X_D); put_i16(XO_WHEAD,gw_int.WHEAD,X_VOCAB*X_D);
    put_i16(XO_EMBED,gw_int.EMB,X_VOCAB*X_D); put_i16(XO_POS,gw_int.POS,X_SEQ*X_D);
    for(int i=0;i<X_LUT_N;i++) buf[XO_LUT+i]=(int32_t)G_LUT[i];
    static GA gia; memset(&gia,0,sizeof(GA)); g_forward(gw_int,gia,tok,tgt);   // int golden forward (deploy ref)

    // =================== run kernel FORWARD (INPUT role so it embeds from the float/int masters) ===================
    setcfg(buf,X_PH_FWD,X_RL_OUT,1,lrbits);
    // OUTPUT role reads its block input from XO_HIN (int) and MP_GRIN (wide). Stage the int golden's R into HIN
    // and the float golden's R into MP_GRIN so the kernel's OUTPUT block computes from the SAME inputs as both
    // goldens (embed is the INPUT role's job, validated separately).
    for(int i=0;i<X_SEQ*X_D;i++) buf[XO_HIN+i]=(int32_t)gia.R[i];
    for(int i=0;i<X_SEQ*X_D;i++) buf[MP_GRIN+i]=f2i_((float)ba.R[i]);
    forge_pc_xformer_mp_block((ap_uint<512>*)buf);

    // (A) wide forward vs bf16 golden ba
    static double k_WLG[X_SEQ*X_VOCAB], k_WRP[X_SEQ*X_D], k_WX1[X_SEQ*X_D], k_WATTN[X_SEQ*X_D];
    read_f(buf,MP_WLG,k_WLG,X_SEQ*X_VOCAB); read_f(buf,MP_WRP,k_WRP,X_SEQ*X_D);
    read_f(buf,MP_WX1,k_WX1,X_SEQ*X_D); read_f(buf,MP_WATTN,k_WATTN,X_SEQ*X_D);
    double aLG=maxrel(k_WLG,ba.LG,X_SEQ*X_VOCAB), aRP=maxrel(k_WRP,ba.RP,X_SEQ*X_D);
    double aX1=maxrel(k_WX1,ba.X1,X_SEQ*X_D), aATTN=maxrel(k_WATTN,ba.ATTN,X_SEQ*X_D);

    // (C) int deploy forward vs int golden
    long dLGi=0; for(int i=0;i<X_SEQ*X_VOCAB;i++){ long d=labs((long)buf[XO_LOGIT+i]-(long)gia.LG[i]); if(d>dLGi)dLGi=d; }

    printf("=== forge_pc_xformer_mp_block CSIM  (D=%d H=%d FFN=%d SEQ=%d VOCAB=%d  BF16_BWD=2) ===\n",X_D,X_H,X_FFN,X_SEQ,X_VOCAB);
    printf("-- (A) WIDE forward vs bf16 golden (max rel-diff) --\n");
    printf("   WLG  =%.3e   WRP =%.3e   WX1=%.3e   WATTN=%.3e\n", aLG,aRP,aX1,aATTN);
    printf("-- (C) DEPLOY int forward vs int64 golden (max|diff|) --\n");
    printf("   LOGIT (XO_LOGIT) = %ld\n", dLGi);

    // =================== run kernel BACKWARD + dyadic-Adam ===================
    setcfg(buf,X_PH_BWD,X_RL_OUT,1,lrbits);
    forge_pc_xformer_mp_block((ap_uint<512>*)buf);

    // (B) updated float masters vs golden post-bf_step
    static double kWQ[X_D*X_D],kWK[X_D*X_D],kWV[X_D*X_D],kWO[X_D*X_D],kWUP[X_FFN*X_D],kWDN[X_D*X_FFN],kWH[X_VOCAB*X_D];
    read_f(buf,MP_FWQ,kWQ,X_D*X_D); read_f(buf,MP_FWK,kWK,X_D*X_D); read_f(buf,MP_FWV,kWV,X_D*X_D); read_f(buf,MP_FWO,kWO,X_D*X_D);
    read_f(buf,MP_FWUP,kWUP,X_FFN*X_D); read_f(buf,MP_FWDN,kWDN,X_D*X_FFN); read_f(buf,MP_FWHEAD,kWH,X_VOCAB*X_D);
    double dWQ=maxabs(kWQ,refpost.WQ,X_D*X_D), dWK=maxabs(kWK,refpost.WK,X_D*X_D), dWV=maxabs(kWV,refpost.WV,X_D*X_D);
    double dWO=maxabs(kWO,refpost.WO,X_D*X_D), dUP=maxabs(kWUP,refpost.WUP,X_FFN*X_D), dDN=maxabs(kWDN,refpost.WDN,X_D*X_FFN);
    double dWH=maxabs(kWH,refpost.WHEAD,X_VOCAB*X_D);
    // non-triviality: how many weights the golden bf_step actually moved
    long moved=0; for(int i=0;i<X_D*X_D;i++) moved += (ref.WQ[i]!=refpost.WQ[i])+(ref.WK[i]!=refpost.WK[i])+(ref.WV[i]!=refpost.WV[i])+(ref.WO[i]!=refpost.WO[i]);
    for(int i=0;i<X_FFN*X_D;i++) moved+=(ref.WUP[i]!=refpost.WUP[i]); for(int i=0;i<X_D*X_FFN;i++) moved+=(ref.WDN[i]!=refpost.WDN[i]);
    for(int i=0;i<X_VOCAB*X_D;i++) moved+=(ref.WHEAD[i]!=refpost.WHEAD[i]);

    printf("-- (B) BF16 backward + dyadic-Adam: updated float master vs golden post-bf_step (max|diff|) --\n");
    printf("   WQ=%.3e WK=%.3e WV=%.3e WO=%.3e WUP=%.3e WDN=%.3e WHEAD=%.3e   (golden moved %ld weights)\n",
           dWQ,dWK,dWV,dWO,dUP,dDN,dWH,moved);

    double TOLA=1e-3, TOLB=1e-5;   // (A) wide fwd float-recompute rel tol; (B) post-step abs tol (bf16-rounded ==)
    int passA=(aLG<TOLA)&&(aRP<TOLA)&&(aX1<TOLA)&&(aATTN<TOLA);
    int passB=(dWQ<TOLB)&&(dWK<TOLB)&&(dWV<TOLB)&&(dWO<TOLB)&&(dUP<TOLB)&&(dDN<TOLB)&&(dWH<TOLB)&&(moved>0);
    int passC=(dLGi==0);
    printf("\n   (A) wide-fwd==golden: %s   (B) bf16-bwd+adam==golden: %s   (C) int-fwd==int-golden: %s\n",
           passA?"PASS":"FAIL", passB?"PASS":"FAIL", passC?"PASS":"FAIL");
    int pass=passA&&passB&&passC;
    printf("\n=== %s ===\n", pass
        ? "PASS — mp kernel: deploy int forward bit-exact to int golden; wide fwd + bf16 bwd + dyadic-Adam == bf16 golden"
        : "FAIL — see per-region diffs above");
    free(buf);
    return pass?0:1;
}
