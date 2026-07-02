// forge_pc_xformer_mp_pipe_tb.cpp — CROSS-SUB-BATCH PIPELINING bit-equality csim.
// The parity design (cfg lane 6 = PARITY, slot-indexed wide cache MP_W* / MP_GRIN / MP_GROUT / MP_TOK / MP_TGT)
// exists so sub-batch g+1's FORWARD can run BEFORE sub-batch g's BACKWARD without clobbering the wide activation
// cache the backward needs. This TB proves that INVARIANT on a single OUTPUT block: it drives 4 sub-batches of
// DIFFERENT inputs (tok/tgt/GRIN) two ways and asserts the results are BIT-IDENTICAL:
//   (a) SERIAL      : F(0),B(0),F(1),B(1),F(2),B(2),F(3),B(3)           (fwd(g) then bwd(g), always)
//   (b) INTERLEAVED : F(0),F(1),B(0),F(2),B(1),F(3),B(2),B(3)           (fwd(g+1) BEFORE bwd(g))
// Every kick carries PARITY = g&1. GZERO on g==0, DO_UP on g==3 (a fixed lr). The grad accumulators (MP_GACC_*)
// and the float masters (MP_F*) are single-slot: in BOTH orders the four backwards run in ascending g, so the
// GACC RMW + the DO_UP dyadic-Adam are identical — the ONLY thing that can differ is a leaked wide-cache slot,
// which the interleaving (a fwd(g+1) landing between fwd(g) and bwd(g)) exposes. With correct parity the two
// orders are bit-identical; without it, F(1) (slot g&1) would clobber F(0)'s cache before B(0) reads it.
//
// PASS: the final updated float master (read MP_FW* after DO_UP) AND the per-g up-message gRout (MP_GROUT+slot)
//       are BIT-IDENTICAL between (a) and (b). Printed PASS/FAIL per tensor.
//
//   g++ -O2 -std=c++14 -DMP_DEBUG_GRADS -D__AP_INT_SPECIAL_H__ -I<apshim> -I<hlsshim> -I. -I<mpinc> \
//       forge_pc_xformer_mp_block.cpp forge_pc_xformer_mp_pipe_tb.cpp -o /tmp/mppipe
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ap_int.h>
#include "forge_pc_xformer_bf16_golden_block.hpp"
#include "forge_pc_xformer_mp_config.hpp"
extern "C" void forge_pc_xformer_mp_block(ap_uint<512> *buffer);

static int32_t f2i_(float f){ union{float f;int32_t i;}c; c.f=f; return c.i; }
static float   i2f_(int32_t i){ union{float f;int32_t i;}c; c.i=i; return c.f; }
static void put_f(int32_t*b,int o,const double*s,int n){ for(int i=0;i<n;i++) b[o+i]=f2i_((float)s[i]); }
static void put_i(int32_t*b,int o,const int*s,int n){ for(int i=0;i<n;i++) b[o+i]=s[i]; }
static void read_f(const int32_t*b,int o,double*d,int n){ for(int i=0;i<n;i++) d[i]=(double)i2f_(b[o+i]); }

static int32_t* buf;
// OUTPUT-role cfg with the PARITY lane (lane 6). gzero/do_up only matter in the BWD phase.
static void setcfg(int phase,int do_up,int gzero,int lrbits,int parity){
    buf[XO_CFG+0]=phase; buf[XO_CFG+1]=X_RL_OUT; buf[XO_CFG+2]=do_up; buf[XO_CFG+3]=lrbits;
    buf[XO_CFG+4]=gzero; buf[XO_CFG+5]=1 /*TRAIN: skip deploy-int fwd (wide path only)*/; buf[XO_CFG+6]=parity;
}
static void stage_master(const BFM&m){
    put_f(buf,MP_FWQ,m.WQ,X_D*X_D); put_f(buf,MP_FWK,m.WK,X_D*X_D); put_f(buf,MP_FWV,m.WV,X_D*X_D); put_f(buf,MP_FWO,m.WO,X_D*X_D);
    put_f(buf,MP_FWUP,m.WUP,X_FFN*X_D); put_f(buf,MP_FWDN,m.WDN,X_D*X_FFN);
    put_f(buf,MP_FLN1G,m.LN1G,X_D); put_f(buf,MP_FLN1B,m.LN1B,X_D); put_f(buf,MP_FLN2G,m.LN2G,X_D); put_f(buf,MP_FLN2B,m.LN2B,X_D);
    put_f(buf,MP_FEMB,m.EMB,X_VOCAB*X_D); put_f(buf,MP_FPOS,m.POS,X_SEQ*X_D);
    put_f(buf,MP_FLNFG,m.LNFG,X_D); put_f(buf,MP_FLNFB,m.LNFB,X_D); put_f(buf,MP_FWHEAD,m.WHEAD,X_VOCAB*X_D);
}

// the trainable float masters that DO_UP updates (LN gammas/betas frozen) — flattened for a bit-exact compare
struct Reg { int off; int n; const char* name; };
static const Reg MASTERS[] = {
    {MP_FWQ, X_D*X_D, "FWQ"}, {MP_FWK, X_D*X_D, "FWK"}, {MP_FWV, X_D*X_D, "FWV"}, {MP_FWO, X_D*X_D, "FWO"},
    {MP_FWUP, X_FFN*X_D, "FWUP"}, {MP_FWDN, X_D*X_FFN, "FWDN"}, {MP_FWHEAD, X_VOCAB*X_D, "FWHEAD"},
};
static const int NMASTER = sizeof(MASTERS)/sizeof(MASTERS[0]);

#define NSUB 4
static double R4[NSUB][X_SEQ*X_D];      // per-sub-batch OUTPUT block wide input (golden ba.R on distinct tokens)
static int    tok4[NSUB][X_SEQ], tgt4[NSUB][X_SEQ];

// run ONE ordering of the 8 kicks; capture the final master (per region) + the per-g gRout up-message.
static void run_order(const BFM&bm, const int order[2*NSUB][2], int lrbits,
                      double masterOut[/*NMASTER*/][X_FFN*X_D], double groutOut[NSUB][X_SEQ*X_D]){
    // fresh buffer: zero (=> GACC=0, MOM=0, VAR=0), stage the initial master, benign LUT + zero int HIN.
    for(int i=0;i<MP_BUFLEN;i++) buf[i]=0;
    stage_master(bm);
    for(int i=0;i<X_LUT_N;i++) buf[XO_LUT+i]=0;
    for(int i=0;i<X_SEQ*X_D;i++) buf[XO_HIN+i]=0;

    for(int s=0;s<2*NSUB;s++){
        int isbwd = order[s][0], g = order[s][1];
        int slot = g&1; int po = slot? MP_PAR_OFF : 0;
        if(!isbwd){
            // FORWARD F(g): stage this sub-batch's inputs into ITS slot, then kick.
            put_f(buf, MP_GRIN+po, R4[g], X_SEQ*X_D);            // OUTPUT block wide input
            put_i(buf, MP_TGT+slot*X_SEQ, tgt4[g], X_SEQ);       // CE targets (read in the BWD)
            put_i(buf, MP_TOK+slot*X_SEQ, tok4[g], X_SEQ);       // (OUTPUT doesn't embed, staged for completeness)
            setcfg(X_PH_FWD, 0, 0, 0, slot);
            forge_pc_xformer_mp_block((ap_uint<512>*)buf);
        } else {
            // BACKWARD B(g): GZERO on the first sub-batch, DO_UP (dyadic-Adam) on the last.
            setcfg(X_PH_BWD, (g==NSUB-1)?1:0, (g==0)?1:0, lrbits, slot);
            forge_pc_xformer_mp_block((ap_uint<512>*)buf);
            read_f(buf, MP_GROUT+po, groutOut[g], X_SEQ*X_D);    // per-g up-message (OUTPUT bwd writes MP_GROUT)
        }
    }
    for(int m=0;m<NMASTER;m++) read_f(buf, MASTERS[m].off, masterOut[m], MASTERS[m].n);
}

static long ndiff_d(const double*a,const double*b,int n){ long c=0; for(int i=0;i<n;i++) if(a[i]!=b[i])c++; return c; }
static double maxabs_d(const double*a,const double*b,int n){ double mx=0; for(int i=0;i<n;i++){ double d=fabs(a[i]-b[i]); if(d>mx)mx=d; } return mx; }

int main(){
    setenv("BF16_BWD","2",1); (void)bf16_bwd(); g_lut_init(); g_exp_init();
    buf=(int32_t*)aligned_alloc(64,sizeof(int32_t)*MP_BUFLEN);

    static BFM bm; bf_init(bm,7);
    // 4 DISTINCT sub-batches: distinct tokens -> distinct golden block input R (via a golden full forward).
    static BFA ba;
    for(int g=0; g<NSUB; g++){
        for(int t=0;t<X_SEQ;t++){ tok4[g][t]=(t*7 + 13*g + 3)%X_VOCAB; tgt4[g][t]=(t*11 + 5*g + 5)%X_VOCAB; }
        bf_forward(bm, ba, tok4[g]);
        for(int i=0;i<X_SEQ*X_D;i++) R4[g][i]=ba.R[i];
    }
    int lrbits=(int)lround(3e-3*1048576.0);   // fixed lr for the DO_UP dyadic-Adam step

    // (a) SERIAL       : F0 B0 F1 B1 F2 B2 F3 B3     — order[s] = {isbwd, g}
    const int serial[2*NSUB][2]      = {{0,0},{1,0},{0,1},{1,1},{0,2},{1,2},{0,3},{1,3}};
    // (b) INTERLEAVED  : F0 F1 B0 F2 B1 F3 B2 B3     — a fwd(g+1) lands between fwd(g) and bwd(g)
    const int interleaved[2*NSUB][2] = {{0,0},{0,1},{1,0},{0,2},{1,1},{0,3},{1,2},{1,3}};

    static double mA[16][X_FFN*X_D], mB[16][X_FFN*X_D];   // per-master (>=NMASTER); X_FFN*X_D is the largest region
    static double gA[NSUB][X_SEQ*X_D], gB[NSUB][X_SEQ*X_D];
    run_order(bm, serial,      lrbits, mA, gA);
    run_order(bm, interleaved, lrbits, mB, gB);

    printf("=== forge_pc_xformer_mp_block PIPELINE bit-equality  (serial vs interleaved, 4 sub-batches, PARITY=g&1) ===\n");
    printf("    D=%d H=%d FFN=%d SEQ=%d VOCAB=%d | MP_PAR_OFF=%d MP_TOK=%d MP_TGT=%d MP_BUFLEN=%d\n",
           X_D,X_H,X_FFN,X_SEQ,X_VOCAB, MP_PAR_OFF, MP_TOK, MP_TGT, MP_BUFLEN);
    int pass=1;
    printf("  final updated float master (after DO_UP on g=3) — must be BIT-IDENTICAL:\n");
    for(int m=0;m<NMASTER;m++){
        long nd=ndiff_d(mA[m],mB[m],MASTERS[m].n); double mx=maxabs_d(mA[m],mB[m],MASTERS[m].n);
        printf("    %-7s : %s  (entries-differing=%ld/%d  max|diff|=%.3e)\n",
               MASTERS[m].name, nd==0?"PASS":"FAIL", nd, MASTERS[m].n, mx);
        if(nd!=0) pass=0;
    }
    printf("  per-sub-batch up-message gRout (MP_GROUT+slot) — must be BIT-IDENTICAL:\n");
    for(int g=0; g<NSUB; g++){
        long nd=ndiff_d(gA[g],gB[g],X_SEQ*X_D); double mx=maxabs_d(gA[g],gB[g],X_SEQ*X_D);
        printf("    gRout[g=%d] : %s  (entries-differing=%ld/%d  max|diff|=%.3e)\n",
               g, nd==0?"PASS":"FAIL", nd, X_SEQ*X_D, mx);
        if(nd!=0) pass=0;
    }
    printf("=== %s — interleaved (fwd(g+1) before bwd(g)) is BIT-IDENTICAL to serial => parity slots do not leak ===\n",
           pass?"PASS":"FAIL");
    free(buf);
    return pass?0:1;
}
