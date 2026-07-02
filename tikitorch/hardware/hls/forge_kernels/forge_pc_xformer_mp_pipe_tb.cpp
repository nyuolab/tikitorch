// forge_pc_xformer_mp_pipe_tb.cpp — CROSS-SUB-BATCH PIPELINING bit-equality csim.
// REBUILD #4: pipeline slot depth S=MP_SLOTS (was 2). The slot design (cfg lane 6 = SLOT index 0..MP_SLOTS-1,
// slot-indexed wide cache MP_W* / MP_GRIN / MP_GROUT / MP_TOK / MP_TGT) exists so a LATER sub-batch's FORWARD can
// run BEFORE an earlier sub-batch's BACKWARD without clobbering the wide activation cache the backward needs.
// This TB proves that INVARIANT on a single OUTPUT block: it drives NSUB(=8) sub-batches of DIFFERENT inputs
// (tok/tgt/GRIN) THREE ways and asserts the results are BIT-IDENTICAL to the serial baseline:
//   (a) SERIAL     : F0 B0 F1 B1 ... F7 B7                       (fwd(g) then bwd(g), always)               — baseline
//   (b) AGGRESSIVE : F0 F1 F2 ... F7  B0 B1 ... B7               (ALL 8 forwards before ANY backward; needs S=8)
//   (c) PAIRWISE   : F0 F1 B0 F2 B1 ... F7 B6 B7                 (one fwd ahead — the old shallow 2-slot case)
// Each kick carries SLOT = g%MP_SLOTS (distinct per g since NSUB<=MP_SLOTS). GZERO on g==0, DO_UP on g==NSUB-1
// (a fixed lr). The grad accumulators (MP_GACC_*) and the float masters (MP_F*) are single-slot: in ALL orders
// the NSUB backwards run in ascending g, so the GACC RMW + the DO_UP dyadic-Adam are identical — the ONLY thing
// that can differ is a leaked wide-cache slot, which the deep interleaves expose. With correct slotting all
// orders are bit-identical; a slot collision would let a later fwd clobber an earlier fwd's cache before its bwd.
//
// PASS: the final updated float master (read MP_FW* after DO_UP) AND every per-g up-message gRout (MP_GROUT+slot)
//       are BIT-IDENTICAL between the serial baseline and each interleave. Printed PASS/FAIL per tensor.
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
static void setcfg(int phase,int do_up,int gzero,int lrbits,int slot){
    buf[XO_CFG+0]=phase; buf[XO_CFG+1]=X_RL_OUT; buf[XO_CFG+2]=do_up; buf[XO_CFG+3]=lrbits;
    buf[XO_CFG+4]=gzero; buf[XO_CFG+5]=1 /*TRAIN: skip deploy-int fwd (wide path only)*/; buf[XO_CFG+6]=slot;
}
// REBUILD #4: DDR slot offset for a per-sub-batch wide/msg region (== the kernel's `po`): slot 0 = base region,
// slots 1..MP_SLOTS-1 = the appended mirrors (stride MP_PAR_SPAN, first at MP_PAR_OFF).
static inline int slot_po(int slot){ return slot ? (MP_PAR_OFF + (slot-1)*MP_PAR_SPAN) : 0; }
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

#define NSUB 8                          // REBUILD #4: 8 sub-batches -> exercise all MP_SLOTS distinct slots
static_assert(NSUB <= MP_SLOTS, "NSUB must fit the pipeline slot depth (each sub-batch g uses slot g%MP_SLOTS)");
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
        int slot = g%MP_SLOTS; int po = slot_po(slot);
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
    // NSUB DISTINCT sub-batches: distinct tokens -> distinct golden block input R (via a golden full forward).
    static BFA ba;
    for(int g=0; g<NSUB; g++){
        for(int t=0;t<X_SEQ;t++){ tok4[g][t]=(t*7 + 13*g + 3)%X_VOCAB; tgt4[g][t]=(t*11 + 5*g + 5)%X_VOCAB; }
        bf_forward(bm, ba, tok4[g]);
        for(int i=0;i<X_SEQ*X_D;i++) R4[g][i]=ba.R[i];
    }
    int lrbits=(int)lround(3e-3*1048576.0);   // fixed lr for the DO_UP dyadic-Adam step

    // Build three kick orderings (order[s] = {isbwd, g}). In EVERY order the NSUB backwards run in ASCENDING g so
    // the single-slot GACC RMW + the DO_UP dyadic-Adam are identical; the ONLY thing that can differ between orders
    // is a leaked wide-cache slot (a fwd of a later sub-batch landing before the bwd of an earlier one). Each
    // sub-batch g uses slot g%MP_SLOTS -> with NSUB<=MP_SLOTS every g has a DISTINCT slot.
    static int serial[2*NSUB][2], aggressive[2*NSUB][2], pairwise[2*NSUB][2];
    // (a) SERIAL      : F0 B0 F1 B1 ... F7 B7                              (fwd(g) then bwd(g), always) — baseline
    for(int g=0; g<NSUB; g++){ serial[2*g][0]=0; serial[2*g][1]=g; serial[2*g+1][0]=1; serial[2*g+1][1]=g; }
    // (b) AGGRESSIVE  : F0 F1 F2 ... F7  B0 B1 ... B7   — ALL NSUB forwards before ANY backward (needs NSUB slots;
    //     the deepest possible interleave: fwd(g+7) can precede bwd(g)). Legal iff S=MP_SLOTS>=NSUB.
    for(int g=0; g<NSUB; g++){ aggressive[g][0]=0; aggressive[g][1]=g; aggressive[NSUB+g][0]=1; aggressive[NSUB+g][1]=g; }
    // (c) PAIRWISE    : F0 F1 B0 F2 B1 ... F7 B6 B7    — the old 2-slot-style interleave (one fwd ahead), kept as a
    //     cheap regression of the shallow case. F0, then (F(g),B(g-1)) for g=1..NSUB-1, then B(NSUB-1).
    { int s=0; pairwise[s][0]=0; pairwise[s][1]=0; s++;
      for(int g=1; g<NSUB; g++){ pairwise[s][0]=0; pairwise[s][1]=g;   s++;
                                 pairwise[s][0]=1; pairwise[s][1]=g-1; s++; }
      pairwise[s][0]=1; pairwise[s][1]=NSUB-1; s++; }

    static double mS[16][X_FFN*X_D], mAg[16][X_FFN*X_D], mPw[16][X_FFN*X_D];  // per-master (>=NMASTER); largest region X_FFN*X_D
    static double gS[NSUB][X_SEQ*X_D], gAg[NSUB][X_SEQ*X_D], gPw[NSUB][X_SEQ*X_D];
    run_order(bm, serial,     lrbits, mS,  gS);
    run_order(bm, aggressive, lrbits, mAg, gAg);
    run_order(bm, pairwise,   lrbits, mPw, gPw);

    printf("=== forge_pc_xformer_mp_block PIPELINE bit-equality  (serial vs interleaved, %d sub-batches, slot=g%%%d) ===\n", NSUB, MP_SLOTS);
    printf("    D=%d H=%d FFN=%d SEQ=%d VOCAB=%d | MP_SLOTS=%d MP_PAR_OFF=%d MP_PAR_SPAN=%d MP_TOK=%d MP_TGT=%d MP_BUFLEN=%d\n",
           X_D,X_H,X_FFN,X_SEQ,X_VOCAB, MP_SLOTS, MP_PAR_OFF, MP_PAR_SPAN, MP_TOK, MP_TGT, MP_BUFLEN);
    int pass=1;
    // compare one interleaved order vs the serial baseline: master (post DO_UP on g=NSUB-1) + every per-g gRout.
    auto compare=[&](const char* tag, double m[16][X_FFN*X_D], double g[NSUB][X_SEQ*X_D])->void{
        printf("  --- %s vs SERIAL — must be BIT-IDENTICAL ---\n", tag);
        for(int mm=0; mm<NMASTER; mm++){
            long nd=ndiff_d(mS[mm],m[mm],MASTERS[mm].n); double mx=maxabs_d(mS[mm],m[mm],MASTERS[mm].n);
            printf("    master %-7s : %s  (entries-differing=%ld/%d  max|diff|=%.3e)\n",
                   MASTERS[mm].name, nd==0?"PASS":"FAIL", nd, MASTERS[mm].n, mx);
            if(nd!=0) pass=0;
        }
        long gnd=0; double gmx=0;
        for(int gg=0; gg<NSUB; gg++){ gnd+=ndiff_d(gS[gg],g[gg],X_SEQ*X_D); double mx=maxabs_d(gS[gg],g[gg],X_SEQ*X_D); if(mx>gmx)gmx=mx; }
        printf("    per-g gRout (all %d)  : %s  (entries-differing=%ld/%d  max|diff|=%.3e)\n",
               NSUB, gnd==0?"PASS":"FAIL", gnd, NSUB*X_SEQ*X_D, gmx);
        if(gnd!=0) pass=0;
    };
    compare("AGGRESSIVE (all 8 fwds before any bwd)", mAg, gAg);
    compare("PAIRWISE (one fwd ahead, 2-slot style)", mPw, gPw);
    printf("=== %s — every interleave is BIT-IDENTICAL to serial => the %d pipeline slots do not leak ===\n",
           pass?"PASS":"FAIL", MP_SLOTS);
    free(buf);
    return pass?0:1;
}
