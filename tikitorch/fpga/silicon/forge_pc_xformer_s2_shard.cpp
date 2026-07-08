// forge_pc_xformer_s2_shard.cpp — train the SCALE-2 (30M, D=512) DDR-STREAMING PC-transformer MODEL-PARALLEL
// across the FPGAs of an f2.48xlarge: ONE transformer block per physical chip (forge_pc_xformer_s2_block kernel;
// role/phase CFG). Chip 0 = INPUT (float embed/pos, block 0), chips 1..L-2 = HIDDEN, chip L-1 = OUTPUT (block,
// final-LN, LM head, CE). Each chip holds ONLY its block's FLOAT master + dyadic-Adam m/v + gradient accumulators
// + the wide fwd-for-bwd activation cache in its OWN on-card DDR — all STREAMED tile-by-tile inside the kernel
// (weights are no longer on-chip resident; that is the s2 footprint-wall fix). The ONLY data crossing chips is a
// WIDE (float) neighbour hidden-state (down, forward) + a WIDE (float/bf16) neighbour error (up, backward). No
// global model / gradient / all-reduce / barrier — same no-global-sync contract as the mp shard.
//
// This is the D=512 twin of forge_pc_xformer_mp_shard.cpp. It mirrors that driver's structure verbatim (open the
// slots via the AWS fpga_mgmt/fpga_pci API, host-shadow each chip's DDR, burst-stage regions, kick with a CFG
// word, read back DDR, relay boundary messages over PCIe, persist weights). The DIFFERENCES vs the mp driver:
//   * TRAIN-ONLY layout: everything lives in the S2_* FLOAT DDR layout — there are NO XO_* int deploy regions,
//     NO exp2 LUT, NO int forward, NO int neighbour relay (the s2 kernel omits the deploy-int forward);
//   * the FLOAT master init is the mp bf_init recipe scaled to the D=512 dims (matmul weights ~ U(-1/sqrt(fanin)),
//     EMB/POS ~ N(0,1), LN gamma=1/beta=0 frozen), per-chip seed = seed + 1000*l (BYTE-FOR-BYTE the mp contract,
//     so a CPU twin — /tmp/mp_chain_30m.py — can replay via INIT_DIR in this exact FMaster field order);
//   * neighbour messages are the WIDE float strips S2_GROUT -> S2_GRIN (SEQ*D floats = 512 KB each), bursted;
//   * bpb is read from the OUTPUT chip's WIDE float logits S2_WLG;
//   * PIPE=1 enables the cross-sub-batch dependency EVENT LOOP (ported from the mp shard) over the s2 kernel's
//     S=8 SLOT parity: up to S2_SLOTS in-flight sub-batches overlap (slot=g%S2_SLOTS), backward-first dispatch,
//     per-chip monotonic bwd order for GACC, a per-step barrier. PIPE=0 (default) is the serial bit-compat path;
//   * KICKMEAS=<n>: the Phase-1.5 single-chip roofline. Run ONE chip in OUTPUT role (the heaviest block: block +
//     final-LN + LM head + CE + full backward + adam) for n FWD+BWD steps and print per-phase kick ms. INPUT/HID
//     chips are strictly cheaper (no head), so this is an upper bound on the per-chip kick cost.
//
//   g++ -O2 -std=c++14 -I<hls/forge_kernels> -I$SDK/include forge_pc_xformer_s2_shard.cpp -lfpga_mgmt -o xfs2shard
//   sudo ./xfs2shard <steps> <bindir> <slot0> <slot1> ... <slotL-1>      (env: ACCUM, XF_LR, XF_WARMUP, SEED,
//                                                                          INIT_DIR, SAMPLE, DUMP_DIR, LR_DECAY,
//                                                                          LR_MINFRAC, ACTSTATS, FIXEDOFF, KICKMEAS)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <ctime>
extern "C" {
#include <fpga_pci.h>
#include <fpga_mgmt.h>
}
#include "forge_pc_xformer_s2_config.hpp"   // S2_* dims + the float-only DDR layout (no XO_* int regions)

#define MAXL 8
#define R_AP_CTRL 0x00
#define R_BUF0 0x10
#define R_BUF1 0x14
#define AP_START 0x01
#define AP_DONE  0x02

static int L;
static pci_bar_handle_t bar0[MAXL], bar4[MAXL], bar4r[MAXL];
static int32_t *hbuf[MAXL];                 // host shadow of each chip's DDR buffer (S2_BUFLEN int32 lanes)
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// float <-> int32 DDR lane bit-reinterpret (one float per int32 lane; exactly the kernel's i2f/f2i)
static inline int32_t f2i(float f){ union{float f; int32_t i;} c; c.f=f; return c.i; }
static inline float   i2f(int32_t i){ union{float f; int32_t i;} c; c.i=i; return c.f; }

// ---- burst-stage helpers (identical to the mp shard's stage/kick/readw) ----
static void stage(int l,long off,long n){ for(long o=off;o<off+n;o+=262144){ long k=(off+n-o<262144)?(off+n-o):262144;
    fpga_pci_write_burst(bar4[l],(uint64_t)o*4,(uint32_t*)(hbuf[l]+o),(uint64_t)k);} }
// per-phase kick timing: attribute wall-clock per chip x phase — fwd vs bwd kernel-exec ms (kick() tight-spins
// AP_DONE, so the measured span IS kernel execution). g_phase set by the sweep loops / the KICKMEAS loop.
static int g_phase = 0;                       // 0=fwd 1=bwd (index into the accumulators)
static double g_kick_s[2][8]; static long g_kick_n[2][8];
static void kick(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    double t0=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); uint32_t c=0;
    while(now_s()-t0<60.0){ fpga_pci_peek(bar0[l],R_AP_CTRL,&c);                       // s2 kicks are heavier -> 60s cap
        if(c&AP_DONE){ g_kick_s[g_phase][l]+=now_s()-t0; g_kick_n[g_phase][l]++; return; } }
    fprintf(stderr,"KICK HUNG slot %d\n",l); exit(6); }
static void kick_report(){
    for(int p=0;p<2;p++) for(int l=0;l<8;l++) if(g_kick_n[p][l])
        printf("[kicktime] %s chip %d: %.1f ms/kick (n=%ld)\n", p?"BWD":"FWD", l, 1e3*g_kick_s[p][l]/g_kick_n[p][l], g_kick_n[p][l]);
    fflush(stdout); }

// ---- ASYNC kick (PIPE scheduler): split kick() into start (poke AP_START) + poll (peek AP_DONE) so the
//      scheduler can run ONE kick at a time on EACH chip concurrently and overlap sub-batches. Timing is
//      attributed on completion by the phase of the finished kick (g_kick_s/g_kick_n stay meaningful). ----
static double g_kick_t0[MAXL];
static void kick_start(int l){ fpga_pci_poke(bar0[l],R_BUF0,0); fpga_pci_poke(bar0[l],R_BUF1,0);
    g_kick_t0[l]=now_s(); fpga_pci_poke(bar0[l],R_AP_CTRL,AP_START); }
static int kick_poll(int l){ uint32_t c=0; fpga_pci_peek(bar0[l],R_AP_CTRL,&c); return (c&AP_DONE)?1:0; }
// cross-sub-batch PIPELINING: env PIPE (default 0 = the existing serial per-sub-batch sweeps, bit-compatible)
static int g_pipe=-1;
static inline int pipe_on(){ if(g_pipe<0){ const char*e=getenv("PIPE"); g_pipe=e?atoi(e):0; } return g_pipe; }
// DDR slot offset for a per-sub-batch wide/msg region: slot 0 = the base region (offset 0), slots 1..S2_SLOTS-1 =
// the appended mirrors (stride S2_PAR_SPAN, first at S2_PAR_OFF) — the exact formula the kernel's `po` uses
// (forge_pc_xformer_s2_block.cpp, cfg lane 6 = slot). Returns long: offsets reach ~39.3M words at S2_SLOTS=8.
static inline long slot_po(int slot){ return slot ? ((long)S2_PAR_OFF + (long)(slot-1)*S2_PAR_SPAN) : 0L; }

// BURSTED readback: one mmap'd memcpy from the BAR4 read window instead of per-word peek. Env READBURST=0 forces
// the old per-word path. CHUNKED so tensors larger than the map window (e.g. WUP = FFN*D = 1,048,576 words at
// D=512) read correctly — the mp driver's single-shot get_address only covered <=1<<18 words.
static int32_t g_rbuf[1<<18];                 // 262144-word (1 MB) staging window
static int g_readburst = -1;
static inline int readburst_on(){ if(g_readburst<0){ const char*e=getenv("READBURST"); g_readburst = e?atoi(e):1; } return g_readburst; }
static void readw(int l,long off,int32_t*dst,long n){
    void*p=NULL;
    for(long b=0;b<n;){
        long k=(n-b<(1<<18))?(n-b):(1<<18);
        if(readburst_on() && fpga_pci_get_address(bar4r[l],(uint64_t)(off+b)*4,(uint64_t)k,&p)==0 && p){ memcpy(dst+b,p,(size_t)k*4); b+=k; continue; }
        uint32_t v; for(long j=0;j<k;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+b+j)*4,&v); dst[b+j]=(int32_t)v; } b+=k;
    }
}
// float-lane read (chunked): bulk-read int32 lanes then reinterpret as float.
static void readf(int l,long off,float*dst,long n){
    void*p=NULL;
    for(long b=0;b<n;){
        long k=(n-b<(1<<18))?(n-b):(1<<18);
        if(readburst_on() && fpga_pci_get_address(bar4r[l],(uint64_t)(off+b)*4,(uint64_t)k,&p)==0 && p){
            memcpy(g_rbuf,p,(size_t)k*4); for(long j=0;j<k;j++) dst[b+j]=i2f(g_rbuf[j]); b+=k; continue; }
        uint32_t v; for(long j=0;j<k;j++){ fpga_pci_peek(bar4r[l],(uint64_t)(off+b+j)*4,&v); dst[b+j]=i2f((int32_t)v); } b+=k;
    }
}
// stage a host float array into a DDR float region (lane bit-reinterpret then burst)
static void put_f(int l,long off,const float*src,long n){ for(long j=0;j<n;j++) hbuf[l][off+j]=f2i(src[j]); stage(l,off,n); }

// CFG word lanes: [phase, role, do_update, lrbits, gzero, train, slot, tile_o, tile_i, seq_tile]
//   lane 3 = fixed-point lr (kernel: LR = lrbits*2^-20). lanes 7/8/9 = 0 => the compile-time tile defaults.
static void setcfg(int l,int phase,int role,int doupd,int lrbits,int gzero=0,int slot=0){
    for(int i=0;i<S2_WL;i++) hbuf[l][S2O_CFG+i]=0;
    hbuf[l][S2O_CFG+0]=phase; hbuf[l][S2O_CFG+1]=role; hbuf[l][S2O_CFG+2]=doupd; hbuf[l][S2O_CFG+3]=lrbits;
    hbuf[l][S2O_CFG+4]=gzero;   // ACCUM>1: 1 on the first sub-batch -> kernel zeroes the grad accumulators
    hbuf[l][S2O_CFG+5]=1;       // TRAIN (train-only kernel ignores it; keep 1 like the s2 TB)
    hbuf[l][S2O_CFG+6]=slot;    // SLOT parity — SERIAL-ONLY v1 keeps this 0 (pipelining is a later optimization)
    // lanes 7/8/9 (tile_o/tile_i/seq_tile) left 0 => kernel uses its compile-time S2_TILE_* defaults
    stage(l,S2O_CFG,S2_WL); }
static int role_of(int l){ return (l==0)?S2_RL_INPUT : (l==L-1)?S2_RL_OUT : S2_RL_HID; }
// HOST-SIDE EMBED: the on-chip INPUT embed's bload_f reads FEMB as 0 on silicon (a DDR-path bug csim does NOT
// reproduce — WR=0 while the driver's readf sees FEMB=4.2; localized via S2_DIAG). Workaround: the driver
// computes R = EMB[tok]+POS host-side (a memory gather, not compute) and feeds it to chip 0 via the PROVEN
// GRIN->block path (chip 0 runs as a HID block), and trains EMB/POS host-side from chip 0's up-message gR.
// Driver-only; NO AFI rebuild. Env HOST_EMBED (default 1). krole() = the role staged to the kernel.
static int g_he=-1; static inline int host_embed(){ if(g_he<0){const char*e=getenv("HOST_EMBED"); g_he=e?atoi(e):1;} return g_he; }
static int krole(int l){ return (host_embed() && l==0 && L>1) ? S2_RL_HID : role_of(l); }
static float *hEMB=0,*hPOS=0,*hmE=0,*hvE=0,*hmP=0,*hvP=0,*hgE=0,*hgP=0;
// dyadic-Adam step (matches the s2 kernel recipe: pow2 step from |m|/v ratio).
static inline float dyad_adam(float g,float&m,float&v,float lr){ m=0.875f*m+0.125f*g; v=0.875f*v+0.125f*fabsf(g);
    float r=fabsf(m)/(v+1e-9f); float st=(r>0)?ldexpf(1.0f,(int)floorf(log2f(r))):0.0f; return lr*((m>0)-(m<0))*st; }

// =====================================================================================================
// HOST_BACKWARD (env, default 0=OFF): a DIAGNOSTIC/HYBRID training path that BYPASSES every in-kernel backward
// GEMM. The kernel does ONLY the FORWARD (proven faithful on silicon by S2_DIAG); the driver reads back the
// forward activations from each chip's DDR wide-cache and computes the ENTIRE backward + dyadic-Adam ON THE HOST
// in C++, MIRRORING the CPU twin /tmp/mp_chain_30m.py (bwd_block) EXACTLY — same bf16 STE round points, dyadic
// weights for the dx-route, qact, dyadic-mantissa-LN scales (read from S2_WLN*S), softmax PRAW RECOMPUTE from the
// cached post-qact Q,K, wgrad/dx per linbwd, the CE grad, and the cross-chip up-error passed host-side (replacing
// the broken in-kernel GROUT/GRIN backward relay). Updated weights are re-staged to silicon DDR for the NEXT
// step's forward. If this DESCENDS on silicon while the all-kernel path diverges, the backward GEMMs are the SOLE
// bug. NOTE: SLOW (host compute + full activation readback per sub-batch) and NOT the on-fabric final — the
// kernel transposed-strided-load fix + rebuild remains the real solution. This block is arithmetic-only; it adds
// NO new DDR layout and NO AFI rebuild. All math below is BYTE-for-BYTE the twin's (which descends).
static int g_hb=-1; static inline int host_backward(){ if(g_hb<0){const char*e=getenv("HOST_BACKWARD"); g_hb=e?atoi(e):0;} return g_hb; }

// =====================================================================================================
// FWD_DUMP (env, default 0=OFF): a pure DIAGNOSTIC for the s2 silicon-vs-csim forward-divergence hunt. After the
// FORWARD sweep of step 0 / sub-batch 0 (before any backward can clobber the wide cache), read back EACH forward
// activation from EACH chip's DDR wide-cache (SLOT 0) and dump the FULL fp32 tensor to <DUMP_DIR>/fwd_<chip>_<name>.f32.
// Diff these bit-for-bit against the bit-faithful shim's same-named dumps (same matched init + fixed data window)
// to pin the FIRST forward stage where silicon drifts from source -> localizes the broken (suspected DDR-streamed) op.
// NO backward is needed for this mode; it piggybacks on the readf path and adds ZERO DDR layout / AFI rebuild.
// Forward order dumped (== kernel forge_pc_xformer_s2_block fwd): R, X1(LN1), Q,K,V(proj+qact), ATTN(attn+qact),
// M(=R+AO), X2(LN2), HD(FFN+relu+qact), RP(=M+FO); + XF(LNF) and LG(logits) on the OUTPUT chip only.
static int g_fd=-1; static inline int fwd_dump(){ if(g_fd<0){const char*e=getenv("FWD_DUMP"); g_fd=e?atoi(e):0;} return g_fd; }
static void dump_f32(const char* dir, int chip, const char* name, const float* x, long n){
    char op[600]; snprintf(op,600,"%s/fwd_%d_%s.f32", dir, chip, name);
    FILE* f=fopen(op,"wb"); if(!f){ fprintf(stderr,"[FWD_DUMP] open %s failed\n",op); return; }
    fwrite(x,sizeof(float),(size_t)n,f); fclose(f);
    fprintf(stderr,"[FWD_DUMP] chip %d %-4s -> %s (%ld floats)\n", chip, name, op, n);
}
// Largest tensor any host-side diagnostic reads: SEQ*FFN (HD), SEQ*VOCAB (LG), or VOCAB*D (FEMB probe).
// MUST cover all three — at S2_VOCAB=8192 the old S2_SEQ*S2_FFN sizing overflowed on the LG/FEMB reads
// (16 MB read into a 2 MB static = host bus error / silent BSS corruption).
static const size_t S2_HOSTBUF =
    ((size_t)S2_SEQ*S2_FFN   > (size_t)S2_SEQ*S2_VOCAB ? (size_t)S2_SEQ*S2_FFN : (size_t)S2_SEQ*S2_VOCAB) >
     (size_t)S2_VOCAB*S2_D ?
    ((size_t)S2_SEQ*S2_FFN   > (size_t)S2_SEQ*S2_VOCAB ? (size_t)S2_SEQ*S2_FFN : (size_t)S2_SEQ*S2_VOCAB) :
     (size_t)S2_VOCAB*S2_D;

// dump one chip's forward wide-cache. `is_out` also emits XF + LG (only the OUTPUT chip writes them).
static void fwd_dump_chip(int l, const char* dir, int is_out){
    static float db[S2_HOSTBUF];
    readf(l,S2_WR,   db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"R",   db,(long)S2_SEQ*S2_D);
    readf(l,S2_WX1,  db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"X1",  db,(long)S2_SEQ*S2_D);
    readf(l,S2_WQ,   db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"Q",   db,(long)S2_SEQ*S2_D);
    readf(l,S2_WK,   db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"K",   db,(long)S2_SEQ*S2_D);
    readf(l,S2_WV,   db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"V",   db,(long)S2_SEQ*S2_D);
    readf(l,S2_WATTN,db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"ATTN",db,(long)S2_SEQ*S2_D);
    readf(l,S2_WM,   db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"M",   db,(long)S2_SEQ*S2_D);
    readf(l,S2_WX2,  db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"X2",  db,(long)S2_SEQ*S2_D);
    readf(l,S2_WHD,  db,(long)S2_SEQ*S2_FFN);dump_f32(dir,l,"HD",  db,(long)S2_SEQ*S2_FFN);
    readf(l,S2_WRP,  db,(long)S2_SEQ*S2_D);  dump_f32(dir,l,"RP",  db,(long)S2_SEQ*S2_D);
    if(is_out){
        readf(l,S2_WXF, db,(long)S2_SEQ*S2_D);    dump_f32(dir,l,"XF",db,(long)S2_SEQ*S2_D);
        readf(l,S2_WLG, db,(long)S2_SEQ*S2_VOCAB);dump_f32(dir,l,"LG",db,(long)S2_SEQ*S2_VOCAB);
    }
}

// ---- bf16 STE round (== kernel bf16_round / twin P()) ----
static inline float hb_P(float x){ union{float f;uint32_t u;}c; c.f=x; uint32_t u=c.u; uint32_t lsb=(u>>16)&1u; u+=0x7fffu+lsb; u&=0xffff0000u; c.u=u; return c.f; }
// ---- dyad (== kernel bf_dyad_f / twin dyad) ----
static inline float hb_dyad(float w){ float a=w<0?-w:w; if(a<1e-30f) return 0.0f; int e=(int)lroundf(log2f(a)); float p=ldexpf(1.0f,e); return w<0?-p:p; }
// ---- qact IN PLACE over a [n] tensor (== kernel s2_qact_ddr / twin qact): global max, shift-round ----
static void hb_qact(float*x,long n){ float mx=0; for(long i=0;i<n;i++){ float a=x[i]<0?-x[i]:x[i]; if(a>mx)mx=a; } if(mx<1e-9f)mx=1e-9f; int sh=(int)floorf(log2f(mx))-S2_ABITS; float sc=ldexpf(1.0f,sh); for(long i=0;i<n;i++) x[i]=roundf(x[i]/sc)*sc; }
// ---- LN backward on a [SEQ x D] strip in place (== kernel s2_ln_bwd_ddr / twin ln_bwd). gamma[d], s[SEQ]. ----
//   gxhat=P(gy*gamma); mg=mean_d(gxhat); dx=P((gxhat-mg)/s[row]).
static void hb_ln_bwd(float*gy,const float*gamma,const float*s,int M){ static float gxhat[S2_D];
    for(int t=0;t<M;t++){ float*gr=gy+(long)t*S2_D; double mg=0; for(int d=0;d<S2_D;d++){ gxhat[d]=hb_P(gr[d]*gamma[d]); mg+=gxhat[d]; } float mgm=(float)(mg/S2_D); for(int d=0;d<S2_D;d++) gr[d]=hb_P((gxhat[d]-mgm)/s[t]); } }
// ---- linbwd: dx=P(gy @ dyad(W)), dW += P(gy^T @ x). W is [O x I] float master (dyad'd for the dx route). ----
//   gy:[SEQ x O]  x:[SEQ x I]  -> dx:[SEQ x I]  dW:[O x I] (accumulated into dWacc). rows=SEQ.
static void hb_linbwd(const float*gy,const float*Wf,const float*x,int O,int I,float*dx,float*dWacc){
    // dx[r][i] = P( sum_o gy[r][o]*dyad(W[o][i]) )
    for(long r=0;r<S2_SEQ;r++){ const float*gr=gy+r*O; float*dr=dx+r*I;
        for(int i=0;i<I;i++){ double acc=0; for(int o=0;o<O;o++) acc+=(double)gr[o]*hb_dyad(Wf[(long)o*I+i]); dr[i]=hb_P((float)acc); } }
    // dW[o][i] += P( sum_r gy[r][o]*x[r][i] )
    for(int o=0;o<O;o++){ float*dwo=dWacc+(long)o*I;
        for(int i=0;i<I;i++){ double acc=0; for(long r=0;r<S2_SEQ;r++) acc+=(double)gy[r*O+o]*x[r*I+i]; dwo[i]+=hb_P((float)acc); } }
}
// ---- PRAW recompute + attention backward (== kernel s2_attn_bwd / twin _attn_bwd_vec, per-head, causal) ----
//   Q,K,V,gATTN are [SEQ x D] (post-qact Q/K/V). Writes gQ,gK,gV [SEQ x D]. P() applied per twin's rounding pts.
static void hb_attn_bwd(const float*Q,const float*K,const float*V,const float*gATTN,float*gQ,float*gK,float*gV){
    const float scale=1.0f/sqrtf((float)S2_DH);
    for(long i=0;i<(long)S2_SEQ*S2_D;i++){ gQ[i]=0; gK[i]=0; gV[i]=0; }
    static float p[S2_SEQ], gp[S2_SEQ], gsv[S2_SEQ];
    for(int h=0;h<S2_H;h++){ int hb=h*S2_DH;
        for(int i=0;i<S2_SEQ;i++){
            // --- recompute softmax row p[0..i] from cached Q,K (same score/exp/normalize as fwd) ---
            float mx=-1e30f;
            for(int j=0;j<=i;j++){ double dot=0; for(int d=0;d<S2_DH;d++) dot+=(double)Q[(long)i*S2_D+hb+d]*K[(long)j*S2_D+hb+d]; p[j]=(float)dot*scale; if(p[j]>mx)mx=p[j]; }
            double sum=0; for(int j=0;j<=i;j++){ float e=expf(p[j]-mx); p[j]=e; sum+=e; }
            for(int j=0;j<=i;j++) p[j]/=(float)sum;
            // --- gp[j]=<gATTN_i,V_j> ; gV[j] += P(p[j]*gATTN_i) ---
            for(int j=0;j<=i;j++){ double g=0; for(int d=0;d<S2_DH;d++) g+=(double)gATTN[(long)i*S2_D+hb+d]*V[(long)j*S2_D+hb+d]; gp[j]=(float)g;
                for(int d=0;d<S2_DH;d++) gV[(long)j*S2_D+hb+d]+=hb_P(p[j]*gATTN[(long)i*S2_D+hb+d]); }
            double ex=0; for(int j=0;j<=i;j++) ex+=(double)p[j]*gp[j];
            for(int j=0;j<=i;j++) gsv[j]=hb_P(p[j]*(gp[j]-(float)ex))*scale;
            // gQ[i] += sum_j gsv[j]*K[j] ; gK[j] += gsv[j]*Q[i]
            for(int d=0;d<S2_DH;d++){ double qa=0; for(int j=0;j<=i;j++) qa+=(double)gsv[j]*K[(long)j*S2_D+hb+d]; gQ[(long)i*S2_D+hb+d]+=(float)qa; }
            for(int j=0;j<=i;j++) for(int d=0;d<S2_DH;d++) gK[(long)j*S2_D+hb+d]+=gsv[j]*Q[(long)i*S2_D+hb+d];
        }
    }
    // final P() per head-slice write (== kernel post-loop P / twin P(gQ),P(gK),P(gV))
    for(long i=0;i<(long)S2_SEQ*S2_D;i++){ gQ[i]=hb_P(gQ[i]); gK[i]=hb_P(gK[i]); gV[i]=hb_P(gV[i]); }
}

// =====================================================================================================
// FLOAT-master init — BYTE-FOR-BYTE the mp bf_init contract (forge_pc_xformer_mp_shard.cpp / the bf16 golden),
// scaled to the D=512 dims:
//   matmul weights ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in)) via the bf_u LCG;  EMB/POS ~ N(0,1) via bf_n (sum of
//   12 uniforms - 6);  LN gamma=1, beta=0 (frozen).  RNG: g=seed*2654435761u+1; draw advances g=g*1103515245+12345.
//   DRAW ORDER (must match the CPU twin EXACTLY): WQ,WK,WV,WO,WUP,WDN,WHEAD, then EMB, then POS.  Per-chip seed
//   offset seed + 1000*l.  fan_in: WQ/WK/WV/WO/WUP/WHEAD = D ; WDN = FFN.
// The CPU twin (/tmp/mp_chain_30m.py) can replay by loading INIT_DIR/master<l>.f32 = the FMaster struct dumped
// below (fields in exactly this order: WQ,WK,WV,WO,WUP,WDN,WHEAD,EMB,POS,LN1G,LN1B,LN2G,LN2B,LNFG,LNFB).
// =====================================================================================================
static unsigned g_rng = 1;
static inline double bf_u(){ g_rng = g_rng*1103515245u + 12345u; return ((g_rng>>8)&0xffffff)/(double)0x1000000; }
static inline double bf_n(){ double u=0; for(int k=0;k<12;k++) u+=bf_u(); return u-6.0; }
static void uni(float*W,long n,int fan){ double a=1.0/sqrt((double)fan); for(long i=0;i<n;i++) W[i]=(float)((bf_u()*2-1)*a); }

// per-chip FLOAT master (host side; what we stage into the S2_F* region). Heap-allocated (~14 MB @ D=512).
struct FMaster {
    float WQ[S2_D*S2_D], WK[S2_D*S2_D], WV[S2_D*S2_D], WO[S2_D*S2_D];
    float WUP[S2_FFN*S2_D], WDN[S2_D*S2_FFN], WHEAD[S2_VOCAB*S2_D];
    float EMB[S2_VOCAB*S2_D], POS[S2_SEQ*S2_D];
    float LN1G[S2_D], LN1B[S2_D], LN2G[S2_D], LN2B[S2_D], LNFG[S2_D], LNFB[S2_D];
};
static void init_master(FMaster &m, unsigned seed){
    g_rng = seed*2654435761u + 1u;
    uni(m.WQ,(long)S2_D*S2_D,S2_D); uni(m.WK,(long)S2_D*S2_D,S2_D); uni(m.WV,(long)S2_D*S2_D,S2_D); uni(m.WO,(long)S2_D*S2_D,S2_D);
    uni(m.WUP,(long)S2_FFN*S2_D,S2_D); uni(m.WDN,(long)S2_D*S2_FFN,S2_FFN); uni(m.WHEAD,(long)S2_VOCAB*S2_D,S2_D);
    for(long i=0;i<(long)S2_VOCAB*S2_D;i++) m.EMB[i]=(float)bf_n();
    for(long i=0;i<(long)S2_SEQ*S2_D;i++)   m.POS[i]=(float)bf_n();
    for(int d=0;d<S2_D;d++){ m.LN1G[d]=1;m.LN2G[d]=1;m.LNFG[d]=1; m.LN1B[d]=0;m.LN2B[d]=0;m.LNFB[d]=0; }
}
static bool load_master_file(FMaster &m, const char* path){
    FILE* f=fopen(path,"rb"); if(!f) return false;
    size_t n = fread(&m, sizeof(float), sizeof(FMaster)/sizeof(float), f); fclose(f);
    return n == sizeof(FMaster)/sizeof(float);
}
static void dump_master_file(const FMaster &m, const char* path){
    FILE* f=fopen(path,"wb"); if(!f) return; fwrite(&m, sizeof(float), sizeof(FMaster)/sizeof(float), f); fclose(f);
}
// stage a fully-populated FMaster into a chip's S2_F* region (INPUT-only EMB/POS, OUTPUT-only LNF/HEAD).
static void stage_master(int l, const FMaster &m){
    put_f(l, S2_FWQ,  m.WQ,  (long)S2_D*S2_D);   put_f(l, S2_FWK,  m.WK,  (long)S2_D*S2_D);
    put_f(l, S2_FWV,  m.WV,  (long)S2_D*S2_D);   put_f(l, S2_FWO,  m.WO,  (long)S2_D*S2_D);
    put_f(l, S2_FWUP, m.WUP, (long)S2_FFN*S2_D); put_f(l, S2_FWDN, m.WDN, (long)S2_D*S2_FFN);
    put_f(l, S2_FLN1G,m.LN1G,S2_D);              put_f(l, S2_FLN1B,m.LN1B,S2_D);
    put_f(l, S2_FLN2G,m.LN2G,S2_D);              put_f(l, S2_FLN2B,m.LN2B,S2_D);
    if(l==0){   put_f(l, S2_FEMB, m.EMB, (long)S2_VOCAB*S2_D); put_f(l, S2_FPOS, m.POS, (long)S2_SEQ*S2_D); }
    if(l==L-1){ put_f(l, S2_FLNFG,m.LNFG,S2_D); put_f(l, S2_FLNFB,m.LNFB,S2_D); put_f(l, S2_FWHEAD,m.WHEAD,(long)S2_VOCAB*S2_D); }
}

// per-chip activation-statistics probe (ACTSTATS=1; no-op otherwise) — does the WIDE residual stream / logits
// collapse, explode, or saturate across chips 0->L-1 or over training? Printed at step 0 and every 500 steps.
static int g_actstats=-1;
static int actstats_on(){ if(g_actstats<0){ const char*e=getenv("ACTSTATS"); g_actstats=e?atoi(e):0; } return g_actstats; }
static void actstat(const char* tag, int chip, int step, const float* x, long n){
    if(!actstats_on() || !(step==0 || step%500==0)) return;
    double mn=1e300, mx=-1e300, sa=0; long bad=0;
    for(long i=0;i<n;i++){ double v=x[i]; if(!(v==v) || v>1e300 || v<-1e300){bad++;continue;} if(v<mn)mn=v; if(v>mx)mx=v; sa+=v<0?-v:v; }
    printf("[actstat] step %5d  %-8s chip %d  min=%+.3e max=%+.3e absmean=%.3e nan/inf=%ld/%ld\n",
           step, tag, chip, mn, mx, sa/(n>0?n:1), bad, n); fflush(stdout);
}

int main(int argc,char**argv){
    int steps = argc>1?atoi(argv[1]):2000;
    const char*dir = argc>2?argv[2]:"/tmp/pc_enwik8";
    L = argc-3;
    // KICKMEAS single-chip roofline allows L==1; the training sweeps need L>=2.
    int KICKMEAS = getenv("KICKMEAS")?atoi(getenv("KICKMEAS")):0;
    int Lmin = KICKMEAS?1:2;
    if(L<Lmin||L>MAXL){ fprintf(stderr,"usage: %s <steps> <bindir> <slot0..slotL-1> (%d<=L<=%d)\n",argv[0],Lmin,MAXL); return 1; }
    int slot[MAXL]; for(int l=0;l<L;l++) slot[l]=atoi(argv[3+l]);

    // training hyper-params (env; defaults == the mp golden's char/byte-LM recipe)
    unsigned seed   = getenv("SEED")?(unsigned)atoi(getenv("SEED")):7u;
    int   ACCUM     = getenv("ACCUM")?atoi(getenv("ACCUM")):1;
    double LR       = getenv("XF_LR")?atof(getenv("XF_LR")):3e-3;
    int   WARMUP    = getenv("XF_WARMUP")?atoi(getenv("XF_WARMUP")):200;
    const char* INIT_DIR = getenv("INIT_DIR");
    const char* DUMP_DIR = getenv("DUMP_DIR")?getenv("DUMP_DIR"):"/tmp";
    if(ACCUM<1) ACCUM=1;

    // ---- load byte-level enwik8 token stream (uint16 ids 0-255), SEQ=256 windows ----
    char p[512]; snprintf(p,512,"%s/train.u16",dir); FILE*f=fopen(p,"rb"); if(!f){fprintf(stderr,"open %s\n",p);return 2;}
    fseek(f,0,SEEK_END); long nb=ftell(f); fseek(f,0,SEEK_SET); long ntok=nb/2; uint16_t*data=(uint16_t*)malloc(nb);
    if(fread(data,2,ntok,f)!=(size_t)ntok){fprintf(stderr,"read tokens\n");return 2;} fclose(f);
    printf("[xfs2shard] SCALE-2 D=512 %d-block transformer MODEL-PARALLEL across %d FPGA(s) (slots", L, L);
    for(int l=0;l<L;l++) printf(" %d",slot[l]); printf(")%s\n", KICKMEAS?"  [KICKMEAS roofline]":"");
    printf("[xfs2shard] %ld train tokens | D=%d H=%d FFN=%d SEQ=%d VOCAB=%d K=%d | LR=%.1e warmup=%d ACCUM=%d seed=%u | S2_BUFLEN=%d words=%.1f MB/chip\n",
           ntok, S2_D, S2_H, S2_FFN, S2_SEQ, S2_VOCAB, S2_K, LR, WARMUP, ACCUM, seed, S2_BUFLEN, S2_BUFLEN*4.0/1e6);

    if(fpga_mgmt_init()){fprintf(stderr,"mgmt init\n");return 2;}
    for(int l=0;l<L;l++){
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR0,0,&bar0[l])){fprintf(stderr,"bar0 %d\n",slot[l]);return 3;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,BURST_CAPABLE,&bar4[l])){fprintf(stderr,"bar4 %d\n",slot[l]);return 4;}
        if(fpga_pci_attach(slot[l],FPGA_APP_PF,APP_PF_BAR4,0,&bar4r[l])){fprintf(stderr,"bar4r %d\n",slot[l]);return 4;}
        hbuf[l]=(int32_t*)calloc((size_t)S2_BUFLEN,4);    // shadow the WHOLE S2 float buffer (~166 MB/chip @ S2_SLOTS=8)
        if(!hbuf[l]){ fprintf(stderr,"calloc hbuf %d (%zu bytes)\n",l,(size_t)S2_BUFLEN*4); return 5; }
    }

    // =================================================================================================
    // INIT: per chip, build the FLOAT master (bf_init recipe, or a host INIT_DIR file), stage into the S2_F*
    //       region, zero the dyadic-Adam m/v + grad accumulators (calloc already zeroed them). Persist once.
    // =================================================================================================
    static FMaster* M[MAXL] = {0};
    for(int l=0;l<L;l++){
        M[l]=(FMaster*)calloc(1,sizeof(FMaster)); if(!M[l]){ fprintf(stderr,"calloc master %d\n",l); return 5; }
        // KICKMEAS forces a single OUTPUT chip (the heaviest block: +final-LN +head +CE); role_of would call it
        // INPUT when l==0, so init it as OUTPUT (all of EMB/POS/LNF/HEAD present) via the L==1 -> l==L-1 path.
        init_master(*M[l], seed + 1000u*l);
        if(INIT_DIR){ char ip[512]; snprintf(ip,512,"%s/master%d.f32",INIT_DIR,l);
            if(load_master_file(*M[l],ip)) printf("[xfs2shard] chip %d: loaded host init %s\n", l, ip); }
        stage_master(l, *M[l]);
        stage(l,0,S2_BUFLEN);    // persist the whole float buffer (incl. the zeroed adam m/v + grad accumulators) once
    }

    // HOST-EMBED state: reuse chip-0 master's EMB/POS (bf_init recipe); host adam moments + grad accumulators.
    // HOST_BACKWARD FORCES host_embed (the whole backward — incl. embed — runs host-side), so this block always
    // fires under HOST_BACKWARD; the host EMB/POS train exactly as they already do in the HOST_EMBED serial path.
    if(host_embed()){
        hEMB=M[0]->EMB; hPOS=M[0]->POS;
        hmE=(float*)calloc((size_t)S2_VOCAB*S2_D,4); hvE=(float*)calloc((size_t)S2_VOCAB*S2_D,4);
        hmP=(float*)calloc((size_t)S2_SEQ*S2_D,4);   hvP=(float*)calloc((size_t)S2_SEQ*S2_D,4);
        hgE=(float*)calloc((size_t)S2_VOCAB*S2_D,4); hgP=(float*)calloc((size_t)S2_SEQ*S2_D,4);
        printf("[xfs2shard] HOST_EMBED on: chip 0 runs as a HID block; EMB/POS embedded+trained host-side.\n");
    }
    // HOST_BACKWARD state: per-chip host dyadic-Adam m/v + grad ACCUMULATORS for ALL matmul + head weights, in the
    // FMaster layout so we iterate tensor-by-tensor. The FLOAT master IS M[l] (updated host-side, re-staged each
    // step). EMB/POS reuse the hEMB/hmE... arrays above (chip-0 tail). LN gamma/beta are frozen (no grad). ~42 MB/chip.
    static FMaster *HM[MAXL]={0}, *HV[MAXL]={0}, *HG[MAXL]={0};
    if(host_backward()){
        for(int l=0;l<L;l++){
            HM[l]=(FMaster*)calloc(1,sizeof(FMaster)); HV[l]=(FMaster*)calloc(1,sizeof(FMaster)); HG[l]=(FMaster*)calloc(1,sizeof(FMaster));
            if(!HM[l]||!HV[l]||!HG[l]){ fprintf(stderr,"calloc host-bwd adam %d\n",l); return 5; }
        }
        printf("[xfs2shard] HOST_BACKWARD on: kernel does FORWARD only; driver computes the FULL backward + dyadic-Adam\n"
               "            HOST-SIDE (mirrors /tmp/mp_chain_30m.py bwd_block; PRAW recomputed from cached Q,K). This\n"
               "            BYPASSES every in-kernel backward GEMM. Weights re-staged to DDR each step. SLOW; diagnostic.\n");
        if(pipe_on()){ fprintf(stderr,"FATAL: HOST_BACKWARD requires the SERIAL sweep (PIPE=0); unset PIPE.\n"); return 8; }
    }
    // boundary-message scratch (one float per lane; the wide down/up messages are SEQ*D floats = 512 KB)
    float *hmsg=(float*)malloc(4L*S2_SEQ*S2_D);   // wide down-message (forward): S2_GROUT(l) -> S2_GRIN(l+1)
    float *emsg=(float*)malloc(4L*S2_SEQ*S2_D);   // wide up-message  (backward): S2_GROUT(l) -> S2_GRIN(l-1)
    float *wlg =(float*)malloc(4L*S2_SEQ*S2_VOCAB);  // OUTPUT chip wide logits (for bpb)
    // PIPE scheduler: S2_SLOTS-deep boundary buffers indexed by (g % S2_SLOTS) so an in-flight sub-batch g's
    // message can't be clobbered by another slot's message; per-step sub-batch offsets (drawn once up-front).
    float *hmsg2[S2_SLOTS], *emsg2[S2_SLOTS];
    for(int s=0;s<S2_SLOTS;s++){ hmsg2[s]=(float*)malloc(4L*S2_SEQ*S2_D); emsg2[s]=(float*)malloc(4L*S2_SEQ*S2_D); }
    long  *soff=(long*)malloc(sizeof(long)*(size_t)(ACCUM>0?ACCUM:1));
    double t0=now_s(); double bpb_ema=-1, acc_ema=-1;
    srand(123);
    static long g_fixedoff = getenv("FIXEDOFF") ? atol(getenv("FIXEDOFF")) : -1;
    // lr -> fixed-point code (lr = lrbits*2^-20)
    auto lrcode=[&](double lr){ long c=(long)llround(lr * (double)(1<<20)); if(c<0)c=0; if(c>0x7fffffffL)c=0x7fffffffL; return (int)c; };

    // =================================================================================================
    // KICKMEAS — Phase-1.5 single-chip roofline. Run chip 0 in OUTPUT role (block + final-LN + LM head + CE +
    // full BF16-STE backward + streamed dyadic-Adam), FWD then BWD, KICKMEAS times. Feed a fixed synthetic block
    // input into S2_GRIN (OUTPUT reads S2_GRIN as its block input R) + real targets into S2_TGT. Timing does not
    // depend on the values (kick() spins to AP_DONE and the kernel does the same work), so this measures the
    // heaviest per-chip kick cost. INPUT/HID chips are cheaper (no head); print per-phase kick ms + a summary.
    // =================================================================================================
    if(KICKMEAS){
        int msteps = KICKMEAS>0?KICKMEAS:20;
        printf("[KICKMEAS] single chip (slot %d) OUTPUT-role, %d FWD+BWD steps (block+LNF+head+CE+adam)\n", slot[0], msteps);
        // synthetic wide block-input R (N(0,1)*0.5) staged once into S2_GRIN; targets rolled from the stream.
        for(long i=0;i<(long)S2_SEQ*S2_D;i++) hmsg[i]=(float)(bf_n()*0.5);
        put_f(0, S2_GRIN, hmsg, (long)S2_SEQ*S2_D);
        int lrb=lrcode(LR);
        // FIX F: DO_UP semantics match real training. Default (KICKMEAS_DOUP unset => -1): DO_UP only on the LAST
        // sub-batch of each ACCUM window (adam-apply amortized over ACCUM steps), GZERO on the first — so the
        // measured per-kick cost reflects the training roofline, not an adam-every-step upper bound. Set
        // KICKMEAS_DOUP=1 to force DO_UP every kick (old behavior), =0 to measure pure fwd+bwd (no optimizer).
        int KM_DOUP = getenv("KICKMEAS_DOUP") ? atoi(getenv("KICKMEAS_DOUP")) : -1;
        printf("[KICKMEAS] DO_UP mode: %s (ACCUM=%d)\n",
               KM_DOUP<0 ? "last-of-ACCUM-window (train-like)" : (KM_DOUP?"every kick":"never (fwd+bwd only)"), ACCUM);
        for(int s=0;s<msteps;s++){
            long off = g_fixedoff>=0 ? g_fixedoff : (long)(rand() % (ntok - S2_SEQ - 2));
            int doup = (KM_DOUP>=0) ? KM_DOUP : (((s % ACCUM)==(ACCUM-1)) ? 1 : 0);
            int gz   = (KM_DOUP>=0) ? 1       : (((s % ACCUM)==0)         ? 1 : 0);
            for(int t=0;t<S2_SEQ;t++) hbuf[0][S2_TGT+t]=data[off+t+1]; stage(0,S2_TGT,S2_SEQ);
            g_phase=0; setcfg(0, S2_PH_FWD, S2_RL_OUT, 0, 0, gz, 0);            kick(0);
            g_phase=1; setcfg(0, S2_PH_BWD, S2_RL_OUT, doup, doup?lrb:0, gz, 0); kick(0);
        }
        kick_report();
        double fwd = g_kick_n[0][0]?1e3*g_kick_s[0][0]/g_kick_n[0][0]:0;
        double bwd = g_kick_n[1][0]?1e3*g_kick_s[1][0]/g_kick_n[1][0]:0;
        double stepms = fwd+bwd;
        printf("\n========== KICKMEAS ROOFLINE (single OUTPUT chip, D=%d K=%d) ==========\n", S2_D, S2_K);
        printf("[roofline] fwd %.1f ms/kick | bwd %.1f ms/kick | step(fwd+bwd) %.1f ms | %.3f steps/s | %.0f tok/s (SEQ=%d, one block)\n",
               fwd, bwd, stepms, stepms>0?1000.0/stepms:0, stepms>0?1000.0/stepms*S2_SEQ:0, S2_SEQ);
        printf("[roofline] NOTE: OUTPUT chip is the heaviest (includes LM head+CE); INPUT/HID blocks are cheaper. "
               "8-chip model-parallel throughput ~ 1/max(per-chip step ms) once pipelined.\n");
        for(int l=0;l<L;l++){ free(M[l]); fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
        return 0;
    }

    static int g_trace = getenv("STEPTRACE") ? atoi(getenv("STEPTRACE")) : 0;
#define STRACE(...) do{ if(g_trace){ fprintf(stderr,"[trace] " __VA_ARGS__); fflush(stderr);} }while(0)
    // persist_masters(prefix): read every chip's TRAINED float master back from DDR -> <DUMP_DIR>/<prefix>_master<l>.f32.
    // Used by the end-of-run dump (prefix "xfs2", unchanged filenames) and the rolling CKPT_EVERY checkpoint ("ckpt").
    auto persist_masters=[&](const char* prefix){
        for(int l=0;l<L;l++){
            FMaster &m = *M[l];
            readf(l, S2_FWQ,  m.WQ,  (long)S2_D*S2_D);   readf(l, S2_FWK,  m.WK,  (long)S2_D*S2_D);
            readf(l, S2_FWV,  m.WV,  (long)S2_D*S2_D);   readf(l, S2_FWO,  m.WO,  (long)S2_D*S2_D);
            readf(l, S2_FWUP, m.WUP, (long)S2_FFN*S2_D); readf(l, S2_FWDN, m.WDN, (long)S2_D*S2_FFN);
            readf(l, S2_FLN1G,m.LN1G,S2_D);              readf(l, S2_FLN1B,m.LN1B,S2_D);
            readf(l, S2_FLN2G,m.LN2G,S2_D);              readf(l, S2_FLN2B,m.LN2B,S2_D);
            // HYGIENE: under host_embed, EMB/POS are trained host-side in hEMB/hPOS (aliased to M[0]->EMB/POS at init);
            // the on-chip S2_FEMB/S2_FPOS region is the STALE init (chip 0 runs as HID and never writes it) -> skip.
            if(l==0 && !host_embed()){ readf(l, S2_FEMB, m.EMB, (long)S2_VOCAB*S2_D); readf(l, S2_FPOS, m.POS, (long)S2_SEQ*S2_D); }
            if(l==L-1){ readf(l, S2_FLNFG,m.LNFG,S2_D); readf(l, S2_FLNFB,m.LNFB,S2_D); readf(l, S2_FWHEAD,m.WHEAD,(long)S2_VOCAB*S2_D); }
            char op[512]; snprintf(op,512,"%s/%s_master%d.f32",DUMP_DIR,prefix,l); dump_master_file(m,op);
        }
    };
    // START_STEP: resume support — start the loop (and thus the LR schedule) at a later step after reloading
    // a checkpoint via INIT_DIR (see the CKPT_EVERY comment below for the recipe).
    int step0 = getenv("START_STEP") ? atoi(getenv("START_STEP")) : 0;
    for(int step=step0; step<steps; step++){
        STRACE("==== step %d BEGIN ====\n", step);
        // linear warmup then optional COSINE DECAY to LR*LR_MINFRAC (LR_DECAY=1). WHY: the silicon's constant-LR
        // loss floor sits higher than a CPU ref's (noisier effective gradient) — ride it down with a schedule.
        static int   LR_DECAY  = getenv("LR_DECAY")  ? atoi(getenv("LR_DECAY"))   : 0;
        static double LR_MINFR = getenv("LR_MINFRAC")? atof(getenv("LR_MINFRAC")) : 0.1;
        double lr = LR * fmin(1.0, (step+1)/(double)(WARMUP>0?WARMUP:1));
        if (LR_DECAY && step >= WARMUP) {
            double t = (double)(step - WARMUP) / (double)(steps - WARMUP > 0 ? steps - WARMUP : 1);
            lr = LR * (LR_MINFR + (1.0 - LR_MINFR) * 0.5 * (1.0 + cos(M_PI * t)));
        }
        long last_off = 0;
      if(!pipe_on()){
        // ============================ SERIAL sweep (PIPE=0, default, SLOT=0 throughout, bit-compatible) ==========
        // Per sub-batch: a full FWD sweep 0->L-1 then a full BWD sweep L-1->0. ACCUM = effective BATCH: GZERO=1 on
        // the first sub-batch zeroes the persistent DDR grad accumulators, each later BWD adds, DO_UPDATE=1 on the
        // last sub-batch runs streamed dyadic-Adam on the accumulated sum. All ACCUM sub-batches see the same weights.
        for(int k=0;k<ACCUM;k++){
            long off = g_fixedoff>=0 ? g_fixedoff : (long)(rand() % (ntok - S2_SEQ - 2)); last_off = off;
            int do_update = (k==ACCUM-1) ? 1 : 0;

            // ===================== FORWARD SWEEP: chip 0 -> L-1 =====================
            g_phase=0;
            for(int l=0;l<L;l++){
                setcfg(l, S2_PH_FWD, krole(l), 0, 0, 0, 0);
                if(l==0){
                    if(host_embed()){   // R = EMB[tok] + POS computed host-side, fed to chip 0 (HID) via GRIN
                        for(int t=0;t<S2_SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=S2_VOCAB)v=S2_VOCAB-1;
                            for(int d=0;d<S2_D;d++) hmsg[t*S2_D+d]=hEMB[(long)v*S2_D+d]+hPOS[t*S2_D+d]; }
                        put_f(0, S2_GRIN, hmsg, (long)S2_SEQ*S2_D);
                    } else { for(int t=0;t<S2_SEQ;t++) hbuf[0][S2_TOK+t]=data[off+t]; stage(0,S2_TOK,S2_SEQ); }
                } else {
                    put_f(l, S2_GRIN, hmsg, (long)S2_SEQ*S2_D);       // upstream chip's WIDE R' -> this chip's down-message
                }
                if(l==L-1){ for(int t=0;t<S2_SEQ;t++) hbuf[l][S2_TGT+t]=data[off+t+1]; stage(l,S2_TGT,S2_SEQ); }  // OUTPUT targets
                STRACE("s%d k%d FWD chip%d kick...\n", step, k, l);
                kick(l);
                if(l<L-1){
                    STRACE("s%d k%d FWD chip%d readf GROUT...\n", step, k, l);
                    readf(l, S2_GROUT, hmsg, (long)S2_SEQ*S2_D);      // WIDE down-message: this block's R' -> next chip S2_GRIN
                    actstat("R'(wide)", l, step, hmsg, (long)S2_SEQ*S2_D);
                }
            }
            // S2_DIAG: read the forward-chain maxabs RIGHT AFTER the forward sweep (before the backward can
            // clobber WR/GROUT), on the first sub-batch of step 0 — the unambiguous forward-bug localizer.
            if(getenv("S2_DIAG") && step==0 && k==0){
                static float db[S2_HOSTBUF];   // sized for the FEMB (VOCAB*D) + WLG (SEQ*VOCAB) probes
                auto mx=[&](int l,long off,long n){ readf(l,off,db,n); double m=0; for(long j=0;j<n;j++){double a=fabs(db[j]);if(a>m)m=a;} return m; };
                static int32_t tk[S2_SEQ]; readw(0,S2_TOK,tk,S2_SEQ); int tmn=1<<30,tmx=-1; for(int t=0;t<S2_SEQ;t++){if(tk[t]<tmn)tmn=tk[t];if(tk[t]>tmx)tmx=tk[t];}
                fprintf(stderr,"[S2_DIAG-FWD] chip0 role=%d: TOK[%d..%d] FEMB=%.3e FPOS=%.3e WR=%.3e WX1=%.3e WRP=%.3e GROUT=%.3e\n",
                    role_of(0),tmn,tmx, mx(0,S2_FEMB,(long)S2_VOCAB*S2_D), mx(0,S2_FPOS,(long)S2_SEQ*S2_D),
                    mx(0,S2_WR,(long)S2_SEQ*S2_D), mx(0,S2_WX1,(long)S2_SEQ*S2_D), mx(0,S2_WRP,(long)S2_SEQ*S2_D), mx(0,S2_GROUT,(long)S2_SEQ*S2_D));
                fprintf(stderr,"[S2_DIAG-FWD] chip%d role=%d: GRIN=%.3e WRP=%.3e WXF=%.3e WLG=%.3e\n",
                    L-1,role_of(L-1), mx(L-1,S2_GRIN,(long)S2_SEQ*S2_D), mx(L-1,S2_WRP,(long)S2_SEQ*S2_D), mx(L-1,S2_WXF,(long)S2_SEQ*S2_D), mx(L-1,S2_WLG,(long)S2_SEQ*S2_VOCAB));
                fflush(stderr);
            }
            // FWD_DUMP: after the forward sweep of step 0 / sub-batch 0, dump every chip's full forward wide-cache
            // (SLOT 0) to <DUMP_DIR>/fwd_<chip>_<name>.f32 for the bit-level silicon-vs-shim per-stage compare.
            // Done BEFORE the backward sweep (below) so the wide cache still holds this forward's activations.
            if(fwd_dump() && step==0 && k==0){
                fprintf(stderr,"[FWD_DUMP] dumping per-chip forward wide-cache -> %s (fwd_<chip>_<name>.f32)\n", DUMP_DIR);
                for(int l=0;l<L;l++) fwd_dump_chip(l, DUMP_DIR, (l==L-1));
                fflush(stderr);
            }
            // ===================== BACKWARD SWEEP: chip L-1 -> 0 =====================
            g_phase=1;
          if(!host_backward()){
            for(int l=L-1;l>=0;l--){
                setcfg(l, S2_PH_BWD, krole(l), do_update, do_update?lrcode(lr):0, (k==0)?1:0, 0);   // DO_UP -> adam; k==0 -> GZERO
                if(l<L-1){ put_f(l, S2_GRIN, emsg, (long)S2_SEQ*S2_D); }  // neighbour WIDE error in
                STRACE("s%d k%d BWD chip%d kick (do_update=%d)...\n", step, k, l, do_update);
                kick(l);
                if(l>0){ STRACE("s%d k%d BWD chip%d readf GROUT...\n", step, k, l);
                    readf(l, S2_GROUT, emsg, (long)S2_SEQ*S2_D); }  // WIDE up-message: this block's input-grad -> prev chip
                else if(host_embed()){   // chip 0 (HID) up-message gR = the embed input gradient; train EMB/POS host-side
                    readf(0, S2_GROUT, emsg, (long)S2_SEQ*S2_D);
                    if(k==0){ memset(hgE,0,(size_t)S2_VOCAB*S2_D*4); memset(hgP,0,(size_t)S2_SEQ*S2_D*4); }  // GZERO
                    for(int t=0;t<S2_SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=S2_VOCAB)v=S2_VOCAB-1;
                        for(int d=0;d<S2_D;d++){ float g=emsg[t*S2_D+d]; hgE[(long)v*S2_D+d]+=g; hgP[t*S2_D+d]+=g; } }
                    if(do_update && !getenv("EMB_FREEZE")){
                        for(long i=0;i<(long)S2_VOCAB*S2_D;i++) hEMB[i]-=dyad_adam(hgE[i],hmE[i],hvE[i],lr);
                        for(long i=0;i<(long)S2_SEQ*S2_D;i++)   hPOS[i]-=dyad_adam(hgP[i],hmP[i],hvP[i],lr);
                    }
                }
            }
          } else {
            // ================= HOST BACKWARD SWEEP: chip L-1 -> 0, computed ENTIRELY on the host =================
            // The kernel already ran the FORWARD only (loop above). For each chip we readf its DDR wide-cache
            // activations (R,X1,Q,K,V,ATTN,M,X2,HD,RP,RMASK,LN scalars; +XF,LG on OUT), recompute PRAW from Q,K,
            // and run the twin's bwd_block: CE grad (OUT), head + LNF bwd, FFN bwd (relu mask), LN2 bwd, attention
            // bwd (PRAW recompute), WQ/WK/WV dx+wgrad, LN1 bwd -> gR. gR is the cross-chip up-error (host relay,
            // replacing the broken in-kernel GROUT/GRIN). Grads accumulate in HG[l]; adam applies on the last
            // sub-batch to M[l] and re-stages the trained master to DDR for the next step's forward.
            // Per-chip scratch (static: reused across chips/sub-batches; SEQ*FFN-sized).
            static float aR[S2_SEQ*S2_D], aX1[S2_SEQ*S2_D], aQ[S2_SEQ*S2_D], aK[S2_SEQ*S2_D], aV[S2_SEQ*S2_D];
            static float aATTN[S2_SEQ*S2_D], aM[S2_SEQ*S2_D], aX2[S2_SEQ*S2_D], aHD[S2_SEQ*S2_FFN], aRP[S2_SEQ*S2_D];
            static float aRMASK[S2_SEQ*S2_FFN], aXF[S2_SEQ*S2_D], aLG[S2_SEQ*S2_VOCAB];
            static float ln1s[S2_SEQ], ln2s[S2_SEQ], lnfs[S2_SEQ];
            static float gRP[S2_SEQ*S2_D], gM[S2_SEQ*S2_D], gHD[S2_SEQ*S2_FFN], gX2[S2_SEQ*S2_D], gM2[S2_SEQ*S2_D];
            static float gR[S2_SEQ*S2_D], gATTN[S2_SEQ*S2_D], gQ[S2_SEQ*S2_D], gK[S2_SEQ*S2_D], gV[S2_SEQ*S2_D];
            static float gX1[S2_SEQ*S2_D], gTMP[S2_SEQ*S2_D], gXF[S2_SEQ*S2_D], gLG[S2_SEQ*S2_VOCAB];
            for(int l=L-1;l>=0;l--){
                FMaster &W=*M[l]; FMaster &G=*HG[l];
                if(k==0) memset(&G,0,sizeof(FMaster));                   // GZERO: zero this chip's grad accumulators
                // ---- read forward activations from this chip's DDR wide-cache (SLOT 0) ----
                readf(l,S2_WR,aR,(long)S2_SEQ*S2_D);      readf(l,S2_WX1,aX1,(long)S2_SEQ*S2_D);
                readf(l,S2_WQ,aQ,(long)S2_SEQ*S2_D);      readf(l,S2_WK,aK,(long)S2_SEQ*S2_D);   readf(l,S2_WV,aV,(long)S2_SEQ*S2_D);
                readf(l,S2_WATTN,aATTN,(long)S2_SEQ*S2_D);readf(l,S2_WM,aM,(long)S2_SEQ*S2_D);   readf(l,S2_WX2,aX2,(long)S2_SEQ*S2_D);
                readf(l,S2_WHD,aHD,(long)S2_SEQ*S2_FFN);  readf(l,S2_WRP,aRP,(long)S2_SEQ*S2_D); readf(l,S2_WRMASK,aRMASK,(long)S2_SEQ*S2_FFN);
                readf(l,S2_WLN1S,ln1s,S2_SEQ);            readf(l,S2_WLN2S,ln2s,S2_SEQ);
                // ---- OUTPUT chip: CE grad -> head bwd -> LNF bwd -> gRP ; else gRP = incoming host up-error emsg ----
                if(l==L-1){
                    readf(l,S2_WXF,aXF,(long)S2_SEQ*S2_D); readf(l,S2_WLG,aLG,(long)S2_SEQ*S2_VOCAB); readf(l,S2_WLNFS,lnfs,S2_SEQ);
                    for(int t=0;t<S2_SEQ;t++){ float*lr_=aLG+(long)t*S2_VOCAB; float mx=-1e30f;
                        for(int v=0;v<S2_VOCAB;v++) if(lr_[v]>mx)mx=lr_[v];
                        double sum=0; static float pr[S2_VOCAB]; for(int v=0;v<S2_VOCAB;v++){ pr[v]=expf(lr_[v]-mx); sum+=pr[v]; }
                        int tg=(int)data[off+t+1]; for(int v=0;v<S2_VOCAB;v++){ float p=pr[v]/(float)sum, oh=(v==tg)?1.0f:0.0f; gLG[(long)t*S2_VOCAB+v]=hb_P((p-oh)/S2_SEQ); } }
                    // gXF = P(gLG @ dyad(WHEAD)) ; gWHEAD += P(gLG^T @ XF)
                    hb_linbwd(gLG,W.WHEAD,aXF,S2_VOCAB,S2_D,gXF,G.WHEAD);
                    memcpy(gRP,gXF,sizeof(float)*(long)S2_SEQ*S2_D);
                    hb_ln_bwd(gRP,W.LNFG,lnfs,S2_SEQ);                   // gRP = ln_bwd(gXF, LNFG, lnfs)
                } else {
                    memcpy(gRP,emsg,sizeof(float)*(long)S2_SEQ*S2_D);    // incoming wide error (host relay from chip l+1)
                }
                // ---- FFN backward: RP = M + FO ; FO = WDN(HD) ; HD = qact(relu(WUP(X2))) ----
                memcpy(gM,gRP,sizeof(float)*(long)S2_SEQ*S2_D);         // gM = gRP
                hb_linbwd(gRP,W.WDN,aHD,S2_D,S2_FFN,gHD,G.WDN);        // gHD = P(gRP@dyad(WDN)) ; gWDN += P(gRP^T@HD)
                for(long i=0;i<(long)S2_SEQ*S2_FFN;i++) if(aRMASK[i]<=0.0f) gHD[i]=0.0f;   // relu mask
                hb_linbwd(gHD,W.WUP,aX2,S2_FFN,S2_D,gX2,G.WUP);        // gX2 = P(gHD@dyad(WUP)) ; gWUP += P(gHD^T@X2)
                // ---- LN2 over M: gM = P(gM + ln_bwd(gX2, LN2G, ln2s)) ----
                memcpy(gM2,gX2,sizeof(float)*(long)S2_SEQ*S2_D); hb_ln_bwd(gM2,W.LN2G,ln2s,S2_SEQ);
                for(long i=0;i<(long)S2_SEQ*S2_D;i++) gM[i]=hb_P(gM[i]+gM2[i]);
                // ---- attention backward: M = R + AO ; AO = WO(ATTN) ; ATTN = qact(P@V) ----
                memcpy(gR,gM,sizeof(float)*(long)S2_SEQ*S2_D);         // gR = gM
                hb_linbwd(gM,W.WO,aATTN,S2_D,S2_D,gATTN,G.WO);         // gATTN = P(gM@dyad(WO)) ; gWO += P(gM^T@ATTN)
                hb_attn_bwd(aQ,aK,aV,gATTN,gQ,gK,gV);                  // PRAW recompute + gQ,gK,gV (P'd)
                // ---- gX1 += dyad(W{Q,K,V})-route ; wgrad each ; then LN1 ----
                memset(gX1,0,sizeof(float)*(long)S2_SEQ*S2_D);
                hb_linbwd(gQ,W.WQ,aX1,S2_D,S2_D,gTMP,G.WQ); for(long i=0;i<(long)S2_SEQ*S2_D;i++) gX1[i]+=gTMP[i];
                hb_linbwd(gK,W.WK,aX1,S2_D,S2_D,gTMP,G.WK); for(long i=0;i<(long)S2_SEQ*S2_D;i++) gX1[i]+=gTMP[i];
                hb_linbwd(gV,W.WV,aX1,S2_D,S2_D,gTMP,G.WV); for(long i=0;i<(long)S2_SEQ*S2_D;i++) gX1[i]+=gTMP[i];
                for(long i=0;i<(long)S2_SEQ*S2_D;i++) gX1[i]=hb_P(gX1[i]);   // gX1 = P(gX1)
                memcpy(gTMP,gX1,sizeof(float)*(long)S2_SEQ*S2_D); hb_ln_bwd(gTMP,W.LN1G,ln1s,S2_SEQ);   // gR1 -> TMP
                for(long i=0;i<(long)S2_SEQ*S2_D;i++) gR[i]=hb_P(gR[i]+gTMP[i]);   // gR = P(gR + gR1)
                // ---- emit: INPUT/host-embed chip scatters gR -> EMB/POS host grad ; else relay gR as the up-error ----
                if(l==0){
                    if(k==0){ memset(hgE,0,(size_t)S2_VOCAB*S2_D*4); memset(hgP,0,(size_t)S2_SEQ*S2_D*4); }  // GZERO
                    for(int t=0;t<S2_SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=S2_VOCAB)v=S2_VOCAB-1;
                        for(int d=0;d<S2_D;d++){ float g=gR[t*S2_D+d]; hgE[(long)v*S2_D+d]+=g; hgP[t*S2_D+d]+=g; } }
                } else {
                    for(long i=0;i<(long)S2_SEQ*S2_D;i++) emsg[i]=hb_P(gR[i]);   // wide up-message = P(gR) (host relay to chip l-1)
                }
            }
            // ---- last sub-batch: dyadic-Adam apply on HOST-staged masters, then re-stage to DDR for next fwd ----
            if(do_update){
                for(int l=0;l<L;l++){
                    FMaster &W=*M[l]; FMaster &Mm=*HM[l]; FMaster &Vv=*HV[l]; FMaster &G=*HG[l];
                    auto adamT=[&](float*w,float*m,float*v,float*g,long n){ for(long i=0;i<n;i++) w[i]-=dyad_adam(g[i],m[i],v[i],(float)lr); };
                    adamT(W.WQ,Mm.WQ,Vv.WQ,G.WQ,(long)S2_D*S2_D); adamT(W.WK,Mm.WK,Vv.WK,G.WK,(long)S2_D*S2_D);
                    adamT(W.WV,Mm.WV,Vv.WV,G.WV,(long)S2_D*S2_D); adamT(W.WO,Mm.WO,Vv.WO,G.WO,(long)S2_D*S2_D);
                    adamT(W.WUP,Mm.WUP,Vv.WUP,G.WUP,(long)S2_FFN*S2_D); adamT(W.WDN,Mm.WDN,Vv.WDN,G.WDN,(long)S2_D*S2_FFN);
                    if(l==L-1) adamT(W.WHEAD,Mm.WHEAD,Vv.WHEAD,G.WHEAD,(long)S2_VOCAB*S2_D);
                    // re-stage the trained matmul/head masters into DDR for the NEXT step's forward (LN frozen; EMB/POS host-staged into GRIN each fwd).
                    put_f(l,S2_FWQ,W.WQ,(long)S2_D*S2_D); put_f(l,S2_FWK,W.WK,(long)S2_D*S2_D);
                    put_f(l,S2_FWV,W.WV,(long)S2_D*S2_D); put_f(l,S2_FWO,W.WO,(long)S2_D*S2_D);
                    put_f(l,S2_FWUP,W.WUP,(long)S2_FFN*S2_D); put_f(l,S2_FWDN,W.WDN,(long)S2_D*S2_FFN);
                    if(l==L-1) put_f(l,S2_FWHEAD,W.WHEAD,(long)S2_VOCAB*S2_D);
                }
                // EMB/POS host dyadic-Adam (chip-0 tail; same recipe as HOST_EMBED serial path)
                if(!getenv("EMB_FREEZE")){
                    for(long i=0;i<(long)S2_VOCAB*S2_D;i++) hEMB[i]-=dyad_adam(hgE[i],hmE[i],hvE[i],lr);
                    for(long i=0;i<(long)S2_SEQ*S2_D;i++)   hPOS[i]-=dyad_adam(hgP[i],hmP[i],hvP[i],lr);
                }
            }
          }
        }
      } else {
        // HYGIENE: PIPE slot buffers (hmsg2/emsg2) and the kernel's single-slot GACC alias if ACCUM>S2_SLOTS
        // (two sub-batches map to slot g%S2_SLOTS) -> silent activation/gradient corruption. Guard it.
        if(ACCUM > S2_SLOTS){ fprintf(stderr,"FATAL: PIPE=1 requires ACCUM<=S2_SLOTS (%d>%d); lower ACCUM or use serial PIPE=0\n", ACCUM, S2_SLOTS); return 8; }
        // ==================== PIPELINED (PIPE=1): overlap sub-batches via S2_SLOTS(=8) slots ====================
        // Instead of 2L serial kicks/sub-batch, run a dependency EVENT LOOP over the whole step's 2*L*ACCUM kicks
        // keyed by the GLOBAL sub-batch index g (=k within this step; windows never cross a step -> the per-step
        // barrier below structurally satisfies the g%ACCUM==0 window-boundary constraint). A chip runs ONE kick at a
        // time; it picks any READY op with BACKWARD-FIRST preference. slot(g)=g%S2_SLOTS: up to S2_SLOTS in-flight
        // sub-batches never share a slot, so F(g+k) (k<S2_SLOTS) can precede B(g). GACC RMW needs strict ASCENDING
        // bwd order per chip (monotonic bnext[]); GZERO on g==0, DO_UP on g==ACCUM-1 (first/last bwd of the window).
        for(int k=0;k<ACCUM;k++) soff[k] = g_fixedoff>=0 ? g_fixedoff : (long)(rand() % (ntok - S2_SEQ - 2));
        last_off = soff[ACCUM-1];
        int fdone[MAXL], bdone[MAXL], fnext[MAXL], bnext[MAXL], busy[MAXL], bph[MAXL], bg[MAXL];
        for(int l=0;l<L;l++){ fdone[l]=bdone[l]=fnext[l]=bnext[l]=0; busy[l]=0; bph[l]=0; bg[l]=0; }
        int total=2*L*ACCUM, ndone=0; double last_progress=now_s();
        while(ndone<total){
            // (1) reap completions: attribute timing + read this op's outgoing neighbour message into its slot buffer
            for(int l=0;l<L;l++) if(busy[l] && kick_poll(l)){
                int ph=bph[l], g=bg[l], slot=g%S2_SLOTS; long po=slot_po(slot);
                g_kick_s[ph][l]+=now_s()-g_kick_t0[l]; g_kick_n[ph][l]++;
                if(ph==0){ if(l<L-1){ readf(l, S2_GROUT+po, hmsg2[slot], (long)S2_SEQ*S2_D); } fdone[l]++; }   // fwd down-msg R'
                else     { if(l>0)  { readf(l, S2_GROUT+po, emsg2[slot], (long)S2_SEQ*S2_D); }
                           else if(host_embed()){   // chip 0 (HID) up-msg gR -> host embed gradient (slot-aware GZERO/DO_UP)
                               readf(0, S2_GROUT+po, emsg2[slot], (long)S2_SEQ*S2_D); long off=soff[g];
                               if(g==0){ memset(hgE,0,(size_t)S2_VOCAB*S2_D*4); memset(hgP,0,(size_t)S2_SEQ*S2_D*4); }
                               for(int t=0;t<S2_SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=S2_VOCAB)v=S2_VOCAB-1;
                                   for(int d=0;d<S2_D;d++){ float gg=emsg2[slot][t*S2_D+d]; hgE[(long)v*S2_D+d]+=gg; hgP[t*S2_D+d]+=gg; } }
                               if(g==ACCUM-1 && !getenv("EMB_FREEZE")){
                                   for(long i=0;i<(long)S2_VOCAB*S2_D;i++) hEMB[i]-=dyad_adam(hgE[i],hmE[i],hvE[i],lr);
                                   for(long i=0;i<(long)S2_SEQ*S2_D;i++)   hPOS[i]-=dyad_adam(hgP[i],hmP[i],hvP[i],lr); }
                           }
                           bdone[l]++; }   // bwd up-msg gR
                busy[l]=0; ndone++; last_progress=now_s();
            }
            // (2) dispatch: for each idle chip pick a ready op, BACKWARD-FIRST (drains slots), else a ready forward
            for(int l=0;l<L;l++) if(!busy[l]){
                int gb=bnext[l];
                int bwok = (gb<ACCUM) && (fdone[l]>gb) && (l==L-1 || bdone[l+1]>gb);           // B_l(g): F_l(g) & B_{l+1}(g)
                int gf=fnext[l];
                // F_l(g): F_{l-1}(g) done AND the slot g%S2_SLOTS is free (its previous occupant B_l(g-S2_SLOTS) done).
                int fwok = (gf<ACCUM) && (l==0 || fdone[l-1]>gf) && (gf<S2_SLOTS || bdone[l]>gf-S2_SLOTS);
                if(bwok){
                    int slot=gb%S2_SLOTS; long po=slot_po(slot);
                    int doup=(gb==ACCUM-1)?1:0, gz=(gb==0)?1:0;
                    setcfg(l, S2_PH_BWD, krole(l), doup, doup?lrcode(lr):0, gz, slot);
                    if(l<L-1){ put_f(l, S2_GRIN+po, emsg2[slot], (long)S2_SEQ*S2_D); }  // neighbour WIDE error in
                    kick_start(l); busy[l]=1; bph[l]=1; bg[l]=gb; bnext[l]++;
                } else if(fwok){
                    int slot=gf%S2_SLOTS; long po=slot_po(slot); long off=soff[gf];
                    setcfg(l, S2_PH_FWD, krole(l), 0, 0, 0, slot);
                    if(l==0){
                        if(host_embed()){   // R = EMB[tok]+POS host-side -> chip 0 (HID) GRIN slot
                            for(int t=0;t<S2_SEQ;t++){ int v=(int)data[off+t]; if(v<0)v=0; if(v>=S2_VOCAB)v=S2_VOCAB-1;
                                for(int d=0;d<S2_D;d++) hmsg2[slot][t*S2_D+d]=hEMB[(long)v*S2_D+d]+hPOS[t*S2_D+d]; }
                            put_f(0, S2_GRIN+po, hmsg2[slot], (long)S2_SEQ*S2_D);
                        } else { for(int t=0;t<S2_SEQ;t++) hbuf[0][S2_TOK+slot*S2_SEQ+t]=data[off+t]; stage(0,S2_TOK+slot*S2_SEQ,S2_SEQ); }
                    } else {
                        put_f(l, S2_GRIN+po, hmsg2[slot], (long)S2_SEQ*S2_D);   // upstream chip's R' -> down-message
                    }
                    if(l==L-1){ for(int t=0;t<S2_SEQ;t++) hbuf[l][S2_TGT+slot*S2_SEQ+t]=data[off+t+1]; stage(l,S2_TGT+slot*S2_SEQ,S2_SEQ); }  // OUTPUT targets (read in this g's BWD)
                    kick_start(l); busy[l]=1; bph[l]=0; bg[l]=gf; fnext[l]++;
                }
            }
            if(now_s()-last_progress>60.0){ fprintf(stderr,"PIPE SCHED STALL: no progress 60s (done %d/%d)\n",ndone,total); return 7; }
        }
      }

        // ===================== per-step bpb/acc from the OUTPUT chip's WIDE logits =====================
        // slot of the LAST sub-batch (k=ACCUM-1): 0 in serial, (ACCUM-1)%S2_SLOTS in pipe mode.
        long metric_po = pipe_on() ? slot_po((ACCUM-1)%S2_SLOTS) : 0;
        // S2_DIAG: at step 0, trace the forward chain maxabs to localize the all-zero-logit bug (relay/block/LNF/head).
        if(getenv("S2_DIAG") && step==0){
            static float dbuf[S2_HOSTBUF]; long po0 = pipe_on()?slot_po((ACCUM-1)%S2_SLOTS):0;   // FEMB/WLG probes are VOCAB-sized
            auto mx=[&](int l,int off,long n){ readf(l,off+po0,dbuf,n); double m=0; for(long j=0;j<n;j++){double a=fabs(dbuf[j]); if(a>m)m=a;} return m; };
            // chip0 tokens (int) + INPUT forward chain: FEMB, FPOS, WR(embed), WX1(LN1), WRP(block), GROUT
            static int32_t tk[S2_SEQ]; readw(0,S2_TOK,tk,S2_SEQ); int tmin=1<<30,tmax=-1; for(int t=0;t<S2_SEQ;t++){ if(tk[t]<tmin)tmin=tk[t]; if(tk[t]>tmax)tmax=tk[t]; }
            fprintf(stderr,"[S2_DIAG] chip0: TOK[%d..%d] FEMB=%.3e FPOS=%.3e WR(embed)=%.3e WX1(LN1)=%.3e WRP(block)=%.3e GROUT=%.3e\n",
                tmin,tmax, mx(0,S2_FEMB,(long)S2_VOCAB*S2_D), mx(0,S2_FPOS,(long)S2_SEQ*S2_D),
                mx(0,S2_WR,(long)S2_SEQ*S2_D), mx(0,S2_WX1,(long)S2_SEQ*S2_D), mx(0,S2_WRP,(long)S2_SEQ*S2_D), mx(0,S2_GROUT,(long)S2_SEQ*S2_D));
            fprintf(stderr,"[S2_DIAG] chip%d GRIN=%.3e WRP=%.3e WXF=%.3e WLG=%.3e FWHEAD=%.3e\n", L-1,
                mx(L-1,S2_GRIN,(long)S2_SEQ*S2_D), mx(L-1,S2_WRP,(long)S2_SEQ*S2_D),
                mx(L-1,S2_WXF,(long)S2_SEQ*S2_D), mx(L-1,S2_WLG,(long)S2_SEQ*S2_VOCAB),
                ({ readf(L-1,S2_FWHEAD,dbuf,(long)S2_SEQ*S2_D); double m=0; for(long j=0;j<(long)S2_SEQ*S2_D;j++){double a=fabs(dbuf[j]);if(a>m)m=a;} m; }));
            fflush(stderr);
        }
        readf(L-1, S2_WLG + metric_po, wlg, (long)S2_SEQ*S2_VOCAB);
        actstat("logits", L-1, step, wlg, (long)S2_SEQ*S2_VOCAB);
        { int c=0; double bits=0; const double L2=log(2.0);
          for(int t=0;t<S2_SEQ;t++){ int tg=data[last_off+t+1];
              double mx=-1e300; int am=0; for(int v=0;v<S2_VOCAB;v++){ double lv=wlg[t*S2_VOCAB+v]; if(lv>mx){mx=lv;am=v;} }
              if(am==tg)c++;
              double sum=0,pt=0; for(int v=0;v<S2_VOCAB;v++){ double e=exp(wlg[t*S2_VOCAB+v]-mx); sum+=e; if(v==tg)pt=e; }
              double pp=pt/sum; if(pp<1e-12)pp=1e-12; bits+=-log(pp)/L2; }
          double ac=(double)c/S2_SEQ, bp=bits/S2_SEQ;
          acc_ema = acc_ema<0?ac:0.99*acc_ema+0.01*ac; bpb_ema = bpb_ema<0?bp:0.99*bpb_ema+0.01*bp;
          static int PRINT_EVERY = getenv("PRINT_EVERY") ? atoi(getenv("PRINT_EVERY")) : 1;   // per-step by default (readback happens anyway)
          if(PRINT_EVERY>0 && (step%PRINT_EVERY==0 || step==steps-1))
              printf("[xfs2shard] step %5d  bpb=%.3f bpb_ema=%.3f  acc=%.3f acc_ema=%.3f  lr=%.1e  (%.3f steps/s)\n",
                     step, bp, bpb_ema, ac, acc_ema, lr, (step+1-step0)/(now_s()-t0));
          if(step%200==0 && step>0) kick_report();
          fflush(stdout);
          // ---- rolling checkpoint (CKPT_EVERY>0): masters -> DUMP_DIR/ckpt_master<l>.f32 + ckpt_meta.txt.
          // Resume: copy ckpt_master<l>.f32 -> <INIT_DIR>/master<l>.f32, relaunch with INIT_S3/INIT_DIR +
          // START_STEP=<step+1> (restores weights + LR phase; Adam m/v stay in DDR and are NOT persisted —
          // a resume resets the moments, which re-settle within ~1/(1-0.875)=8 steps).
          static int CKPT_EVERY = getenv("CKPT_EVERY") ? atoi(getenv("CKPT_EVERY")) : 0;
          if(CKPT_EVERY>0 && step>0 && (step%CKPT_EVERY==0 || step==steps-1)){
              persist_masters("ckpt");
              char mp[512]; snprintf(mp,512,"%s/ckpt_meta.txt",DUMP_DIR); FILE*mf=fopen(mp,"w");
              if(mf){ fprintf(mf,"step=%d\nlr=%.6e\nbpb_ema=%.4f\nacc_ema=%.4f\nresume: INIT_DIR<-ckpt_master*.f32 (rename to master<l>.f32) + START_STEP=%d\n",
                              step, lr, bpb_ema, acc_ema, step+1); fclose(mf); }
              printf("[ckpt] step %d -> %s/ckpt_master*.f32\n", step, DUMP_DIR); fflush(stdout);
          }
        }
    }

    double wall=now_s()-t0;
    kick_report();
    printf("\n========== DISTRIBUTED SCALE-2 D=512 PC TRANSFORMER on %d FPGAs ==========\n", L);
    printf("[throughput] %d steps (ACCUM=%d), %.1fs, %.3f steps/s | %.0f tok/s | final bpb_ema=%.3f acc_ema=%.3f\n",
           steps-step0, ACCUM, wall, (steps-step0)/wall, (steps-step0)*(double)S2_SEQ*ACCUM/wall, bpb_ema, acc_ema);
    printf("[gateC] only neighbour wide hidden-states (down) + wide errors (up) crossed chips; NO global model/gradient/all-reduce\n");

    // ---- persist each chip's TRAINED float master back from DDR to host (fetch trained weights) ----
    persist_masters("xfs2");
    for(int l=0;l<L;l++) printf("[xfs2shard] chip %d: trained float master -> %s/xfs2_master%d.f32\n", l, DUMP_DIR, l);

    // ---- optional coherent-text sample from the OUTPUT chip (roll a SEQ context through the chain) ----
    if(getenv("SAMPLE")){
        int NS = atoi(getenv("SAMPLE")); if(NS<=0) NS=300;
        int ctx[S2_SEQ]; for(int t=0;t<S2_SEQ;t++) ctx[t]=0;     // start from token 0
        printf("\n===== SAMPLE (%d bytes, token ids) =====\n", NS);
        unsigned rng=12345;
        for(int s=0;s<NS;s++){
            for(int l=0;l<L;l++){
                setcfg(l, S2_PH_FWD, role_of(l), 0, 0, 0, 0);   // sampling: serial, SLOT 0
                if(l==0){ for(int t=0;t<S2_SEQ;t++) hbuf[0][S2_TOK+t]=ctx[t]; stage(0,S2_TOK,S2_SEQ); }
                else    { put_f(l, S2_GRIN, hmsg, (long)S2_SEQ*S2_D); }
                kick(l);
                if(l<L-1) readf(l, S2_GROUT, hmsg, (long)S2_SEQ*S2_D);
            }
            readf(L-1, S2_WLG, wlg, (long)S2_SEQ*S2_VOCAB);
            int last=S2_SEQ-1; double mx=-1e300; for(int v=0;v<S2_VOCAB;v++){ double l2=wlg[last*S2_VOCAB+v]; if(l2>mx)mx=l2; }
            double sum=0,pr[S2_VOCAB]; for(int v=0;v<S2_VOCAB;v++){ pr[v]=exp(wlg[last*S2_VOCAB+v]-mx); sum+=pr[v]; }
            rng=rng*1103515245u+12345u; double r=((rng>>8)&0xffffff)/(double)0x1000000*sum; int nxt=0; double acc=0;
            for(int v=0;v<S2_VOCAB;v++){ acc+=pr[v]; if(acc>=r){ nxt=v; break; } }
            printf("%d ", nxt);
            for(int t=0;t<S2_SEQ-1;t++) ctx[t]=ctx[t+1]; ctx[S2_SEQ-1]=nxt;
        }
        printf("\n(token ids = byte values 0-255; map back to chars host-side)\n");
    }

    for(int l=0;l<L;l++){ free(M[l]); fpga_pci_detach(bar0[l]); fpga_pci_detach(bar4[l]); fpga_pci_detach(bar4r[l]); }
    return 0;
}
