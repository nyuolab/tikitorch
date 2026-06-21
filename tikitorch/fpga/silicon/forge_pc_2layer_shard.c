// forge_pc_2layer_shard.c — the multi-F2 scale-out demonstration: a 2-layer predictive-coding BACKWARD
// sweep SHARDED ACROSS TWO PHYSICAL FPGAs (slots 0 and 1 of an f2.12xlarge), each holding one layer,
// exchanging ONLY a single neighbour error message (e_down) — NO global gradient, NO weight transport,
// NO barrier. Verifies the sharded result is bit-for-bit identical to cascade_grads.
//
// This is the silicon realization of pc_model_parallel.py (the gloo Test-1 correctness proof): the PC
// credit-assignment backward, which removes backprop's global ops, run across real separate accelerators.
//   - slot 1 (layer 1): forge_pc_wgrad(e_top, W1, acts1, mask1, skip1) -> gW1, gB1, e_down1
//   - NEIGHBOUR MESSAGE: e_down1 (slot 1 -> host -> slot 0). The ONLY cross-FPGA data.
//   - slot 0 (layer 0): forge_pc_wgrad(e_down1, W0, acts0, mask0, skip0) -> gW0, gB0, e_down0
// Each FPGA produces ONLY its own layer's gradient locally (no chip ever holds the global gradient or the
// other layer's weights). The host computes the forward + the reference (the forward is a standard sweep,
// equally shardable; the THESIS — no global backward ops — lives in the BACKWARD, which is what is sharded).
//
// Build on an f2.12xlarge (2 FPGAs): gcc -O2 -I$SDK/include forge_pc_2layer_shard.c -lfpga_mgmt -o ...
//   sudo ./forge_pc_2layer_shard            (loads the AGFI on both slots beforehand via the runner)
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

#define PC_D 256
#define PC_N 16
#define O_E    0
#define O_W    (O_E    + PC_N*PC_D)
#define O_ACT  (O_W    + PC_D*PC_D)
#define O_MASK (O_ACT  + PC_N*PC_D)
#define O_SKIP (O_MASK + PC_N*PC_D)
#define O_GW   (O_SKIP + PC_N*PC_D)
#define O_EDN  (O_GW   + PC_D*PC_D)
#define O_GB   (O_EDN  + PC_N*PC_D)
#define BUFLEN (O_GB   + PC_D)

#define R_AP_CTRL  0x00
#define R_BUFFER_0 0x10
#define R_BUFFER_1 0x14
#define AP_START   0x01
#define AP_DONE    0x02
#define DDR_BUF_ADDR 0x0ULL

static long ref_dyad(long v) {                          // == kernel dyad: table-free 1.5x round
    long a = v < 0 ? -v : v; if (a == 0) return 0;
    int msb = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { msb = b; break; }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;
    long m = 1L << msb; return v < 0 ? -m : m;
}
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }

// CPU reference for one layer-step (writes gW/e_down/gB into the buffer's output regions, like the kernel)
static void ref_layer(int32_t* h) {
    static long delta[PC_N][PC_D];
    for (int n = 0; n < PC_N; n++) for (int o = 0; o < PC_D; o++)
        delta[n][o] = (h[O_MASK+n*PC_D+o] && h[O_SKIP+n*PC_D+o]) ? h[O_E+n*PC_D+o] : 0;
    for (int o = 0; o < PC_D; o++) {
        long gb = 0;
        for (int i = 0; i < PC_D; i++) {
            long acc = 0;
            for (int n = 0; n < PC_N; n++) acc += ref_dyad(delta[n][o]) * ref_dyad(h[O_ACT+n*PC_D+i]);
            h[O_GW + o*PC_D + i] = (int32_t)acc;
        }
        for (int n = 0; n < PC_N; n++) gb += ref_dyad(delta[n][o]);
        h[O_GB + o] = (int32_t)gb;
    }
    for (int n = 0; n < PC_N; n++) for (int i = 0; i < PC_D; i++) {
        long acc = 0;
        for (int o = 0; o < PC_D; o++) acc += delta[n][o] * ref_dyad(h[O_W+o*PC_D+i]);
        if (!h[O_SKIP+n*PC_D+i]) acc += h[O_E+n*PC_D+i];
        h[O_EDN + n*PC_D + i] = (int32_t)acc;
    }
}

// run one layer's backward on a specific FPGA slot: stage inputs -> kick -> read gW/e_down/gB back
static int run_layer_on_fpga(int slot, int32_t* h) {
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT, bar4 = PCI_BAR_HANDLE_INIT, bar4r = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0) { fprintf(stderr,"slot %d BAR0 attach failed\n",slot); return 1; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &bar4) != 0) { fprintf(stderr,"slot %d BAR4 burst attach failed\n",slot); return 1; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, 0, &bar4r) != 0) { fprintf(stderr,"slot %d BAR4 read attach failed\n",slot); return 1; }
    // zero the output regions in h before staging (so a non-write would be visibly 0)
    for (int j = 0; j < PC_D*PC_D; j++) h[O_GW + j] = 0;
    for (int j = 0; j < PC_N*PC_D; j++) h[O_EDN + j] = 0;
    for (int j = 0; j < PC_D; j++) h[O_GB + j] = 0;
    const long CHW = 256*1024;
    for (long off = 0; off < BUFLEN; off += CHW) {
        long n = (BUFLEN - off < CHW) ? (BUFLEN - off) : CHW;
        if (fpga_pci_write_burst(bar4, DDR_BUF_ADDR + (uint64_t)off*4, (uint32_t*)(h+off), (uint64_t)n) != 0) {
            fprintf(stderr,"slot %d write_burst @%ld failed\n",slot,off); return 1; }
    }
    fpga_pci_poke(bar0, R_BUFFER_0, (uint32_t)(DDR_BUF_ADDR & 0xFFFFFFFFULL));
    fpga_pci_poke(bar0, R_BUFFER_1, (uint32_t)(DDR_BUF_ADDR >> 32));
    fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
    double t0 = now_s(); int done = 0; uint32_t ctrl = 0;
    while (now_s() - t0 < 10.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { done = 1; break; } }
    if (!done) { fprintf(stderr,"slot %d KICK HUNG\n",slot); return 1; }
    printf("[shard] slot %d: AP_DONE in %.3f ms\n", slot, (now_s()-t0)*1e3);
    uint32_t v;
    for (int j = 0; j < PC_D*PC_D; j++) { fpga_pci_peek(bar4r, DDR_BUF_ADDR+(uint64_t)(O_GW+j)*4, &v);  h[O_GW+j]  = (int32_t)v; }
    for (int j = 0; j < PC_N*PC_D; j++) { fpga_pci_peek(bar4r, DDR_BUF_ADDR+(uint64_t)(O_EDN+j)*4, &v); h[O_EDN+j] = (int32_t)v; }
    for (int j = 0; j < PC_D; j++)      { fpga_pci_peek(bar4r, DDR_BUF_ADDR+(uint64_t)(O_GB+j)*4, &v);  h[O_GB+j]  = (int32_t)v; }
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return 0;
}

static long maxdiff(const int32_t* a, const int32_t* b, int n) { long m=0; for (int j=0;j<n;j++){ long d=labs((long)a[j]-b[j]); if(d>m)m=d; } return m; }

int main(int argc, char** argv) {
    if (fpga_mgmt_init() != 0) { fprintf(stderr,"fpga_mgmt_init failed\n"); return 2; }
    srand(424242);
    // two square layers' cached forward state + the top error (random in the deployed fixed-point range;
    // the backward is a function of these cached values, so random exercises it fully)
    int32_t *h1 = calloc(BUFLEN,4), *h0 = calloc(BUFLEN,4);
    int32_t *W0 = malloc(4*PC_D*PC_D), *W1 = malloc(4*PC_D*PC_D);
    int32_t *acts0 = malloc(4*PC_N*PC_D), *acts1 = malloc(4*PC_N*PC_D);
    int32_t *m0 = malloc(4*PC_N*PC_D), *m1 = malloc(4*PC_N*PC_D), *s0 = malloc(4*PC_N*PC_D), *s1 = malloc(4*PC_N*PC_D);
    int32_t *etop = malloc(4*PC_N*PC_D);
    for (int j=0;j<PC_D*PC_D;j++){ W0[j]=(rand()%1201)-600; W1[j]=(rand()%1201)-600; }
    for (int j=0;j<PC_N*PC_D;j++){
        acts0[j]=rand()%601; acts1[j]=rand()%601;
        m0[j]=(rand()%100)<60; m1[j]=(rand()%100)<60; s0[j]=(rand()%100)<70; s1[j]=(rand()%100)<70;
        etop[j]=(rand()%1201)-600;
    }
    // pack a layer's inputs into a buffer
    #define PACK(h,E,W,A,M,S) do{ memcpy(h+O_E,E,4*PC_N*PC_D); memcpy(h+O_W,W,4*PC_D*PC_D); \
        memcpy(h+O_ACT,A,4*PC_N*PC_D); memcpy(h+O_MASK,M,4*PC_N*PC_D); memcpy(h+O_SKIP,S,4*PC_N*PC_D); }while(0)

    // ---- reference + FPGA, LAYER 1 (top) ----
    int32_t* r1 = calloc(BUFLEN,4); int32_t* r0 = calloc(BUFLEN,4);
    PACK(r1, etop, W1, acts1, m1, s1); ref_layer(r1);                 // reference layer 1
    PACK(h1, etop, W1, acts1, m1, s1);
    if (run_layer_on_fpga(1, h1)) return 3;                           // slot 1 = layer 1 (top)
    printf("[shard] slot 1 produced gW1, gB1, e_down1 LOCALLY (never saw layer 0's weights/grad).\n");

    // ---- NEIGHBOUR MESSAGE: e_down1 (slot 1) -> layer 0 input. The ONLY cross-FPGA data. ----
    // Propagated PC errors grow each layer (e_down = sum of D terms), so the inter-layer message must be
    // RESCALED to the int16 fixed-point range (a single host-side arithmetic shift -- standard fixed-point
    // cascade dynamics, NOT a global op). Both the reference and the FPGA use the SAME rescaled message.
    long maxe = 1; for (int j=0;j<PC_N*PC_D;j++){ long a=labs((long)h1[O_EDN+j]); if(a>maxe)maxe=a; }
    // target |e0| < ~8000: fits int16 input AND keeps the layer-0 output e_down0 (sum of D=256 terms,
    // each up to |e0|*q(W)~|e0|*1024) inside int32.
    int kshift = 0; while ((maxe >> kshift) >= 8000) kshift++;
    int32_t* e0 = malloc(4*PC_N*PC_D);
    for (int j=0;j<PC_N*PC_D;j++) e0[j] = (int32_t)((long)h1[O_EDN+j] >> kshift);
    printf("[shard] neighbour message e_down1 (max|.|=%ld) rescaled >>%d to int16 fixed-point.\n", maxe, kshift);

    // ---- reference + FPGA, LAYER 0 (from the rescaled neighbour message) ----
    PACK(r0, e0, W0, acts0, m0, s0); ref_layer(r0);                   // reference layer 0
    PACK(h0, e0, W0, acts0, m0, s0);
    if (run_layer_on_fpga(0, h0)) return 3;                           // slot 0 = layer 0
    printf("[shard] slot 0 produced gW0, gB0, e_down0 LOCALLY from the e_down neighbour message only.\n");

    // ---- verify the sharded-across-2-FPGAs result == the single-process reference ----
    long d_gw1 = maxdiff(h1+O_GW, r1+O_GW, PC_D*PC_D), d_ed1 = maxdiff(h1+O_EDN, r1+O_EDN, PC_N*PC_D), d_gb1 = maxdiff(h1+O_GB, r1+O_GB, PC_D);
    long d_gw0 = maxdiff(h0+O_GW, r0+O_GW, PC_D*PC_D), d_ed0 = maxdiff(h0+O_EDN, r0+O_EDN, PC_N*PC_D), d_gb0 = maxdiff(h0+O_GB, r0+O_GB, PC_D);
    printf("\n[shard] layer 1 (slot 1): max|gW|=%ld max|e_down|=%ld max|gB|=%ld\n", d_gw1, d_ed1, d_gb1);
    printf("[shard] layer 0 (slot 0): max|gW|=%ld max|e_down|=%ld max|gB|=%ld\n", d_gw0, d_ed0, d_gb0);
    int ok = (d_gw1|d_ed1|d_gb1|d_gw0|d_ed0|d_gb0) == 0;
    printf("[shard] -> %s\n", ok ?
        "BIT-EXACT: the 2-layer PC backward SHARDED ACROSS 2 FPGAs (neighbour-message-only, no global "
        "gradient/transport/barrier) == cascade_grads" : "MISMATCH");
    return ok ? 0 : 1;
}
