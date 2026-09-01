// forge_pc_ddr_verify.c — drive the multiplier-free PC backward kernel (forge_pc_wgrad) on F2 reading/
// writing ON-CARD DDR4, and verify the gradient computed ON SILICON is bit-exact to cascade_grads.
//
// The drivable on-card-DDR path (no host IOMMU; same as muzero_ddr_power.c): the host STAGES the I/O
// buffer into DDR over the shell DMA-PCIS via BAR4 (host -> DMA-PCIS -> the CL 2->1 crossbar -> sh_ddr ->
// DDR4), kicks the kernel (BUFFER ptr reg = 0), the kernel reads inputs + writes gW/e_down/gB BACK to DDR
// via its m_axi, then the host READS THEM BACK over BAR4 and compares to an int64 cascade_grads reference
// computed on the host. UNLIKE the muzero power driver (write+kick only), this reads outputs back and
// checks them — the point is correctness on silicon, not power.
//
// Build on an F2 with the aws-fpga SDK:
//   gcc -O2 -I$SDK/include forge_pc_ddr_verify.c -lfpga_mgmt -o forge_pc_ddr_verify
//   sudo ./forge_pc_ddr_verify <slot>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <fpga_pci.h>
#include <fpga_mgmt.h>

// must match forge_pc_wgrad.cpp (square interior layer)
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

// forge_pc_wgrad s_axi_control map: single `buffer` pointer arg + ap_ctrl -> the standard Vitis HLS layout
// (5-bit address space). Confirm against the build's forge_pc_wgrad_control_s_axi.v before a real run.
#define R_AP_CTRL  0x00
#define R_BUFFER_0 0x10
#define R_BUFFER_1 0x14
#define AP_START   0x01
#define AP_DONE    0x02
#define DDR_BUF_ADDR 0x0ULL

static long ref_dyad(long v) {                          // == kernel dyad: table-free 1.5x round (no ROM)
    long a = v < 0 ? -v : v; if (a == 0) return 0;
    int msb = 0; for (int b = 15; b >= 1; b--) if ((a >> b) & 1) { msb = b; break; }
    if (msb >= 1 && ((a >> (msb - 1)) & 1)) msb++;       // bump on bit below MSB
    long m = 1L << msb; return v < 0 ? -m : m;
}
static double now_s(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char** argv) {
    int slot = (argc > 1) ? atoi(argv[1]) : 0;
    int kat  = (argc > 2 && strcmp(argv[2], "kat") == 0);   // known-answer test: all inputs = 1
    const size_t bytes = (size_t)BUFLEN * 4;
    int32_t* h = (int32_t*)calloc(BUFLEN, 4);             // host buffer (inputs in, outputs read back)

    // --- build inputs. KAT (all 1s): answer is trivially gW=gB=N, e_down=D everywhere, and lane order
    // is IRRELEVANT (every value is 1) -> isolates a value/lane READ issue from a structural compute bug.
    // Otherwise: deployed fixed-point range (same recipe as the CSIM tb). ---
    srand(20260621);
    for (int i = 0; i < PC_N*PC_D; i++) {
        h[O_E + i]    = kat ? 1 : (rand() % 1201) - 600;
        h[O_ACT + i]  = kat ? 1 : rand() % 601;
        h[O_MASK + i] = kat ? 1 : ((rand() % 100) < 60 ? 1 : 0);
        h[O_SKIP + i] = kat ? 1 : ((rand() % 100) < 70 ? 1 : 0);
    }
    for (int i = 0; i < PC_D*PC_D; i++) h[O_W + i] = kat ? 1 : (rand() % 1201) - 600;
    if (kat) printf("[pc] KNOWN-ANSWER TEST: all inputs=1 -> expect gW=gB=%d and e_down=%d everywhere\n", PC_N, PC_D);

    // --- host reference (int64 cascade_grads layer-step) BEFORE staging (outputs zeroed in h) ---
    static long delta[PC_N][PC_D];
    for (int n = 0; n < PC_N; n++) for (int o = 0; o < PC_D; o++)
        delta[n][o] = (h[O_MASK+n*PC_D+o] && h[O_SKIP+n*PC_D+o]) ? h[O_E+n*PC_D+o] : 0;
    long* ref_gw = (long*)malloc(sizeof(long)*PC_D*PC_D);
    long* ref_ed = (long*)malloc(sizeof(long)*PC_N*PC_D);
    long* ref_gb = (long*)malloc(sizeof(long)*PC_D);
    for (int o = 0; o < PC_D; o++) {
        long gb = 0;
        for (int i = 0; i < PC_D; i++) {
            long acc = 0;
            for (int n = 0; n < PC_N; n++) acc += ref_dyad(delta[n][o]) * ref_dyad(h[O_ACT+n*PC_D+i]);
            ref_gw[o*PC_D+i] = acc;
        }
        for (int n = 0; n < PC_N; n++) gb += ref_dyad(delta[n][o]);
        ref_gb[o] = gb;
    }
    for (int n = 0; n < PC_N; n++) for (int i = 0; i < PC_D; i++) {
        long acc = 0;
        for (int o = 0; o < PC_D; o++) acc += delta[n][o] * ref_dyad(h[O_W+o*PC_D+i]);
        if (!h[O_SKIP+n*PC_D+i]) acc += h[O_E+n*PC_D+i];
        ref_ed[n*PC_D+i] = acc;
    }

    // --- attach, stage inputs to DDR (BAR4), kick, read outputs back, compare ---
    if (fpga_mgmt_init() != 0) { fprintf(stderr,"fpga_mgmt_init failed\n"); return 2; }
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT, bar4 = PCI_BAR_HANDLE_INIT, bar4r = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0) { fprintf(stderr,"attach BAR0 failed\n"); return 3; }
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &bar4) != 0) { fprintf(stderr,"attach BAR4 (burst) failed\n"); return 4; }
    // separate non-burst BAR4 handle for single-word peek READS (burst handle is write-combined)
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR4, 0, &bar4r) != 0) { fprintf(stderr,"attach BAR4 (read) failed\n"); return 4; }

    const long CHW = 256*1024;
    for (long off = 0; off < BUFLEN; off += CHW) {
        long n = (BUFLEN - off < CHW) ? (BUFLEN - off) : CHW;
        if (fpga_pci_write_burst(bar4, DDR_BUF_ADDR + (uint64_t)off*4, (uint32_t*)(h+off), (uint64_t)n) != 0) {
            fprintf(stderr,"write_burst @%ld failed\n", off); return 5; }
    }
    // staging sanity: peek back EVERY input word and count mismatches per region (isolates a DDR
    // staging bug from a kernel bug). Covers e/W/acts/relu_mask/skip_gate — esp. mask/skip whose
    // mis-staging would break the gating exactly like the observed "too-large" outputs.
    { uint32_t rv; long me=0,mw=0,ma=0,mk=0,ms=0; long s_mk=0,s_sk=0;
      for (int j=0;j<PC_N*PC_D;j++){ fpga_pci_peek(bar4r,DDR_BUF_ADDR+(uint64_t)(O_E+j)*4,&rv);    if((int32_t)rv!=h[O_E+j]) me++; }
      for (int j=0;j<PC_D*PC_D;j++){ fpga_pci_peek(bar4r,DDR_BUF_ADDR+(uint64_t)(O_W+j)*4,&rv);    if((int32_t)rv!=h[O_W+j]) mw++; }
      for (int j=0;j<PC_N*PC_D;j++){ fpga_pci_peek(bar4r,DDR_BUF_ADDR+(uint64_t)(O_ACT+j)*4,&rv);  if((int32_t)rv!=h[O_ACT+j]) ma++; }
      for (int j=0;j<PC_N*PC_D;j++){ fpga_pci_peek(bar4r,DDR_BUF_ADDR+(uint64_t)(O_MASK+j)*4,&rv); if((int32_t)rv!=h[O_MASK+j]) mk++; s_mk+=h[O_MASK+j]; }
      for (int j=0;j<PC_N*PC_D;j++){ fpga_pci_peek(bar4r,DDR_BUF_ADDR+(uint64_t)(O_SKIP+j)*4,&rv); if((int32_t)rv!=h[O_SKIP+j]) ms++; s_sk+=h[O_SKIP+j]; }
      printf("[pc] staging mismatches: e %ld/%d  W %ld/%d  acts %ld/%d  mask %ld/%d  skip %ld/%d\n",
             me,PC_N*PC_D, mw,PC_D*PC_D, ma,PC_N*PC_D, mk,PC_N*PC_D, ms,PC_N*PC_D);
      printf("[pc] gate density (host): mask=1 in %ld/%d, skip=1 in %ld/%d\n", s_mk,PC_N*PC_D, s_sk,PC_N*PC_D); }

    fpga_pci_poke(bar0, R_BUFFER_0, (uint32_t)(DDR_BUF_ADDR & 0xFFFFFFFFULL));
    fpga_pci_poke(bar0, R_BUFFER_1, (uint32_t)(DDR_BUF_ADDR >> 32));
    fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
    double t0 = now_s(); int done = 0; uint32_t ctrl = 0;
    while (now_s() - t0 < 10.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { done = 1; break; } }
    if (!done) { fprintf(stderr,"[pc] KICK HUNG (no AP_DONE in 10s)\n"); return 6; }
    printf("[pc] AP_DONE in %.3f ms -> kernel ran from DDR\n", (now_s()-t0)*1e3);

    // read gW, e_down, gB back from DDR via BAR4 single-word peeks (the SDK has no read_burst;
    // a peek at BAR4 offset X -> DMA-PCIS read X -> crossbar -> DDR[X]). ~70k peeks, a few ms.
    int32_t* gw = (int32_t*)malloc(4*PC_D*PC_D); int32_t* ed = (int32_t*)malloc(4*PC_N*PC_D); int32_t* gb = (int32_t*)malloc(4*PC_D);
    uint32_t v;
    for (int j = 0; j < PC_D*PC_D; j++) { fpga_pci_peek(bar4r, DDR_BUF_ADDR + (uint64_t)(O_GW + j)*4,  &v); gw[j] = (int32_t)v; }
    for (int j = 0; j < PC_N*PC_D; j++) { fpga_pci_peek(bar4r, DDR_BUF_ADDR + (uint64_t)(O_EDN + j)*4, &v); ed[j] = (int32_t)v; }
    for (int j = 0; j < PC_D; j++)      { fpga_pci_peek(bar4r, DDR_BUF_ADDR + (uint64_t)(O_GB + j)*4,  &v); gb[j] = (int32_t)v; }

    // diagnostics: are outputs zero (kernel writes not landing) or non-zero-but-wrong (compute/layout)?
    long nz_gw = 0, nz_ed = 0, nz_gb = 0;
    for (int j = 0; j < PC_D*PC_D; j++) if (gw[j] != 0) nz_gw++;
    for (int j = 0; j < PC_N*PC_D; j++) if (ed[j] != 0) nz_ed++;
    for (int j = 0; j < PC_D; j++) if (gb[j] != 0) nz_gb++;
    printf("[pc] readback nonzero: gw %ld/%d  e_down %ld/%d  gb %ld/%d\n",
           nz_gw, PC_D*PC_D, nz_ed, PC_N*PC_D, nz_gb, PC_D);
    printf("[pc] samples: gw[0]=%d(ref %ld) gw[100]=%d(ref %ld) ed[0]=%d(ref %ld) gb[0]=%d(ref %ld)\n",
           gw[0], ref_gw[0], gw[100], ref_gw[100], ed[0], ref_ed[0], gb[0], ref_gb[0]);

    long mgw = 0, med = 0, mgb = 0;
    for (int j = 0; j < PC_D*PC_D; j++) { long d = labs((long)gw[j] - ref_gw[j]); if (d > mgw) mgw = d; }
    for (int j = 0; j < PC_N*PC_D; j++) { long d = labs((long)ed[j] - ref_ed[j]); if (d > med) med = d; }
    for (int j = 0; j < PC_D; j++)      { long d = labs((long)gb[j] - ref_gb[j]); if (d > mgb) mgb = d; }
    printf("[pc] ON-SILICON vs cascade_grads:  max|gW|=%ld  max|e_down|=%ld  max|gB|=%ld\n", mgw, med, mgb);
    int ok = (mgw == 0 && med == 0 && mgb == 0);
    printf("[pc] -> %s\n", ok ? "BIT-EXACT ON SILICON (the PC gradient computed on F2 == cascade_grads)" : "MISMATCH");
    fpga_pci_detach(bar0); fpga_pci_detach(bar4); fpga_pci_detach(bar4r);
    return ok ? 0 : 1;
}
