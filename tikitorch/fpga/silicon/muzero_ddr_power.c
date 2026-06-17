// muzero_ddr_power.c — drive the forward MuZero chess kernel on F2 reading ON-CARD DDR4, in a kick loop, so
// on-board Vccint can be sampled DURING inference (the dynamic / per-inference power). This is the DRIVABLE
// path the PCIM variant couldn't reach (no host IOMMU needed): the host STAGES the ~12 MB weight buffer into
// DDR over the shell DMA-PCIS via fpga_dma (host -> DMA-PCIS -> the CL's 2->1 crossbar -> sh_ddr -> DDR4),
// then the kernel READS it via m_axi (-> the same crossbar -> DDR). Buffer lives at DDR address 0; the
// kernel's BUFFER pointer register is set to 0.
//
// Build on an F2 with the aws-fpga SDK:
//   g++ -O2 -std=c++17 -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096 \
//       -I$SDK/include -I<hls>/forge_kernels -I<hls> muzero_ddr_power.c -lfpga_mgmt -o muzero_ddr_power
//   sudo ./muzero_ddr_power <slot> <n_kicks>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include <inttypes.h>
extern "C" {
#include <fpga_pci.h>
#include <fpga_mgmt.h>
#include <utils/lcd.h>
#include <fpga_dma.h>
}
#include "forge_buffer_muzero.hpp"   // MZ_BUFFER_WORDS (chess via -DMZ_*_OVERRIDE)

// s_axilite control register offsets (confirmed from control_s_axi.v; identical to bridge.cpp) ----
#define R_AP_CTRL   0x00
#define R_AP_RETURN 0x10
#define R_BUFFER_0  0x18
#define R_BUFFER_1  0x1c
#define R_MODE      0x24
#define R_ARG1      0x2c
#define AP_START    0x01
#define AP_DONE     0x02

#define DDR_BUF_ADDR 0x0ULL          // buffer staged at DDR address 0 (crossbar M00 base = 0)

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    int slot = (argc > 1) ? atoi(argv[1]) : 0;
    long n_kicks = (argc > 2) ? atol(argv[2]) : 4000;
    const size_t bytes = (size_t)MZ_BUFFER_WORDS * 4;
    printf("[ddr] MZ_BUFFER_WORDS=%d (%.2f MB); n_kicks=%ld; mode=0 (forward); DDR addr=0x%llx\n",
           MZ_BUFFER_WORDS, bytes / 1048576.0, n_kicks, (unsigned long long)DDR_BUF_ADDR);

    if (fpga_mgmt_init() != 0) { fprintf(stderr, "fpga_mgmt_init failed\n"); return 2; }
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0) { fprintf(stderr, "attach BAR0 failed\n"); return 3; }

    // --- stage the buffer into DDR over DMA-PCIS (XDMA) ---
    int wfd = fpga_dma_open_queue(FPGA_DMA_XDMA, slot, /*channel*/0, /*is_read*/false);
    if (wfd < 0) { fprintf(stderr, "fpga_dma_open_queue(write) failed: %d\n", wfd); return 4; }
    uint8_t* hbuf = (uint8_t*)calloc(1, bytes);    // weights=0: do_fwd still streams + shift-adds (real toggling)
    const size_t CH = 4UL * 1024 * 1024;           // 4 MB chunks
    double sdma = now_s();
    for (size_t off = 0; off < bytes; off += CH) {
        size_t n = (bytes - off < CH) ? (bytes - off) : CH;
        int rc = fpga_dma_burst_write(wfd, hbuf + off, n, (size_t)(DDR_BUF_ADDR + off));
        if (rc != 0) { fprintf(stderr, "fpga_dma_burst_write @0x%zx rc=%d\n", off, rc); return 5; }
    }
    printf("[ddr] staged %.2f MB into DDR in %.3f s\n", bytes / 1048576.0, now_s() - sdma);

    // --- program the kernel's m_axi buffer pointer (= DDR address) + mode ---
    fpga_pci_poke(bar0, R_BUFFER_0, (uint32_t)(DDR_BUF_ADDR & 0xFFFFFFFFULL));
    fpga_pci_poke(bar0, R_BUFFER_1, (uint32_t)(DDR_BUF_ADDR >> 32));
    fpga_pci_poke(bar0, R_MODE, 0);
    fpga_pci_poke(bar0, R_ARG1, 0);

    // --- warm kick: does the kernel complete reading DDR? (the drivability test) ---
    fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
    double t0 = now_s(); int done = 0; uint32_t ctrl = 0;
    while (now_s() - t0 < 10.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { done = 1; break; } }
    if (!done) { fprintf(stderr, "[ddr] WARM KICK HUNG (no AP_DONE in 10s)\n"); return 6; }
    uint32_t ret = 0; fpga_pci_peek(bar0, R_AP_RETURN, &ret);
    printf("[ddr] warm kick: AP_DONE in %.3f ms, ret=%u  -> KERNEL RUNS FROM DDR\n", (now_s() - t0) * 1e3, ret);

    // --- timed kick loop (the window during which the runner samples Vccint) ---
    double s = now_s(); long ok = 0;
    for (long k = 0; k < n_kicks; ++k) {
        fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
        double k0 = now_s(); int d = 0;
        while (now_s() - k0 < 2.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { d = 1; break; } }
        if (d) ok++;
    }
    double el = now_s() - s;
    printf("[ddr] LOOP: %ld/%ld kicks in %.3f s -> %.3f ms/kick, %.1f inf/s\n", ok, n_kicks, el, el / ok * 1e3, ok / el);
    free(hbuf); fpga_pci_detach(bar0);
    return 0;
}
