// muzero_pcim_power.c — drive the forward MuZero CHESS kernel on real F2 silicon over the
// PCIM-to-host-DRAM path, in a tight kick loop, so on-board power can be sampled DURING inference
// (the dynamic / per-inference Vccint the energy axis wants). Unlike the morph-LM bridge (buffer in
// BAR4/on-CL-URAM), our CL masters HOST DRAM via PCIM, so the host must:
//   1. allocate a physically-CONTIGUOUS host buffer (a 1 GB hugepage -> contiguous in guest-physical),
//   2. resolve its GUEST-PHYSICAL address (/proc/self/pagemap),
//   3. program that address into the kernel's s_axilite buffer-pointer register (BUFFER_0/1),
//   4. kick AP_START / poll AP_DONE (the proven bridge.cpp control flow; offsets confirmed identical).
// The kernel then issues PCIM reads/writes to that host address. Whether the Nitro IOMMU accepts a raw
// guest-physical PCIM master access is THE open question — read pcim-read-count / pcim-range-error from
// fpga-describe-local-image -M after a run: read-count>0 + 0 range errors => it reached host DRAM.
//
// Build on an F2 with the aws-fpga SDK:
//   g++ -O2 -std=c++17 -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096 \
//       -I$SDK/include -I<hls>/forge_kernels muzero_pcim_power.c -lfpga_mgmt -o muzero_pcim_power
//   sudo ./muzero_pcim_power <slot> <n_kicks>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <time.h>
#include <inttypes.h>
extern "C" {
#include <fpga_pci.h>
#include <fpga_mgmt.h>
}
#include "forge_buffer_muzero.hpp"   // MZ_BUFFER_WORDS, OFF_* (chess via -DMZ_*_OVERRIDE)

// s_axilite control register offsets (confirmed from forge_train_step_muzero_control_s_axi.v) ----
#define R_AP_CTRL   0x00
#define R_AP_RETURN 0x10
#define R_BUFFER_0  0x18
#define R_BUFFER_1  0x1c
#define R_MODE      0x24
#define R_ARG1      0x2c
#define AP_START    0x01
#define AP_DONE     0x02

#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)   // 2^30
#endif

static uint64_t virt_to_phys(void* p) {
    uint64_t v = (uint64_t)p;
    int fd = open("/proc/self/pagemap", O_RDONLY);
    if (fd < 0) return 0;
    uint64_t entry = 0;
    if (pread(fd, &entry, 8, (v / 4096) * 8) != 8) { close(fd); return 0; }
    close(fd);
    if (!(entry & (1ULL << 63))) return 0;                 // page not present
    return (entry & ((1ULL << 55) - 1)) * 4096 + (v % 4096);
}

static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char** argv) {
    int slot = (argc > 1) ? atoi(argv[1]) : 0;
    long n_kicks = (argc > 2) ? atol(argv[2]) : 2000;
    const size_t need = (size_t)MZ_BUFFER_WORDS * 4;
    printf("[pcim] MZ_BUFFER_WORDS=%d (%.2f MB); n_kicks=%ld; mode=0 (forward)\n",
           MZ_BUFFER_WORDS, need / 1048576.0, n_kicks);

    if (fpga_mgmt_init() != 0) { fprintf(stderr, "fpga_mgmt_init failed\n"); return 2; }
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT;
    if (fpga_pci_attach(slot, FPGA_APP_PF, APP_PF_BAR0, 0, &bar0) != 0) {
        fprintf(stderr, "fpga_pci_attach BAR0 failed\n"); return 3;
    }

    // 1 GB hugepage -> guest-physically contiguous, enough for the ~6 MB buffer.
    const size_t HP = 1024UL * 1024 * 1024;
    void* buf = mmap(NULL, HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
    if (buf == MAP_FAILED) {
        fprintf(stderr, "1GB hugepage mmap failed (reserve nr_hugepages first): %s\n", strerror(errno));
        return 4;
    }
    memset(buf, 0, need);                 // weights=0: do_fwd still streams + shift-adds -> real toggling
    if (mlock(buf, need) != 0) fprintf(stderr, "warn: mlock failed: %s\n", strerror(errno));
    ((volatile char*)buf)[0] = 0;         // fault the first page in before pagemap
    uint64_t phys = virt_to_phys(buf);
    printf("[pcim] hugepage vaddr=%p guest-phys=0x%" PRIx64 "\n", buf, phys);
    if (!phys) { fprintf(stderr, "pagemap returned 0 (need sudo + present page)\n"); return 5; }

    // program the kernel's m_axi buffer pointer (PCIM target) + mode
    fpga_pci_poke(bar0, R_BUFFER_0, (uint32_t)(phys & 0xFFFFFFFFULL));
    fpga_pci_poke(bar0, R_BUFFER_1, (uint32_t)(phys >> 32));
    fpga_pci_poke(bar0, R_MODE, 0);
    fpga_pci_poke(bar0, R_ARG1, 0);

    // one warm kick with a generous timeout + return-code check (catches a hang/IOMMU stall)
    fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
    double t0 = now_s(); int done = 0; uint32_t ctrl = 0;
    while (now_s() - t0 < 5.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { done = 1; break; } }
    if (!done) { fprintf(stderr, "[pcim] WARM KICK HUNG (no AP_DONE in 5s) -> kernel stalled on PCIM (IOMMU?)\n"); return 6; }
    uint32_t ret = 0; fpga_pci_peek(bar0, R_AP_RETURN, &ret);
    printf("[pcim] warm kick: AP_DONE in %.3f ms, ret=%u\n", (now_s() - t0) * 1e3, ret);

    // timed kick loop — the window during which the runner samples Vccint
    double s = now_s();
    long ok = 0;
    for (long k = 0; k < n_kicks; ++k) {
        fpga_pci_poke(bar0, R_AP_CTRL, AP_START);
        double k0 = now_s(); int d = 0;
        while (now_s() - k0 < 2.0) { fpga_pci_peek(bar0, R_AP_CTRL, &ctrl); if (ctrl & AP_DONE) { d = 1; break; } }
        if (d) ok++;
    }
    double el = now_s() - s;
    printf("[pcim] LOOP: %ld/%ld kicks done in %.3f s -> %.3f ms/kick, %.1f inf/s\n",
           ok, n_kicks, el, el / ok * 1e3, ok / el);
    fpga_pci_detach(bar0);
    return 0;
}
