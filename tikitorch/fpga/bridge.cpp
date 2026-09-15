// tikitorch/fpga/bridge.cpp — the reference native bridge for `_libtikifpga`.
//
// Implements the extern-C symbols that tikitorch/fpga/_native.py binds, in three
// transports selected at compile time:
//
//   LOCAL (default): runs the bundled kernel math ON THE HOST (no board) so
//     the whole tikitorch.fpga chain (.fpga() -> bound -> train_step -> runner ->
//     tt_fpga_run -> the silicon-validated kernel) is validated bit-exactly on a
//     laptop. Build with -DCSIM_ONLY -DTT_KERNEL_1LAYER + the kernel .cpp.
//
//   FPGA-PCI (-DTT_FPGA_PCI): drives a REAL AWS F1/F2 board via the AWS HDK
//     `fpga_pci`/`fpga_mgmt` OCL-register flow -- the mechanism proven by
//     scripts/train_shakespeare_on_silicon_fast.c (BAR0 control regs, the int32
//     buffer resident in BAR4 at 0x01000000, busy-poll AP_DONE). Build on an F2
//     instance with the aws-fpga SDK; link -lfpga_mgmt. No kernel .cpp needed
//     (the kernel is the loaded AFI).
//
//   XRT (-DTT_FPGA_XRT): Vitis/XRT transport (kept for xclbin-based flows).
//
// The buffer is addressed by int32 WORD offset in tt_fpga_{write,read}_region /
// tt_fpga_run's implicit buffer, matching forge_buffer.hpp exactly. The on-board
// runner uses write_region/read_region (not a raw host pointer) so ONE runner
// drives LOCAL and silicon identically.

#include <cstdint>
#include <cstdio>   // fprintf (clkgen bring-up diagnostics)
#include <cstdlib>
#include <cstring>
#include <ctime>    // nanosleep (clkgen MMCM lock poll)
#include <new>  // std::nothrow
#include <limits>

#if defined(TT_FPGA_PCI)
#include <map>
extern "C" {
#include <fpga_pci.h>
#include <fpga_mgmt.h>
}
#elif defined(TT_FPGA_XRT)
#include <map>
#include <string>
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_uuid.h"
#endif

// --- bundled kernel prototypes (LOCAL only; only the selected ones are linked) --
#if !defined(TT_FPGA_PCI) && !defined(TT_FPGA_XRT)
extern "C" {
#ifdef TT_KERNEL_1LAYER
int forge_train_step(int32_t* buffer, int mode, int a1, int a2, int a3);
#endif
#ifdef TT_KERNEL_2LAYER
int forge_train_step_2layer(int32_t* buffer, int mode, int a1, int a2, int a3, int layer_id);
#endif
#ifdef TT_KERNEL_3LAYER
int forge_train_step_3layer(int32_t* buffer, int mode, int a1, int a2, int a3, int layer_id);
#endif
#ifdef TT_KERNEL_MORPH
int forge_train_step_morph(int32_t* buffer, int mode, int a1, int a2, int a3);
#endif
#ifdef TT_KERNEL_MORPH_ECG
int forge_train_step_morph_ecg(int32_t* buffer, int mode, int a1, int a2, int a3);
#endif
#ifdef TT_KERNEL_MORPH_LM
int forge_train_step_morph_lm(int32_t* buffer, int mode, int a1, int a2, int a3);
#endif
}
#endif

// =============================================================================
// FPGA-PCI register map (matches scripts/train_shakespeare_on_silicon_fast.c and
// the silicon-validated v5/v6 monolithic-train_step AGFIs).
// =============================================================================
#ifdef TT_FPGA_PCI
#define TT_ADDR_AP_CTRL   0x00
#define TT_ADDR_AP_RETURN 0x10
#define TT_ADDR_BUFFER_0  0x18
#define TT_ADDR_BUFFER_1  0x1c
#define TT_ADDR_MODE      0x24
#define TT_ADDR_ARG1      0x2c
#define TT_ADDR_ARG2      0x34
#define TT_ADDR_ARG3      0x3c
#define TT_AP_START       0x01
#define TT_AP_DONE        0x02
#define TT_BUFFER_ADDR    0x01000000ULL
// --- AWS_CLK_GEN bring-up (env-gated TT_CLKGEN_BRINGUP=1; OFF by default) ----
// Clkgen-enabled AFIs (the d512 v3 CL and any --aws_clk_gen build) boot with
// SYS_RST_REG=0xFFFF_FFFE: every CL reset asserted and clk_extra_a1 static-low
// until MMCM_A locks — the ENTIRE 125 MHz user design (CDCs, crossbars, banks,
// kernel) is dead and the first poke/DMA hangs forever. The host must poll
// MMCM_LOCK_REG then clear SYS_RST_REG BEFORE any staging/kick (the S1 lesson;
// see forge_train_step_s1_f2.c clkgen_bringup + AWS_CLK_GEN_spec Boot/Reset).
// ENV-GATED because the deployed non-clkgen AFIs (d128 agfi-0433/N4) have no
// AWS_CLK_GEN — poking clkgen registers there would be undefined. Set
// TT_CLKGEN_BRINGUP=1 in the runner for clkgen AFIs only (the d512 flow).
// Registers (explicit-register flavor, no -lfpga_clkgen link dependency):
#define TT_CLKGEN_BASE_REG    0x00058000UL  // AWS_CLK_REGS base in the mgmt (SDA) space
#define TT_CLKGEN_SYS_RST     0x14U         // SYS_RST_REG; boot=0xFFFF_FFFE
#define TT_CLKGEN_MMCM_LOCK   0x20U         // MMCM_LOCK_REG; bit0 = MMCM_A locked
#define TT_CLKGEN_A_LOCKED    0x00000001U   // group-A-only mask (CLK_GRP_A_EN only)
#endif

namespace {
struct TtCtx {
    int present = 0;
    int last_error = 0;
#if defined(TT_FPGA_PCI)
    int slot = 0;
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT;   // OCL control regs
    pci_bar_handle_t bar4 = PCI_BAR_HANDLE_INIT;   // int32 buffer (DDR)
    int buffer_set = 0;
    std::map<void*, size_t> allocations;
#elif defined(TT_FPGA_XRT)
    xrt::device device;
    xrt::uuid uuid;
    std::map<std::string, xrt::kernel> kernels;
    std::map<void*, xrt::bo> bos;
    xrt::kernel& kernel_for(const char* entry) {
        std::string e(entry);
        auto it = kernels.find(e);
        if (it == kernels.end())
            it = kernels.emplace(e, xrt::kernel(device, uuid, e)).first;
        return it->second;
    }
#endif
};

// Preserve the void transfer ABI while making PCI failures observable. Errors
// are sticky: a partially completed device write requires a fresh context and
// reinitialization, not clearing a status bit and using uncertain device state.
constexpr int TT_INVALID_ARGUMENT = -201;
constexpr int TT_INVALID_REGION = -202;
constexpr int TT_ALLOCATION_FAILED = -203;

int latch_error(TtCtx* ctx, int error) {
    if (ctx && error && !ctx->last_error) ctx->last_error = error;
    return ctx ? ctx->last_error : TT_INVALID_ARGUMENT;
}

#if defined(TT_FPGA_PCI)
bool pci_write(TtCtx* ctx, pci_bar_handle_t bar, uint64_t address, uint32_t value) {
    return ctx && !ctx->last_error &&
           latch_error(ctx, fpga_pci_poke(bar, address, value)) == 0;
}

bool pci_read(TtCtx* ctx, pci_bar_handle_t bar, uint64_t address, uint32_t* value) {
    return ctx && !ctx->last_error &&
           latch_error(ctx, fpga_pci_peek(bar, address, value)) == 0;
}

int pci_detach(TtCtx* ctx, pci_bar_handle_t bar) {
    // Cleanup must still run after an earlier error. close() has a void ABI and
    // destroys the context, so also report detach failures to stderr.
    const int error = fpga_pci_detach(bar);
    latch_error(ctx, error);
    if (error) std::fprintf(stderr, "[tt] PCI detach failed: %d\n", error);
    return error;
}

bool valid_region(TtCtx* ctx, void* buffer, long offset, long count,
                  const void* data = nullptr) {
    if (!ctx || ctx->last_error) return false;
    if (!ctx->present || offset < 0 || count < 0 || (count && !data)) {
        latch_error(ctx, TT_INVALID_ARGUMENT);
        return false;
    }
    const auto allocation = ctx->allocations.find(buffer);
    if (allocation == ctx->allocations.end() ||
        static_cast<size_t>(offset) > allocation->second ||
        static_cast<size_t>(count) > allocation->second - static_cast<size_t>(offset)) {
        latch_error(ctx, TT_INVALID_REGION);
        return false;
    }
    return true;
}

bool pci_kick(TtCtx* ctx, int mode, int a1, int a2, int a3) {
    return pci_write(ctx, ctx->bar0, TT_ADDR_MODE, static_cast<uint32_t>(mode)) &&
           pci_write(ctx, ctx->bar0, TT_ADDR_ARG1, static_cast<uint32_t>(a1)) &&
           pci_write(ctx, ctx->bar0, TT_ADDR_ARG2, static_cast<uint32_t>(a2)) &&
           pci_write(ctx, ctx->bar0, TT_ADDR_ARG3, static_cast<uint32_t>(a3)) &&
           pci_write(ctx, ctx->bar0, TT_ADDR_AP_CTRL, TT_AP_START);
}
#endif
}  // namespace

extern "C" {

void* tt_fpga_open(const char* arg) {
#if defined(TT_FPGA_PCI)
    if (fpga_mgmt_init() != 0) return nullptr;
    TtCtx* ctx = new (std::nothrow) TtCtx();
    if (!ctx) return nullptr;
    ctx->slot = (arg && *arg) ? atoi(arg) : 0;
    // Clkgen AFIs: release the 125 MHz user-design resets BEFORE the first
    // attach/poke (see the TT_CLKGEN_* note above). Env-gated, default OFF.
    {
        const char* cg = std::getenv("TT_CLKGEN_BRINGUP");
        if (cg && cg[0] == '1') {
            pci_bar_handle_t mbar = PCI_BAR_HANDLE_INIT;
            if (fpga_pci_attach(ctx->slot, FPGA_MGMT_PF, MGMT_PF_BAR4, 0, &mbar) != 0) {
                std::fprintf(stderr, "[tt] clkgen: attach mgmt BAR failed\n");
                delete ctx; return nullptr;
            }
            uint32_t lock = 0;
            for (int i = 0; i < 500; ++i) {   // ~5 s @ 10 ms
                if (!pci_read(ctx, mbar, TT_CLKGEN_BASE_REG + TT_CLKGEN_MMCM_LOCK, &lock)) {
                    pci_detach(ctx, mbar); delete ctx; return nullptr;
                }
                if (lock & TT_CLKGEN_A_LOCKED) break;
                struct timespec ts = {0, 10 * 1000 * 1000}; nanosleep(&ts, nullptr);
            }
            if (!(lock & TT_CLKGEN_A_LOCKED)) {
                std::fprintf(stderr, "[tt] clkgen: MMCM_A did NOT lock (MMCM_LOCK_REG=0x%08x) in 5s — "
                                     "clkgen AFI dead or TT_CLKGEN_BRINGUP set on a non-clkgen AFI\n", lock);
                pci_detach(ctx, mbar); delete ctx; return nullptr;
            }
            if (!pci_write(ctx, mbar, TT_CLKGEN_BASE_REG + TT_CLKGEN_SYS_RST, 0x00000000U)) {
                std::fprintf(stderr, "[tt] clkgen: SYS_RST_REG deassert write failed\n");
                pci_detach(ctx, mbar); delete ctx; return nullptr;
            }
            uint32_t rstv = 0xdeadbeefU;
            if (!pci_read(ctx, mbar, TT_CLKGEN_BASE_REG + TT_CLKGEN_SYS_RST, &rstv) ||
                (rstv & 0x3FFu)) {
                std::fprintf(stderr, "[tt] clkgen: reset readback failed or resets remain asserted\n");
                pci_detach(ctx, mbar); delete ctx; return nullptr;
            }
            // Reserved bits [31:10] read back SET per spec; only [9:0] must be 0.
            std::fprintf(stderr, "[tt] clkgen: MMCM_A locked (0x%08x); SYS_RST cleared "
                                 "(readback 0x%08x, writable[9:0]=0x%03x expect 0x000) — user domain live\n",
                         lock, rstv, rstv & 0x3FFu);
            if (pci_detach(ctx, mbar) != 0) { delete ctx; return nullptr; }
        }
    }
    if (fpga_pci_attach(ctx->slot, FPGA_APP_PF, APP_PF_BAR0, 0, &ctx->bar0) != 0) {
        delete ctx; return nullptr;
    }
    if (fpga_pci_attach(ctx->slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &ctx->bar4) != 0) {
        pci_detach(ctx, ctx->bar0); delete ctx; return nullptr;
    }
    // Point the kernel's s_axilite buffer-address registers at BAR4 base (once).
    if (!pci_write(ctx, ctx->bar0, TT_ADDR_BUFFER_0, (uint32_t)(TT_BUFFER_ADDR & 0xFFFFFFFFULL)) ||
        !pci_write(ctx, ctx->bar0, TT_ADDR_BUFFER_1, (uint32_t)((TT_BUFFER_ADDR >> 32) & 0xFFFFFFFFULL))) {
        pci_detach(ctx, ctx->bar0); pci_detach(ctx, ctx->bar4);
        delete ctx; return nullptr;
    }
    ctx->buffer_set = 1;
    ctx->present = 1;
    return ctx;
#elif defined(TT_FPGA_XRT)
    try {
        TtCtx* ctx = new TtCtx();
        const char* path = (arg && *arg) ? arg : std::getenv("TT_FPGA_XCLBIN");
        if (!path) { delete ctx; return nullptr; }
        ctx->device = xrt::device(0);
        ctx->uuid = ctx->device.load_xclbin(path);
        ctx->present = 1;
        return ctx;
    } catch (...) { return nullptr; }
#else
    (void)arg;
    TtCtx* ctx = new (std::nothrow) TtCtx();
    if (ctx) ctx->present = 1;   // LOCAL: the "board" is the host CPU
    return ctx;
#endif
}

void tt_fpga_close(void* h) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (ctx) { pci_detach(ctx, ctx->bar0); pci_detach(ctx, ctx->bar4); }
#endif
    delete static_cast<TtCtx*>(h);
}

int tt_fpga_board_present(void* h) {
    TtCtx* ctx = static_cast<TtCtx*>(h);
    return (ctx != nullptr && ctx->present) ? 1 : 0;
}

int tt_fpga_last_error(void* h) {
    TtCtx* ctx = static_cast<TtCtx*>(h);
    return ctx ? ctx->last_error : TT_INVALID_ARGUMENT;
}

void* tt_fpga_alloc(void* h, long words) {
    if (words <= 0 || static_cast<size_t>(words) > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        latch_error(static_cast<TtCtx*>(h), TT_INVALID_ARGUMENT);
        return nullptr;
    }
#if defined(TT_FPGA_XRT)
    try {
        TtCtx* ctx = static_cast<TtCtx*>(h);
        xrt::kernel& k = ctx->kernel_for(std::getenv("TT_FPGA_ENTRY")
                                         ? std::getenv("TT_FPGA_ENTRY") : "forge_train_step");
        xrt::bo bo(ctx->device, (size_t)words * 4, k.group_id(0));
        void* host = bo.map<void*>();
        std::memset(host, 0, (size_t)words * 4);
        ctx->bos.emplace(host, std::move(bo));
        return host;
    } catch (...) { return nullptr; }
#elif defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!ctx || !ctx->present || ctx->last_error) return nullptr;
    void* buffer = std::calloc(static_cast<size_t>(words), sizeof(int32_t));
    if (!buffer) { latch_error(ctx, TT_ALLOCATION_FAILED); return nullptr; }
    try {
        ctx->allocations.emplace(buffer, static_cast<size_t>(words));
    } catch (...) {
        std::free(buffer); latch_error(ctx, TT_ALLOCATION_FAILED); return nullptr;
    }
    return buffer;
#else
    // LOCAL: the malloc IS the kernel buffer. FPGA-PCI: a host shadow (the device
    // buffer lives in BAR4); regions are pushed/pulled by write_region/read_region.
    (void)h;
    return std::calloc((size_t)words, sizeof(int32_t));
#endif
}

void tt_fpga_free(void* h, void* buf) {
#if defined(TT_FPGA_XRT)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (ctx) ctx->bos.erase(buf);
#elif defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!ctx || !buf) return;
    if (!ctx->allocations.erase(buf)) { latch_error(ctx, TT_INVALID_REGION); return; }
    std::free(buf);
#else
    (void)h; std::free(buf);
#endif
}

// --- region transfer (int32 WORD offsets), the runner's portable buffer API ----
void tt_fpga_write_region(void* h, void* buf, long off_words, const int32_t* src, long n) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!valid_region(ctx, buf, off_words, n, src) || !n) return;
    uint64_t base = TT_BUFFER_ADDR + (uint64_t)off_words * 4;
    for (long i = 0; i < n; ++i)
        if (!pci_write(ctx, ctx->bar4, base + (uint64_t)i * 4, (uint32_t)src[i])) return;
    std::memcpy((int32_t*)buf + off_words, src, (size_t)n * 4);
#else
    (void)h;
    std::memcpy((int32_t*)buf + off_words, src, (size_t)n * 4);
#endif
}

void tt_fpga_read_region(void* h, void* buf, long off_words, int32_t* dst, long n) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!valid_region(ctx, buf, off_words, n, dst) || !n) return;
    int32_t* staged = static_cast<int32_t*>(std::malloc(static_cast<size_t>(n) * sizeof(int32_t)));
    if (!staged) { latch_error(ctx, TT_ALLOCATION_FAILED); return; }
    uint64_t base = TT_BUFFER_ADDR + (uint64_t)off_words * 4;
    for (long i = 0; i < n; ++i) {
        uint32_t v = 0;
        if (!pci_read(ctx, ctx->bar4, base + (uint64_t)i * 4, &v)) {
            std::free(staged); return;
        }
        staged[i] = static_cast<int32_t>(v);
    }
    std::memcpy(dst, staged, (size_t)n * 4);
    std::memcpy((int32_t*)buf + off_words, staged, (size_t)n * 4);
    std::free(staged);
#else
    (void)h;
    std::memcpy(dst, (int32_t*)buf + off_words, (size_t)n * 4);
#endif
}

void tt_fpga_load_weights(void* h, void* buf, const int16_t* w, long n) {
    // Convenience: contiguous int16 weights -> the int32 buffer from word 0.
#if defined(TT_FPGA_PCI)
    if (!valid_region(static_cast<TtCtx*>(h), buf, 0, n, w) || !n) return;
#endif
    int32_t* tmp = (int32_t*)std::malloc((size_t)n * 4);
    if (!tmp) { latch_error(static_cast<TtCtx*>(h), TT_ALLOCATION_FAILED); return; }
    for (long i = 0; i < n; ++i) tmp[i] = (int32_t)w[i];
    tt_fpga_write_region(h, buf, 0, tmp, n);
    std::free(tmp);
}

void tt_fpga_save_weights(void* h, void* buf, int16_t* w, long n) {
#if defined(TT_FPGA_PCI)
    if (!valid_region(static_cast<TtCtx*>(h), buf, 0, n, w) || !n) return;
#endif
    int32_t* tmp = (int32_t*)std::malloc((size_t)n * 4);
    if (!tmp) { latch_error(static_cast<TtCtx*>(h), TT_ALLOCATION_FAILED); return; }
    tt_fpga_read_region(h, buf, 0, tmp, n);
#if defined(TT_FPGA_PCI)
    if (tt_fpga_last_error(h)) { std::free(tmp); return; }
#endif
    for (long i = 0; i < n; ++i) w[i] = (int16_t)tmp[i];
    std::free(tmp);
}

int tt_fpga_run(void* h, const char* entry, void* buf, int mode,
                int a1, int a2, int a3, int layer_id) {
#if defined(TT_FPGA_PCI)
    (void)entry; (void)layer_id;
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!valid_region(ctx, buf, 0, 0)) return tt_fpga_last_error(h);
    if (!pci_kick(ctx, mode, a1, a2, a3)) return tt_fpga_last_error(h);
    // Spin cap ~= wall-clock ceiling on ONE kick (each peek ~1.5us PCIe read ->
    // 200M spins ~= 5-7 min). Long fused whole-run kicks (d512 at 125 MHz:
    // ~0.21 s/step -> 2000 steps ~= 7 min) legitimately exceed it: override
    // with TT_KICK_SPIN (spins, not seconds).
    long spin_cap = 200000000L;
    { const char* s = std::getenv("TT_KICK_SPIN"); if (s && *s) { long v = atol(s); if (v > 0) spin_cap = v; } }
    for (long spin = 0; spin < spin_cap; ++spin) {
        uint32_t ctrl = 0;
        if (!pci_read(ctx, ctx->bar0, TT_ADDR_AP_CTRL, &ctrl)) return tt_fpga_last_error(h);
        if (ctrl & TT_AP_DONE) {
            uint32_t ret = 0;
            if (!pci_read(ctx, ctx->bar0, TT_ADDR_AP_RETURN, &ret)) return tt_fpga_last_error(h);
            return (int)ret;
        }
    }
    return latch_error(ctx, -200);   // kernel hung; never reuse uncertain device state
#elif defined(TT_FPGA_XRT)
    try {
        TtCtx* ctx = static_cast<TtCtx*>(h);
        auto it = ctx->bos.find(buf);
        if (it == ctx->bos.end()) return -98;
        xrt::bo& bo = it->second;
        xrt::kernel& k = ctx->kernel_for(entry);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        std::string e(entry);
        bool has_layer = (e.find("2layer") != std::string::npos ||
                          e.find("3layer") != std::string::npos);
        xrt::run r = has_layer ? k(bo, mode, a1, a2, a3, layer_id) : k(bo, mode, a1, a2, a3);
        r.wait();
        bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        return (int)r.return_code();
    } catch (...) { return -97; }
#else
    (void)h; int32_t* b = static_cast<int32_t*>(buf); (void)layer_id;
#ifdef TT_KERNEL_1LAYER
    if (std::strcmp(entry, "forge_train_step") == 0)
        return forge_train_step(b, mode, a1, a2, a3);
#endif
#ifdef TT_KERNEL_2LAYER
    if (std::strcmp(entry, "forge_train_step_2layer") == 0)
        return forge_train_step_2layer(b, mode, a1, a2, a3, layer_id);
#endif
#ifdef TT_KERNEL_3LAYER
    if (std::strcmp(entry, "forge_train_step_3layer") == 0)
        return forge_train_step_3layer(b, mode, a1, a2, a3, layer_id);
#endif
#ifdef TT_KERNEL_MORPH
    if (std::strcmp(entry, "forge_train_step_morph") == 0)
        return forge_train_step_morph(b, mode, a1, a2, a3);
#endif
#ifdef TT_KERNEL_MORPH_ECG
    if (std::strcmp(entry, "forge_train_step_morph_ecg") == 0)
        return forge_train_step_morph_ecg(b, mode, a1, a2, a3);
#endif
#ifdef TT_KERNEL_MORPH_LM
    if (std::strcmp(entry, "forge_train_step_morph_lm") == 0)
        return forge_train_step_morph_lm(b, mode, a1, a2, a3);
#endif
    return -99;
#endif
}

// --- N-way replicated CL helpers (v3 multi-kernel) -------------------------
// The N-way CL BROADCASTS OCL control writes to all kernels and MUXes reads by
// the control-address window (64 KB stride, bits [17:16] select the instance).
// tt_fpga_kick pokes mode/args + AP_START at window 0 -> broadcast -> all N
// instances start at once (non-blocking; caller polls). tt_fpga_done reads the
// AP_CTRL of instance `inst` via its 64 KB window and returns its ap_done bit.
void tt_fpga_kick(void* h, int mode, int a1, int a2, int a3) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!ctx || ctx->last_error) return;
    pci_kick(ctx, mode, a1, a2, a3);
#else
    (void)h; (void)mode; (void)a1; (void)a2; (void)a3;
#endif
}

int tt_fpga_done(void* h, int inst) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (!ctx || ctx->last_error) return 0;
    if (inst < 0) { latch_error(ctx, TT_INVALID_ARGUMENT); return 0; }
    uint32_t ctrl = 0;
    if (!pci_read(ctx, ctx->bar0, (uint64_t)inst * 0x10000ULL + TT_ADDR_AP_CTRL, &ctrl)) return 0;
    return (ctrl & TT_AP_DONE) ? 1 : 0;
#else
    (void)h; (void)inst; return 1;
#endif
}

}  // extern "C"
