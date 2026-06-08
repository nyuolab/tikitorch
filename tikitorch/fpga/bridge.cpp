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
#include <cstdlib>
#include <cstring>
#include <new>  // std::nothrow

#if defined(TT_FPGA_PCI)
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
#endif

namespace {
struct TtCtx {
    int present = 0;
#if defined(TT_FPGA_PCI)
    int slot = 0;
    pci_bar_handle_t bar0 = PCI_BAR_HANDLE_INIT;   // OCL control regs
    pci_bar_handle_t bar4 = PCI_BAR_HANDLE_INIT;   // int32 buffer (DDR)
    int buffer_set = 0;
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
}  // namespace

extern "C" {

void* tt_fpga_open(const char* arg) {
#if defined(TT_FPGA_PCI)
    if (fpga_mgmt_init() != 0) return nullptr;
    TtCtx* ctx = new (std::nothrow) TtCtx();
    if (!ctx) return nullptr;
    ctx->slot = (arg && *arg) ? atoi(arg) : 0;
    if (fpga_pci_attach(ctx->slot, FPGA_APP_PF, APP_PF_BAR0, 0, &ctx->bar0) != 0) {
        delete ctx; return nullptr;
    }
    if (fpga_pci_attach(ctx->slot, FPGA_APP_PF, APP_PF_BAR4, BURST_CAPABLE, &ctx->bar4) != 0) {
        fpga_pci_detach(ctx->bar0); delete ctx; return nullptr;
    }
    // Point the kernel's s_axilite buffer-address registers at BAR4 base (once).
    fpga_pci_poke(ctx->bar0, TT_ADDR_BUFFER_0, (uint32_t)(TT_BUFFER_ADDR & 0xFFFFFFFFULL));
    fpga_pci_poke(ctx->bar0, TT_ADDR_BUFFER_1, (uint32_t)((TT_BUFFER_ADDR >> 32) & 0xFFFFFFFFULL));
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
    if (ctx) { fpga_pci_detach(ctx->bar0); fpga_pci_detach(ctx->bar4); }
#endif
    delete static_cast<TtCtx*>(h);
}

int tt_fpga_board_present(void* h) {
    TtCtx* ctx = static_cast<TtCtx*>(h);
    return (ctx != nullptr && ctx->present) ? 1 : 0;
}

void* tt_fpga_alloc(void* h, long words) {
    if (words <= 0) return nullptr;
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
#else
    (void)h; std::free(buf);
#endif
}

// --- region transfer (int32 WORD offsets), the runner's portable buffer API ----
void tt_fpga_write_region(void* h, void* buf, long off_words, const int32_t* src, long n) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    if (buf) std::memcpy((int32_t*)buf + off_words, src, (size_t)n * 4);   // keep shadow in sync
    uint64_t base = TT_BUFFER_ADDR + (uint64_t)off_words * 4;
    for (long i = 0; i < n; ++i) fpga_pci_poke(ctx->bar4, base + (uint64_t)i * 4, (uint32_t)src[i]);
#else
    (void)h;
    std::memcpy((int32_t*)buf + off_words, src, (size_t)n * 4);
#endif
}

void tt_fpga_read_region(void* h, void* buf, long off_words, int32_t* dst, long n) {
#if defined(TT_FPGA_PCI)
    TtCtx* ctx = static_cast<TtCtx*>(h);
    uint64_t base = TT_BUFFER_ADDR + (uint64_t)off_words * 4;
    for (long i = 0; i < n; ++i) {
        uint32_t v = 0; fpga_pci_peek(ctx->bar4, base + (uint64_t)i * 4, &v); dst[i] = (int32_t)v;
    }
    if (buf) std::memcpy((int32_t*)buf + off_words, dst, (size_t)n * 4);
#else
    (void)h;
    std::memcpy(dst, (int32_t*)buf + off_words, (size_t)n * 4);
#endif
}

void tt_fpga_load_weights(void* h, void* buf, const int16_t* w, long n) {
    // Convenience: contiguous int16 weights -> the int32 buffer from word 0.
    int32_t* tmp = (int32_t*)std::malloc((size_t)n * 4);
    for (long i = 0; i < n; ++i) tmp[i] = (int32_t)w[i];
    tt_fpga_write_region(h, buf, 0, tmp, n);
    std::free(tmp);
}

void tt_fpga_save_weights(void* h, void* buf, int16_t* w, long n) {
    int32_t* tmp = (int32_t*)std::malloc((size_t)n * 4);
    tt_fpga_read_region(h, buf, 0, tmp, n);
    for (long i = 0; i < n; ++i) w[i] = (int16_t)tmp[i];
    std::free(tmp);
}

int tt_fpga_run(void* h, const char* entry, void* buf, int mode,
                int a1, int a2, int a3, int layer_id) {
#if defined(TT_FPGA_PCI)
    (void)entry; (void)buf; (void)layer_id;
    TtCtx* ctx = static_cast<TtCtx*>(h);
    fpga_pci_poke(ctx->bar0, TT_ADDR_MODE, (uint32_t)mode);
    fpga_pci_poke(ctx->bar0, TT_ADDR_ARG1, (uint32_t)a1);
    fpga_pci_poke(ctx->bar0, TT_ADDR_ARG2, (uint32_t)a2);
    fpga_pci_poke(ctx->bar0, TT_ADDR_ARG3, (uint32_t)a3);
    fpga_pci_poke(ctx->bar0, TT_ADDR_AP_CTRL, TT_AP_START);
    for (long spin = 0; spin < 200000000L; ++spin) {
        uint32_t ctrl = 0;
        fpga_pci_peek(ctx->bar0, TT_ADDR_AP_CTRL, &ctrl);
        if (ctrl & TT_AP_DONE) {
            uint32_t ret = 0; fpga_pci_peek(ctx->bar0, TT_ADDR_AP_RETURN, &ret);
            return (int)ret;
        }
    }
    return -200;   // kernel hung
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
    return -99;
#endif
}

}  // extern "C"
