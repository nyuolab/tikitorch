// CPU-only fault-injection transport. Never linked into a production bridge.
#include "fpga_pci.h"
#include "fpga_mgmt.h"
#include <map>
#include <utility>

namespace {
std::map<std::pair<int, uint64_t>, uint32_t> memory;
int operation = 0, fault_kind = 0, fault_bar = 0, fault_after = 0, fault_status = 0;
uint64_t fault_address = 0;
int live_handles = 0;
bool complete = true;
bool resets_asserted = false;
int fail(int kind, int bar, uint64_t address) {
    ++operation;
    if (fault_kind == kind && fault_bar == bar && fault_address == address && --fault_after == 0)
        return fault_status;
    return 0;
}
}

extern "C" {
void stub_reset() {
    memory.clear(); operation = 0; fault_kind = 0; fault_after = 0;
    live_handles = 0; complete = true; resets_asserted = false;
}
void stub_fault(int kind, int bar, uint64_t address, int after, int status) {
    fault_kind = kind; fault_bar = bar; fault_address = address;
    fault_after = after; fault_status = status;
}
int stub_operations() { return operation; }
int stub_live_handles() { return live_handles; }
void stub_complete(int value) { complete = value != 0; }
void stub_reset_asserted(int value) { resets_asserted = value != 0; }
void stub_store(int bar, uint64_t address, uint32_t value) { memory[{bar, address}] = value; }
uint32_t stub_load(int bar, uint64_t address) { return memory[{bar, address}]; }
int fpga_mgmt_init() { return 0; }
int fpga_pci_attach(int, int pf, int bar, unsigned, pci_bar_handle_t* handle) {
    const int id = pf == FPGA_MGMT_PF ? 14 : bar;
    const int error = fail(3, id, 0);
    if (error) return error;
    *handle = id; ++live_handles; return 0;
}
int fpga_pci_detach(pci_bar_handle_t handle) {
    --live_handles;
    return fail(4, handle, 0);
}
int fpga_pci_poke(pci_bar_handle_t handle, uint64_t address, uint32_t value) {
    const int error = fail(1, handle, address);
    if (error) return error;
    memory[{handle, address}] = value;
    return 0;
}
int fpga_pci_peek(pci_bar_handle_t handle, uint64_t address, uint32_t* value) {
    const int error = fail(2, handle, address);
    if (error) { *value = 0xBAD0BAD0; return error; } // failed reads poison output
    if (handle == 14 && address == 0x58020) *value = 1;
    else if (handle == 14 && address == 0x58014) *value = resets_asserted ? 1 : 0xFFFFFC00;
    else if (handle == 0 && (address % 0x10000) == 0) *value = complete ? 2 : 0;
    else *value = memory[{handle, address}];
    return 0;
}
// The LOCAL build regression uses a tiny stand-in, not FPGA arithmetic claims.
int forge_train_step(int32_t* buffer, int mode, int a1, int a2, int a3) {
    buffer[0] = mode + a1 + a2 + a3;
    return 0;
}
}
