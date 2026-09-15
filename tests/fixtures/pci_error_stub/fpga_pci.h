#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int pci_bar_handle_t;
#define PCI_BAR_HANDLE_INIT (-1)
#define FPGA_APP_PF 0
#define FPGA_MGMT_PF 1
#define APP_PF_BAR0 0
#define APP_PF_BAR4 4
#define MGMT_PF_BAR4 4
#define BURST_CAPABLE 1
int fpga_pci_attach(int slot, int pf, int bar, unsigned flags, pci_bar_handle_t* handle);
int fpga_pci_detach(pci_bar_handle_t handle);
int fpga_pci_poke(pci_bar_handle_t handle, uint64_t address, uint32_t value);
int fpga_pci_peek(pci_bar_handle_t handle, uint64_t address, uint32_t* value);
#ifdef __cplusplus
}
#endif
