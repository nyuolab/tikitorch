# gen_ip.tcl — IP for cl_forge_muzero_ddr: the OCL AXI4-Lite register slice + a 2->1 AXI crossbar that
# merges the kernel m_axi (reads) and the shell DMA-PCIS (host staging) onto the single sh_ddr DDR AXI slave.
# SASD (shared-address/shared-data, serialized) connectivity keeps the master-side ID width = 16 (matching
# cl_sh_ddr_axi's 16-bit ID) instead of widening to 17 — and serialized access is fine because staging and
# kernel-read happen sequentially, never concurrently.
#   vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>
set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
# OCL control slice (AXI4-Lite 32b)
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_light -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips axi_register_slice_light]
# 2 SI (kernel + DMA-PCIS) -> 1 MI (DDR), 512b, 64-bit addr, 16-bit ID, serialized (SASD)
create_ip -name axi_crossbar -vendor xilinx.com -library ip -module_name axi_crossbar_ddr -dir $ipdir
set_property -dict [list \
  CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {16} \
  CONFIG.CONNECTIVITY_MODE {SASD} \
  CONFIG.R_REGISTER {1} \
  CONFIG.M00_A00_BASE_ADDR {0x0000000000000000} CONFIG.M00_A00_ADDR_WIDTH {34} \
] [get_ips axi_crossbar_ddr]
generate_target all [get_ips]
