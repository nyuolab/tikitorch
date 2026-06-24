# gen_ip.tcl — IP for cl_forge_train_step_s1: the OCL AXI4-Lite register slice + a 2->1 AXI crossbar that
# merges the kernel m_axi (reads AND writes) and the shell DMA-PCIS (host staging) onto the single sh_ddr DDR
# AXI slave. SASD (shared-address/shared-data, serialized) connectivity keeps the master-side ID width = 16
# (matching cl_sh_ddr_axi's 16-bit ID) instead of widening to 17 — and serialized access is fine because
# staging and kernel-read/write happen sequentially, never concurrently.
#   vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>
set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
# OCL control slice (AXI4-Lite 32b). Named s1_ocl_slice (NOT axi_register_slice_light) so it does not
# collide with the HDK's own axi_register_slice_light that the DDR path (sh_ddr deps) reads from HDK_IP_SRC_DIR.
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name s1_ocl_slice -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips s1_ocl_slice]
# 2 SI (kernel + DMA-PCIS) -> 1 MI (DDR), 512b, 64-bit addr, ID_WIDTH=16. With NUM_MI=1 the axi_crossbar does
# NOT widen the master ID (it tracks SIs internally) — master thread ID = ID_WIDTH = 16, matching
# cl_sh_ddr_axi's 16-bit ID (smoke #2 showed ID_WIDTH=15 -> 15-bit master, hence 16). Default SAMD.
create_ip -name axi_crossbar -vendor xilinx.com -library ip -module_name axi_crossbar_ddr -dir $ipdir
set_property -dict [list \
  CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {16} \
  CONFIG.R_REGISTER {1} \
  CONFIG.M00_A00_BASE_ADDR {0x0000000000000000} CONFIG.M00_A00_ADDR_WIDTH {34} \
] [get_ips axi_crossbar_ddr]

# ---- CLKGEN+CDC rearchitecture: the kernel runs on clk_extra_a1 (125 MHz); two axi_clock_converter CDC IPs
#      bridge the kernel's two AXI boundaries to/from the 250 MHz shell domain. SELF-OWNED module names
#      (s1_axil_cdc / s1_axi_cdc) so they do NOT collide with the HDK's pre-built cl_axi_clock_converter*
#      that sh_ddr reads from HDK_IP_SRC_DIR — same renaming trick as s1_ocl_slice.
# CDC-1: OCL control 250->125 MHz, AXI4-Lite 32b (slave clk_main_a0 -> master clk_extra_a1).
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name s1_axil_cdc -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4LITE} CONFIG.DATA_WIDTH {32} CONFIG.ADDR_WIDTH {32} \
  CONFIG.ID_WIDTH {0} CONFIG.SYNCHRONIZATION_STAGES {3} \
] [get_ips s1_axil_cdc]
# CDC-2: kernel m_axi 125->250 MHz, AXI4 512b, 64-bit addr, 1-bit ID (matches the kernel)
#        (slave clk_extra_a1 -> master clk_main_a0; integer 2:1 ratio).
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name s1_axi_cdc -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} \
  CONFIG.ID_WIDTH {1} CONFIG.SYNCHRONIZATION_STAGES {3} CONFIG.ACLK_RATIO {2:1} \
] [get_ips s1_axi_cdc]

generate_target all [get_ips]
