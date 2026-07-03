# gen_ip.tcl — IP for cl_forge_pc_xformer_s2: the OCL AXI4-Lite register slice + a 2->1 AXI crossbar that
# merges the kernel m_axi (reads+writes) and the shell DMA-PCIS (host staging) onto the single sh_ddr DDR AXI
# slave. IDENTICAL IP set + config to the mp CL (cl_forge_pc_xformer) — the s2 kernel keeps the same 512-bit DDR
# interface width, same clk_kernel CDC, same crossbar; only the instantiated kernel top differs. SASD
# (shared-address/shared-data, serialized) connectivity keeps the master-side ID width = 16 (matching
# cl_sh_ddr_axi's 16-bit ID) instead of widening to 17 — and serialized access is fine because staging and
# kernel-read happen sequentially, never concurrently.
#   vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>
set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
# OCL control slice (AXI4-Lite 32b). Named pcblock_ocl_slice (NOT axi_register_slice_light) so it does not
# collide with the HDK's own axi_register_slice_light that the DDR path (sh_ddr deps) reads from HDK_IP_SRC_DIR.
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name pcblock_ocl_slice -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips pcblock_ocl_slice]
# 2 SI (kernel + DMA-PCIS) -> 1 MI (DDR), 512b, 64-bit addr, ID_WIDTH=16. With NUM_MI=1 the axi_crossbar does
# NOT widen the master ID (it tracks SIs internally) — master thread ID = ID_WIDTH = 16, matching
# cl_sh_ddr_axi's 16-bit ID (smoke #2 showed ID_WIDTH=15 -> 15-bit master, hence 16). Default SAMD.
# M00_A00_ADDR_WIDTH {34} = a 16 GB DDR region at base 0 — plenty for the s2 buffer (S2_BUFLEN ~= 101 MB/chip),
# so the larger footprint needs NO change here (same decode as the mp CL).
create_ip -name axi_crossbar -vendor xilinx.com -library ip -module_name axi_crossbar_ddr -dir $ipdir
set_property -dict [list \
  CONFIG.NUM_SI {2} CONFIG.NUM_MI {1} \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} CONFIG.ID_WIDTH {16} \
  CONFIG.R_REGISTER {1} \
  CONFIG.M00_A00_BASE_ADDR {0x0000000000000000} CONFIG.M00_A00_ADDR_WIDTH {34} \
] [get_ips axi_crossbar_ddr]

# =============================================================================
# CDC: run ONLY the kernel on a slower derived clock (clk_kernel ~150 MHz) than
# the shell's fixed 250 MHz clk_main_a0, because the kernel's INTERNAL compute
# misses timing at 250 MHz (post-route WNS -1.025 ns). Everything else (OCL
# slice, crossbar, sh_ddr, DMA-PCIS) stays on clk_main_a0.  Three IP:
#   clk_div_kernel  : Clocking Wizard, 250 MHz in -> 150 MHz out + locked
#   axi_cc_gmem     : AXI4   clock converter for the kernel's 512b m_axi master
#   axi_cc_ctrl     : AXI4LITE clock converter for the OCL control path
# =============================================================================

# ---- Clocking Wizard: clk_main_a0 (250 MHz) -> clk_kernel (~150 MHz) ----
# Simple fixed-ratio divider: NO dynamic-reconfig AXI interface (USE_DYN_RECONFIG
# false) and NO reset port (USE_RESET false) -> ports are just clk_in1 / clk_out1 /
# locked. Dropping the reset removes the only port-name ambiguity (resetn vs reset);
# the MMCM locks on power-up from the always-present clk_main_a0, and `locked` gates
# the kernel-domain reset (rst_kernel_n) in the CL top, so no clk_wiz reset is needed.
# PRIM_IN_FREQ 250.000 = clk_main_a0; CLKOUT1_REQUESTED_OUT_FREQ 150.000 = clk_kernel.
create_ip -name clk_wiz -vendor xilinx.com -library ip -module_name clk_div_kernel -dir $ipdir
set_property -dict [list \
  CONFIG.PRIMITIVE {MMCM} \
  CONFIG.PRIM_IN_FREQ {250.000} \
  CONFIG.CLKOUT1_REQUESTED_OUT_FREQ {150.000} \
  CONFIG.USE_DYN_RECONFIG {false} \
  CONFIG.USE_LOCKED {true} \
  CONFIG.USE_RESET {false} \
  CONFIG.PRIM_SOURCE {Global_buffer} \
  CONFIG.CLKIN1_JITTER_PS {40.0} \
] [get_ips clk_div_kernel]

# ---- AXI4 clock converter for the kernel m_axi gmem (512b) ----
# SLAVE side  = clk_kernel  (the kernel's m_axi_gmem master attaches here)
# MASTER side = clk_main_a0 (attaches to crossbar SI0 where the kernel used to go)
# Widths MUST match the kernel master EXACTLY (see cl_forge_pc_xformer_s2.sv krn_* nets):
#   ADDR_WIDTH 64  (krn_awaddr/araddr[63:0])
#   DATA_WIDTH 512 (krn_wdata/rdata[511:0], krn_wstrb[63:0])
#   ID_WIDTH   1   (krn_awid/arid/bid/rid are 1-bit; the crossbar widens 1->16 DOWNSTREAM)
#   AWUSER/ARUSER width 0 (kernel m_axi_gmem AWUSER/ARUSER/WUSER/BUSER/RUSER tied/unused)
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name axi_cc_gmem -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4} \
  CONFIG.ADDR_WIDTH {64} \
  CONFIG.DATA_WIDTH {512} \
  CONFIG.ID_WIDTH {1} \
  CONFIG.AWUSER_WIDTH {0} CONFIG.ARUSER_WIDTH {0} \
  CONFIG.WUSER_WIDTH {0}  CONFIG.RUSER_WIDTH {0}  CONFIG.BUSER_WIDTH {0} \
] [get_ips axi_cc_gmem]

# ---- AXI4-LITE clock converter for the OCL control path (32b) ----
# SLAVE side  = clk_main_a0 (the OCL register-slice m_axi / ocl_*_pre nets attach here)
# MASTER side = clk_kernel  (the kernel's s_axi_control attaches here)
# AXI4LITE has no ID/USER/burst signals; ADDR/DATA 32 match the OCL slice + kernel ctrl map.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name axi_cc_ctrl -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4LITE} \
  CONFIG.ADDR_WIDTH {32} \
  CONFIG.DATA_WIDTH {32} \
] [get_ips axi_cc_ctrl]

generate_target all [get_ips]
