set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
# 1. OCL AXI4-Lite register slice (32b)
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_light -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips axi_register_slice_light]
# 2. DMA-PCIS AXI4 BRAM controller (512b, ID16, external single-port, 4MB/bank).
#    MEM_DEPTH 65536 = the IP's MAXIMUM (262144 was out of range — the reason
#    for the two-bank v2). 65536 x 512b lines = 4 MB; the CL instantiates one
#    per bank (2 banks = 8 MB staging).
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_dma -dir $ipdir
set_property -dict [list CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.PROTOCOL {AXI4} CONFIG.BMG_INSTANCE {EXTERNAL} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.MEM_DEPTH {65536} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {1} CONFIG.SUPPORTS_NARROW_BURST {1}] [get_ips axi_bram_ctrl_dma]
# 3. kernel-side AXI4 BRAM controller (512b, ID1, 4MB/bank; one per bank)
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_krn -dir $ipdir
set_property -dict [list CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {1} CONFIG.PROTOCOL {AXI4} CONFIG.BMG_INSTANCE {EXTERNAL} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.MEM_DEPTH {65536} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {1} CONFIG.SUPPORTS_NARROW_BURST {1}] [get_ips axi_bram_ctrl_krn]
# 4. DMA-PCIS crossbar: 1 SI -> 2 MI (512b AXI4, ID16). Two 4 MB bank windows,
#    CONTIGUOUS from 0x0100_0000 (host sees one 8 MB buffer). NUM_MI MUST match
#    the CL localparam NB (=2).
create_ip -name axi_crossbar -vendor xilinx.com -library ip -module_name axi_crossbar_dma -dir $ipdir
set_property -dict [list \
  CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.ADDR_WIDTH {64} \
  CONFIG.CONNECTIVITY_MODE {SAMD} CONFIG.STRATEGY {2} \
  CONFIG.M00_A00_BASE_ADDR {0x0000000001000000} CONFIG.M00_A00_ADDR_WIDTH {22} \
  CONFIG.M01_A00_BASE_ADDR {0x0000000001400000} CONFIG.M01_A00_ADDR_WIDTH {22} \
] [get_ips axi_crossbar_dma]
# 5. kernel-side crossbar: 1 SI -> 2 MI (512b AXI4, ID1). The kernel's single
#    m_axi gmem emits absolute addresses from base 0x0100_0000 (bridge.cpp
#    TT_BUFFER_ADDR poked into the s_axilite pointer regs), so the bank decode
#    uses the SAME bases as the DMA side. NUM_MI MUST match the CL NB (=2).
create_ip -name axi_crossbar -vendor xilinx.com -library ip -module_name axi_crossbar_krn -dir $ipdir
set_property -dict [list \
  CONFIG.NUM_SI {1} CONFIG.NUM_MI {2} \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {1} CONFIG.ADDR_WIDTH {64} \
  CONFIG.CONNECTIVITY_MODE {SAMD} CONFIG.STRATEGY {2} \
  CONFIG.M00_A00_BASE_ADDR {0x0000000001000000} CONFIG.M00_A00_ADDR_WIDTH {22} \
  CONFIG.M01_A00_BASE_ADDR {0x0000000001400000} CONFIG.M01_A00_ADDR_WIDTH {22} \
] [get_ips axi_crossbar_krn]
# 6. Full-AXI4 register slice on each crossbar MI -> BRAM_CTL_DMA path. Placed in
#    the kernel's SLR pblock so the crossbar(shared)->slice(kernel SLR) crossing
#    is reg-to-reg (Laguna) instead of a combinational SLL hop (the recurring
#    N>=3 inter-SLR failure into BRAM_CTL_DMA's bram_addr/bram_wrdata regs).
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_dma -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.ADDR_WIDTH {64} \
  CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.REG_AR {1} CONFIG.REG_R {1}] [get_ips axi_register_slice_dma]
# 7. Full-AXI4 register slice on each XBAR_KRN MI -> BRAM_CTL_KRN path (ID1,
#    kernel side). The bare crossbar->controller hop is the twice-measured
#    (-1.75 / -1.147 ns) inter-SLR failure mode of this design family; the
#    pblock-excluded slice floats to the SLR boundary so both hops route
#    reg-to-reg through Laguna — the exact RS_DMA fix, mirrored.
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_krn -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {1} CONFIG.ADDR_WIDTH {64} \
  CONFIG.REG_AW {1} CONFIG.REG_W {1} CONFIG.REG_B {1} CONFIG.REG_AR {1} CONFIG.REG_R {1}] [get_ips axi_register_slice_krn]
# 8. v3 CLKGEN CDCs — the whole user design runs on clk_extra_a1 (125 MHz;
#    measured v2 250 MHz P&R WNS -2.726 ns inside the URAM staging banks).
#    ASYNC converters (clock-agnostic; the S1 pattern, cut at the shell edge).
#    SELF-OWNED module names so they never collide with HDK pre-built CDC IP.
# CDC-1: shell DMA-PCIS 250 -> 125 MHz, AXI4 512b ID16.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name mld512_pcis_cdc -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} \
  CONFIG.ID_WIDTH {16} CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3} \
] [get_ips mld512_pcis_cdc]
# CDC-2: shell OCL control 250 -> 125 MHz, AXI4-Lite 32b.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name mld512_ocl_cdc -dir $ipdir
set_property -dict [list \
  CONFIG.PROTOCOL {AXI4LITE} CONFIG.DATA_WIDTH {32} CONFIG.ADDR_WIDTH {32} \
  CONFIG.ID_WIDTH {0} CONFIG.ACLK_ASYNC {1} CONFIG.SYNCHRONIZATION_STAGES {3} \
] [get_ips mld512_ocl_cdc]
puts "=== generated IPs ==="
foreach ip {axi_register_slice_light axi_bram_ctrl_dma axi_bram_ctrl_krn axi_crossbar_dma axi_crossbar_krn axi_register_slice_dma axi_register_slice_krn mld512_pcis_cdc mld512_ocl_cdc} {
  generate_target {synthesis} [get_ips $ip]
  puts "$ip OK"
}
