set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
# 1. OCL AXI4-Lite register slice (32b)
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_light -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips axi_register_slice_light]
# 2. DMA-PCIS AXI4 BRAM controller (512b, ID16, external single-port, 2MB)
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_dma -dir $ipdir
set_property -dict [list CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {16} CONFIG.PROTOCOL {AXI4} CONFIG.BMG_INSTANCE {EXTERNAL} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.MEM_DEPTH {32768} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {1} CONFIG.SUPPORTS_NARROW_BURST {1}] [get_ips axi_bram_ctrl_dma]
# 3. kernel-side AXI4 BRAM controller (512b, ID1)
create_ip -name axi_bram_ctrl -vendor xilinx.com -library ip -module_name axi_bram_ctrl_krn -dir $ipdir
set_property -dict [list CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {1} CONFIG.PROTOCOL {AXI4} CONFIG.BMG_INSTANCE {EXTERNAL} CONFIG.SINGLE_PORT_BRAM {1} CONFIG.MEM_DEPTH {32768} CONFIG.ECC_TYPE {0} CONFIG.READ_LATENCY {1} CONFIG.SUPPORTS_NARROW_BURST {1}] [get_ips axi_bram_ctrl_krn]
puts "=== generated IPs ==="
foreach ip {axi_register_slice_light axi_bram_ctrl_dma axi_bram_ctrl_krn} {
  generate_target {synthesis} [get_ips $ip]
  puts "$ip OK"
}
