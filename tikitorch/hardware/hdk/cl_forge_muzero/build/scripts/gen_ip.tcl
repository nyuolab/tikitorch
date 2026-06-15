# gen_ip.tcl — IP for cl_forge_muzero. Unlike morph-LM (URAM staging + N-way crossbar), the chess
# MuZero buffer (~12 MB fwd / ~6 MB train) lives in HOST DRAM and the kernel reaches it via PCIM, so the
# only IP needed is the OCL AXI4-Lite register slice + an AXI4 register slice on the kernel->PCIM master
# (timing). vivado -mode batch -source gen_ip.tcl -tclargs <out_ip_dir>
set part xcvu47p-fsvh2892-2-e
set ipdir [lindex $argv 0]
create_project -in_memory -part $part
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_light -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.ADDR_WIDTH {32} CONFIG.DATA_WIDTH {32}] [get_ips axi_register_slice_light]
# full AXI4 512b register slice on the kernel m_axi -> PCIM path (helps timing across to the shell PCIM)
create_ip -name axi_register_slice -vendor xilinx.com -library ip -module_name axi_register_slice_pcim -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.ADDR_WIDTH {64} CONFIG.DATA_WIDTH {512} CONFIG.ID_WIDTH {6}] [get_ips axi_register_slice_pcim]
generate_target all [get_ips]
