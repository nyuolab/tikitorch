source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl
print "Reading encrypted user source (CL.sv + flattened HLS kernel verilog)"
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]
print "Reading + synthesizing CL IP blocks (regenerated this build)"
# v3 CLKGEN: aws_clk_gen.sv itself is added by the --aws_clk_gen build flag — do
# NOT read it here (double-defines the module; the S1 v8 lesson). But its
# INTERNAL dependency IPs must be readable so the module elaborates: MMCM_A +
# the clkgen AXI-Lite xbar/slice/converter, pre-built in the HDK cl_ip tree
# (paths + file-exists guards copied from the PROVEN S1 synth tcl).
foreach f [list \
    $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/clk_mmcm_a/clk_mmcm_a.xci \
    $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_clk_axil_xbar/cl_clk_axil_xbar.xci \
    ${HDK_IP_SRC_DIR}/cl_axi_register_slice_light/cl_axi_register_slice_light.xci \
    ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci \
  ] {
  if {[file exists $f]} { read_ip $f }
}
read_ip $::env(CL_DIR)/design/ip/axi_register_slice_light/axi_register_slice_light.xci
read_ip $::env(CL_DIR)/design/ip/axi_bram_ctrl_dma/axi_bram_ctrl_dma.xci
read_ip $::env(CL_DIR)/design/ip/axi_bram_ctrl_krn/axi_bram_ctrl_krn.xci
read_ip $::env(CL_DIR)/design/ip/axi_crossbar_dma/axi_crossbar_dma.xci
read_ip $::env(CL_DIR)/design/ip/axi_crossbar_krn/axi_crossbar_krn.xci
read_ip $::env(CL_DIR)/design/ip/axi_register_slice_dma/axi_register_slice_dma.xci
read_ip $::env(CL_DIR)/design/ip/axi_register_slice_krn/axi_register_slice_krn.xci
read_ip $::env(CL_DIR)/design/ip/mld512_pcis_cdc/mld512_pcis_cdc.xci
read_ip $::env(CL_DIR)/design/ip/mld512_ocl_cdc/mld512_ocl_cdc.xci
generate_target all [get_ips]
synth_ip [get_ips]
print "Reading user constraints"
read_xdc [ list ${constraints_dir}/cl_synth_user.xdc ${constraints_dir}/cl_timing_user.xdc ]
set_property PROCESSING_ORDER LATE [get_files cl_synth_user.xdc]
set_property PROCESSING_ORDER LATE [get_files cl_timing_user.xdc]
print "Starting synthesis of ${CL}"
update_compile_order -fileset sources_1
synth_design -mode out_of_context -top ${CL} -verilog_define XSDB_SLV_DIS -part ${DEVICE_TYPE} -keep_equivalent_registers
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
