# synth_cl_forge_pc_xformer.tcl — CL synth driver for the DDR variant. Mirrors the F2 CL_TEMPLATE flow
# (header/footer from $HDK_SHELL_DIR/build/scripts) but ALSO pulls in the sh_ddr DDR controller module +
# its DDR4 IP (the CL instantiates sh_ddr with DDR_PRESENT=1). The exact sh_ddr source/IP wiring is the one
# thing the reference cl_dram_dma build sets up; this driver reads sh_ddr.sv from the shell design dir and
# the two CL IP. If the build needs the DDR4 IP generated (sh_ddr's internal MIG), the build runner injects
# the HDK DDR-IP step (see run_muzero_ddr_afi_build.sh) — confirmed against the reference DDR build.
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

# CL top + kernel RTL (+ shell .inc), staged into the post-encryption source dir by encrypt.tcl.
# NOTE: do NOT read_verilog sh_ddr.sv — it is an ENCRYPTED shell module already provided by the header/shell
# (the PCIM build instantiated it via unused_ddr_template with no extra read); a raw read_verilog of the
# encrypted file fails with "Error generated from encrypted envelope". We just instantiate sh_ddr in the .sv.
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]

# sh_ddr (DDR_PRESENT=1) dependency IPs from the HDK IP src dir: the DDR4 MIG + the AXI clock-converters /
# register slices sh_ddr instantiates internally (DDR clock-domain crossing). These are pre-built in
# HDK_IP_SRC_DIR; reading them makes their modules available so sh_ddr's encrypted body elaborates (the
# "encrypted envelope" error was sh_ddr referencing these missing deps). Matches the reference cl_dram_hbm_dma
# read_ip list, DDR subset.
read_ip [list \
  ${HDK_IP_SRC_DIR}/cl_ddr4/cl_ddr4.xci \
  ${HDK_IP_SRC_DIR}/axi_register_slice/axi_register_slice.xci \
  ${HDK_IP_SRC_DIR}/axi_register_slice_light/axi_register_slice_light.xci \
  ${HDK_IP_SRC_DIR}/cl_axi3_256b_reg_slice/cl_axi3_256b_reg_slice.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter/cl_axi_clock_converter.xci \
  ${HDK_IP_SRC_DIR}/cl_axi_clock_converter_light/cl_axi_clock_converter_light.xci \
]

# our IP: OCL AXI4-Lite slice (pcblock_ocl_slice — renamed to avoid the HDK axi_register_slice_light clash)
# + the 2->1 DDR crossbar. OOC flow needs synth_ip for the CL's own IP.
read_ip [glob $CL_DIR/design/ip/pcblock_ocl_slice/pcblock_ocl_slice.xci]
read_ip [glob $CL_DIR/design/ip/axi_crossbar_ddr/axi_crossbar_ddr.xci]
synth_ip [get_ips pcblock_ocl_slice]
synth_ip [get_ips axi_crossbar_ddr]

# CDC IP: clk_div_kernel (clk_wiz 250->150 MHz) + the two AXI clock converters that
# bridge the kernel's clk_kernel domain to the clk_main_a0 fabric (gmem AXI4 512b /
# control AXI4LITE 32b). Same read_ip + synth_ip OOC pattern as the CL IP above.
read_ip [glob $CL_DIR/design/ip/clk_div_kernel/clk_div_kernel.xci]
read_ip [glob $CL_DIR/design/ip/axi_cc_gmem/axi_cc_gmem.xci]
read_ip [glob $CL_DIR/design/ip/axi_cc_ctrl/axi_cc_ctrl.xci]
synth_ip [get_ips clk_div_kernel]
synth_ip [get_ips axi_cc_gmem]
synth_ip [get_ips axi_cc_ctrl]

# HLS BF16 floating-point operator IP — the mp kernel's BF16 backward uses single-precision fadd/fsub/faddfsub/
# fcmp/fmul/fexp/fdiv + sitofp. Vitis HLS csynth emits only the operator WRAPPERS (.v, read via read_verilog
# above) plus a companion *_ip.tcl generator for each; csynth never RUNS them, so the underlying floating_point
# _ip cores are missing -> "module forge_pc_xformer_mp_block_..._ip not found" at synth (the AFI build #1 wall).
# Source the generators here: each does create_ip (floating_point 7.1) + generate_target {synthesis simulation}
# and sets generate_synth_checkpoint=false, so synth_design below synthesizes them INLINE with the CL (the path
# HLS intends). The wrappers' _ip instantiations resolve at elaboration. (The int kernel had zero float ops,
# hence never needed this.) The AFI build stages these into $CL_DIR/design/fpip/ from the HLS verilog dir.
set fpip_tcls [lsort [glob -nocomplain $CL_DIR/design/fpip/*_ip.tcl]]
puts "FPIP: [llength $fpip_tcls] HLS floating-point IP generators to source"
foreach iptcl $fpip_tcls {
  puts "FPIP: sourcing [file tail $iptcl]"
  source $iptcl
}

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
