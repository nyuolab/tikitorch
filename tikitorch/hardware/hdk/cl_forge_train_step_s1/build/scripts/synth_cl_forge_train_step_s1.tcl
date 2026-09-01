# synth_cl_forge_train_step_s1.tcl — CL synth driver for the S1 trainer DDR variant. Mirrors the F2
# CL_TEMPLATE flow (header/footer from $HDK_SHELL_DIR/build/scripts) but ALSO pulls in the sh_ddr DDR
# controller module + its DDR4 IP (the CL instantiates sh_ddr with DDR_PRESENT=1). The exact sh_ddr
# source/IP wiring is the one thing the reference cl_dram_dma build sets up; this driver reads sh_ddr.sv
# from the shell design dir and the two CL IP. If the build needs the DDR4 IP generated (sh_ddr's internal
# MIG), the build runner injects the HDK DDR-IP step (see run_s1_afi_build.sh) — confirmed against the
# reference DDR build, byte-identical to the proven synth_cl_forge_muzero_ddr.tcl except the OCL-slice IP name.
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

# ---- CLKGEN+CDC rearchitecture: the kernel runs on clk_extra_a1 (125 MHz, Recipe A1) ----
# aws_clk_gen.sv (instantiated as AWS_CLK_GEN in the CL) + aws_clk_regs.sv are added by the --aws_clk_gen build
# flag -- do NOT read them here (cl_mem_perf's synth tcl does not either; a manual read double-defines the module,
# and $HDK_DIR is not a Tcl var in this context -> "can't read HDK_DIR" was the v8 fail). The flag handles them.
# AWS_CLK_GEN's MMCM (group A) + AXI-Lite clock-xbar + reg-slice IP (pre-built in the HDK cl_ip src tree, the
# same locations cl_mem_perf's synth reads). Only group A is enabled in the CL, so only clk_mmcm_a is needed;
# cl_clk_axil_xbar + cl_axi_register_slice_light + cl_axi_clock_converter_light are aws_clk_gen-internal deps.
foreach f [list \
    $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/clk_mmcm_a/clk_mmcm_a.xci \
    $HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/cl_ip.srcs/sources_1/ip/cl_clk_axil_xbar/cl_clk_axil_xbar.xci \
    ${HDK_IP_SRC_DIR}/cl_axi_register_slice_light/cl_axi_register_slice_light.xci \
  ] {
  if {[file exists $f]} { read_ip $f }
}

# our IP: OCL AXI4-Lite slice (s1_ocl_slice — renamed to avoid the HDK axi_register_slice_light clash)
# + the 2->1 DDR crossbar + the TWO self-owned CDC clock converters (s1_axil_cdc OCL 250->125,
# s1_axi_cdc kernel-mem 125->250). OOC flow needs synth_ip for the CL's own IP.
read_ip [glob $CL_DIR/design/ip/s1_ocl_slice/s1_ocl_slice.xci]
read_ip [glob $CL_DIR/design/ip/axi_crossbar_ddr/axi_crossbar_ddr.xci]
read_ip [glob $CL_DIR/design/ip/s1_axil_cdc/s1_axil_cdc.xci]
read_ip [glob $CL_DIR/design/ip/s1_axi_cdc/s1_axi_cdc.xci]
# s1_pcis_rs: fully-registered AXI4 reg slice on the shell DMA-PCIS <-> crossbar SI1 boundary (timing-close fix)
read_ip [glob $CL_DIR/design/ip/s1_pcis_rs/s1_pcis_rs.xci]
synth_ip [get_ips s1_ocl_slice]
synth_ip [get_ips axi_crossbar_ddr]
synth_ip [get_ips s1_axil_cdc]
synth_ip [get_ips s1_axi_cdc]
synth_ip [get_ips s1_pcis_rs]

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
