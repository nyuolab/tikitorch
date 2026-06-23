# synth_cl_forge_sparsecoding_ddr.tcl — CL synth driver for the DDR variant. Mirrors the F2 CL_TEMPLATE flow
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

# our IP: OCL AXI4-Lite slice (scpc_ocl_slice — renamed to avoid the HDK axi_register_slice_light clash)
# + the 2->1 DDR crossbar. OOC flow needs synth_ip for the CL's own IP.
read_ip [glob $CL_DIR/design/ip/scpc_ocl_slice/scpc_ocl_slice.xci]
read_ip [glob $CL_DIR/design/ip/axi_crossbar_ddr/axi_crossbar_ddr.xci]
synth_ip [get_ips scpc_ocl_slice]
synth_ip [get_ips axi_crossbar_ddr]

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
