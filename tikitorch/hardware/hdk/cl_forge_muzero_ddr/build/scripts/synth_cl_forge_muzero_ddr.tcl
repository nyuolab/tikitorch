# synth_cl_forge_muzero_ddr.tcl — CL synth driver for the DDR variant. Mirrors the F2 CL_TEMPLATE flow
# (header/footer from $HDK_SHELL_DIR/build/scripts) but ALSO pulls in the sh_ddr DDR controller module +
# its DDR4 IP (the CL instantiates sh_ddr with DDR_PRESENT=1). The exact sh_ddr source/IP wiring is the one
# thing the reference cl_dram_dma build sets up; this driver reads sh_ddr.sv from the shell design dir and
# the two CL IP. If the build needs the DDR4 IP generated (sh_ddr's internal MIG), the build runner injects
# the HDK DDR-IP step (see run_muzero_ddr_afi_build.sh) — confirmed against the reference DDR build.
source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

# CL top + kernel RTL (+ shell .inc), staged into the post-encryption source dir by encrypt.tcl
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]

# the sh_ddr DDR controller module (synthesizable source shipped with the shell)
read_verilog -sv [glob ${HDK_SHELL_DIR}/design/sh_ddr/synth/*.sv]

# our IP: OCL AXI4-Lite slice + the 2->1 DDR crossbar
read_ip [glob $CL_DIR/design/ip/axi_register_slice_light/axi_register_slice_light.xci]
read_ip [glob $CL_DIR/design/ip/axi_crossbar_ddr/axi_crossbar_ddr.xci]
synth_ip [get_ips axi_register_slice_light]
synth_ip [get_ips axi_crossbar_ddr]

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define XSDB_SLV_DIS \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
