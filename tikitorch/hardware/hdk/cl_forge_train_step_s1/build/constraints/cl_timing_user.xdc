# cl_timing_user.xdc — S1 trainer CL user timing constraints.
#
# The kernel runs on the AWS_CLK_GEN-generated clk_extra_a1 (125 MHz); the shell interfaces (OCL/DMA-PCIS/
# sh_ddr/crossbar) run on clk_main_a0 (250 MHz). The two domains are ASYNCHRONOUS and cross ONLY through the
# axi_clock_converter CDC IPs (s1_axil_cdc, s1_axi_cdc), which carry their own internal CDC synchronizers.
# Without an explicit clock-group declaration Vivado TIMES every clk_main_a0<->clk_extra_a1 crossing (255 such
# references in the v9 report) as if synchronous, wasting router effort and reporting false inter-clock paths.
# Declaring the groups asynchronous tells Vivado these crossings are handled by the CDC IPs, so it stops timing
# them and focuses on the real single-clock paths (the v9 worst path was a congestion-starved 250 MHz sh_ddr
# DDR-controller async FIFO at WNS -0.409 ns; freeing the phantom crossings lets P&R close it).
set_clock_groups -asynchronous \
  -group [get_clocks -include_generated_clocks clk_main_a0] \
  -group [get_clocks -include_generated_clocks clk_extra_a1]
