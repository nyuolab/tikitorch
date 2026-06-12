// =============================================================================
// cl_forge_morph_lm.sv — single-kernel training CL.
//
// Wires the monolithic `forge_train_step_morph_lm` HLS kernel's m_axi master directly
// to the AWS shell's PCIM port. This is the EXACT same wiring as
// cl_forge_signsgd.sv (silicon-validated v6 path) — the only difference is
// the kernel module name.
//
// Kernel does fwd_loss / bwd / signsgd dispatched by a `mode` register arg.
// Host writes (mode, arg1, arg2, arg3) to OCL register window then kicks
// AP_START to run one of the three ops.
//
// File rationale: F2 silicon testing of the multi-kernel CL (cl_forge_train_4kernels)
// showed that BOTH custom Nto1 mux AND chained PoC bridges drop kernel writes.
// Only the single-kernel→PCIM path is silicon-proven. This CL uses that path.
// =============================================================================

`include "cl_forge_morph_lm_defines.vh"

module cl_forge_morph_lm

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
`include "unused_ddr_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_apppf_irq_template.inc"
`include "unused_dma_pcis_template.inc"

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_status0 = 32'h0;
  assign cl_sh_status1 = 32'h0;
  assign cl_sh_status2 = 32'h0;
  assign cl_sh_status_vled = 16'h0;
  assign cl_sh_dma_wr_full = 1'b0;
  assign cl_sh_dma_rd_full = 1'b0;

  // -------------------------------------------------------------------------
  // OCL register slice (timing closure between shell and our s_axilite)
  // -------------------------------------------------------------------------
  logic        ocl_aw_valid_pre, ocl_aw_ready_pre;
  logic [31:0] ocl_aw_addr_pre;
  logic        ocl_w_valid_pre, ocl_w_ready_pre;
  logic [31:0] ocl_w_data_pre;
  logic  [3:0] ocl_w_strb_pre;
  logic        ocl_b_valid_pre, ocl_b_ready_pre;
  logic  [1:0] ocl_b_resp_pre;
  logic        ocl_ar_valid_pre, ocl_ar_ready_pre;
  logic [31:0] ocl_ar_addr_pre;
  logic        ocl_r_valid_pre, ocl_r_ready_pre;
  logic [31:0] ocl_r_data_pre;
  logic  [1:0] ocl_r_resp_pre;

  axi_register_slice_light OCL_REG_SLC (
    .aclk          (clk_main_a0),
    .aresetn       (rst_main_n),
    .s_axi_awaddr  (ocl_cl_awaddr),
    .s_axi_awprot  (3'h0),
    .s_axi_awvalid (ocl_cl_awvalid),
    .s_axi_awready (cl_ocl_awready),
    .s_axi_wdata   (ocl_cl_wdata),
    .s_axi_wstrb   (ocl_cl_wstrb),
    .s_axi_wvalid  (ocl_cl_wvalid),
    .s_axi_wready  (cl_ocl_wready),
    .s_axi_bresp   (cl_ocl_bresp),
    .s_axi_bvalid  (cl_ocl_bvalid),
    .s_axi_bready  (ocl_cl_bready),
    .s_axi_araddr  (ocl_cl_araddr),
    .s_axi_arvalid (ocl_cl_arvalid),
    .s_axi_arready (cl_ocl_arready),
    .s_axi_rdata   (cl_ocl_rdata),
    .s_axi_rresp   (cl_ocl_rresp),
    .s_axi_rvalid  (cl_ocl_rvalid),
    .s_axi_rready  (ocl_cl_rready),

    .m_axi_awaddr  (ocl_aw_addr_pre),
    .m_axi_awprot  (),
    .m_axi_awvalid (ocl_aw_valid_pre),
    .m_axi_awready (ocl_aw_ready_pre),
    .m_axi_wdata   (ocl_w_data_pre),
    .m_axi_wstrb   (ocl_w_strb_pre),
    .m_axi_wvalid  (ocl_w_valid_pre),
    .m_axi_wready  (ocl_w_ready_pre),
    .m_axi_bresp   (ocl_b_resp_pre),
    .m_axi_bvalid  (ocl_b_valid_pre),
    .m_axi_bready  (ocl_b_ready_pre),
    .m_axi_araddr  (ocl_ar_addr_pre),
    .m_axi_arvalid (ocl_ar_valid_pre),
    .m_axi_arready (ocl_ar_ready_pre),
    .m_axi_rdata   (ocl_r_data_pre),
    .m_axi_rresp   (ocl_r_resp_pre),
    .m_axi_rvalid  (ocl_r_valid_pre),
    .m_axi_rready  (ocl_r_ready_pre)
  );

  // -------------------------------------------------------------------------
  // PCIM master tie-offs
  // -------------------------------------------------------------------------
  assign cl_sh_pcim_awid    = 16'h0;
  assign cl_sh_pcim_awsize  = 3'h6;       // 64 bytes per beat (512-bit data)
  assign cl_sh_pcim_awburst = 2'b01;      // INCR
  assign cl_sh_pcim_awcache = 4'h3;
  assign cl_sh_pcim_awlock  = 1'b0;
  assign cl_sh_pcim_awprot  = 3'h0;
  assign cl_sh_pcim_awqos   = 4'h0;
  assign cl_sh_pcim_awuser  = 55'h0;
  assign cl_sh_pcim_arid    = 16'h0;
  assign cl_sh_pcim_arsize  = 3'h6;
  assign cl_sh_pcim_arburst = 2'b01;
  assign cl_sh_pcim_arcache = 4'h3;
  assign cl_sh_pcim_arlock  = 1'b0;
  assign cl_sh_pcim_arprot  = 3'h0;
  assign cl_sh_pcim_arqos   = 4'h0;
  assign cl_sh_pcim_aruser  = 55'h0;
  assign cl_sh_pcim_wuser   = 64'h0;
  assign cl_sh_pcim_wid     = 16'h0;

  // -------------------------------------------------------------------------
  // Monolithic training kernel: forge_train_step
  // s_axilite control window is 4 KB (matches signsgd v6 width).
  // m_axi master is 512-bit wide (max_widen_bitwidth=512 pragma in HLS).
  // -------------------------------------------------------------------------
  logic kernel_interrupt;

  forge_train_step_morph_lm u_train (
    .ap_clk     (clk_main_a0),
    .ap_rst_n   (rst_main_n),
    .interrupt  (kernel_interrupt),

    .s_axi_control_AWADDR  (ocl_aw_addr_pre[12:0]),
    .s_axi_control_AWVALID (ocl_aw_valid_pre),
    .s_axi_control_AWREADY (ocl_aw_ready_pre),
    .s_axi_control_WDATA   (ocl_w_data_pre),
    .s_axi_control_WSTRB   (ocl_w_strb_pre),
    .s_axi_control_WVALID  (ocl_w_valid_pre),
    .s_axi_control_WREADY  (ocl_w_ready_pre),
    .s_axi_control_BRESP   (ocl_b_resp_pre),
    .s_axi_control_BVALID  (ocl_b_valid_pre),
    .s_axi_control_BREADY  (ocl_b_ready_pre),
    .s_axi_control_ARADDR  (ocl_ar_addr_pre[12:0]),
    .s_axi_control_ARVALID (ocl_ar_valid_pre),
    .s_axi_control_ARREADY (ocl_ar_ready_pre),
    .s_axi_control_RDATA   (ocl_r_data_pre),
    .s_axi_control_RRESP   (ocl_r_resp_pre),
    .s_axi_control_RVALID  (ocl_r_valid_pre),
    .s_axi_control_RREADY  (ocl_r_ready_pre),

    .m_axi_gmem_AWADDR  (cl_sh_pcim_awaddr),
    .m_axi_gmem_AWLEN   (cl_sh_pcim_awlen),
    .m_axi_gmem_AWVALID (cl_sh_pcim_awvalid),
    .m_axi_gmem_AWREADY (sh_cl_pcim_awready),
    .m_axi_gmem_WDATA   (cl_sh_pcim_wdata),
    .m_axi_gmem_WSTRB   (cl_sh_pcim_wstrb),
    .m_axi_gmem_WLAST   (cl_sh_pcim_wlast),
    .m_axi_gmem_WVALID  (cl_sh_pcim_wvalid),
    .m_axi_gmem_WREADY  (sh_cl_pcim_wready),
    .m_axi_gmem_BRESP   (sh_cl_pcim_bresp),
    .m_axi_gmem_BVALID  (sh_cl_pcim_bvalid),
    .m_axi_gmem_BREADY  (cl_sh_pcim_bready),
    .m_axi_gmem_ARADDR  (cl_sh_pcim_araddr),
    .m_axi_gmem_ARLEN   (cl_sh_pcim_arlen),
    .m_axi_gmem_ARVALID (cl_sh_pcim_arvalid),
    .m_axi_gmem_ARREADY (sh_cl_pcim_arready),
    .m_axi_gmem_RDATA   (sh_cl_pcim_rdata),
    .m_axi_gmem_RRESP   (sh_cl_pcim_rresp),
    .m_axi_gmem_RLAST   (sh_cl_pcim_rlast),
    .m_axi_gmem_RVALID  (sh_cl_pcim_rvalid),
    .m_axi_gmem_RREADY  (cl_sh_pcim_rready)
  );

endmodule
