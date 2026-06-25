// =============================================================================
// cl_forge_pc_xformer.sv — DDR CL for the distributed PC TRANSFORMER-BLOCK kernel (forge_pc_xformer_block):
// one kick = forward + cascade backward + SignSGD-momentum update; weights+momentum persist in on-card DDR.
//
// Deploys the DSP=0, bit-exact PC cascade backward layer-step (routed adjoint + weight grad + bias grad
// + error route + residual skip highway; results/predictive_coding/PC_BACKWARD_SILICON.md) on F2 silicon
// so the gradient produced ON the chip can be compared to tikitorch's cascade_grads. Same drivable
// on-card-DDR architecture as the proven cl_forge_muzero_ddr (host stages the I/O buffer into DDR over
// DMA-PCIS, the kernel reads/writes it via m_axi — both reach DDR through a 2->1 crossbar; no host IOMMU):
//
//   ocl  (AXI4-Lite) --[reg slice]--> kernel s_axi_control  (forge_pc_wgrad: 5-bit control map)
//   sh_cl_dma_pcis (512b, host staging) --\
//                                          >--[axi_crossbar 2->1]--> sh_ddr cl_sh_ddr_axi --> on-card DDR4
//   kernel m_axi gmem (512b, reads+writes) --/
//
// Buffer at DDR addr 0 holds [e | W | acts | relu_mask | skip_gate | gW | e_down | gB]: the host DMA-PCIS-
// writes the inputs, kicks the kernel (BUFFER ptr reg = 0), the kernel reads inputs + writes gW/e_down/gB
// back to DDR, the host reads them out. The kernel m_axi gmem is 512b (wide burst-stage; see the kernel's
// burst_load/burst_store) so it wires straight to the proven 512b crossbar — no width converter. PCIM is
// unused (tied off); DMA-PCIS + DDR are used.
//
// Reuses the proven cl_forge_muzero_ddr wiring for OCL control + reset sync + crossbar + sh_ddr verbatim;
// only the kernel instance (forge_pc_wgrad) and its 5-bit control slice differ.
// =============================================================================

`include "cl_forge_pc_xformer_defines.vh"

module cl_forge_pc_xformer

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_apppf_irq_template.inc"
`include "unused_pcim_template.inc"     // PCIM unused in the DDR variant (kernel reads on-card DDR)
// DDR + DMA-PCIS are USED below (not tied off).

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_status0 = 32'h0;
  assign cl_sh_status1 = 32'h0;
  assign cl_sh_status2 = 32'h0;
  assign cl_sh_status_vled = 16'h0;
  assign cl_sh_dma_wr_full = 1'b0;
  assign cl_sh_dma_rd_full = 1'b0;

  // ---- reset synchronizers (async assert / sync deassert) ----
  (* ASYNC_REG = "true" *) logic [3:0] rst_pipe;
  always_ff @(posedge clk_main_a0 or negedge rst_main_n)
    if (!rst_main_n) rst_pipe <= 4'b0;
    else             rst_pipe <= {rst_pipe[2:0], 1'b1};
  wire rst_n_local      = rst_pipe[3];
  wire rst_main_n_sync  = rst_pipe[3];   // name expected by the sh_ddr template

  // =========================================================================
  // OCL register slice -> kernel s_axi_control (proven; single kernel, no fan-out)
  // =========================================================================
  logic        ocl_aw_valid_pre, ocl_aw_ready_pre;  logic [31:0] ocl_aw_addr_pre;
  logic        ocl_w_valid_pre, ocl_w_ready_pre;    logic [31:0] ocl_w_data_pre;  logic [3:0] ocl_w_strb_pre;
  logic        ocl_b_valid_pre, ocl_b_ready_pre;    logic  [1:0] ocl_b_resp_pre;
  logic        ocl_ar_valid_pre, ocl_ar_ready_pre;  logic [31:0] ocl_ar_addr_pre;
  logic        ocl_r_valid_pre, ocl_r_ready_pre;    logic [31:0] ocl_r_data_pre;  logic [1:0] ocl_r_resp_pre;

  pcblock_ocl_slice OCL_REG_SLC (
    .aclk          (clk_main_a0),       .aresetn       (rst_main_n),
    .s_axi_awaddr  (ocl_cl_awaddr),     .s_axi_awprot  (3'h0),
    .s_axi_awvalid (ocl_cl_awvalid),    .s_axi_awready (cl_ocl_awready),
    .s_axi_wdata   (ocl_cl_wdata),      .s_axi_wstrb   (ocl_cl_wstrb),
    .s_axi_wvalid  (ocl_cl_wvalid),     .s_axi_wready  (cl_ocl_wready),
    .s_axi_bresp   (cl_ocl_bresp),      .s_axi_bvalid  (cl_ocl_bvalid),     .s_axi_bready (ocl_cl_bready),
    .s_axi_araddr  (ocl_cl_araddr),     .s_axi_arvalid (ocl_cl_arvalid),    .s_axi_arready(cl_ocl_arready),
    .s_axi_rdata   (cl_ocl_rdata),      .s_axi_rresp   (cl_ocl_rresp),
    .s_axi_rvalid  (cl_ocl_rvalid),     .s_axi_rready  (ocl_cl_rready),
    .m_axi_awaddr  (ocl_aw_addr_pre),   .m_axi_awprot  (),
    .m_axi_awvalid (ocl_aw_valid_pre),  .m_axi_awready (ocl_aw_ready_pre),
    .m_axi_wdata   (ocl_w_data_pre),    .m_axi_wstrb   (ocl_w_strb_pre),
    .m_axi_wvalid  (ocl_w_valid_pre),   .m_axi_wready  (ocl_w_ready_pre),
    .m_axi_bresp   (ocl_b_resp_pre),    .m_axi_bvalid  (ocl_b_valid_pre),  .m_axi_bready (ocl_b_ready_pre),
    .m_axi_araddr  (ocl_ar_addr_pre),   .m_axi_arvalid (ocl_ar_valid_pre), .m_axi_arready(ocl_ar_ready_pre),
    .m_axi_rdata   (ocl_r_data_pre),    .m_axi_rresp   (ocl_r_resp_pre),
    .m_axi_rvalid  (ocl_r_valid_pre),   .m_axi_rready  (ocl_r_ready_pre)
  );

  // =========================================================================
  // Kernel m_axi (gmem) nets -> crossbar SI0. 1-bit ID (single outstanding).
  // =========================================================================
  logic [63:0]  krn_awaddr,  krn_araddr;   logic [7:0] krn_awlen,  krn_arlen;
  logic  [2:0]  krn_awsize,  krn_arsize;    logic [1:0] krn_awburst, krn_arburst;
  logic  [3:0]  krn_awcache, krn_arcache;   logic [2:0] krn_awprot,  krn_arprot;
  logic  [3:0]  krn_awqos,   krn_arqos;     logic [1:0] krn_awlock,  krn_arlock;
  logic         krn_awid,    krn_arid,    krn_bid,     krn_rid;
  logic         krn_awvalid, krn_awready, krn_arvalid, krn_arready;
  logic [511:0] krn_wdata,   krn_rdata;     logic [63:0] krn_wstrb;
  logic         krn_wlast,   krn_wvalid,  krn_wready;
  logic  [1:0]  krn_bresp,   krn_rresp;
  logic         krn_bvalid,  krn_bready,  krn_rlast, krn_rvalid, krn_rready, krn_irq;

  forge_pc_xformer_block u_kernel (
    .ap_clk     (clk_main_a0),  .ap_rst_n (rst_n_local),  .interrupt (krn_irq),
    .s_axi_control_AWADDR  (ocl_aw_addr_pre[4:0]), .s_axi_control_AWVALID (ocl_aw_valid_pre), .s_axi_control_AWREADY (ocl_aw_ready_pre),
    .s_axi_control_WDATA   (ocl_w_data_pre),  .s_axi_control_WSTRB (ocl_w_strb_pre), .s_axi_control_WVALID (ocl_w_valid_pre), .s_axi_control_WREADY (ocl_w_ready_pre),
    .s_axi_control_BRESP   (ocl_b_resp_pre),  .s_axi_control_BVALID (ocl_b_valid_pre), .s_axi_control_BREADY (ocl_b_ready_pre),
    .s_axi_control_ARADDR  (ocl_ar_addr_pre[4:0]), .s_axi_control_ARVALID (ocl_ar_valid_pre), .s_axi_control_ARREADY (ocl_ar_ready_pre),
    .s_axi_control_RDATA   (ocl_r_data_pre),  .s_axi_control_RRESP (ocl_r_resp_pre), .s_axi_control_RVALID (ocl_r_valid_pre), .s_axi_control_RREADY (ocl_r_ready_pre),
    .m_axi_gmem_AWADDR (krn_awaddr), .m_axi_gmem_AWLEN (krn_awlen), .m_axi_gmem_AWSIZE (krn_awsize), .m_axi_gmem_AWBURST (krn_awburst),
    .m_axi_gmem_AWCACHE (krn_awcache), .m_axi_gmem_AWPROT (krn_awprot), .m_axi_gmem_AWQOS (krn_awqos), .m_axi_gmem_AWLOCK (krn_awlock),
    .m_axi_gmem_AWID (krn_awid), .m_axi_gmem_AWUSER (), .m_axi_gmem_AWREGION (), .m_axi_gmem_AWVALID (krn_awvalid), .m_axi_gmem_AWREADY (krn_awready),
    .m_axi_gmem_WDATA (krn_wdata), .m_axi_gmem_WSTRB (krn_wstrb), .m_axi_gmem_WLAST (krn_wlast), .m_axi_gmem_WID (), .m_axi_gmem_WUSER (),
    .m_axi_gmem_WVALID (krn_wvalid), .m_axi_gmem_WREADY (krn_wready),
    .m_axi_gmem_BRESP (krn_bresp), .m_axi_gmem_BID (krn_bid), .m_axi_gmem_BUSER (1'b0), .m_axi_gmem_BVALID (krn_bvalid), .m_axi_gmem_BREADY (krn_bready),
    .m_axi_gmem_ARADDR (krn_araddr), .m_axi_gmem_ARLEN (krn_arlen), .m_axi_gmem_ARSIZE (krn_arsize), .m_axi_gmem_ARBURST (krn_arburst),
    .m_axi_gmem_ARCACHE (krn_arcache), .m_axi_gmem_ARPROT (krn_arprot), .m_axi_gmem_ARQOS (krn_arqos), .m_axi_gmem_ARLOCK (krn_arlock),
    .m_axi_gmem_ARID (krn_arid), .m_axi_gmem_ARUSER (), .m_axi_gmem_ARREGION (), .m_axi_gmem_ARVALID (krn_arvalid), .m_axi_gmem_ARREADY (krn_arready),
    .m_axi_gmem_RDATA (krn_rdata), .m_axi_gmem_RRESP (krn_rresp), .m_axi_gmem_RLAST (krn_rlast), .m_axi_gmem_RID (krn_rid), .m_axi_gmem_RUSER (1'b0),
    .m_axi_gmem_RVALID (krn_rvalid), .m_axi_gmem_RREADY (krn_rready)
  );

  // =========================================================================
  // 2->1 AXI crossbar: SI0 = kernel m_axi (ID widened 1->16), SI1 = sh_cl_dma_pcis (16-bit ID),
  // MI = the DDR AXI slave. Both masters address the single DDR region (base 0). axi_crossbar_ddr
  // (gen_ip.tcl): NUM_SI=2, NUM_MI=1, 512b, 64-bit addr, 16-bit ID.
  // =========================================================================
  // DDR-side (crossbar master output -> sh_ddr cl_sh_ddr_axi)
  logic [15:0]  ddr_awid;   logic [63:0] ddr_awaddr;  logic [7:0] ddr_awlen;  logic [2:0] ddr_awsize;
  logic  [1:0]  ddr_awburst; logic ddr_awlock; logic [3:0] ddr_awcache; logic [2:0] ddr_awprot; logic [3:0] ddr_awqos; logic [3:0] ddr_awregion;
  logic         ddr_awvalid, ddr_awready;
  logic [511:0] ddr_wdata;  logic [63:0] ddr_wstrb;  logic ddr_wlast, ddr_wvalid, ddr_wready;
  logic [15:0]  ddr_bid;    logic [1:0] ddr_bresp;   logic ddr_bvalid, ddr_bready;
  logic [15:0]  ddr_arid;   logic [63:0] ddr_araddr;  logic [7:0] ddr_arlen;  logic [2:0] ddr_arsize;
  logic  [1:0]  ddr_arburst; logic ddr_arlock; logic [3:0] ddr_arcache; logic [2:0] ddr_arprot; logic [3:0] ddr_arqos; logic [3:0] ddr_arregion;
  logic         ddr_arvalid, ddr_arready;
  logic [15:0]  ddr_rid;    logic [511:0] ddr_rdata;  logic [1:0] ddr_rresp; logic ddr_rlast, ddr_rvalid, ddr_rready;
  logic [15:0]  krn_bid_16, krn_rid_16;   // 16-bit kernel-side response IDs from the crossbar (kernel uses bit 0)

  axi_crossbar_ddr XBAR_DDR (
    .aclk    (clk_main_a0),  .aresetn (rst_main_n_sync),
    // ---- SI: kernel m_axi (ID 1->16) high, sh_cl_dma_pcis (16-bit ID) low ----
    .s_axi_awid    ({ {15'h0, krn_awid}, sh_cl_dma_pcis_awid }),
    .s_axi_awaddr  ({ krn_awaddr,        sh_cl_dma_pcis_awaddr }),
    .s_axi_awlen   ({ krn_awlen,         sh_cl_dma_pcis_awlen }),
    .s_axi_awsize  ({ krn_awsize,        sh_cl_dma_pcis_awsize }),
    .s_axi_awburst ({ krn_awburst,       sh_cl_dma_pcis_awburst }),
    .s_axi_awlock  ({ krn_awlock[0],     sh_cl_dma_pcis_awlock }),
    .s_axi_awcache ({ krn_awcache,       sh_cl_dma_pcis_awcache }),
    .s_axi_awprot  ({ krn_awprot,        sh_cl_dma_pcis_awprot }),
    .s_axi_awqos   ({ krn_awqos,         4'h0 }),
    .s_axi_awvalid ({ krn_awvalid,       sh_cl_dma_pcis_awvalid }),
    .s_axi_awready ({ krn_awready,       cl_sh_dma_pcis_awready }),
    .s_axi_wdata   ({ krn_wdata,         sh_cl_dma_pcis_wdata }),
    .s_axi_wstrb   ({ krn_wstrb,         sh_cl_dma_pcis_wstrb }),
    .s_axi_wlast   ({ krn_wlast,         sh_cl_dma_pcis_wlast }),
    .s_axi_wvalid  ({ krn_wvalid,        sh_cl_dma_pcis_wvalid }),
    .s_axi_wready  ({ krn_wready,        cl_sh_dma_pcis_wready }),
    .s_axi_bid     ({ krn_bid_16,        cl_sh_dma_pcis_bid }),
    .s_axi_bresp   ({ krn_bresp,         cl_sh_dma_pcis_bresp }),
    .s_axi_bvalid  ({ krn_bvalid,        cl_sh_dma_pcis_bvalid }),
    .s_axi_bready  ({ krn_bready,        sh_cl_dma_pcis_bready }),
    .s_axi_arid    ({ {15'h0, krn_arid}, sh_cl_dma_pcis_arid }),
    .s_axi_araddr  ({ krn_araddr,        sh_cl_dma_pcis_araddr }),
    .s_axi_arlen   ({ krn_arlen,         sh_cl_dma_pcis_arlen }),
    .s_axi_arsize  ({ krn_arsize,        sh_cl_dma_pcis_arsize }),
    .s_axi_arburst ({ krn_arburst,       sh_cl_dma_pcis_arburst }),
    .s_axi_arlock  ({ krn_arlock[0],     sh_cl_dma_pcis_arlock }),
    .s_axi_arcache ({ krn_arcache,       sh_cl_dma_pcis_arcache }),
    .s_axi_arprot  ({ krn_arprot,        sh_cl_dma_pcis_arprot }),
    .s_axi_arqos   ({ krn_arqos,         4'h0 }),
    .s_axi_arvalid ({ krn_arvalid,       sh_cl_dma_pcis_arvalid }),
    .s_axi_arready ({ krn_arready,       cl_sh_dma_pcis_arready }),
    .s_axi_rid     ({ krn_rid_16,        cl_sh_dma_pcis_rid }),
    .s_axi_rdata   ({ krn_rdata,         cl_sh_dma_pcis_rdata }),
    .s_axi_rresp   ({ krn_rresp,         cl_sh_dma_pcis_rresp }),
    .s_axi_rlast   ({ krn_rlast,         cl_sh_dma_pcis_rlast }),
    .s_axi_rvalid  ({ krn_rvalid,        cl_sh_dma_pcis_rvalid }),
    .s_axi_rready  ({ krn_rready,        sh_cl_dma_pcis_rready }),
    // ---- M00 = DDR ----
    .m_axi_awid (ddr_awid), .m_axi_awaddr (ddr_awaddr), .m_axi_awlen (ddr_awlen), .m_axi_awsize (ddr_awsize),
    .m_axi_awburst (ddr_awburst), .m_axi_awlock (ddr_awlock), .m_axi_awcache (ddr_awcache), .m_axi_awprot (ddr_awprot),
    .m_axi_awregion (ddr_awregion), .m_axi_awqos (ddr_awqos), .m_axi_awvalid (ddr_awvalid), .m_axi_awready (ddr_awready),
    .m_axi_wdata (ddr_wdata), .m_axi_wstrb (ddr_wstrb), .m_axi_wlast (ddr_wlast), .m_axi_wvalid (ddr_wvalid), .m_axi_wready (ddr_wready),
    .m_axi_bid (ddr_bid), .m_axi_bresp (ddr_bresp), .m_axi_bvalid (ddr_bvalid), .m_axi_bready (ddr_bready),
    .m_axi_arid (ddr_arid), .m_axi_araddr (ddr_araddr), .m_axi_arlen (ddr_arlen), .m_axi_arsize (ddr_arsize),
    .m_axi_arburst (ddr_arburst), .m_axi_arlock (ddr_arlock), .m_axi_arcache (ddr_arcache), .m_axi_arprot (ddr_arprot),
    .m_axi_arregion (ddr_arregion), .m_axi_arqos (ddr_arqos), .m_axi_arvalid (ddr_arvalid), .m_axi_arready (ddr_arready),
    .m_axi_rid (ddr_rid), .m_axi_rdata (ddr_rdata), .m_axi_rresp (ddr_rresp), .m_axi_rlast (ddr_rlast), .m_axi_rvalid (ddr_rvalid), .m_axi_rready (ddr_rready)
  );
  // kernel response IDs come back 16-bit from the crossbar; the kernel uses bit 0 (DMA-PCIS bid/rid direct above)
  assign krn_bid = krn_bid_16[0];
  assign krn_rid = krn_rid_16[0];

  // =========================================================================
  // sh_ddr (DDR_PRESENT=1): DDR controller. AXI slave <- crossbar MI; DDR4 pins <- shell; stats.
  // =========================================================================
  logic sh_cl_ddr_is_ready;
  sh_ddr #(.DDR_PRESENT(1)) SH_DDR (
    .clk (clk_main_a0), .rst_n (rst_main_n_sync), .stat_clk (clk_main_a0), .stat_rst_n (rst_main_n_sync),
    .CLK_DIMM_DP (CLK_DIMM_DP), .CLK_DIMM_DN (CLK_DIMM_DN), .M_ACT_N (M_ACT_N), .M_MA (M_MA), .M_BA (M_BA),
    .M_BG (M_BG), .M_CKE (M_CKE), .M_ODT (M_ODT), .M_CS_N (M_CS_N), .M_CLK_DN (M_CLK_DN), .M_CLK_DP (M_CLK_DP),
    .M_PAR (M_PAR), .M_DQ (M_DQ), .M_ECC (M_ECC), .M_DQS_DP (M_DQS_DP), .M_DQS_DN (M_DQS_DN), .cl_RST_DIMM_N (RST_DIMM_N),
    .cl_sh_ddr_axi_awid (ddr_awid), .cl_sh_ddr_axi_awaddr (ddr_awaddr), .cl_sh_ddr_axi_awlen (ddr_awlen),
    .cl_sh_ddr_axi_awsize (ddr_awsize), .cl_sh_ddr_axi_awvalid (ddr_awvalid), .cl_sh_ddr_axi_awburst (ddr_awburst), .cl_sh_ddr_axi_awready (ddr_awready),
    .cl_sh_ddr_axi_wdata (ddr_wdata), .cl_sh_ddr_axi_wstrb (ddr_wstrb), .cl_sh_ddr_axi_wlast (ddr_wlast), .cl_sh_ddr_axi_wvalid (ddr_wvalid), .cl_sh_ddr_axi_wready (ddr_wready),
    .cl_sh_ddr_axi_bid (ddr_bid), .cl_sh_ddr_axi_bresp (ddr_bresp), .cl_sh_ddr_axi_bvalid (ddr_bvalid), .cl_sh_ddr_axi_bready (ddr_bready),
    .cl_sh_ddr_axi_arid (ddr_arid), .cl_sh_ddr_axi_araddr (ddr_araddr), .cl_sh_ddr_axi_arlen (ddr_arlen),
    .cl_sh_ddr_axi_arsize (ddr_arsize), .cl_sh_ddr_axi_arvalid (ddr_arvalid), .cl_sh_ddr_axi_arburst (ddr_arburst), .cl_sh_ddr_axi_arready (ddr_arready),
    .cl_sh_ddr_axi_rid (ddr_rid), .cl_sh_ddr_axi_rdata (ddr_rdata), .cl_sh_ddr_axi_rresp (ddr_rresp), .cl_sh_ddr_axi_rlast (ddr_rlast), .cl_sh_ddr_axi_rvalid (ddr_rvalid), .cl_sh_ddr_axi_rready (ddr_rready),
    .sh_ddr_stat_bus_addr (sh_cl_ddr_stat_addr), .sh_ddr_stat_bus_wdata (sh_cl_ddr_stat_wdata), .sh_ddr_stat_bus_wr (sh_cl_ddr_stat_wr),
    .sh_ddr_stat_bus_rd (sh_cl_ddr_stat_rd), .sh_ddr_stat_bus_ack (cl_sh_ddr_stat_ack), .sh_ddr_stat_bus_rdata (cl_sh_ddr_stat_rdata),
    .ddr_sh_stat_int (cl_sh_ddr_stat_int), .sh_cl_ddr_is_ready (sh_cl_ddr_is_ready)
  );

endmodule
