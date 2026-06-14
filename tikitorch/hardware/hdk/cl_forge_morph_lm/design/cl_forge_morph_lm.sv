// =============================================================================
// cl_forge_morph_lm.sv — single-kernel MorphSSM-LM training CL with an
// ON-CL BUFFER RAM (v2 architecture).
//
// The bridge stages the flat int32 training buffer over BAR4/DMA-PCIS at
// 0x0100_0000 and the kernel masters the SAME buffer through its m_axi —
// in v1 the kernel pointed at PCIM (host memory) while BAR4 had no backing
// store (measured: dma-pcis timeouts + kernel reading unrelated host
// memory). v2 backs the window with a 2 MB URAM, dual-ported:
//
//   sh_cl_dma_pcis (512b AXI4, ID16) -> axi_bram_ctrl DMA  -> URAM port A
//   kernel m_axi gmem (512b AXI4)    -> axi_bram_ctrl KRN  -> URAM port B
//
// Both initiators address 0x0100_0000 + offset; the CL masks to the low
// 21 bits (2 MB window >= ML_BUFFER_WORDS*4 = ~1.74 MB). PCIM is unused.
// OCL AXI-Lite -> register slice -> kernel s_axilite (2026 F2 HDK port
// names: ocl_cl_* in / cl_ocl_* out — the renamed family; May-era names
// synthesize as floating implicit nets).
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
`include "unused_pcim_template.inc"

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
  // Kernel: s_axilite from OCL; m_axi gmem into BRAM controller KRN.
  // -------------------------------------------------------------------------
  logic kernel_interrupt;

  logic [63:0]  krn_awaddr,  krn_araddr;
  logic  [7:0]  krn_awlen,   krn_arlen;
  logic  [2:0]  krn_awsize,  krn_arsize;
  logic  [1:0]  krn_awburst, krn_arburst;
  logic  [3:0]  krn_awcache, krn_arcache;
  logic  [2:0]  krn_awprot,  krn_arprot;
  logic  [3:0]  krn_awqos,   krn_arqos;
  logic  [1:0]  krn_awlock,  krn_arlock;
  logic         krn_awid,    krn_arid;
  logic         krn_bid,     krn_rid;
  logic         krn_awvalid, krn_awready, krn_arvalid, krn_arready;
  logic [511:0] krn_wdata,   krn_rdata;
  logic  [63:0] krn_wstrb;
  logic         krn_wlast,   krn_wvalid, krn_wready;
  logic  [1:0]  krn_bresp,   krn_rresp;
  logic         krn_bvalid,  krn_bready;
  logic         krn_rlast,   krn_rvalid, krn_rready;

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

    .m_axi_gmem_AWADDR  (krn_awaddr),
    .m_axi_gmem_AWLEN   (krn_awlen),
    .m_axi_gmem_AWSIZE  (krn_awsize),
    .m_axi_gmem_AWBURST (krn_awburst),
    .m_axi_gmem_AWCACHE (krn_awcache),
    .m_axi_gmem_AWPROT  (krn_awprot),
    .m_axi_gmem_AWQOS   (krn_awqos),
    .m_axi_gmem_AWLOCK  (krn_awlock),
    .m_axi_gmem_AWID    (krn_awid),
    .m_axi_gmem_AWUSER  (),
    .m_axi_gmem_AWREGION(),
    .m_axi_gmem_AWVALID (krn_awvalid),
    .m_axi_gmem_AWREADY (krn_awready),
    .m_axi_gmem_WDATA   (krn_wdata),
    .m_axi_gmem_WSTRB   (krn_wstrb),
    .m_axi_gmem_WLAST   (krn_wlast),
    .m_axi_gmem_WID     (),
    .m_axi_gmem_WUSER   (),
    .m_axi_gmem_WVALID  (krn_wvalid),
    .m_axi_gmem_WREADY  (krn_wready),
    .m_axi_gmem_BRESP   (krn_bresp),
    .m_axi_gmem_BID     (krn_bid),
    .m_axi_gmem_BUSER   (1'b0),
    .m_axi_gmem_BVALID  (krn_bvalid),
    .m_axi_gmem_BREADY  (krn_bready),
    .m_axi_gmem_ARADDR  (krn_araddr),
    .m_axi_gmem_ARLEN   (krn_arlen),
    .m_axi_gmem_ARSIZE  (krn_arsize),
    .m_axi_gmem_ARBURST (krn_arburst),
    .m_axi_gmem_ARCACHE (krn_arcache),
    .m_axi_gmem_ARPROT  (krn_arprot),
    .m_axi_gmem_ARQOS   (krn_arqos),
    .m_axi_gmem_ARLOCK  (krn_arlock),
    .m_axi_gmem_ARID    (krn_arid),
    .m_axi_gmem_ARUSER  (),
    .m_axi_gmem_ARREGION(),
    .m_axi_gmem_ARVALID (krn_arvalid),
    .m_axi_gmem_ARREADY (krn_arready),
    .m_axi_gmem_RDATA   (krn_rdata),
    .m_axi_gmem_RRESP   (krn_rresp),
    .m_axi_gmem_RLAST   (krn_rlast),
    .m_axi_gmem_RID     (krn_rid),
    .m_axi_gmem_RUSER   (1'b0),
    .m_axi_gmem_RVALID  (krn_rvalid),
    .m_axi_gmem_RREADY  (krn_rready)
  );

  // -------------------------------------------------------------------------
  // 2 MB buffer RAM (URAM), dual-ported between two AXI BRAM controllers.
  // Both initiators address 0x0100_0000+x; the CL masks to [20:0].
  // -------------------------------------------------------------------------
  logic        bram_a_clk, bram_a_rst, bram_a_en;
  logic [63:0] bram_a_we;
  logic [20:0] bram_a_addr;
  logic [511:0] bram_a_wrdata, bram_a_rddata;
  logic        bram_b_clk, bram_b_rst, bram_b_en;
  logic [63:0] bram_b_we;
  logic [20:0] bram_b_addr;
  logic [511:0] bram_b_wrdata, bram_b_rddata;

  axi_bram_ctrl_dma BRAM_CTL_DMA (
    .s_axi_aclk    (clk_main_a0),
    .s_axi_aresetn (rst_main_n),
    .s_axi_awid    (sh_cl_dma_pcis_awid),
    .s_axi_awaddr  (sh_cl_dma_pcis_awaddr[20:0]),
    .s_axi_awlen   (sh_cl_dma_pcis_awlen),
    .s_axi_awsize  (sh_cl_dma_pcis_awsize),
    .s_axi_awburst (sh_cl_dma_pcis_awburst),
    .s_axi_awlock  (sh_cl_dma_pcis_awlock),
    .s_axi_awcache (sh_cl_dma_pcis_awcache),
    .s_axi_awprot  (sh_cl_dma_pcis_awprot),
    .s_axi_awvalid (sh_cl_dma_pcis_awvalid),
    .s_axi_awready (cl_sh_dma_pcis_awready),
    .s_axi_wdata   (sh_cl_dma_pcis_wdata),
    .s_axi_wstrb   (sh_cl_dma_pcis_wstrb),
    .s_axi_wlast   (sh_cl_dma_pcis_wlast),
    .s_axi_wvalid  (sh_cl_dma_pcis_wvalid),
    .s_axi_wready  (cl_sh_dma_pcis_wready),
    .s_axi_bid     (cl_sh_dma_pcis_bid),
    .s_axi_bresp   (cl_sh_dma_pcis_bresp),
    .s_axi_bvalid  (cl_sh_dma_pcis_bvalid),
    .s_axi_bready  (sh_cl_dma_pcis_bready),
    .s_axi_arid    (sh_cl_dma_pcis_arid),
    .s_axi_araddr  (sh_cl_dma_pcis_araddr[20:0]),
    .s_axi_arlen   (sh_cl_dma_pcis_arlen),
    .s_axi_arsize  (sh_cl_dma_pcis_arsize),
    .s_axi_arburst (sh_cl_dma_pcis_arburst),
    .s_axi_arlock  (sh_cl_dma_pcis_arlock),
    .s_axi_arcache (sh_cl_dma_pcis_arcache),
    .s_axi_arprot  (sh_cl_dma_pcis_arprot),
    .s_axi_arvalid (sh_cl_dma_pcis_arvalid),
    .s_axi_arready (cl_sh_dma_pcis_arready),
    .s_axi_rid     (cl_sh_dma_pcis_rid),
    .s_axi_rdata   (cl_sh_dma_pcis_rdata),
    .s_axi_rresp   (cl_sh_dma_pcis_rresp),
    .s_axi_rlast   (cl_sh_dma_pcis_rlast),
    .s_axi_rvalid  (cl_sh_dma_pcis_rvalid),
    .s_axi_rready  (sh_cl_dma_pcis_rready),
    .bram_rst_a    (bram_a_rst),
    .bram_clk_a    (bram_a_clk),
    .bram_en_a     (bram_a_en),
    .bram_we_a     (bram_a_we),
    .bram_addr_a   (bram_a_addr),
    .bram_wrdata_a (bram_a_wrdata),
    .bram_rddata_a (bram_a_rddata)
  );

  axi_bram_ctrl_krn BRAM_CTL_KRN (
    .s_axi_aclk    (clk_main_a0),
    .s_axi_aresetn (rst_main_n),
    .s_axi_awid    (krn_awid),
    .s_axi_awaddr  (krn_awaddr[20:0]),
    .s_axi_awlen   (krn_awlen),
    .s_axi_awsize  (krn_awsize),
    .s_axi_awburst (krn_awburst),
    .s_axi_awlock  (krn_awlock[0]),
    .s_axi_awcache (krn_awcache),
    .s_axi_awprot  (krn_awprot),
    .s_axi_awvalid (krn_awvalid),
    .s_axi_awready (krn_awready),
    .s_axi_wdata   (krn_wdata),
    .s_axi_wstrb   (krn_wstrb),
    .s_axi_wlast   (krn_wlast),
    .s_axi_wvalid  (krn_wvalid),
    .s_axi_wready  (krn_wready),
    .s_axi_bid     (krn_bid),
    .s_axi_bresp   (krn_bresp),
    .s_axi_bvalid  (krn_bvalid),
    .s_axi_bready  (krn_bready),
    .s_axi_arid    (krn_arid),
    .s_axi_araddr  (krn_araddr[20:0]),
    .s_axi_arlen   (krn_arlen),
    .s_axi_arsize  (krn_arsize),
    .s_axi_arburst (krn_arburst),
    .s_axi_arlock  (krn_arlock[0]),
    .s_axi_arcache (krn_arcache),
    .s_axi_arprot  (krn_arprot),
    .s_axi_arvalid (krn_arvalid),
    .s_axi_arready (krn_arready),
    .s_axi_rid     (krn_rid),
    .s_axi_rdata   (krn_rdata),
    .s_axi_rresp   (krn_rresp),
    .s_axi_rlast   (krn_rlast),
    .s_axi_rvalid  (krn_rvalid),
    .s_axi_rready  (krn_rready),
    .bram_rst_a    (bram_b_rst),
    .bram_clk_a    (bram_b_clk),
    .bram_en_a     (bram_b_en),
    .bram_we_a     (bram_b_we),
    .bram_addr_a   (bram_b_addr),
    .bram_wrdata_a (bram_b_wrdata),
    .bram_rddata_a (bram_b_rddata)
  );

  // True-dual-port URAM, 512b x 32768 (2 MB); byte write enables; 1-cycle
  // read latency (the axi_bram_ctrl expectation). bram_addr_* are BYTE
  // addresses from the controllers -> word index = addr[20:6].
  xpm_memory_tdpram #(
    .MEMORY_SIZE        (16777216),     // bits = 2 MB
    .MEMORY_PRIMITIVE   ("ultra"),
    .CLOCKING_MODE      ("common_clock"),
    .WRITE_DATA_WIDTH_A (512),
    .READ_DATA_WIDTH_A  (512),
    .BYTE_WRITE_WIDTH_A (8),
    .ADDR_WIDTH_A       (15),
    .READ_LATENCY_A     (1),
    .WRITE_MODE_A       ("no_change"),
    .WRITE_DATA_WIDTH_B (512),
    .READ_DATA_WIDTH_B  (512),
    .BYTE_WRITE_WIDTH_B (8),
    .ADDR_WIDTH_B       (15),
    .READ_LATENCY_B     (1),
    .WRITE_MODE_B       ("no_change")
  ) BUF_URAM (
    .clka   (clk_main_a0),
    .rsta   (bram_a_rst),
    .ena    (bram_a_en),
    .wea    (bram_a_we),
    .addra  (bram_a_addr[20:6]),
    .dina   (bram_a_wrdata),
    .douta  (bram_a_rddata),
    .regcea (1'b1),
    .injectsbiterra (1'b0),
    .injectdbiterra (1'b0),
    .sbiterra (),
    .dbiterra (),
    .clkb   (clk_main_a0),
    .rstb   (bram_b_rst),
    .enb    (bram_b_en),
    .web    (bram_b_we),
    .addrb  (bram_b_addr[20:6]),
    .dinb   (bram_b_wrdata),
    .doutb  (bram_b_rddata),
    .regceb (1'b1),
    .injectsbiterrb (1'b0),
    .injectdbiterrb (1'b0),
    .sbiterrb (),
    .dbiterrb (),
    .sleep  (1'b0)
  );

endmodule
