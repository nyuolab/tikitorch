// =============================================================================
// cl_forge_morph_lm_d512.sv — D=512 SINGLE-KERNEL MorphSSM-LM training CL.
//
// PARALLEL to cl_forge_morph_lm.sv (the DEPLOYED D=128 NK=4 sentinel, agfi-0433...,
// which stays byte-identical). This D=512 variant is the A3 reduced-clock build:
// the D=512 training buffer is 12.25 MB (vs 1.66 MB at D=128), so a single kernel
// needs a 16 MB staging URAM — larger than one SLR's URAM budget — forcing NK 4->1
// and an ALL-SLR floorplan (see cl_pnr_user_floorplan.xdc). Deltas vs the D=128
// package (all from the HDK width-prep report, wf_007d5672):
//   * NK 4 -> 1 (one kernel; the 16 MB URAM cannot be replicated per-SLR)
//   * BUF_URAM 2 MB -> 16 MB (MEMORY_SIZE bits 16777216 -> 134217728,
//       ADDR_WIDTH 15 -> 18, address line-select [20:6] -> [23:6])
//   * BRAM-controller address decode [20:0] -> [23:0] (16 MB window)
//   * axi_bram_ctrl MEM_DEPTH 32768 -> 262144  (gen_ip.tcl)
//   * axi_crossbar_dma 1 SI -> 1 MI, M00 ADDR_WIDTH 21 -> 24, 16 MB window (gen_ip.tcl)
//
//   sh_cl_dma_pcis (512b AXI4, ID16) --[xbar_dma 1->1, 16 MB window]-->
//        axi_bram_ctrl_dma -> URAM port A
//   ocl (AXI4-Lite) --[reg slice]--> kernel s_axilite control
//   kernel m_axi gmem (512b) -> axi_bram_ctrl_krn -> URAM port B
//
// Address map (host-visible), NK=1:
//   DMA-PCIS staging window : 0x0100_0000  (16 MB)
//   OCL control window      : 0x0000_0000  (64 KB)
// The axi_bram_ctrl masks the low 24 bits -> 512b line index = addr[23:6].
// PCIM unused. 2026 F2 HDK port family: ocl_cl_* in / cl_ocl_* out.
// The kernel itself (forge_train_step_morph_lm, -DML_D=512) moves its own
// out-proj/head weight arrays to URAM via #if ML_D>=512 guards — that URAM is
// INTERNAL to the kernel and adds to the per-SLR URAM budget the floorplan must
// satisfy (hence the all-SLR pblock).
// =============================================================================

`include "cl_forge_morph_lm_defines.vh"

module cl_forge_morph_lm_d512

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
`include "unused_ddr_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_apppf_irq_template.inc"
`include "unused_pcim_template.inc"

  // -------------------------------------------------------------------------
  // Replication factor = 1 at D=512. The single 16 MB staging URAM already
  // exceeds one SLR's URAM budget (~320 blocks = 11.25 MB) and, with the
  // kernel's own D=512 weight URAM, spans all three SLRs — so there is no room
  // to replicate. (The D=128 sentinel runs NK=4 at 2 MB/kernel.)
  // MUST match axi_crossbar_dma NUM_MI (=1) in gen_ip.tcl.
  // -------------------------------------------------------------------------
  localparam int NK = 1;

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_status0 = 32'h0;
  assign cl_sh_status1 = 32'h0;
  assign cl_sh_status2 = 32'h0;
  assign cl_sh_status_vled = 16'h0;
  assign cl_sh_dma_wr_full = 1'b0;
  assign cl_sh_dma_rd_full = 1'b0;

  // =========================================================================
  // OCL register slice (timing closure between shell and the OCL fan-out)
  // =========================================================================
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

  // OCL fan-out: per-kernel control-port nets. Writes are broadcast; the
  // write-side handshake is taken from instance 0 (all instances are identical
  // RTL on a common clock/reset and receive identical control writes, so their
  // AW/W/B handshakes are cycle-identical). Reads are muxed by window bits.
  logic [NK-1:0]  k_awready, k_wready, k_bvalid, k_arready, k_rvalid;
  logic  [1:0]    k_bresp [NK];
  logic  [1:0]    k_rresp [NK];
  logic [31:0]    k_rdata [NK];
  logic [NK-1:0]  k_arvalid, k_rready;

  // read-side instance select from the control address window (64 KB stride).
  // NK==1 => $clog2(1)==0 is a zero-width slice (illegal), so pin the select to
  // instance 0; the single-kernel read path needs no window decode.
  localparam int SELW = (NK > 1) ? $clog2(NK) : 1;
  logic [SELW-1:0] rd_sel;
  assign rd_sel = (NK > 1) ? ocl_ar_addr_pre[16 +: SELW] : '0;

  always_comb begin
    integer kk;
    // write-side handshake from instance 0 (broadcast => identical)
    ocl_aw_ready_pre = k_awready[0];
    ocl_w_ready_pre  = k_wready[0];
    ocl_b_valid_pre  = k_bvalid[0];
    ocl_b_resp_pre   = k_bresp[0];
    // read-side mux from the selected instance
    ocl_ar_ready_pre = k_arready[rd_sel];
    ocl_r_valid_pre  = k_rvalid[rd_sel];
    ocl_r_data_pre   = k_rdata[rd_sel];
    ocl_r_resp_pre   = k_rresp[rd_sel];
    // route AR/R only to the selected instance
    for (kk = 0; kk < NK; kk++) begin
      k_arvalid[kk] = (kk == rd_sel) ? ocl_ar_valid_pre : 1'b0;
      k_rready[kk]  = (kk == rd_sel) ? ocl_r_ready_pre  : 1'b0;
    end
  end

  // =========================================================================
  // DMA-PCIS crossbar: 1 SI (shell dma_pcis) -> NK MI (staging URAM windows).
  // AXI4, 512b, ID16. MI i decoded at base 0x0100_0000 + i*0x20_0000 (2 MB).
  // =========================================================================
  logic [NK*16-1:0]  xd_awid;    logic [NK*64-1:0]  xd_awaddr;
  logic [NK*8-1:0]   xd_awlen;   logic [NK*3-1:0]   xd_awsize;
  logic [NK*2-1:0]   xd_awburst; logic [NK-1:0]     xd_awlock;
  logic [NK*4-1:0]   xd_awcache; logic [NK*3-1:0]   xd_awprot;
  logic [NK-1:0]     xd_awvalid; logic [NK-1:0]     xd_awready;
  logic [NK*512-1:0] xd_wdata;   logic [NK*64-1:0]  xd_wstrb;
  logic [NK-1:0]     xd_wlast;   logic [NK-1:0]     xd_wvalid;  logic [NK-1:0] xd_wready;
  logic [NK*16-1:0]  xd_bid;     logic [NK*2-1:0]   xd_bresp;
  logic [NK-1:0]     xd_bvalid;  logic [NK-1:0]     xd_bready;
  logic [NK*16-1:0]  xd_arid;    logic [NK*64-1:0]  xd_araddr;
  logic [NK*8-1:0]   xd_arlen;   logic [NK*3-1:0]   xd_arsize;
  logic [NK*2-1:0]   xd_arburst; logic [NK-1:0]     xd_arlock;
  logic [NK*4-1:0]   xd_arcache; logic [NK*3-1:0]   xd_arprot;
  logic [NK-1:0]     xd_arvalid; logic [NK-1:0]     xd_arready;
  logic [NK*16-1:0]  xd_rid;     logic [NK*512-1:0] xd_rdata;
  logic [NK*2-1:0]   xd_rresp;   logic [NK-1:0]     xd_rlast;
  logic [NK-1:0]     xd_rvalid;  logic [NK-1:0]     xd_rready;

  axi_crossbar_dma XBAR_DMA (
    .aclk    (clk_main_a0),
    .aresetn (rst_main_n),
    .s_axi_awid    (sh_cl_dma_pcis_awid),
    .s_axi_awaddr  (sh_cl_dma_pcis_awaddr),
    .s_axi_awlen   (sh_cl_dma_pcis_awlen),
    .s_axi_awsize  (sh_cl_dma_pcis_awsize),
    .s_axi_awburst (sh_cl_dma_pcis_awburst),
    .s_axi_awlock  (sh_cl_dma_pcis_awlock),
    .s_axi_awcache (sh_cl_dma_pcis_awcache),
    .s_axi_awprot  (sh_cl_dma_pcis_awprot),
    .s_axi_awqos   (4'h0),
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
    .s_axi_araddr  (sh_cl_dma_pcis_araddr),
    .s_axi_arlen   (sh_cl_dma_pcis_arlen),
    .s_axi_arsize  (sh_cl_dma_pcis_arsize),
    .s_axi_arburst (sh_cl_dma_pcis_arburst),
    .s_axi_arlock  (sh_cl_dma_pcis_arlock),
    .s_axi_arcache (sh_cl_dma_pcis_arcache),
    .s_axi_arprot  (sh_cl_dma_pcis_arprot),
    .s_axi_arqos   (4'h0),
    .s_axi_arvalid (sh_cl_dma_pcis_arvalid),
    .s_axi_arready (cl_sh_dma_pcis_arready),
    .s_axi_rid     (cl_sh_dma_pcis_rid),
    .s_axi_rdata   (cl_sh_dma_pcis_rdata),
    .s_axi_rresp   (cl_sh_dma_pcis_rresp),
    .s_axi_rlast   (cl_sh_dma_pcis_rlast),
    .s_axi_rvalid  (cl_sh_dma_pcis_rvalid),
    .s_axi_rready  (sh_cl_dma_pcis_rready),

    .m_axi_awid    (xd_awid),
    .m_axi_awaddr  (xd_awaddr),
    .m_axi_awlen   (xd_awlen),
    .m_axi_awsize  (xd_awsize),
    .m_axi_awburst (xd_awburst),
    .m_axi_awlock  (xd_awlock),
    .m_axi_awcache (xd_awcache),
    .m_axi_awprot  (xd_awprot),
    .m_axi_awregion(),
    .m_axi_awqos   (),
    .m_axi_awvalid (xd_awvalid),
    .m_axi_awready (xd_awready),
    .m_axi_wdata   (xd_wdata),
    .m_axi_wstrb   (xd_wstrb),
    .m_axi_wlast   (xd_wlast),
    .m_axi_wvalid  (xd_wvalid),
    .m_axi_wready  (xd_wready),
    .m_axi_bid     (xd_bid),
    .m_axi_bresp   (xd_bresp),
    .m_axi_bvalid  (xd_bvalid),
    .m_axi_bready  (xd_bready),
    .m_axi_arid    (xd_arid),
    .m_axi_araddr  (xd_araddr),
    .m_axi_arlen   (xd_arlen),
    .m_axi_arsize  (xd_arsize),
    .m_axi_arburst (xd_arburst),
    .m_axi_arlock  (xd_arlock),
    .m_axi_arcache (xd_arcache),
    .m_axi_arprot  (xd_arprot),
    .m_axi_arregion(),
    .m_axi_arqos   (),
    .m_axi_arvalid (xd_arvalid),
    .m_axi_arready (xd_arready),
    .m_axi_rid     (xd_rid),
    .m_axi_rdata   (xd_rdata),
    .m_axi_rresp   (xd_rresp),
    .m_axi_rlast   (xd_rlast),
    .m_axi_rvalid  (xd_rvalid),
    .m_axi_rready  (xd_rready)
  );

  // =========================================================================
  // NK replicated instances: kernel + 2 BRAM controllers + 2 MB URAM each.
  // =========================================================================
  genvar gi;
  generate
  for (gi = 0; gi < NK; gi++) begin : KINST

    // ---- kernel m_axi (gmem) wires ----
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
    logic         krn_irq;

    // BRAM-port wires (per instance, 2 ports of the URAM)
    logic        bram_a_rst, bram_a_en;   logic [63:0] bram_a_we;
    logic [23:0] bram_a_addr;             logic [511:0] bram_a_wrdata, bram_a_rddata;  // [20:0]->[23:0]: 16 MB window
    logic        bram_b_rst, bram_b_en;   logic [63:0] bram_b_we;
    logic [23:0] bram_b_addr;             logic [511:0] bram_b_wrdata, bram_b_rddata;  // [20:0]->[23:0]: 16 MB window

    // ---- per-instance reset synchronizer (async assert / sync deassert) ----
    // The shell reset is a single high-fanout net; driving each replicated
    // instance (in a possibly-different SLR) directly from it created the
    // inter-SLR reset paths that failed timing. Re-synchronize it locally per
    // instance so each instance's reset is registered SLR-local; only one
    // low-fanout net per instance crosses the SLR boundary (async assert, no
    // timing requirement; 4-cycle synchronized deassert).
    (* ASYNC_REG = "true" *) logic [3:0] rst_pipe;
    always_ff @(posedge clk_main_a0 or negedge rst_main_n)
      if (!rst_main_n) rst_pipe <= 4'b0;
      else             rst_pipe <= {rst_pipe[2:0], 1'b1};
    wire rst_n_local = rst_pipe[3];

    // ---- OCL register slice (control): broadcast/mux drives the slice s_axi
    // (central, near OCL_REG_SLC); the slice m_axi drives THIS kernel. Excluded
    // from the kernel pblock so it floats to the SLR boundary -> the slice's
    // registered output crosses to the kernel reg-to-reg (the OCL readback path
    // kernel->OCL_REG_SLC was the residual -1.147 ns inter-SLR crossing).
    logic [31:0] rso_awaddr, rso_araddr, rso_wdata, rso_rdata;
    logic  [3:0] rso_wstrb;
    logic  [1:0] rso_bresp, rso_rresp;
    logic        rso_awvalid, rso_awready, rso_wvalid, rso_wready;
    logic        rso_bvalid, rso_bready, rso_arvalid, rso_arready, rso_rvalid, rso_rready;
    axi_register_slice_light RS_OCL (
      .aclk          (clk_main_a0),
      .aresetn       (rst_n_local),
      .s_axi_awaddr  (ocl_aw_addr_pre),
      .s_axi_awprot  (3'h0),
      .s_axi_awvalid (ocl_aw_valid_pre),
      .s_axi_awready (k_awready[gi]),
      .s_axi_wdata   (ocl_w_data_pre),
      .s_axi_wstrb   (ocl_w_strb_pre),
      .s_axi_wvalid  (ocl_w_valid_pre),
      .s_axi_wready  (k_wready[gi]),
      .s_axi_bresp   (k_bresp[gi]),
      .s_axi_bvalid  (k_bvalid[gi]),
      .s_axi_bready  (ocl_b_ready_pre),
      .s_axi_araddr  (ocl_ar_addr_pre),
      .s_axi_arvalid (k_arvalid[gi]),
      .s_axi_arready (k_arready[gi]),
      .s_axi_rdata   (k_rdata[gi]),
      .s_axi_rresp   (k_rresp[gi]),
      .s_axi_rvalid  (k_rvalid[gi]),
      .s_axi_rready  (k_rready[gi]),
      .m_axi_awaddr  (rso_awaddr),
      .m_axi_awprot  (),
      .m_axi_awvalid (rso_awvalid),
      .m_axi_awready (rso_awready),
      .m_axi_wdata   (rso_wdata),
      .m_axi_wstrb   (rso_wstrb),
      .m_axi_wvalid  (rso_wvalid),
      .m_axi_wready  (rso_wready),
      .m_axi_bresp   (rso_bresp),
      .m_axi_bvalid  (rso_bvalid),
      .m_axi_bready  (rso_bready),
      .m_axi_araddr  (rso_araddr),
      .m_axi_arvalid (rso_arvalid),
      .m_axi_arready (rso_arready),
      .m_axi_rdata   (rso_rdata),
      .m_axi_rresp   (rso_rresp),
      .m_axi_rvalid  (rso_rvalid),
      .m_axi_rready  (rso_rready)
    );

    // ---- kernel: control via per-instance OCL register slice ----
    forge_train_step_morph_lm u_train (
      .ap_clk     (clk_main_a0),
      .ap_rst_n   (rst_n_local),
      .interrupt  (krn_irq),

      .s_axi_control_AWADDR  (rso_awaddr[12:0]),
      .s_axi_control_AWVALID (rso_awvalid),
      .s_axi_control_AWREADY (rso_awready),
      .s_axi_control_WDATA   (rso_wdata),
      .s_axi_control_WSTRB   (rso_wstrb),
      .s_axi_control_WVALID  (rso_wvalid),
      .s_axi_control_WREADY  (rso_wready),
      .s_axi_control_BRESP   (rso_bresp),
      .s_axi_control_BVALID  (rso_bvalid),
      .s_axi_control_BREADY  (rso_bready),
      .s_axi_control_ARADDR  (rso_araddr[12:0]),
      .s_axi_control_ARVALID (rso_arvalid),
      .s_axi_control_ARREADY (rso_arready),
      .s_axi_control_RDATA   (rso_rdata),
      .s_axi_control_RRESP   (rso_rresp),
      .s_axi_control_RVALID  (rso_rvalid),
      .s_axi_control_RREADY  (rso_rready),

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

    // ---- SLR-crossing register slice: crossbar MI i -> (reg) -> BRAM_CTL_DMA.
    // Lives in this instance's pblock (KINST[gi].*), so the long hop from the
    // shared crossbar (STRATEGY=2, registered outputs) to here is reg-to-reg and
    // Vivado can route it through Laguna. Fixes the inter-SLR -1.75 ns failure
    // into BRAM_CTL_DMA's bram_addr/bram_wrdata input registers.
    logic [15:0]  rsd_awid, rsd_arid, rsd_bid, rsd_rid;
    logic [63:0]  rsd_awaddr, rsd_araddr;
    logic  [7:0]  rsd_awlen, rsd_arlen;
    logic  [2:0]  rsd_awsize, rsd_arsize, rsd_awprot, rsd_arprot;
    logic  [1:0]  rsd_awburst, rsd_arburst, rsd_bresp, rsd_rresp;
    logic         rsd_awlock, rsd_arlock;
    logic  [3:0]  rsd_awcache, rsd_arcache;
    logic         rsd_awvalid, rsd_awready, rsd_arvalid, rsd_arready;
    logic [511:0] rsd_wdata, rsd_rdata;
    logic  [63:0] rsd_wstrb;
    logic         rsd_wlast, rsd_wvalid, rsd_wready;
    logic         rsd_bvalid, rsd_bready, rsd_rlast, rsd_rvalid, rsd_rready;

    axi_register_slice_dma RS_DMA (
      .aclk    (clk_main_a0),
      .aresetn (rst_n_local),
      .s_axi_awid    (xd_awid[gi*16 +: 16]),
      .s_axi_awaddr  (xd_awaddr[gi*64 +: 64]),
      .s_axi_awlen   (xd_awlen[gi*8 +: 8]),
      .s_axi_awsize  (xd_awsize[gi*3 +: 3]),
      .s_axi_awburst (xd_awburst[gi*2 +: 2]),
      .s_axi_awlock  (xd_awlock[gi]),
      .s_axi_awcache (xd_awcache[gi*4 +: 4]),
      .s_axi_awprot  (xd_awprot[gi*3 +: 3]),
      .s_axi_awregion(4'h0),
      .s_axi_awqos   (4'h0),
      .s_axi_awvalid (xd_awvalid[gi]),
      .s_axi_awready (xd_awready[gi]),
      .s_axi_wdata   (xd_wdata[gi*512 +: 512]),
      .s_axi_wstrb   (xd_wstrb[gi*64 +: 64]),
      .s_axi_wlast   (xd_wlast[gi]),
      .s_axi_wvalid  (xd_wvalid[gi]),
      .s_axi_wready  (xd_wready[gi]),
      .s_axi_bid     (xd_bid[gi*16 +: 16]),
      .s_axi_bresp   (xd_bresp[gi*2 +: 2]),
      .s_axi_bvalid  (xd_bvalid[gi]),
      .s_axi_bready  (xd_bready[gi]),
      .s_axi_arid    (xd_arid[gi*16 +: 16]),
      .s_axi_araddr  (xd_araddr[gi*64 +: 64]),
      .s_axi_arlen   (xd_arlen[gi*8 +: 8]),
      .s_axi_arsize  (xd_arsize[gi*3 +: 3]),
      .s_axi_arburst (xd_arburst[gi*2 +: 2]),
      .s_axi_arlock  (xd_arlock[gi]),
      .s_axi_arcache (xd_arcache[gi*4 +: 4]),
      .s_axi_arprot  (xd_arprot[gi*3 +: 3]),
      .s_axi_arregion(4'h0),
      .s_axi_arqos   (4'h0),
      .s_axi_arvalid (xd_arvalid[gi]),
      .s_axi_arready (xd_arready[gi]),
      .s_axi_rid     (xd_rid[gi*16 +: 16]),
      .s_axi_rdata   (xd_rdata[gi*512 +: 512]),
      .s_axi_rresp   (xd_rresp[gi*2 +: 2]),
      .s_axi_rlast   (xd_rlast[gi]),
      .s_axi_rvalid  (xd_rvalid[gi]),
      .s_axi_rready  (xd_rready[gi]),

      .m_axi_awid    (rsd_awid),
      .m_axi_awaddr  (rsd_awaddr),
      .m_axi_awlen   (rsd_awlen),
      .m_axi_awsize  (rsd_awsize),
      .m_axi_awburst (rsd_awburst),
      .m_axi_awlock  (rsd_awlock),
      .m_axi_awcache (rsd_awcache),
      .m_axi_awprot  (rsd_awprot),
      .m_axi_awregion(),
      .m_axi_awqos   (),
      .m_axi_awvalid (rsd_awvalid),
      .m_axi_awready (rsd_awready),
      .m_axi_wdata   (rsd_wdata),
      .m_axi_wstrb   (rsd_wstrb),
      .m_axi_wlast   (rsd_wlast),
      .m_axi_wvalid  (rsd_wvalid),
      .m_axi_wready  (rsd_wready),
      .m_axi_bid     (rsd_bid),
      .m_axi_bresp   (rsd_bresp),
      .m_axi_bvalid  (rsd_bvalid),
      .m_axi_bready  (rsd_bready),
      .m_axi_arid    (rsd_arid),
      .m_axi_araddr  (rsd_araddr),
      .m_axi_arlen   (rsd_arlen),
      .m_axi_arsize  (rsd_arsize),
      .m_axi_arburst (rsd_arburst),
      .m_axi_arlock  (rsd_arlock),
      .m_axi_arcache (rsd_arcache),
      .m_axi_arprot  (rsd_arprot),
      .m_axi_arregion(),
      .m_axi_arqos   (),
      .m_axi_arvalid (rsd_arvalid),
      .m_axi_arready (rsd_arready),
      .m_axi_rid     (rsd_rid),
      .m_axi_rdata   (rsd_rdata),
      .m_axi_rresp   (rsd_rresp),
      .m_axi_rlast   (rsd_rlast),
      .m_axi_rvalid  (rsd_rvalid),
      .m_axi_rready  (rsd_rready)
    );

    // ---- DMA-side BRAM controller (reg slice -> URAM port A) ----
    axi_bram_ctrl_dma BRAM_CTL_DMA (
      .s_axi_aclk    (clk_main_a0),
      .s_axi_aresetn (rst_n_local),
      .s_axi_awid    (rsd_awid),
      .s_axi_awaddr  (rsd_awaddr[23:0]),   // [20:0]->[23:0]: 16 MB staging window
      .s_axi_awlen   (rsd_awlen),
      .s_axi_awsize  (rsd_awsize),
      .s_axi_awburst (rsd_awburst),
      .s_axi_awlock  (rsd_awlock),
      .s_axi_awcache (rsd_awcache),
      .s_axi_awprot  (rsd_awprot),
      .s_axi_awvalid (rsd_awvalid),
      .s_axi_awready (rsd_awready),
      .s_axi_wdata   (rsd_wdata),
      .s_axi_wstrb   (rsd_wstrb),
      .s_axi_wlast   (rsd_wlast),
      .s_axi_wvalid  (rsd_wvalid),
      .s_axi_wready  (rsd_wready),
      .s_axi_bid     (rsd_bid),
      .s_axi_bresp   (rsd_bresp),
      .s_axi_bvalid  (rsd_bvalid),
      .s_axi_bready  (rsd_bready),
      .s_axi_arid    (rsd_arid),
      .s_axi_araddr  (rsd_araddr[23:0]),   // [20:0]->[23:0]: 16 MB staging window
      .s_axi_arlen   (rsd_arlen),
      .s_axi_arsize  (rsd_arsize),
      .s_axi_arburst (rsd_arburst),
      .s_axi_arlock  (rsd_arlock),
      .s_axi_arcache (rsd_arcache),
      .s_axi_arprot  (rsd_arprot),
      .s_axi_arvalid (rsd_arvalid),
      .s_axi_arready (rsd_arready),
      .s_axi_rid     (rsd_rid),
      .s_axi_rdata   (rsd_rdata),
      .s_axi_rresp   (rsd_rresp),
      .s_axi_rlast   (rsd_rlast),
      .s_axi_rvalid  (rsd_rvalid),
      .s_axi_rready  (rsd_rready),
      .bram_rst_a    (bram_a_rst),
      .bram_clk_a    (),
      .bram_en_a     (bram_a_en),
      .bram_we_a     (bram_a_we),
      .bram_addr_a   (bram_a_addr),
      .bram_wrdata_a (bram_a_wrdata),
      .bram_rddata_a (bram_a_rddata)
    );

    // ---- kernel-side BRAM controller (kernel m_axi -> URAM port B) ----
    axi_bram_ctrl_krn BRAM_CTL_KRN (
      .s_axi_aclk    (clk_main_a0),
      .s_axi_aresetn (rst_n_local),
      .s_axi_awid    (krn_awid),
      .s_axi_awaddr  (krn_awaddr[23:0]),   // [20:0]->[23:0]: 16 MB window
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
      .s_axi_araddr  (krn_araddr[23:0]),   // [20:0]->[23:0]: 16 MB window
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
      .bram_clk_a    (),
      .bram_en_a     (bram_b_en),
      .bram_we_a     (bram_b_we),
      .bram_addr_a   (bram_b_addr),
      .bram_wrdata_a (bram_b_wrdata),
      .bram_rddata_a (bram_b_rddata)
    );

    // ---- 16 MB true-dual-port URAM (512b x 262144); 1-cycle read latency ----
    // D=512 buffer is 12.25 MB -> 16 MB window. MEMORY_SIZE is in BITS:
    // 16 MB = 134217728 bits (was 16777216 = 2 MB). 262144 lines -> ADDR_WIDTH 18
    // (was 32768/15). This 16 MB URAM spans multiple SLRs (see floorplan).
    xpm_memory_tdpram #(
      .MEMORY_SIZE        (134217728),
      .MEMORY_PRIMITIVE   ("ultra"),
      .CLOCKING_MODE      ("common_clock"),
      .WRITE_DATA_WIDTH_A (512),
      .READ_DATA_WIDTH_A  (512),
      .BYTE_WRITE_WIDTH_A (8),
      .ADDR_WIDTH_A       (18),
      .READ_LATENCY_A     (1),
      .WRITE_MODE_A       ("no_change"),
      .WRITE_DATA_WIDTH_B (512),
      .READ_DATA_WIDTH_B  (512),
      .BYTE_WRITE_WIDTH_B (8),
      .ADDR_WIDTH_B       (18),
      .READ_LATENCY_B     (1),
      .WRITE_MODE_B       ("no_change")
    ) BUF_URAM (
      .clka   (clk_main_a0),
      .rsta   (bram_a_rst),
      .ena    (bram_a_en),
      .wea    (bram_a_we),
      .addra  (bram_a_addr[23:6]),   // [20:6]->[23:6]: 18-bit line select (262144 x 512b)
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
      .addrb  (bram_b_addr[23:6]),   // [20:6]->[23:6]: 18-bit line select (262144 x 512b)
      .dinb   (bram_b_wrdata),
      .doutb  (bram_b_rddata),
      .regceb (1'b1),
      .injectsbiterrb (1'b0),
      .injectdbiterrb (1'b0),
      .sbiterrb (),
      .dbiterrb (),
      .sleep  (1'b0)
    );

  end
  endgenerate

endmodule
