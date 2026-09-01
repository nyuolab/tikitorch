// =============================================================================
// cl_forge_morph_lm_d512.sv — D=512 SINGLE-KERNEL MorphSSM-LM training CL.
//
// PARALLEL to cl_forge_morph_lm.sv (the DEPLOYED D=128 NK=4 sentinel, agfi-0433...,
// which stays byte-identical). This D=512 variant is the A3 build. TWO-BANK v2:
// the original single 16 MB window was IMPOSSIBLE — axi_bram_ctrl caps at
// MEM_DEPTH 65536 (= 4 MB at 512b), and 16 MB of staging URAM (456 blocks) on
// top of the kernel's measured 605 exceeds the device's 960. Instead the D=512
// buffer layout CULLS its dead gradient sections (grads are on-chip-only under
// the mandatory TT_RESIDENT_W+TT_ONCHIP_GRAD; forge_buffer_morph_lm.hpp) down
// to 7.69 MB, which fits an 8 MB staging window built as TWO 4 MB banks — each
// bank at the D=128-proven controller geometry (just deeper: 32768 -> 65536).
// Deltas vs the D=128 package:
//   * NK 4 -> 1 (one kernel), NB = 2 staging banks
//   * per-bank BUF_URAM 4 MB (MEMORY_SIZE 33554432 bits, ADDR_WIDTH 16,
//       line-select [21:6]); 2 banks = 8 MB total (256 URAM blocks)
//   * per-bank BRAM-controller address decode [21:0] (4 MB window each)
//   * axi_bram_ctrl MEM_DEPTH 32768 -> 65536 (the IP's max)  (gen_ip.tcl)
//   * axi_crossbar_dma 1 SI -> 2 MI: M00 0x0100_0000, M01 0x0140_0000, both
//     22-bit (4 MB) windows — host sees ONE contiguous 8 MB buffer (gen_ip.tcl)
//   * NEW axi_crossbar_krn 1 SI -> 2 MI (ID1): the kernel's single m_axi gmem
//     (which emits absolute addresses at base 0x0100_0000 — bridge.cpp
//     TT_BUFFER_ADDR) is decoded to the two kernel-side controllers with the
//     SAME bases as the DMA side. AXI bursts never cross a 4 KB boundary, so
//     no burst straddles the 4 MB bank seam.
//
//   sh_cl_dma_pcis (512b AXI4, ID16) --[xbar_dma 1->2]--> per-bank
//        RS_DMA -> axi_bram_ctrl_dma -> URAM port A
//   ocl (AXI4-Lite) --[reg slice]--> kernel s_axilite control
//   kernel m_axi gmem (512b) --[xbar_krn 1->2]--> per-bank
//        axi_bram_ctrl_krn -> URAM port B
//
// Address map (host-visible):
//   DMA-PCIS staging window : 0x0100_0000  (8 MB contiguous, 2 x 4 MB banks)
//   OCL control window      : 0x0000_0000  (64 KB)
// Each axi_bram_ctrl masks the low 22 bits -> 512b line index = addr[21:6].
// PCIM unused. 2026 F2 HDK port family: ocl_cl_* in / cl_ocl_* out.
// The kernel itself (forge_train_step_morph_lm, -DML_D=512) holds its resident
// weights + on-chip grads in URAM via #if ML_D>=512 guards — that URAM (605
// blocks measured at csynth) is INTERNAL to the kernel; with the 256-block
// staging total = 861/960 device URAM, spanning all SLRs (hence the all-SLR
// pblock; see cl_pnr_user_floorplan.xdc).
// =============================================================================

`include "cl_forge_morph_lm_defines.vh"

module cl_forge_morph_lm_d512

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
`include "unused_ddr_template.inc"
// NOTE: unused_cl_sda_template.inc is DELETED — the SDA AXI-Lite slave drives
// AWS_CLK_GEN's control plane (the S1 clkgen pattern; see run_s1_afi_build.sh).
`include "unused_apppf_irq_template.inc"
`include "unused_pcim_template.inc"

  // -------------------------------------------------------------------------
  // Replication factor = 1 at D=512 (the kernel's own 605-block URAM footprint
  // spans all three SLRs — no room to replicate; the D=128 sentinel runs NK=4
  // at 2 MB/kernel). NB = 2 staging BANKS of 4 MB each (axi_bram_ctrl caps at
  // 65536 lines): the DMA crossbar decodes host addresses across them and the
  // kernel-side crossbar does the same for the kernel's single m_axi.
  // NB MUST match axi_crossbar_dma AND axi_crossbar_krn NUM_MI in gen_ip.tcl.
  // -------------------------------------------------------------------------
  localparam int NK = 1;
  localparam int NB = 2;

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_status0 = 32'h0;
  assign cl_sh_status1 = 32'h0;
  assign cl_sh_status2 = 32'h0;
  assign cl_sh_status_vled = 16'h0;
  assign cl_sh_dma_wr_full = 1'b0;
  assign cl_sh_dma_rd_full = 1'b0;

  // =========================================================================
  // v3 CLKGEN: the v2 250 MHz P&R measured WNS -2.726 ns / TNS -40.8 us with
  // the worst paths INSIDE the 4 MB staging banks (BRAM_CTL write control ->
  // 16-deep URAM cascade CAS_IN_EN pins, ~6.7 ns physical) — unfixable at the
  // shell-fixed 4 ns clk_main_a0. So the ENTIRE user design (crossbars, banks,
  // controllers, kernel) moves to the generated clk_extra_a1 = 125 MHz (8 ns;
  // worst measured path closes with >1.2 ns slack), with async AXI clock
  // converters at the two shell boundaries (DMA-PCIS + OCL). This is the S1
  // CDC rearchitecture (cl_forge_train_step_s1.sv), cut at the shell edge.
  // Build MUST pass --aws_clk_gen --clock_recipe_a A1 (run_morph_lm_d512_afi.sh).
  // =========================================================================
  logic  gen_clk_hbm_ref;
  logic  gen_clk_main_a0;            // 250 MHz pass-through (shell domain)
  logic  gen_clk_extra_a1;           // 125 MHz (Recipe A1)  <-- USER-DESIGN CLOCK
  logic  gen_clk_extra_a2, gen_clk_extra_a3;
  logic  gen_clk_extra_b0, gen_clk_extra_b1, gen_clk_extra_c0, gen_clk_extra_c1;
  logic  gen_clk_hbm_axi;
  logic  gen_rst_hbm_axi_n, gen_rst_hbm_ref_n;
  logic  gen_rst_c1_n, gen_rst_c0_n, gen_rst_b1_n, gen_rst_b0_n;
  logic  gen_rst_a3_n, gen_rst_a2_n;
  logic  gen_rst_a1_n;               // reset sync'd to clk_extra_a1 (125 MHz domain)
  logic  gen_rst_main_n;             // reset sync'd to clk_main_a0 (250 MHz shell domain)

  // SDA reset synchronizer (matches cl_mem_perf / the S1 CL).
  logic sda_sync_rst_n;
  xpm_cdc_async_rst CDC_ASYNC_RST_N_SDA (
    .src_arst  (rst_main_n     ),
    .dest_clk  (clk_main_a0    ),
    .dest_arst (sda_sync_rst_n )
  );

  aws_clk_gen #(
    .CLK_GRP_A_EN (1), .CLK_GRP_B_EN (0), .CLK_GRP_C_EN (0), .CLK_HBM_EN (0)
  ) AWS_CLK_GEN (
    .i_clk_main_a0       (clk_main_a0    ),
    .i_rst_main_n        (sda_sync_rst_n ),
    .i_clk_hbm_ref       (clk_hbm_ref    ),
    .s_axil_ctrl_awaddr  (sda_cl_awaddr  ), .s_axil_ctrl_awvalid (sda_cl_awvalid ), .s_axil_ctrl_awready (cl_sda_awready),
    .s_axil_ctrl_wdata   (sda_cl_wdata   ), .s_axil_ctrl_wstrb   (sda_cl_wstrb   ), .s_axil_ctrl_wvalid  (sda_cl_wvalid ), .s_axil_ctrl_wready (cl_sda_wready),
    .s_axil_ctrl_bresp   (cl_sda_bresp   ), .s_axil_ctrl_bvalid  (cl_sda_bvalid  ), .s_axil_ctrl_bready  (sda_cl_bready ),
    .s_axil_ctrl_araddr  (sda_cl_araddr  ), .s_axil_ctrl_arvalid (sda_cl_arvalid ), .s_axil_ctrl_arready (cl_sda_arready),
    .s_axil_ctrl_rdata   (cl_sda_rdata   ), .s_axil_ctrl_rresp   (cl_sda_rresp   ), .s_axil_ctrl_rvalid  (cl_sda_rvalid ), .s_axil_ctrl_rready (sda_cl_rready),
    .o_clk_hbm_ref       (gen_clk_hbm_ref ), .o_clk_main_a0   (gen_clk_main_a0 ),
    .o_clk_extra_a1      (gen_clk_extra_a1), .o_clk_extra_a2  (gen_clk_extra_a2), .o_clk_extra_a3 (gen_clk_extra_a3),
    .o_clk_extra_b0      (gen_clk_extra_b0), .o_clk_extra_b1  (gen_clk_extra_b1),
    .o_clk_extra_c0      (gen_clk_extra_c0), .o_clk_extra_c1  (gen_clk_extra_c1), .o_clk_hbm_axi  (gen_clk_hbm_axi),
    .o_cl_rst_hbm_axi_n  (gen_rst_hbm_axi_n), .o_cl_rst_hbm_ref_n (gen_rst_hbm_ref_n),
    .o_cl_rst_c1_n       (gen_rst_c1_n    ), .o_cl_rst_c0_n   (gen_rst_c0_n    ), .o_cl_rst_b1_n  (gen_rst_b1_n   ), .o_cl_rst_b0_n (gen_rst_b0_n),
    .o_cl_rst_a3_n       (gen_rst_a3_n    ), .o_cl_rst_a2_n   (gen_rst_a2_n    ), .o_cl_rst_a1_n  (gen_rst_a1_n   ), .o_cl_rst_main_n (gen_rst_main_n)
  );

  // =========================================================================
  // OCL path: shell (250 MHz) -> OCL_CDC (async clock converter) -> register
  // slice + fan-out on the 125 MHz user domain.
  // =========================================================================
  logic        oclc_aw_valid, oclc_aw_ready;
  logic [31:0] oclc_aw_addr;
  logic        oclc_w_valid, oclc_w_ready;
  logic [31:0] oclc_w_data;
  logic  [3:0] oclc_w_strb;
  logic        oclc_b_valid, oclc_b_ready;
  logic  [1:0] oclc_b_resp;
  logic        oclc_ar_valid, oclc_ar_ready;
  logic [31:0] oclc_ar_addr;
  logic        oclc_r_valid, oclc_r_ready;
  logic [31:0] oclc_r_data;
  logic  [1:0] oclc_r_resp;

  mld512_ocl_cdc OCL_CDC (
    .s_axi_aclk    (clk_main_a0),     .s_axi_aresetn (rst_main_n),      // 250 MHz slave = shell OCL
    .s_axi_awaddr  (ocl_cl_awaddr),   .s_axi_awprot  (3'h0),
    .s_axi_awvalid (ocl_cl_awvalid),  .s_axi_awready (cl_ocl_awready),
    .s_axi_wdata   (ocl_cl_wdata),    .s_axi_wstrb   (ocl_cl_wstrb),
    .s_axi_wvalid  (ocl_cl_wvalid),   .s_axi_wready  (cl_ocl_wready),
    .s_axi_bresp   (cl_ocl_bresp),    .s_axi_bvalid  (cl_ocl_bvalid),   .s_axi_bready (ocl_cl_bready),
    .s_axi_araddr  (ocl_cl_araddr),   .s_axi_arprot  (3'h0),
    .s_axi_arvalid (ocl_cl_arvalid),  .s_axi_arready (cl_ocl_arready),
    .s_axi_rdata   (cl_ocl_rdata),    .s_axi_rresp   (cl_ocl_rresp),
    .s_axi_rvalid  (cl_ocl_rvalid),   .s_axi_rready  (ocl_cl_rready),
    .m_axi_aclk    (gen_clk_extra_a1),.m_axi_aresetn (gen_rst_a1_n),    // 125 MHz master
    .m_axi_awaddr  (oclc_aw_addr),    .m_axi_awprot  (),
    .m_axi_awvalid (oclc_aw_valid),   .m_axi_awready (oclc_aw_ready),
    .m_axi_wdata   (oclc_w_data),     .m_axi_wstrb   (oclc_w_strb),
    .m_axi_wvalid  (oclc_w_valid),    .m_axi_wready  (oclc_w_ready),
    .m_axi_bresp   (oclc_b_resp),     .m_axi_bvalid  (oclc_b_valid),    .m_axi_bready (oclc_b_ready),
    .m_axi_araddr  (oclc_ar_addr),    .m_axi_arprot  (),
    .m_axi_arvalid (oclc_ar_valid),   .m_axi_arready (oclc_ar_ready),
    .m_axi_rdata   (oclc_r_data),     .m_axi_rresp   (oclc_r_resp),
    .m_axi_rvalid  (oclc_r_valid),    .m_axi_rready  (oclc_r_ready)
  );

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
    .aclk          (gen_clk_extra_a1),
    .aresetn       (gen_rst_a1_n),
    .s_axi_awaddr  (oclc_aw_addr),
    .s_axi_awprot  (3'h0),
    .s_axi_awvalid (oclc_aw_valid),
    .s_axi_awready (oclc_aw_ready),
    .s_axi_wdata   (oclc_w_data),
    .s_axi_wstrb   (oclc_w_strb),
    .s_axi_wvalid  (oclc_w_valid),
    .s_axi_wready  (oclc_w_ready),
    .s_axi_bresp   (oclc_b_resp),
    .s_axi_bvalid  (oclc_b_valid),
    .s_axi_bready  (oclc_b_ready),
    .s_axi_araddr  (oclc_ar_addr),
    .s_axi_arvalid (oclc_ar_valid),
    .s_axi_arready (oclc_ar_ready),
    .s_axi_rdata   (oclc_r_data),
    .s_axi_rresp   (oclc_r_resp),
    .s_axi_rvalid  (oclc_r_valid),
    .s_axi_rready  (oclc_r_ready),

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
  // DMA-PCIS crossbar: 1 SI (shell dma_pcis) -> NB MI (staging URAM banks).
  // AXI4, 512b, ID16. Bank b decoded at base 0x0100_0000 + b*0x40_0000 (4 MB).
  // =========================================================================
  logic [NB*16-1:0]  xd_awid;    logic [NB*64-1:0]  xd_awaddr;
  logic [NB*8-1:0]   xd_awlen;   logic [NB*3-1:0]   xd_awsize;
  logic [NB*2-1:0]   xd_awburst; logic [NB-1:0]     xd_awlock;
  logic [NB*4-1:0]   xd_awcache; logic [NB*3-1:0]   xd_awprot;
  logic [NB-1:0]     xd_awvalid; logic [NB-1:0]     xd_awready;
  logic [NB*512-1:0] xd_wdata;   logic [NB*64-1:0]  xd_wstrb;
  logic [NB-1:0]     xd_wlast;   logic [NB-1:0]     xd_wvalid;  logic [NB-1:0] xd_wready;
  logic [NB*16-1:0]  xd_bid;     logic [NB*2-1:0]   xd_bresp;
  logic [NB-1:0]     xd_bvalid;  logic [NB-1:0]     xd_bready;
  logic [NB*16-1:0]  xd_arid;    logic [NB*64-1:0]  xd_araddr;
  logic [NB*8-1:0]   xd_arlen;   logic [NB*3-1:0]   xd_arsize;
  logic [NB*2-1:0]   xd_arburst; logic [NB-1:0]     xd_arlock;
  logic [NB*4-1:0]   xd_arcache; logic [NB*3-1:0]   xd_arprot;
  logic [NB-1:0]     xd_arvalid; logic [NB-1:0]     xd_arready;
  logic [NB*16-1:0]  xd_rid;     logic [NB*512-1:0] xd_rdata;
  logic [NB*2-1:0]   xd_rresp;   logic [NB-1:0]     xd_rlast;
  logic [NB-1:0]     xd_rvalid;  logic [NB-1:0]     xd_rready;

  // PCIS CDC: shell dma_pcis (250 MHz) -> async converter -> XBAR_DMA (125 MHz).
  logic [15:0]  pdc_awid, pdc_arid, pdc_bid, pdc_rid;
  logic [63:0]  pdc_awaddr, pdc_araddr;
  logic  [7:0]  pdc_awlen, pdc_arlen;
  logic  [2:0]  pdc_awsize, pdc_arsize, pdc_awprot, pdc_arprot;
  logic  [1:0]  pdc_awburst, pdc_arburst, pdc_bresp, pdc_rresp;
  logic         pdc_awlock, pdc_arlock;
  logic  [3:0]  pdc_awcache, pdc_arcache, pdc_awqos, pdc_arqos;
  logic         pdc_awvalid, pdc_awready, pdc_arvalid, pdc_arready;
  logic [511:0] pdc_wdata, pdc_rdata;
  logic  [63:0] pdc_wstrb;
  logic         pdc_wlast, pdc_wvalid, pdc_wready;
  logic         pdc_bvalid, pdc_bready, pdc_rlast, pdc_rvalid, pdc_rready;

  mld512_pcis_cdc PCIS_CDC (
    .s_axi_aclk    (clk_main_a0), .s_axi_aresetn (rst_main_n),          // 250 MHz slave = shell dma_pcis
    .s_axi_awid    (sh_cl_dma_pcis_awid),
    .s_axi_awaddr  (sh_cl_dma_pcis_awaddr),
    .s_axi_awlen   (sh_cl_dma_pcis_awlen),
    .s_axi_awsize  (sh_cl_dma_pcis_awsize),
    .s_axi_awburst (sh_cl_dma_pcis_awburst),
    .s_axi_awlock  (sh_cl_dma_pcis_awlock),
    .s_axi_awcache (sh_cl_dma_pcis_awcache),
    .s_axi_awprot  (sh_cl_dma_pcis_awprot),
    .s_axi_awqos   (4'h0),
    .s_axi_awregion(4'h0),
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
    .s_axi_arregion(4'h0),
    .s_axi_arvalid (sh_cl_dma_pcis_arvalid),
    .s_axi_arready (cl_sh_dma_pcis_arready),
    .s_axi_rid     (cl_sh_dma_pcis_rid),
    .s_axi_rdata   (cl_sh_dma_pcis_rdata),
    .s_axi_rresp   (cl_sh_dma_pcis_rresp),
    .s_axi_rlast   (cl_sh_dma_pcis_rlast),
    .s_axi_rvalid  (cl_sh_dma_pcis_rvalid),
    .s_axi_rready  (sh_cl_dma_pcis_rready),
    .m_axi_aclk    (gen_clk_extra_a1), .m_axi_aresetn (gen_rst_a1_n),   // 125 MHz master = XBAR_DMA SI
    .m_axi_awid    (pdc_awid),
    .m_axi_awaddr  (pdc_awaddr),
    .m_axi_awlen   (pdc_awlen),
    .m_axi_awsize  (pdc_awsize),
    .m_axi_awburst (pdc_awburst),
    .m_axi_awlock  (pdc_awlock),
    .m_axi_awcache (pdc_awcache),
    .m_axi_awprot  (pdc_awprot),
    .m_axi_awqos   (pdc_awqos),
    .m_axi_awregion(),
    .m_axi_awvalid (pdc_awvalid),
    .m_axi_awready (pdc_awready),
    .m_axi_wdata   (pdc_wdata),
    .m_axi_wstrb   (pdc_wstrb),
    .m_axi_wlast   (pdc_wlast),
    .m_axi_wvalid  (pdc_wvalid),
    .m_axi_wready  (pdc_wready),
    .m_axi_bid     (pdc_bid),
    .m_axi_bresp   (pdc_bresp),
    .m_axi_bvalid  (pdc_bvalid),
    .m_axi_bready  (pdc_bready),
    .m_axi_arid    (pdc_arid),
    .m_axi_araddr  (pdc_araddr),
    .m_axi_arlen   (pdc_arlen),
    .m_axi_arsize  (pdc_arsize),
    .m_axi_arburst (pdc_arburst),
    .m_axi_arlock  (pdc_arlock),
    .m_axi_arcache (pdc_arcache),
    .m_axi_arprot  (pdc_arprot),
    .m_axi_arqos   (pdc_arqos),
    .m_axi_arregion(),
    .m_axi_arvalid (pdc_arvalid),
    .m_axi_arready (pdc_arready),
    .m_axi_rid     (pdc_rid),
    .m_axi_rdata   (pdc_rdata),
    .m_axi_rresp   (pdc_rresp),
    .m_axi_rlast   (pdc_rlast),
    .m_axi_rvalid  (pdc_rvalid),
    .m_axi_rready  (pdc_rready)
  );

  axi_crossbar_dma XBAR_DMA (
    .aclk    (gen_clk_extra_a1),
    .aresetn (gen_rst_a1_n),
    .s_axi_awid    (pdc_awid),
    .s_axi_awaddr  (pdc_awaddr),
    .s_axi_awlen   (pdc_awlen),
    .s_axi_awsize  (pdc_awsize),
    .s_axi_awburst (pdc_awburst),
    .s_axi_awlock  (pdc_awlock),
    .s_axi_awcache (pdc_awcache),
    .s_axi_awprot  (pdc_awprot),
    .s_axi_awqos   (pdc_awqos),
    .s_axi_awvalid (pdc_awvalid),
    .s_axi_awready (pdc_awready),
    .s_axi_wdata   (pdc_wdata),
    .s_axi_wstrb   (pdc_wstrb),
    .s_axi_wlast   (pdc_wlast),
    .s_axi_wvalid  (pdc_wvalid),
    .s_axi_wready  (pdc_wready),
    .s_axi_bid     (pdc_bid),
    .s_axi_bresp   (pdc_bresp),
    .s_axi_bvalid  (pdc_bvalid),
    .s_axi_bready  (pdc_bready),
    .s_axi_arid    (pdc_arid),
    .s_axi_araddr  (pdc_araddr),
    .s_axi_arlen   (pdc_arlen),
    .s_axi_arsize  (pdc_arsize),
    .s_axi_arburst (pdc_arburst),
    .s_axi_arlock  (pdc_arlock),
    .s_axi_arcache (pdc_arcache),
    .s_axi_arprot  (pdc_arprot),
    .s_axi_arqos   (pdc_arqos),
    .s_axi_arvalid (pdc_arvalid),
    .s_axi_arready (pdc_arready),
    .s_axi_rid     (pdc_rid),
    .s_axi_rdata   (pdc_rdata),
    .s_axi_rresp   (pdc_rresp),
    .s_axi_rlast   (pdc_rlast),
    .s_axi_rvalid  (pdc_rvalid),
    .s_axi_rready  (pdc_rready),

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

    // ---- kernel-side crossbar M nets: XBAR_KRN 1 SI -> NB MI (one per bank).
    // ID width 1 (the kernel's single-ID m_axi); bank b decoded at
    // 0x0100_0000 + b*0x40_0000, same bases as the DMA-side crossbar.
    logic [NB-1:0]     kb_awid;    logic [NB*64-1:0]  kb_awaddr;
    logic [NB*8-1:0]   kb_awlen;   logic [NB*3-1:0]   kb_awsize;
    logic [NB*2-1:0]   kb_awburst; logic [NB-1:0]     kb_awlock;
    logic [NB*4-1:0]   kb_awcache; logic [NB*3-1:0]   kb_awprot;
    logic [NB-1:0]     kb_awvalid; logic [NB-1:0]     kb_awready;
    logic [NB*512-1:0] kb_wdata;   logic [NB*64-1:0]  kb_wstrb;
    logic [NB-1:0]     kb_wlast;   logic [NB-1:0]     kb_wvalid;  logic [NB-1:0] kb_wready;
    logic [NB-1:0]     kb_bid;     logic [NB*2-1:0]   kb_bresp;
    logic [NB-1:0]     kb_bvalid;  logic [NB-1:0]     kb_bready;
    logic [NB-1:0]     kb_arid;    logic [NB*64-1:0]  kb_araddr;
    logic [NB*8-1:0]   kb_arlen;   logic [NB*3-1:0]   kb_arsize;
    logic [NB*2-1:0]   kb_arburst; logic [NB-1:0]     kb_arlock;
    logic [NB*4-1:0]   kb_arcache; logic [NB*3-1:0]   kb_arprot;
    logic [NB-1:0]     kb_arvalid; logic [NB-1:0]     kb_arready;
    logic [NB-1:0]     kb_rid;     logic [NB*512-1:0] kb_rdata;
    logic [NB*2-1:0]   kb_rresp;   logic [NB-1:0]     kb_rlast;
    logic [NB-1:0]     kb_rvalid;  logic [NB-1:0]     kb_rready;

    // ---- per-instance reset synchronizer (async assert / sync deassert) ----
    // The shell reset is a single high-fanout net; driving each replicated
    // instance (in a possibly-different SLR) directly from it created the
    // inter-SLR reset paths that failed timing. Re-synchronize it locally per
    // instance so each instance's reset is registered SLR-local; only one
    // low-fanout net per instance crosses the SLR boundary (async assert, no
    // timing requirement; 4-cycle synchronized deassert).
    (* ASYNC_REG = "true" *) logic [3:0] rst_pipe;
    always_ff @(posedge gen_clk_extra_a1 or negedge gen_rst_a1_n)
      if (!gen_rst_a1_n) rst_pipe <= 4'b0;
      else               rst_pipe <= {rst_pipe[2:0], 1'b1};
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
      .aclk          (gen_clk_extra_a1),
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
      .ap_clk     (gen_clk_extra_a1),
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

    // ---- kernel-side crossbar: kernel m_axi gmem -> NB bank controllers.
    // The kernel emits absolute addresses from base 0x0100_0000 (bridge.cpp
    // pokes the s_axilite buffer pointer once); this crossbar decodes bank 0
    // at 0x0100_0000 and bank 1 at 0x0140_0000 (22-bit windows), mirroring
    // XBAR_DMA. Lives inside the kernel pblock (talks only to in-pblock
    // endpoints). STRATEGY=2 registered outputs.
    axi_crossbar_krn XBAR_KRN (
      .aclk    (gen_clk_extra_a1),
      .aresetn (rst_n_local),
      .s_axi_awid    (krn_awid),
      .s_axi_awaddr  (krn_awaddr),
      .s_axi_awlen   (krn_awlen),
      .s_axi_awsize  (krn_awsize),
      .s_axi_awburst (krn_awburst),
      .s_axi_awlock  (krn_awlock[0]),
      .s_axi_awcache (krn_awcache),
      .s_axi_awprot  (krn_awprot),
      .s_axi_awqos   (krn_awqos),
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
      .s_axi_araddr  (krn_araddr),
      .s_axi_arlen   (krn_arlen),
      .s_axi_arsize  (krn_arsize),
      .s_axi_arburst (krn_arburst),
      .s_axi_arlock  (krn_arlock[0]),
      .s_axi_arcache (krn_arcache),
      .s_axi_arprot  (krn_arprot),
      .s_axi_arqos   (krn_arqos),
      .s_axi_arvalid (krn_arvalid),
      .s_axi_arready (krn_arready),
      .s_axi_rid     (krn_rid),
      .s_axi_rdata   (krn_rdata),
      .s_axi_rresp   (krn_rresp),
      .s_axi_rlast   (krn_rlast),
      .s_axi_rvalid  (krn_rvalid),
      .s_axi_rready  (krn_rready),

      .m_axi_awid    (kb_awid),
      .m_axi_awaddr  (kb_awaddr),
      .m_axi_awlen   (kb_awlen),
      .m_axi_awsize  (kb_awsize),
      .m_axi_awburst (kb_awburst),
      .m_axi_awlock  (kb_awlock),
      .m_axi_awcache (kb_awcache),
      .m_axi_awprot  (kb_awprot),
      .m_axi_awregion(),
      .m_axi_awqos   (),
      .m_axi_awvalid (kb_awvalid),
      .m_axi_awready (kb_awready),
      .m_axi_wdata   (kb_wdata),
      .m_axi_wstrb   (kb_wstrb),
      .m_axi_wlast   (kb_wlast),
      .m_axi_wvalid  (kb_wvalid),
      .m_axi_wready  (kb_wready),
      .m_axi_bid     (kb_bid),
      .m_axi_bresp   (kb_bresp),
      .m_axi_bvalid  (kb_bvalid),
      .m_axi_bready  (kb_bready),
      .m_axi_arid    (kb_arid),
      .m_axi_araddr  (kb_araddr),
      .m_axi_arlen   (kb_arlen),
      .m_axi_arsize  (kb_arsize),
      .m_axi_arburst (kb_arburst),
      .m_axi_arlock  (kb_arlock),
      .m_axi_arcache (kb_arcache),
      .m_axi_arprot  (kb_arprot),
      .m_axi_arregion(),
      .m_axi_arqos   (),
      .m_axi_arvalid (kb_arvalid),
      .m_axi_arready (kb_arready),
      .m_axi_rid     (kb_rid),
      .m_axi_rdata   (kb_rdata),
      .m_axi_rresp   (kb_rresp),
      .m_axi_rlast   (kb_rlast),
      .m_axi_rvalid  (kb_rvalid),
      .m_axi_rready  (kb_rready)
    );

    // ---- Per-bank staging: RS_DMA -> BRAM_CTL_DMA -> URAM port A;
    //                        XBAR_KRN MI -> BRAM_CTL_KRN -> URAM port B.
    // The RS_DMA slice lives in the kernel pblock's exclusion set (floats to
    // the SLR boundary) so the hop from the shared crossbar (STRATEGY=2,
    // registered outputs) is reg-to-reg through Laguna — same fix as the
    // D=128 -1.75 ns inter-SLR failure into BRAM_CTL_DMA's input registers.
    for (genvar gb = 0; gb < NB; gb++) begin : BANK

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

    // BRAM-port wires (this bank's URAM, 2 ports)
    logic        bram_a_rst, bram_a_en;   logic [63:0] bram_a_we;
    logic [21:0] bram_a_addr;             logic [511:0] bram_a_wrdata, bram_a_rddata;  // 4 MB bank window
    logic        bram_b_rst, bram_b_en;   logic [63:0] bram_b_we;
    logic [21:0] bram_b_addr;             logic [511:0] bram_b_wrdata, bram_b_rddata;  // 4 MB bank window

    axi_register_slice_dma RS_DMA (
      .aclk    (gen_clk_extra_a1),
      .aresetn (rst_n_local),
      .s_axi_awid    (xd_awid[gb*16 +: 16]),
      .s_axi_awaddr  (xd_awaddr[gb*64 +: 64]),
      .s_axi_awlen   (xd_awlen[gb*8 +: 8]),
      .s_axi_awsize  (xd_awsize[gb*3 +: 3]),
      .s_axi_awburst (xd_awburst[gb*2 +: 2]),
      .s_axi_awlock  (xd_awlock[gb]),
      .s_axi_awcache (xd_awcache[gb*4 +: 4]),
      .s_axi_awprot  (xd_awprot[gb*3 +: 3]),
      .s_axi_awregion(4'h0),
      .s_axi_awqos   (4'h0),
      .s_axi_awvalid (xd_awvalid[gb]),
      .s_axi_awready (xd_awready[gb]),
      .s_axi_wdata   (xd_wdata[gb*512 +: 512]),
      .s_axi_wstrb   (xd_wstrb[gb*64 +: 64]),
      .s_axi_wlast   (xd_wlast[gb]),
      .s_axi_wvalid  (xd_wvalid[gb]),
      .s_axi_wready  (xd_wready[gb]),
      .s_axi_bid     (xd_bid[gb*16 +: 16]),
      .s_axi_bresp   (xd_bresp[gb*2 +: 2]),
      .s_axi_bvalid  (xd_bvalid[gb]),
      .s_axi_bready  (xd_bready[gb]),
      .s_axi_arid    (xd_arid[gb*16 +: 16]),
      .s_axi_araddr  (xd_araddr[gb*64 +: 64]),
      .s_axi_arlen   (xd_arlen[gb*8 +: 8]),
      .s_axi_arsize  (xd_arsize[gb*3 +: 3]),
      .s_axi_arburst (xd_arburst[gb*2 +: 2]),
      .s_axi_arlock  (xd_arlock[gb]),
      .s_axi_arcache (xd_arcache[gb*4 +: 4]),
      .s_axi_arprot  (xd_arprot[gb*3 +: 3]),
      .s_axi_arregion(4'h0),
      .s_axi_arqos   (4'h0),
      .s_axi_arvalid (xd_arvalid[gb]),
      .s_axi_arready (xd_arready[gb]),
      .s_axi_rid     (xd_rid[gb*16 +: 16]),
      .s_axi_rdata   (xd_rdata[gb*512 +: 512]),
      .s_axi_rresp   (xd_rresp[gb*2 +: 2]),
      .s_axi_rlast   (xd_rlast[gb]),
      .s_axi_rvalid  (xd_rvalid[gb]),
      .s_axi_rready  (xd_rready[gb]),

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
      .s_axi_aclk    (gen_clk_extra_a1),
      .s_axi_aresetn (rst_n_local),
      .s_axi_awid    (rsd_awid),
      .s_axi_awaddr  (rsd_awaddr[21:0]),   // 4 MB bank window ([21:0], 65536 lines)
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
      .s_axi_araddr  (rsd_araddr[21:0]),   // 4 MB bank window ([21:0], 65536 lines)
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

    // ---- kernel-side register slice: XBAR_KRN MI gb -> (reg) -> BRAM_CTL_KRN.
    // Mirrors RS_DMA exactly (review-confirmed RISK: the bare crossbar->
    // controller hop is the twice-measured -1.75/-1.147 ns inter-SLR failure
    // mode of this design family). Excluded from the pblock so it floats to
    // the SLR boundary and both hops route reg-to-reg through Laguna.
    logic         rsk_awid, rsk_arid, rsk_bid, rsk_rid;
    logic [63:0]  rsk_awaddr, rsk_araddr;
    logic  [7:0]  rsk_awlen, rsk_arlen;
    logic  [2:0]  rsk_awsize, rsk_arsize, rsk_awprot, rsk_arprot;
    logic  [1:0]  rsk_awburst, rsk_arburst, rsk_bresp, rsk_rresp;
    logic         rsk_awlock, rsk_arlock;
    logic  [3:0]  rsk_awcache, rsk_arcache;
    logic         rsk_awvalid, rsk_awready, rsk_arvalid, rsk_arready;
    logic [511:0] rsk_wdata, rsk_rdata;
    logic  [63:0] rsk_wstrb;
    logic         rsk_wlast, rsk_wvalid, rsk_wready;
    logic         rsk_bvalid, rsk_bready, rsk_rlast, rsk_rvalid, rsk_rready;

    axi_register_slice_krn RS_KRN (
      .aclk    (gen_clk_extra_a1),
      .aresetn (rst_n_local),
      .s_axi_awid    (kb_awid[gb]),
      .s_axi_awaddr  (kb_awaddr[gb*64 +: 64]),
      .s_axi_awlen   (kb_awlen[gb*8 +: 8]),
      .s_axi_awsize  (kb_awsize[gb*3 +: 3]),
      .s_axi_awburst (kb_awburst[gb*2 +: 2]),
      .s_axi_awlock  (kb_awlock[gb]),
      .s_axi_awcache (kb_awcache[gb*4 +: 4]),
      .s_axi_awprot  (kb_awprot[gb*3 +: 3]),
      .s_axi_awregion(4'h0),
      .s_axi_awqos   (4'h0),
      .s_axi_awvalid (kb_awvalid[gb]),
      .s_axi_awready (kb_awready[gb]),
      .s_axi_wdata   (kb_wdata[gb*512 +: 512]),
      .s_axi_wstrb   (kb_wstrb[gb*64 +: 64]),
      .s_axi_wlast   (kb_wlast[gb]),
      .s_axi_wvalid  (kb_wvalid[gb]),
      .s_axi_wready  (kb_wready[gb]),
      .s_axi_bid     (kb_bid[gb]),
      .s_axi_bresp   (kb_bresp[gb*2 +: 2]),
      .s_axi_bvalid  (kb_bvalid[gb]),
      .s_axi_bready  (kb_bready[gb]),
      .s_axi_arid    (kb_arid[gb]),
      .s_axi_araddr  (kb_araddr[gb*64 +: 64]),
      .s_axi_arlen   (kb_arlen[gb*8 +: 8]),
      .s_axi_arsize  (kb_arsize[gb*3 +: 3]),
      .s_axi_arburst (kb_arburst[gb*2 +: 2]),
      .s_axi_arlock  (kb_arlock[gb]),
      .s_axi_arcache (kb_arcache[gb*4 +: 4]),
      .s_axi_arprot  (kb_arprot[gb*3 +: 3]),
      .s_axi_arregion(4'h0),
      .s_axi_arqos   (4'h0),
      .s_axi_arvalid (kb_arvalid[gb]),
      .s_axi_arready (kb_arready[gb]),
      .s_axi_rid     (kb_rid[gb]),
      .s_axi_rdata   (kb_rdata[gb*512 +: 512]),
      .s_axi_rresp   (kb_rresp[gb*2 +: 2]),
      .s_axi_rlast   (kb_rlast[gb]),
      .s_axi_rvalid  (kb_rvalid[gb]),
      .s_axi_rready  (kb_rready[gb]),

      .m_axi_awid    (rsk_awid),
      .m_axi_awaddr  (rsk_awaddr),
      .m_axi_awlen   (rsk_awlen),
      .m_axi_awsize  (rsk_awsize),
      .m_axi_awburst (rsk_awburst),
      .m_axi_awlock  (rsk_awlock),
      .m_axi_awcache (rsk_awcache),
      .m_axi_awprot  (rsk_awprot),
      .m_axi_awregion(),
      .m_axi_awqos   (),
      .m_axi_awvalid (rsk_awvalid),
      .m_axi_awready (rsk_awready),
      .m_axi_wdata   (rsk_wdata),
      .m_axi_wstrb   (rsk_wstrb),
      .m_axi_wlast   (rsk_wlast),
      .m_axi_wvalid  (rsk_wvalid),
      .m_axi_wready  (rsk_wready),
      .m_axi_bid     (rsk_bid),
      .m_axi_bresp   (rsk_bresp),
      .m_axi_bvalid  (rsk_bvalid),
      .m_axi_bready  (rsk_bready),
      .m_axi_arid    (rsk_arid),
      .m_axi_araddr  (rsk_araddr),
      .m_axi_arlen   (rsk_arlen),
      .m_axi_arsize  (rsk_arsize),
      .m_axi_arburst (rsk_arburst),
      .m_axi_arlock  (rsk_arlock),
      .m_axi_arcache (rsk_arcache),
      .m_axi_arprot  (rsk_arprot),
      .m_axi_arregion(),
      .m_axi_arqos   (),
      .m_axi_arvalid (rsk_arvalid),
      .m_axi_arready (rsk_arready),
      .m_axi_rid     (rsk_rid),
      .m_axi_rdata   (rsk_rdata),
      .m_axi_rresp   (rsk_rresp),
      .m_axi_rlast   (rsk_rlast),
      .m_axi_rvalid  (rsk_rvalid),
      .m_axi_rready  (rsk_rready)
    );

    // ---- kernel-side BRAM controller (RS_KRN -> URAM port B) ----
    axi_bram_ctrl_krn BRAM_CTL_KRN (
      .s_axi_aclk    (gen_clk_extra_a1),
      .s_axi_aresetn (rst_n_local),
      .s_axi_awid    (rsk_awid),
      .s_axi_awaddr  (rsk_awaddr[21:0]),   // 4 MB bank window ([21:0])
      .s_axi_awlen   (rsk_awlen),
      .s_axi_awsize  (rsk_awsize),
      .s_axi_awburst (rsk_awburst),
      .s_axi_awlock  (rsk_awlock),
      .s_axi_awcache (rsk_awcache),
      .s_axi_awprot  (rsk_awprot),
      .s_axi_awvalid (rsk_awvalid),
      .s_axi_awready (rsk_awready),
      .s_axi_wdata   (rsk_wdata),
      .s_axi_wstrb   (rsk_wstrb),
      .s_axi_wlast   (rsk_wlast),
      .s_axi_wvalid  (rsk_wvalid),
      .s_axi_wready  (rsk_wready),
      .s_axi_bid     (rsk_bid),
      .s_axi_bresp   (rsk_bresp),
      .s_axi_bvalid  (rsk_bvalid),
      .s_axi_bready  (rsk_bready),
      .s_axi_arid    (rsk_arid),
      .s_axi_araddr  (rsk_araddr[21:0]),   // 4 MB bank window ([21:0])
      .s_axi_arlen   (rsk_arlen),
      .s_axi_arsize  (rsk_arsize),
      .s_axi_arburst (rsk_arburst),
      .s_axi_arlock  (rsk_arlock),
      .s_axi_arcache (rsk_arcache),
      .s_axi_arprot  (rsk_arprot),
      .s_axi_arvalid (rsk_arvalid),
      .s_axi_arready (rsk_arready),
      .s_axi_rid     (rsk_rid),
      .s_axi_rdata   (rsk_rdata),
      .s_axi_rresp   (rsk_rresp),
      .s_axi_rlast   (rsk_rlast),
      .s_axi_rvalid  (rsk_rvalid),
      .s_axi_rready  (rsk_rready),
      .bram_rst_a    (bram_b_rst),
      .bram_clk_a    (),
      .bram_en_a     (bram_b_en),
      .bram_we_a     (bram_b_we),
      .bram_addr_a   (bram_b_addr),
      .bram_wrdata_a (bram_b_wrdata),
      .bram_rddata_a (bram_b_rddata)
    );

    // ---- 4 MB true-dual-port URAM bank (512b x 65536); 1-cycle read latency ----
    // MEMORY_SIZE is in BITS: 4 MB = 33554432 bits. 65536 lines -> ADDR_WIDTH 16.
    // Two banks = 8 MB staging total (~128 URAM blocks each; see floorplan).
    xpm_memory_tdpram #(
      .MEMORY_SIZE        (33554432),
      .MEMORY_PRIMITIVE   ("ultra"),
      .CLOCKING_MODE      ("common_clock"),
      .WRITE_DATA_WIDTH_A (512),
      .READ_DATA_WIDTH_A  (512),
      .BYTE_WRITE_WIDTH_A (8),
      .ADDR_WIDTH_A       (16),
      .READ_LATENCY_A     (1),
      .WRITE_MODE_A       ("no_change"),
      .WRITE_DATA_WIDTH_B (512),
      .READ_DATA_WIDTH_B  (512),
      .BYTE_WRITE_WIDTH_B (8),
      .ADDR_WIDTH_B       (16),
      .READ_LATENCY_B     (1),
      .WRITE_MODE_B       ("no_change")
    ) BUF_URAM (
      .clka   (gen_clk_extra_a1),
      .rsta   (bram_a_rst),
      .ena    (bram_a_en),
      .wea    (bram_a_we),
      .addra  (bram_a_addr[21:6]),   // 16-bit line select (65536 x 512b)
      .dina   (bram_a_wrdata),
      .douta  (bram_a_rddata),
      .regcea (1'b1),
      .injectsbiterra (1'b0),
      .injectdbiterra (1'b0),
      .sbiterra (),
      .dbiterra (),
      .clkb   (gen_clk_extra_a1),
      .rstb   (bram_b_rst),
      .enb    (bram_b_en),
      .web    (bram_b_we),
      .addrb  (bram_b_addr[21:6]),   // 16-bit line select (65536 x 512b)
      .dinb   (bram_b_wrdata),
      .doutb  (bram_b_rddata),
      .regceb (1'b1),
      .injectsbiterrb (1'b0),
      .injectdbiterrb (1'b0),
      .sbiterrb (),
      .dbiterrb (),
      .sleep  (1'b0)
    );

    end  // BANK[gb]

  end
  endgenerate

endmodule
