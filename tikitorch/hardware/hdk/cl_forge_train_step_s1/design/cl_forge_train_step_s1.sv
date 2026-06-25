// =============================================================================
// cl_forge_train_step_s1.sv — DDR variant CL for the S1 transformer-block TRAINER kernel
// forge_train_step_s1 (forward + backward + dyadic-Adam + fused one-kick trainer, multiplier-free
// except the single attention soft p*V blend).
//
// CLKGEN+CDC REARCHITECTURE (this revision): the HLS kernel forge_train_step_s1 is csim-bit-exact and
// post-route CLOSES at ~144 MHz but MISSES the shell-fixed 250 MHz clk_main_a0 (WNS -2.958 ns). clk_main_a0
// cannot be slowed by recipe on F2. The fix: instantiate AWS_CLK_GEN in the CL, generate clk_extra_a1 =
// 125 MHz (Recipe A1; 8 ns >> the 6.94 ns the kernel needs), and run ONLY the kernel on that domain. The
// HLS RTL + 13-bit s_axi_control ABI + 512-bit m_axi_gmem master + DDR-buffer-at-offset-0 ABI are ALL
// UNCHANGED. Two axi_clock_converter (CDC) IPs bridge the two AXI boundaries the kernel touches:
//   - s1_axil_cdc  : OCL control,  AXI4LITE 32b, slave clk_main_a0/250 -> master clk_extra_a1/125
//   - s1_axi_cdc   : kernel m_axi, AXI4 512b/ID1, slave clk_extra_a1/125 -> master clk_main_a0/250
// The crossbar + sh_ddr + DMA-PCIS staging ALL stay on clk_main_a0/250 (the shell domain). The SDA AXI-Lite
// slave (previously tied off via unused_cl_sda_template.inc) now drives AWS_CLK_GEN's s_axil_ctrl_* control
// plane (the host polls MMCM_LOCK_REG @0x0005_8020 and deasserts cl_rst_a1_n via the clkgen SDK after AFI load).
//
// WHY DDR (NOT PCIM): the sibling PCIM->host-DRAM path loads on F2 but cannot be DRIVEN for inference from
// EC2 userspace — a raw host buffer isn't IOMMU-granted to the FPGA, so CL-mastered PCIM reads never complete
// (Stage 3b finding). On-card DDR is the drivable home for the S1 working set (~6.15 MB, S1_BUFLEN int32):
// the host STAGES it into DDR over the shell DMA-PCIS, the kernel READS *and WRITES* it via its m_axi (the
// S1 trainer writes grads / W / loss back) — both reach DDR through a 2->1 AXI crossbar; no host IOMMU in
// the loop. PCIM is unused here (tied off).
//
//   ocl  (AXI4-Lite) --[reg slice 250]--[s1_axil_cdc 250->125]--> kernel s_axi_control[12:0] (125 MHz)
//   sh_cl_dma_pcis (512b, host staging, 250) --\
//                                               >--[axi_crossbar 2->1, SASD, 250]--> sh_ddr --> DDR4
//   kernel m_axi gmem (512b, 125) --[s1_axi_cdc 125->250]--/
//
// Buffer lives at DDR address 0: the host DMA-PCIS-writes the working set to DDR[0..], the kernel's BUFFER
// pointer register is set to 0 and reads/writes DDR[0..]. DMA staging and kernel m_axi never overlap in time
// (host stages fully -> kicks mode 3 -> kernel runs N steps resident in DDR), so the SASD crossbar needs no
// concurrent arbitration and the master ID stays 16-bit.
//
// The S1 control bundle carries buffer offset (0x18/0x1c) + mode (0x24) + a1/a2/a3, so the OCL address slice
// stays [12:0] (13-bit) — IDENTICAL to muzero_ddr, NOT the pc_train 5-bit width (which would silently miss
// the buffer-ptr / mode / arg registers).
// =============================================================================

`include "cl_forge_train_step_s1_defines.vh"

module cl_forge_train_step_s1

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
// NOTE: unused_cl_sda_template.inc is DELETED — the SDA AXI-Lite slave now drives AWS_CLK_GEN's control plane.
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
  // AWS_CLK_GEN — generate clk_extra_a1 = 125 MHz (Recipe A1) for the kernel.
  //   Input clocks: clk_main_a0 (250) + clk_hbm_ref (100, MMCM reference).
  //   Control plane: the shell SDA AXI-Lite slave drives s_axil_ctrl_* directly (no SDA-BRAM, no
  //   cl_sda_axil_xbar — S1 has no SDA-BRAM; AWS_CLK_GEN self-decodes 0x0005_xxxx and ignores below).
  //   Only group A is enabled (we need a1 = 125 MHz); B/C/HBM MMCMs disabled to save area.
  // =========================================================================
  logic  gen_clk_hbm_ref;
  logic  gen_clk_main_a0;            // 250 MHz pass-through (shell domain)
  logic  gen_clk_extra_a1;          // 125 MHz (Recipe A1)  <-- KERNEL CLOCK
  logic  gen_clk_extra_a2, gen_clk_extra_a3;
  logic  gen_clk_extra_b0, gen_clk_extra_b1, gen_clk_extra_c0, gen_clk_extra_c1;
  logic  gen_clk_hbm_axi;
  logic  gen_rst_hbm_axi_n, gen_rst_hbm_ref_n;
  logic  gen_rst_c1_n, gen_rst_c0_n, gen_rst_b1_n, gen_rst_b0_n;
  logic  gen_rst_a3_n, gen_rst_a2_n;
  logic  gen_rst_a1_n;             // reset sync'd to clk_extra_a1 (125 MHz KERNEL domain)
  logic  gen_rst_main_n;            // reset sync'd to clk_main_a0 (250 MHz shell domain)

  // SDA reset synchronizer (matches cl_mem_perf: rst feeding the pre-clkgen SDA path is clk_main_a0-sync'd).
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
    // AXI-Lite control plane from the shell SDA slave (clk_main_a0 domain). Shell SDA carries wstrb.
    .s_axil_ctrl_awaddr  (sda_cl_awaddr  ), .s_axil_ctrl_awvalid (sda_cl_awvalid ), .s_axil_ctrl_awready (cl_sda_awready),
    .s_axil_ctrl_wdata   (sda_cl_wdata   ), .s_axil_ctrl_wstrb   (sda_cl_wstrb   ), .s_axil_ctrl_wvalid  (sda_cl_wvalid ), .s_axil_ctrl_wready (cl_sda_wready),
    .s_axil_ctrl_bresp   (cl_sda_bresp   ), .s_axil_ctrl_bvalid  (cl_sda_bvalid  ), .s_axil_ctrl_bready  (sda_cl_bready ),
    .s_axil_ctrl_araddr  (sda_cl_araddr  ), .s_axil_ctrl_arvalid (sda_cl_arvalid ), .s_axil_ctrl_arready (cl_sda_arready),
    .s_axil_ctrl_rdata   (cl_sda_rdata   ), .s_axil_ctrl_rresp   (cl_sda_rresp   ), .s_axil_ctrl_rvalid  (cl_sda_rvalid ), .s_axil_ctrl_rready (sda_cl_rready),
    // generated clocks:
    .o_clk_hbm_ref       (gen_clk_hbm_ref ), .o_clk_main_a0   (gen_clk_main_a0 ),
    .o_clk_extra_a1      (gen_clk_extra_a1), .o_clk_extra_a2  (gen_clk_extra_a2), .o_clk_extra_a3 (gen_clk_extra_a3),
    .o_clk_extra_b0      (gen_clk_extra_b0), .o_clk_extra_b1  (gen_clk_extra_b1),
    .o_clk_extra_c0      (gen_clk_extra_c0), .o_clk_extra_c1  (gen_clk_extra_c1), .o_clk_hbm_axi  (gen_clk_hbm_axi),
    // generated resets:
    .o_cl_rst_hbm_axi_n  (gen_rst_hbm_axi_n), .o_cl_rst_hbm_ref_n (gen_rst_hbm_ref_n),
    .o_cl_rst_c1_n       (gen_rst_c1_n    ), .o_cl_rst_c0_n   (gen_rst_c0_n    ), .o_cl_rst_b1_n  (gen_rst_b1_n   ), .o_cl_rst_b0_n (gen_rst_b0_n),
    .o_cl_rst_a3_n       (gen_rst_a3_n    ), .o_cl_rst_a2_n   (gen_rst_a2_n    ), .o_cl_rst_a1_n  (gen_rst_a1_n   ), .o_cl_rst_main_n (gen_rst_main_n)
  );

  // =========================================================================
  // OCL register slice -> kernel s_axi_control (proven; single kernel, no fan-out). Stays at 250 MHz.
  // =========================================================================
  logic        ocl_aw_valid_pre, ocl_aw_ready_pre;  logic [31:0] ocl_aw_addr_pre;
  logic        ocl_w_valid_pre, ocl_w_ready_pre;    logic [31:0] ocl_w_data_pre;  logic [3:0] ocl_w_strb_pre;
  logic        ocl_b_valid_pre, ocl_b_ready_pre;    logic  [1:0] ocl_b_resp_pre;
  logic        ocl_ar_valid_pre, ocl_ar_ready_pre;  logic [31:0] ocl_ar_addr_pre;
  logic        ocl_r_valid_pre, ocl_r_ready_pre;    logic [31:0] ocl_r_data_pre;  logic [1:0] ocl_r_resp_pre;

  s1_ocl_slice OCL_REG_SLC (
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
  // CDC-1 — OCL control: 250 MHz (clk_main_a0) -> 125 MHz (gen_clk_extra_a1), AXI4-Lite 32-bit.
  //   Slave  = the ocl_*_pre nets from the reg slice (clk_main_a0 / rst_main_n).
  //   Master = kctl_* nets that feed the kernel s_axi_control (gen_clk_extra_a1 / gen_rst_a1_n).
  // =========================================================================
  logic        kctl_aw_valid, kctl_aw_ready;  logic [31:0] kctl_aw_addr;
  logic        kctl_w_valid,  kctl_w_ready;   logic [31:0] kctl_w_data;  logic [3:0] kctl_w_strb;
  logic        kctl_b_valid,  kctl_b_ready;   logic  [1:0] kctl_b_resp;
  logic        kctl_ar_valid, kctl_ar_ready;  logic [31:0] kctl_ar_addr;
  logic        kctl_r_valid,  kctl_r_ready;   logic [31:0] kctl_r_data;  logic  [1:0] kctl_r_resp;

  s1_axil_cdc OCL_CDC (
    .s_axi_aclk    (clk_main_a0),     .s_axi_aresetn (rst_main_n),          // 250 MHz slave
    .s_axi_awaddr  (ocl_aw_addr_pre), .s_axi_awprot  (3'd0), .s_axi_awvalid (ocl_aw_valid_pre), .s_axi_awready (ocl_aw_ready_pre),
    .s_axi_wdata   (ocl_w_data_pre),  .s_axi_wstrb   (ocl_w_strb_pre),      .s_axi_wvalid  (ocl_w_valid_pre),  .s_axi_wready  (ocl_w_ready_pre),
    .s_axi_bresp   (ocl_b_resp_pre),  .s_axi_bvalid  (ocl_b_valid_pre),     .s_axi_bready  (ocl_b_ready_pre),
    .s_axi_araddr  (ocl_ar_addr_pre), .s_axi_arprot  (3'd0), .s_axi_arvalid (ocl_ar_valid_pre), .s_axi_arready (ocl_ar_ready_pre),
    .s_axi_rdata   (ocl_r_data_pre),  .s_axi_rresp   (ocl_r_resp_pre),      .s_axi_rvalid  (ocl_r_valid_pre),  .s_axi_rready  (ocl_r_ready_pre),
    .m_axi_aclk    (gen_clk_extra_a1),.m_axi_aresetn (gen_rst_a1_n),        // 125 MHz master
    .m_axi_awaddr  (kctl_aw_addr),    .m_axi_awprot  (), .m_axi_awvalid (kctl_aw_valid), .m_axi_awready (kctl_aw_ready),
    .m_axi_wdata   (kctl_w_data),     .m_axi_wstrb   (kctl_w_strb),         .m_axi_wvalid  (kctl_w_valid),  .m_axi_wready  (kctl_w_ready),
    .m_axi_bresp   (kctl_b_resp),     .m_axi_bvalid  (kctl_b_valid),        .m_axi_bready  (kctl_b_ready),
    .m_axi_araddr  (kctl_ar_addr),    .m_axi_arprot  (), .m_axi_arvalid (kctl_ar_valid), .m_axi_arready (kctl_ar_ready),
    .m_axi_rdata   (kctl_r_data),     .m_axi_rresp   (kctl_r_resp),         .m_axi_rvalid  (kctl_r_valid),  .m_axi_rready  (kctl_r_ready)
  );

  // =========================================================================
  // Kernel m_axi (gmem) nets — now on the 125 MHz (gen_clk_extra_a1) domain. 1-bit ID (single outstanding).
  // These feed the slave side of CDC-2; the CDC master (250 MHz) feeds crossbar SI0.
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

  forge_train_step_s1 u_kernel (
    .ap_clk     (gen_clk_extra_a1),  .ap_rst_n (gen_rst_a1_n),  .interrupt (krn_irq),  // KERNEL on 125 MHz clk_extra_a1
    .s_axi_control_AWADDR  (kctl_aw_addr[12:0]), .s_axi_control_AWVALID (kctl_aw_valid), .s_axi_control_AWREADY (kctl_aw_ready),
    .s_axi_control_WDATA   (kctl_w_data),  .s_axi_control_WSTRB (kctl_w_strb), .s_axi_control_WVALID (kctl_w_valid), .s_axi_control_WREADY (kctl_w_ready),
    .s_axi_control_BRESP   (kctl_b_resp),  .s_axi_control_BVALID (kctl_b_valid), .s_axi_control_BREADY (kctl_b_ready),
    .s_axi_control_ARADDR  (kctl_ar_addr[12:0]), .s_axi_control_ARVALID (kctl_ar_valid), .s_axi_control_ARREADY (kctl_ar_ready),
    .s_axi_control_RDATA   (kctl_r_data),  .s_axi_control_RRESP (kctl_r_resp), .s_axi_control_RVALID (kctl_r_valid), .s_axi_control_RREADY (kctl_r_ready),
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
  // CDC-2 — kernel memory: 125 MHz (gen_clk_extra_a1) -> 250 MHz (clk_main_a0), AXI4 512-bit, ID width 1.
  //   Slave  = the kernel m_axi nets krn_* (gen_clk_extra_a1 / gen_rst_a1_n).
  //   Master = kmem_* nets feeding crossbar SI0 (clk_main_a0 / rst_main_n_sync), exactly where krn_* fed before.
  // =========================================================================
  logic [63:0]  kmem_awaddr, kmem_araddr;  logic [7:0] kmem_awlen, kmem_arlen;
  logic  [2:0]  kmem_awsize, kmem_arsize;  logic [1:0] kmem_awburst, kmem_arburst;
  logic  [3:0]  kmem_awcache, kmem_arcache; logic [2:0] kmem_awprot, kmem_arprot;
  logic  [3:0]  kmem_awqos, kmem_arqos;     logic kmem_awlock, kmem_arlock;  // AXI4 lock = 1 bit
  logic         kmem_awid, kmem_arid, kmem_bid, kmem_rid;
  logic         kmem_awvalid, kmem_awready, kmem_arvalid, kmem_arready;
  logic [511:0] kmem_wdata, kmem_rdata;     logic [63:0] kmem_wstrb;
  logic         kmem_wlast, kmem_wvalid, kmem_wready;
  logic  [1:0]  kmem_bresp, kmem_rresp;     logic kmem_bvalid, kmem_bready, kmem_rlast, kmem_rvalid, kmem_rready;

  s1_axi_cdc GMEM_CDC (
    .s_axi_aclk    (gen_clk_extra_a1), .s_axi_aresetn (gen_rst_a1_n),       // 125 MHz slave = kernel m_axi
    .s_axi_awid(krn_awid), .s_axi_awaddr(krn_awaddr), .s_axi_awlen(krn_awlen), .s_axi_awsize(krn_awsize),
    .s_axi_awburst(krn_awburst), .s_axi_awlock(krn_awlock[0]), .s_axi_awcache(krn_awcache), .s_axi_awprot(krn_awprot),
    .s_axi_awqos(krn_awqos), .s_axi_awregion(4'h0), .s_axi_awvalid(krn_awvalid), .s_axi_awready(krn_awready),
    .s_axi_wdata(krn_wdata), .s_axi_wstrb(krn_wstrb), .s_axi_wlast(krn_wlast), .s_axi_wvalid(krn_wvalid), .s_axi_wready(krn_wready),
    .s_axi_bid(krn_bid), .s_axi_bresp(krn_bresp), .s_axi_bvalid(krn_bvalid), .s_axi_bready(krn_bready),
    .s_axi_arid(krn_arid), .s_axi_araddr(krn_araddr), .s_axi_arlen(krn_arlen), .s_axi_arsize(krn_arsize),
    .s_axi_arburst(krn_arburst), .s_axi_arlock(krn_arlock[0]), .s_axi_arcache(krn_arcache), .s_axi_arprot(krn_arprot),
    .s_axi_arqos(krn_arqos), .s_axi_arregion(4'h0), .s_axi_arvalid(krn_arvalid), .s_axi_arready(krn_arready),
    .s_axi_rid(krn_rid), .s_axi_rdata(krn_rdata), .s_axi_rresp(krn_rresp), .s_axi_rlast(krn_rlast), .s_axi_rvalid(krn_rvalid), .s_axi_rready(krn_rready),
    .m_axi_aclk    (clk_main_a0), .m_axi_aresetn (rst_main_n_sync),         // 250 MHz master = crossbar SI0
    .m_axi_awid(kmem_awid), .m_axi_awaddr(kmem_awaddr), .m_axi_awlen(kmem_awlen), .m_axi_awsize(kmem_awsize),
    .m_axi_awburst(kmem_awburst), .m_axi_awlock(kmem_awlock), .m_axi_awcache(kmem_awcache), .m_axi_awprot(kmem_awprot),
    .m_axi_awqos(kmem_awqos), .m_axi_awregion(), .m_axi_awvalid(kmem_awvalid), .m_axi_awready(kmem_awready),
    .m_axi_wdata(kmem_wdata), .m_axi_wstrb(kmem_wstrb), .m_axi_wlast(kmem_wlast), .m_axi_wvalid(kmem_wvalid), .m_axi_wready(kmem_wready),
    .m_axi_bid(kmem_bid), .m_axi_bresp(kmem_bresp), .m_axi_bvalid(kmem_bvalid), .m_axi_bready(kmem_bready),
    .m_axi_arid(kmem_arid), .m_axi_araddr(kmem_araddr), .m_axi_arlen(kmem_arlen), .m_axi_arsize(kmem_arsize),
    .m_axi_arburst(kmem_arburst), .m_axi_arlock(kmem_arlock), .m_axi_arcache(kmem_arcache), .m_axi_arprot(kmem_arprot),
    .m_axi_arqos(kmem_arqos), .m_axi_arregion(), .m_axi_arvalid(kmem_arvalid), .m_axi_arready(kmem_arready),
    .m_axi_rid(kmem_rid), .m_axi_rdata(kmem_rdata), .m_axi_rresp(kmem_rresp), .m_axi_rlast(kmem_rlast), .m_axi_rvalid(kmem_rvalid), .m_axi_rready(kmem_rready)
  );

  // =========================================================================
  // PCIS register slice — breaks the route-bound shell DMA-PCIS <-> crossbar SI1 long net (TIMING-CLOSE fix).
  //   The post-route worst path is sh_ddr DDR4 wready -> XBAR_DDR wready mux -> high-fanout net to the STATIC
  //   shell DMA-PCIS endpoint (~89% route, crossing SLRs, WNS violated at clk_main_a0/250). The CL previously
  //   wired sh_cl_dma_pcis_* / cl_sh_dma_pcis_* DIRECTLY into crossbar SI1 with no register slice. The proven
  //   AWS reference CLs (cl_mem_perf / cl_dram_hbm_dma) pipeline this exact path with per-SLR PCIS reg slices
  //   (AXI4_REG_SLC_PCIS_SLR*). s1_pcis_rs (gen_ip.tcl): AXI4, 512b, 64-bit addr, 16-bit ID, fully registered.
  //   SLAVE port  = shell side (sh_cl_dma_pcis_* / cl_sh_dma_pcis_*, exactly where crossbar SI1 connected).
  //   MASTER port = new pcis_* nets that REPLACE sh_cl_dma_pcis_* / cl_sh_dma_pcis_* in the crossbar SI1 slot.
  //   Net flow: shell -> s1_pcis_rs (slave) -> (master) pcis_* -> crossbar SI1. Responses flow the other way:
  //   crossbar SI1 -> reg-slice master -> reg-slice slave -> shell. PURE clk_main_a0/250 (NOT a CDC), same
  //   clock+reset as the crossbar. Latency-only; staging stays byte-exact. The kernel SI0 (kmem_*) is unchanged.
  // =========================================================================
  logic [15:0]  pcis_awid;   logic [63:0] pcis_awaddr;  logic [7:0] pcis_awlen;  logic [2:0] pcis_awsize;
  logic  [1:0]  pcis_awburst; logic pcis_awlock; logic [3:0] pcis_awcache; logic [2:0] pcis_awprot;
  logic  [3:0]  pcis_awqos;  logic [3:0] pcis_awregion; logic pcis_awvalid, pcis_awready;
  logic [511:0] pcis_wdata;  logic [63:0] pcis_wstrb;  logic pcis_wlast, pcis_wvalid, pcis_wready;
  logic [15:0]  pcis_bid;    logic [1:0] pcis_bresp;   logic pcis_bvalid, pcis_bready;
  logic [15:0]  pcis_arid;   logic [63:0] pcis_araddr;  logic [7:0] pcis_arlen;  logic [2:0] pcis_arsize;
  logic  [1:0]  pcis_arburst; logic pcis_arlock; logic [3:0] pcis_arcache; logic [2:0] pcis_arprot;
  logic  [3:0]  pcis_arqos;  logic [3:0] pcis_arregion; logic pcis_arvalid, pcis_arready;
  logic [15:0]  pcis_rid;    logic [511:0] pcis_rdata;  logic [1:0] pcis_rresp; logic pcis_rlast, pcis_rvalid, pcis_rready;

  s1_pcis_rs PCIS_REG_SLC (
    .aclk          (clk_main_a0),   .aresetn (rst_main_n_sync),
    // ---- SLAVE = shell DMA-PCIS (host staging). Exactly the ports crossbar SI1 used to connect to. ----
    .s_axi_awid    (sh_cl_dma_pcis_awid),    .s_axi_awaddr  (sh_cl_dma_pcis_awaddr),
    .s_axi_awlen   (sh_cl_dma_pcis_awlen),   .s_axi_awsize  (sh_cl_dma_pcis_awsize),
    .s_axi_awburst (sh_cl_dma_pcis_awburst), .s_axi_awlock  (sh_cl_dma_pcis_awlock),
    .s_axi_awcache (sh_cl_dma_pcis_awcache), .s_axi_awprot  (sh_cl_dma_pcis_awprot),
    .s_axi_awregion(4'h0),                   .s_axi_awqos   (4'h0),
    .s_axi_awvalid (sh_cl_dma_pcis_awvalid), .s_axi_awready (cl_sh_dma_pcis_awready),
    .s_axi_wdata   (sh_cl_dma_pcis_wdata),   .s_axi_wstrb   (sh_cl_dma_pcis_wstrb),
    .s_axi_wlast   (sh_cl_dma_pcis_wlast),   .s_axi_wvalid  (sh_cl_dma_pcis_wvalid),
    .s_axi_wready  (cl_sh_dma_pcis_wready),
    .s_axi_bid     (cl_sh_dma_pcis_bid),     .s_axi_bresp   (cl_sh_dma_pcis_bresp),
    .s_axi_bvalid  (cl_sh_dma_pcis_bvalid),  .s_axi_bready  (sh_cl_dma_pcis_bready),
    .s_axi_arid    (sh_cl_dma_pcis_arid),    .s_axi_araddr  (sh_cl_dma_pcis_araddr),
    .s_axi_arlen   (sh_cl_dma_pcis_arlen),   .s_axi_arsize  (sh_cl_dma_pcis_arsize),
    .s_axi_arburst (sh_cl_dma_pcis_arburst), .s_axi_arlock  (sh_cl_dma_pcis_arlock),
    .s_axi_arcache (sh_cl_dma_pcis_arcache), .s_axi_arprot  (sh_cl_dma_pcis_arprot),
    .s_axi_arregion(4'h0),                   .s_axi_arqos   (4'h0),
    .s_axi_arvalid (sh_cl_dma_pcis_arvalid), .s_axi_arready (cl_sh_dma_pcis_arready),
    .s_axi_rid     (cl_sh_dma_pcis_rid),     .s_axi_rdata   (cl_sh_dma_pcis_rdata),
    .s_axi_rresp   (cl_sh_dma_pcis_rresp),   .s_axi_rlast   (cl_sh_dma_pcis_rlast),
    .s_axi_rvalid  (cl_sh_dma_pcis_rvalid),  .s_axi_rready  (sh_cl_dma_pcis_rready),
    // ---- MASTER = pcis_* (toward crossbar SI1). ----
    .m_axi_awid    (pcis_awid),    .m_axi_awaddr  (pcis_awaddr),
    .m_axi_awlen   (pcis_awlen),   .m_axi_awsize  (pcis_awsize),
    .m_axi_awburst (pcis_awburst), .m_axi_awlock  (pcis_awlock),
    .m_axi_awcache (pcis_awcache), .m_axi_awprot  (pcis_awprot),
    .m_axi_awregion(pcis_awregion),.m_axi_awqos   (pcis_awqos),
    .m_axi_awvalid (pcis_awvalid), .m_axi_awready (pcis_awready),
    .m_axi_wdata   (pcis_wdata),   .m_axi_wstrb   (pcis_wstrb),
    .m_axi_wlast   (pcis_wlast),   .m_axi_wvalid  (pcis_wvalid),
    .m_axi_wready  (pcis_wready),
    .m_axi_bid     (pcis_bid),     .m_axi_bresp   (pcis_bresp),
    .m_axi_bvalid  (pcis_bvalid),  .m_axi_bready  (pcis_bready),
    .m_axi_arid    (pcis_arid),    .m_axi_araddr  (pcis_araddr),
    .m_axi_arlen   (pcis_arlen),   .m_axi_arsize  (pcis_arsize),
    .m_axi_arburst (pcis_arburst), .m_axi_arlock  (pcis_arlock),
    .m_axi_arcache (pcis_arcache), .m_axi_arprot  (pcis_arprot),
    .m_axi_arregion(pcis_arregion),.m_axi_arqos   (pcis_arqos),
    .m_axi_arvalid (pcis_arvalid), .m_axi_arready (pcis_arready),
    .m_axi_rid     (pcis_rid),     .m_axi_rdata   (pcis_rdata),
    .m_axi_rresp   (pcis_rresp),   .m_axi_rlast   (pcis_rlast),
    .m_axi_rvalid  (pcis_rvalid),  .m_axi_rready  (pcis_rready)
  );

  // =========================================================================
  // 2->1 AXI crossbar: SI0 = kernel m_axi via CDC (kmem_*, ID widened 1->16), SI1 = sh_cl_dma_pcis via the
  // s1_pcis_rs register slice (pcis_*, 16-bit ID), MI = the DDR AXI slave. Both masters address the single DDR
  // region (base 0). axi_crossbar_ddr (gen_ip.tcl): NUM_SI=2, NUM_MI=1, 512b, 64-bit addr, 16-bit ID. Crossbar
  // stays fully at clk_main_a0/250.
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
  logic [15:0]  kmem_bid_16, kmem_rid_16;   // 16-bit kernel-side response IDs from the crossbar (kernel uses bit 0)

  axi_crossbar_ddr XBAR_DDR (
    .aclk    (clk_main_a0),  .aresetn (rst_main_n_sync),
    // ---- SI: kernel m_axi via CDC (ID 1->16) high, DMA-PCIS via s1_pcis_rs reg slice (pcis_*, 16-bit ID) low ----
    .s_axi_awid    ({ {15'h0, kmem_awid}, pcis_awid }),
    .s_axi_awaddr  ({ kmem_awaddr,        pcis_awaddr }),
    .s_axi_awlen   ({ kmem_awlen,         pcis_awlen }),
    .s_axi_awsize  ({ kmem_awsize,        pcis_awsize }),
    .s_axi_awburst ({ kmem_awburst,       pcis_awburst }),
    .s_axi_awlock  ({ kmem_awlock,        pcis_awlock }),
    .s_axi_awcache ({ kmem_awcache,       pcis_awcache }),
    .s_axi_awprot  ({ kmem_awprot,        pcis_awprot }),
    .s_axi_awqos   ({ kmem_awqos,         pcis_awqos }),
    .s_axi_awvalid ({ kmem_awvalid,       pcis_awvalid }),
    .s_axi_awready ({ kmem_awready,       pcis_awready }),
    .s_axi_wdata   ({ kmem_wdata,         pcis_wdata }),
    .s_axi_wstrb   ({ kmem_wstrb,         pcis_wstrb }),
    .s_axi_wlast   ({ kmem_wlast,         pcis_wlast }),
    .s_axi_wvalid  ({ kmem_wvalid,        pcis_wvalid }),
    .s_axi_wready  ({ kmem_wready,        pcis_wready }),
    .s_axi_bid     ({ kmem_bid_16,        pcis_bid }),
    .s_axi_bresp   ({ kmem_bresp,         pcis_bresp }),
    .s_axi_bvalid  ({ kmem_bvalid,        pcis_bvalid }),
    .s_axi_bready  ({ kmem_bready,        pcis_bready }),
    .s_axi_arid    ({ {15'h0, kmem_arid}, pcis_arid }),
    .s_axi_araddr  ({ kmem_araddr,        pcis_araddr }),
    .s_axi_arlen   ({ kmem_arlen,         pcis_arlen }),
    .s_axi_arsize  ({ kmem_arsize,        pcis_arsize }),
    .s_axi_arburst ({ kmem_arburst,       pcis_arburst }),
    .s_axi_arlock  ({ kmem_arlock,        pcis_arlock }),
    .s_axi_arcache ({ kmem_arcache,       pcis_arcache }),
    .s_axi_arprot  ({ kmem_arprot,        pcis_arprot }),
    .s_axi_arqos   ({ kmem_arqos,         pcis_arqos }),
    .s_axi_arvalid ({ kmem_arvalid,       pcis_arvalid }),
    .s_axi_arready ({ kmem_arready,       pcis_arready }),
    .s_axi_rid     ({ kmem_rid_16,        pcis_rid }),
    .s_axi_rdata   ({ kmem_rdata,         pcis_rdata }),
    .s_axi_rresp   ({ kmem_rresp,         pcis_rresp }),
    .s_axi_rlast   ({ kmem_rlast,         pcis_rlast }),
    .s_axi_rvalid  ({ kmem_rvalid,        pcis_rvalid }),
    .s_axi_rready  ({ kmem_rready,        pcis_rready }),
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
  // kernel response IDs come back 16-bit from the crossbar; the CDC master expects 1-bit (kernel uses bit 0).
  assign kmem_bid = kmem_bid_16[0];
  assign kmem_rid = kmem_rid_16[0];

  // =========================================================================
  // sh_ddr (DDR_PRESENT=1): DDR controller. AXI slave <- crossbar MI; DDR4 pins <- shell; stats. Stays at 250 MHz.
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
