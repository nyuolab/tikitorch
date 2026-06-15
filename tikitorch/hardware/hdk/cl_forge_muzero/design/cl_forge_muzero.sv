// =============================================================================
// cl_forge_muzero.sv — CL top for the dyadic (zero-DSP) Tropical MuZero CHESS kernel.
//
// WHY THIS DIFFERS FROM cl_forge_morph_lm.sv (the sibling package it adapts):
//   morph-LM staged ONE ~2 MB int32 training buffer into an ON-CL URAM over
//   DMA-PCIS, and the kernel mastered that URAM through its m_axi. That works
//   because the morph-LM buffer fits on-chip (~2 MB << the device's ~13 MB URAM).
//
//   The CHESS MuZero buffer does NOT fit: ~12 MB forward / ~6 MB train (the
//   repr.enc WR1 matrix alone is ~4.7 MB; weights+io+targets together blow past
//   what we can keep resident while still leaving URAM for the kernel's own
//   on-chip working set). So the buffer lives in HOST DRAM and the kernel reaches
//   it by MASTERING the shell PCIM interface (CL -> host memory), instead of an
//   on-CL URAM. The kernel's HLS interface is unchanged — it still sees one flat
//   `int32_t* buffer` over `m_axi bundle=gmem` (512-bit, offset=slave). Only the
//   *destination* of that master changes: PCIM (host DRAM) here vs. URAM there.
//
// DATA PATH:
//   ocl (AXI4-Lite) --[reg slice]--> kernel s_axi_control
//        host writes: buffer base addr (PCIM/host-DRAM bus addr), mode, arg1..3,
//        AP_START; polls AP_DONE / reads the scalar loss back.
//   kernel m_axi gmem (512b AXI4) --[reg slice]--> sh_cl/cl_sh PCIM  --> host DRAM
//        all weight/io/target traffic streams here (see forge_buffer_muzero*.hpp:
//        WR1 streamed in aligned 512-bit beats, resident matrices burst-loaded).
//   DMA-PCIS is UNUSED (no on-CL staging buffer) -> tied off.
//   PCIM is USED (this is the whole point) -> NOT tied off.
//
// SINGLE KERNEL (NK=1): morph-LM replicated NK=4 because each kernel+URAM was
//   cheap and SLR-local; here a single kernel already saturates one PCIM port's
//   bandwidth with the 12 MB streaming working set, and replication would
//   contend on the shared PCIM. Start at 1; multi-PCIM fan-out is future work.
//
// 2026 F2 shell port family: OCL is ocl_cl_* (in) / cl_ocl_* (out) — the renamed
//   family (May-era ocl_* names synthesize with a FLOATING control plane; see
//   SILICON_FINDINGS.md). PCIM keeps the cl_sh_pcim_* / sh_cl_pcim_* family.
//
// !!! F2-VALIDATION REQUIRED (see README §"What needs the F2 build box") !!!
//   The PCIM section below is written to the standard AWS HDK PCIM signal names
//   and widths. The EXACT port names / vector widths / *user fields for the 2026
//   F2 shell are defined in `cl_ports.vh` + the `unused_*_template.inc` files,
//   which ship with the HDK on the build box (NOT in this repo). Reconcile this
//   section against that `cl_ports.vh` on the F2 box before the first build —
//   that reconciliation is a ~30-line edit, not a redesign. Everything ABOVE the
//   PCIM section (IDs, status, reset sync, OCL slice, kernel control) is the
//   proven morph-LM wiring and should synthesize as-is.
// =============================================================================

`include "cl_forge_muzero_defines.vh"

module cl_forge_muzero

(
  `include "cl_ports.vh"
);

`include "cl_id_defines.vh"
`include "unused_flr_template.inc"
`include "unused_ddr_template.inc"
`include "unused_cl_sda_template.inc"
`include "unused_apppf_irq_template.inc"
`include "unused_dma_pcis_template.inc"   // DMA-PCIS unused here (no on-CL staging buffer)
// NOTE: unused_pcim_template.inc is intentionally NOT included — we drive PCIM below.

  assign cl_sh_id0 = `CL_SH_ID0;
  assign cl_sh_id1 = `CL_SH_ID1;
  assign cl_sh_status0 = 32'h0;
  assign cl_sh_status1 = 32'h0;
  assign cl_sh_status2 = 32'h0;
  assign cl_sh_status_vled = 16'h0;
  assign cl_sh_dma_wr_full = 1'b0;
  assign cl_sh_dma_rd_full = 1'b0;

  // ---- global reset synchronizer (async assert / sync deassert) ----
  (* ASYNC_REG = "true" *) logic [3:0] rst_pipe;
  always_ff @(posedge clk_main_a0 or negedge rst_main_n)
    if (!rst_main_n) rst_pipe <= 4'b0;
    else             rst_pipe <= {rst_pipe[2:0], 1'b1};
  wire rst_n_local = rst_pipe[3];

  // =========================================================================
  // OCL register slice (timing closure between shell and the kernel control).
  // Identical to morph-LM; for NK=1 the post-slice signals drive the single
  // kernel's s_axi_control directly (no broadcast/mux fan-out).
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

  // =========================================================================
  // Kernel m_axi (gmem) nets -> PCIM register slice. The HLS kernel emits a
  // 512-bit AXI4 master with a narrow ID (1 bit for our single-outstanding-id
  // config); the PCIM slice widens ID to 6 (gen_ip.tcl) to match the shell.
  // =========================================================================
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

  // ---- the kernel: single instance, control wired straight to the OCL slice ----
  forge_train_step_muzero u_kernel (
    .ap_clk     (clk_main_a0),
    .ap_rst_n   (rst_n_local),
    .interrupt  (krn_irq),

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

  // =========================================================================
  // PCIM path: kernel m_axi gmem --[axi_register_slice_pcim]--> shell PCIM.
  // ========================== F2-VALIDATION REQUIRED =======================
  // The slice's m_axi_* side connects to the shell PCIM ports (cl_sh_pcim_* out,
  // sh_cl_pcim_* in). The names/widths below follow the standard AWS HDK PCIM
  // contract; CONFIRM each against the F2 `cl_ports.vh` on the build box (the
  // 2026 shell may rename/resize *user fields and the ID width). The kernel side
  // is the local `krn_*` nets above and is correct as written.
  //
  //   - Kernel AWID/ARID are 1 bit; PCIM ID width is 6 (gen_ip.tcl ID_WIDTH=6).
  //     Zero-extend kernel->slice ({5'h0, krn_awid}) and drop the high bits on
  //     the response (slice->kernel) — a single in-flight ID, so this is safe.
  //   - cl_sh_pcim_awuser/aruser: tie to the shell's required default (0 unless
  //     the F2 shell documents a non-zero data-mover encoding).
  //   - The shell PCIM address space maps to host DRAM; the host passes the
  //     buffer's PCIM bus address into the kernel via the OCL `buffer` offset
  //     register (offset=slave), so no address translation is done in the CL.
  //
  // Sketch (UNCOMMENT + reconcile names on the F2 box):
  //
  //   axi_register_slice_pcim PCIM_REG_SLC (
  //     .aclk    (clk_main_a0),
  //     .aresetn (rst_n_local),
  //     // ---- slave side: kernel master (widen ID 1 -> 6) ----
  //     .s_axi_awid    ({5'h0, krn_awid}),  .s_axi_awaddr (krn_awaddr),
  //     .s_axi_awlen   (krn_awlen),         .s_axi_awsize (krn_awsize),
  //     .s_axi_awburst (krn_awburst),       .s_axi_awcache(krn_awcache),
  //     .s_axi_awprot  (krn_awprot),        .s_axi_awlock (krn_awlock[0]),
  //     .s_axi_awvalid (krn_awvalid),       .s_axi_awready(krn_awready),
  //     .s_axi_wdata   (krn_wdata),         .s_axi_wstrb  (krn_wstrb),
  //     .s_axi_wlast   (krn_wlast),         .s_axi_wvalid (krn_wvalid),
  //     .s_axi_wready  (krn_wready),
  //     .s_axi_bid     (/*-> krn_bid*/),    .s_axi_bresp  (krn_bresp),
  //     .s_axi_bvalid  (krn_bvalid),        .s_axi_bready (krn_bready),
  //     .s_axi_arid    ({5'h0, krn_arid}),  .s_axi_araddr (krn_araddr),
  //     .s_axi_arlen   (krn_arlen),         .s_axi_arsize (krn_arsize),
  //     .s_axi_arburst (krn_arburst),       .s_axi_arcache(krn_arcache),
  //     .s_axi_arprot  (krn_arprot),        .s_axi_arlock (krn_arlock[0]),
  //     .s_axi_arvalid (krn_arvalid),       .s_axi_arready(krn_arready),
  //     .s_axi_rid     (/*-> krn_rid*/),    .s_axi_rdata  (krn_rdata),
  //     .s_axi_rresp   (krn_rresp),         .s_axi_rlast  (krn_rlast),
  //     .s_axi_rvalid  (krn_rvalid),        .s_axi_rready (krn_rready),
  //     // ---- master side: shell PCIM (confirm exact F2 names/widths) ----
  //     .m_axi_awid    (cl_sh_pcim_awid),   .m_axi_awaddr (cl_sh_pcim_awaddr),
  //     .m_axi_awlen   (cl_sh_pcim_awlen),  .m_axi_awsize (cl_sh_pcim_awsize),
  //     .m_axi_awburst (cl_sh_pcim_awburst),.m_axi_awcache(cl_sh_pcim_awcache),
  //     .m_axi_awprot  (cl_sh_pcim_awprot), .m_axi_awvalid(cl_sh_pcim_awvalid),
  //     .m_axi_awready (sh_cl_pcim_awready),
  //     .m_axi_wdata   (cl_sh_pcim_wdata),  .m_axi_wstrb  (cl_sh_pcim_wstrb),
  //     .m_axi_wlast   (cl_sh_pcim_wlast),  .m_axi_wvalid (cl_sh_pcim_wvalid),
  //     .m_axi_wready  (sh_cl_pcim_wready),
  //     .m_axi_bid     (sh_cl_pcim_bid),    .m_axi_bresp  (sh_cl_pcim_bresp),
  //     .m_axi_bvalid  (sh_cl_pcim_bvalid), .m_axi_bready (cl_sh_pcim_bready),
  //     .m_axi_arid    (cl_sh_pcim_arid),   .m_axi_araddr (cl_sh_pcim_araddr),
  //     .m_axi_arlen   (cl_sh_pcim_arlen),  .m_axi_arsize (cl_sh_pcim_arsize),
  //     .m_axi_arburst (cl_sh_pcim_arburst),.m_axi_arcache(cl_sh_pcim_arcache),
  //     .m_axi_arprot  (cl_sh_pcim_arprot), .m_axi_arvalid(cl_sh_pcim_arvalid),
  //     .m_axi_arready (sh_cl_pcim_arready),
  //     .m_axi_rid     (sh_cl_pcim_rid),    .m_axi_rdata  (sh_cl_pcim_rdata),
  //     .m_axi_rresp   (sh_cl_pcim_rresp),  .m_axi_rlast  (sh_cl_pcim_rlast),
  //     .m_axi_rvalid  (sh_cl_pcim_rvalid), .m_axi_rready (cl_sh_pcim_rready)
  //   );
  //   assign cl_sh_pcim_awuser = '0;   // confirm field exists/width on F2
  //   assign cl_sh_pcim_aruser = '0;
  //
  // Until the PCIM slice is wired on the F2 box, the kernel master nets are left
  // unconnected on their far side (krn_*ready stay low) — the kernel will stall,
  // which is the intended "not yet wired" state for a synth-only smoke check.
  // =========================================================================

endmodule
