# S1 CLKGEN+CDC CL plan — run the forge_train_step_s1 kernel at ~125 MHz (clk_extra_a1) while the shell stays at 250 MHz

**Goal.** The kernel `forge_train_step_s1` is verified and frozen (Vitis csim bit-exact; csynth closes 3.395 ns; post-route it closes ~144 MHz but MISSES the shell-fixed 250 MHz, WNS -2.958 ns). `clk_main_a0` is fixed at 250 MHz on F2 and cannot be slowed by recipe. The fix is to instantiate `AWS_CLK_GEN` in the CL, generate **clk_extra_a1 = 125 MHz** (Recipe A1; 8 ns >> the 6.94 ns the kernel needs), move ONLY `u_kernel` onto that domain, keep every shell interface (DMA-PCIS, OCL, sh_ddr, crossbar) on `clk_main_a0`=250 MHz, and bridge the two AXI boundaries the kernel touches with `cl_axi_clock_converter` (CDC) IP.

**Invariants (do NOT change).** The HLS kernel RTL (`forge_train_step_s1`), the 13-bit `s_axi_control` ABI (buffer ptr 0x18/0x1c + mode 0x24 + a1/a2/a3), the 512-bit `m_axi_gmem` master, the DDR buffer-at-offset-0 ABI, and the host staging protocol are ALL unchanged. Only the CL wrapper (`cl_forge_train_step_s1.sv`), `gen_ip.tcl`, the synth driver, and the build recipe flag change.

**Reference (canonical, read-verified).** `/tmp/clkref_local/clkref/cl_mem_perf_design/cl_mem_perf.sv` (AWS_CLK_GEN instantiation lines 188-239; SDA AXIL xbar 100-162; per-domain `xpm_cdc_async_rst` 246-251/495-500/539-544); `/tmp/clkref_local/clkref/aws_clk_gen.sv` (ports 29-92; internal AXIL CDC 171-215; per-clock reset sync 778-791); `/tmp/clkref_local/clkref/AWS_CLK_GEN_spec.md` (named-instance mandate; SDA control mandatory; SYS_RST/MMCM_LOCK boot sequence); `/tmp/clkref_local/clkref/Clock_Recipes_User_Guide.md` (Recipe A1: clk_extra_a1=125 MHz; build flags).

---

## 1. AWS_CLK_GEN instantiation in cl_forge_train_step_s1.sv

### 1a. Generated-clock/reset wires (declare near the top of the module, after the reset synchronizer)
```systemverilog
  logic  gen_clk_main_a0;            // 250 MHz pass-through (shell domain; drives all shell-side logic)
  logic  gen_clk_extra_a1;          // 125 MHz (Recipe A1)  <-- KERNEL CLOCK
  logic  gen_clk_extra_a2, gen_clk_extra_a3;
  logic  gen_clk_extra_b0, gen_clk_extra_b1, gen_clk_extra_c0, gen_clk_extra_c1;
  logic  gen_clk_hbm_ref, gen_clk_hbm_axi;
  logic  gen_rst_main_n;            // reset sync'd to gen_clk_main_a0 (250 MHz shell domain)
  logic  gen_rst_a1_n;             // reset sync'd to gen_clk_extra_a1 (125 MHz KERNEL domain)
  logic  gen_rst_a2_n, gen_rst_a3_n, gen_rst_b0_n, gen_rst_b1_n, gen_rst_c0_n, gen_rst_c1_n;
  logic  gen_rst_hbm_ref_n, gen_rst_hbm_axi_n;
```

### 1b. STOP tying off SDA — AWS_CLK_GEN's control plane MUST be driven by SDA
- DELETE the `` `include "unused_cl_sda_template.inc" `` line (it currently drives `cl_sda_*ready/valid/data` to 0).
- The shell SDA slave port (`sda_cl_*` / `cl_sda_*` in `cl_ports.vh`) becomes the AXI-Lite source for AWS_CLK_GEN. We do NOT need an SDA-BRAM, so we can route the SDA AXI-Lite slave **directly** into AWS_CLK_GEN's `s_axil_ctrl_*` (no `cl_sda_axil_xbar` demux — that xbar in cl_mem_perf only exists to also serve an SDA-BRAM at 0x0000–0x0004_FFFF, which S1 does not have). AWS_CLK_GEN decodes its own 0x0005_xxxx space internally; addresses below 0x5_0000 simply read `0xBAAD_DEC0` / are ignored, which is acceptable since the driver only ever touches 0x5_xxxx.
- Add an SDA reset synchronizer (matches cl_mem_perf line 107):
```systemverilog
  logic sda_sync_rst_n;
  xpm_cdc_async_rst CDC_ASYNC_RST_N_SDA (
    .src_arst (rst_main_n), .dest_clk (clk_main_a0), .dest_arst (sda_sync_rst_n));
```

### 1c. The AWS_CLK_GEN instance (named EXACTLY `AWS_CLK_GEN` — spec mandate)
Wire `s_axil_ctrl_*` straight from the shell SDA slave port (`sda_cl_*` driven by shell; `cl_sda_*` driven by IP). Input clocks are the shell's `clk_main_a0` (250 MHz) and `clk_hbm_ref` (100 MHz, present in `cl_ports.vh`). Enable only group A (we only need a1); leave B/C/HBM enabled is harmless but to minimize area set `CLK_GRP_B_EN/C_EN/HBM_EN=0` (group A still gives a1/a2/a3 from MMCM_A). Recommended: enable A only.
```systemverilog
  aws_clk_gen #(
    .CLK_GRP_A_EN(1), .CLK_GRP_B_EN(0), .CLK_GRP_C_EN(0), .CLK_HBM_EN(0)
  ) AWS_CLK_GEN (
    .i_clk_main_a0 (clk_main_a0),  .i_rst_main_n (sda_sync_rst_n),  .i_clk_hbm_ref (clk_hbm_ref),
    // AXI-Lite control from the shell SDA slave (clk_main_a0 domain):
    .s_axil_ctrl_awaddr (sda_cl_awaddr),  .s_axil_ctrl_awvalid(sda_cl_awvalid), .s_axil_ctrl_awready(cl_sda_awready),
    .s_axil_ctrl_wdata  (sda_cl_wdata),   .s_axil_ctrl_wstrb (4'hF),            .s_axil_ctrl_wvalid (sda_cl_wvalid),  .s_axil_ctrl_wready(cl_sda_wready),
    .s_axil_ctrl_bresp  (cl_sda_bresp),   .s_axil_ctrl_bvalid(cl_sda_bvalid),   .s_axil_ctrl_bready (sda_cl_bready),
    .s_axil_ctrl_araddr (sda_cl_araddr),  .s_axil_ctrl_arvalid(sda_cl_arvalid), .s_axil_ctrl_arready(cl_sda_arready),
    .s_axil_ctrl_rdata  (cl_sda_rdata),   .s_axil_ctrl_rresp (cl_sda_rresp),    .s_axil_ctrl_rvalid (cl_sda_rvalid), .s_axil_ctrl_rready(sda_cl_rready),
    // generated clocks:
    .o_clk_hbm_ref(gen_clk_hbm_ref), .o_clk_main_a0(gen_clk_main_a0),
    .o_clk_extra_a1(gen_clk_extra_a1), .o_clk_extra_a2(gen_clk_extra_a2), .o_clk_extra_a3(gen_clk_extra_a3),
    .o_clk_extra_b0(gen_clk_extra_b0), .o_clk_extra_b1(gen_clk_extra_b1),
    .o_clk_extra_c0(gen_clk_extra_c0), .o_clk_extra_c1(gen_clk_extra_c1), .o_clk_hbm_axi(gen_clk_hbm_axi),
    // generated resets:
    .o_cl_rst_hbm_axi_n(gen_rst_hbm_axi_n), .o_cl_rst_hbm_ref_n(gen_rst_hbm_ref_n),
    .o_cl_rst_c1_n(gen_rst_c1_n), .o_cl_rst_c0_n(gen_rst_c0_n), .o_cl_rst_b1_n(gen_rst_b1_n), .o_cl_rst_b0_n(gen_rst_b0_n),
    .o_cl_rst_a3_n(gen_rst_a3_n), .o_cl_rst_a2_n(gen_rst_a2_n), .o_cl_rst_a1_n(gen_rst_a1_n), .o_cl_rst_main_n(gen_rst_main_n)
  );
```
Note: exact `sda_cl_*` / `cl_sda_*` signal names come from `cl_ports.vh` (same names the `unused_cl_sda_template.inc` was driving); `wstrb` is tied to `4'hF` exactly as cl_mem_perf does (the shell SDA path doesn't carry per-byte strobe). If the shell `sda_cl_awaddr`/`araddr` are narrower than 32 bits, zero-extend; AWS_CLK_GEN expects `[31:0]`.

### 1d. Which generated clock = 125 MHz
**`gen_clk_extra_a1`** under Recipe A1 = exactly 125 MHz (`Clock_Recipes_User_Guide.md` group-A table: A1 -> clk_extra_a1=125). Its matching synchronized reset is **`gen_rst_a1_n`** (active-low, async-assert/sync-deassert to clk_extra_a1, produced internally by `xpm_cdc_async_rst`, aws_clk_gen.sv 778-791). Use `gen_rst_a1_n` directly as the kernel ap_rst_n — do NOT feed the kernel the old `rst_n_local` (that is sync'd to clk_main_a0 only).

---

## 2. Move u_kernel to the 125 MHz domain

Only two pins change on the (unchanged) kernel instance:
```systemverilog
  forge_train_step_s1 u_kernel (
    .ap_clk   (gen_clk_extra_a1),   // was clk_main_a0
    .ap_rst_n (gen_rst_a1_n),       // was rst_n_local (sync'd to a1 by aws_clk_gen)
    ...all s_axi_control_* and m_axi_gmem_* ports unchanged in NAME, but now connect to the CDC IPs (sec 3)...
  );
```
Everything else inside the kernel (the s_axi_control 13-bit address slice, the 512-bit m_axi, ID widths) is byte-identical. The kernel's `interrupt` (krn_irq) is unused/unconnected today and stays so; if ever wired to the shell it would need a 1-bit `xpm_cdc_single` 125->250 — not needed now.

---

## 3. The two axi_clock_converter (CDC) instances

The kernel's two AXI boundaries each get one `cl_axi_clock_converter` (full AXI4) / `cl_axi_clock_converter_light` (AXI4-Lite) instance. Slave side = the existing 250 MHz net group; master side = the kernel's 125 MHz pins. The CDC handles READY/VALID handshake across the async clocks; both directions are supported by the IP.

### CDC-1 — OCL control: 250 MHz -> 125 MHz, AXI4-Lite 32-bit (cl_axi_clock_converter_light)
- **Slave (250 MHz):** the `ocl_*_pre` nets currently produced by `s1_ocl_slice` (`ocl_aw_addr_pre`, `ocl_aw_valid_pre`, …). Clock = `clk_main_a0`, reset = `rst_main_n` (slice stays exactly where it is).
- **Master (125 MHz):** new nets `kctl_*` that feed the kernel `s_axi_control_*`. Clock = `gen_clk_extra_a1`, reset = `gen_rst_a1_n`.
- The kernel `s_axi_control` consumes only `[12:0]` of the address — drive `s_axi_control_AWADDR(kctl_awaddr[12:0])` exactly as today, but `kctl_awaddr` now comes from the CDC master, not the slice.
```systemverilog
  // 125 MHz kernel-control nets
  logic        kctl_aw_valid, kctl_aw_ready;  logic [31:0] kctl_aw_addr;
  logic        kctl_w_valid,  kctl_w_ready;   logic [31:0] kctl_w_data;  logic [3:0] kctl_w_strb;
  logic        kctl_b_valid,  kctl_b_ready;   logic  [1:0] kctl_b_resp;
  logic        kctl_ar_valid, kctl_ar_ready;  logic [31:0] kctl_ar_addr;
  logic        kctl_r_valid,  kctl_r_ready;   logic [31:0] kctl_r_data;  logic  [1:0] kctl_r_resp;

  cl_axi_clock_converter_light OCL_CDC (
    .s_axi_aclk (clk_main_a0), .s_axi_aresetn (rst_main_n),       // 250 MHz slave
    .s_axi_awaddr(ocl_aw_addr_pre), .s_axi_awprot(3'd0), .s_axi_awvalid(ocl_aw_valid_pre), .s_axi_awready(ocl_aw_ready_pre),
    .s_axi_wdata (ocl_w_data_pre),  .s_axi_wstrb(ocl_w_strb_pre), .s_axi_wvalid (ocl_w_valid_pre), .s_axi_wready (ocl_w_ready_pre),
    .s_axi_bresp (ocl_b_resp_pre),  .s_axi_bvalid(ocl_b_valid_pre), .s_axi_bready(ocl_b_ready_pre),
    .s_axi_araddr(ocl_ar_addr_pre), .s_axi_arprot(3'd0), .s_axi_arvalid(ocl_ar_valid_pre), .s_axi_arready(ocl_ar_ready_pre),
    .s_axi_rdata (ocl_r_data_pre),  .s_axi_rresp(ocl_r_resp_pre),  .s_axi_rvalid(ocl_r_valid_pre), .s_axi_rready(ocl_r_ready_pre),
    .m_axi_aclk (gen_clk_extra_a1), .m_axi_aresetn(gen_rst_a1_n),  // 125 MHz master
    .m_axi_awaddr(kctl_aw_addr), .m_axi_awprot(), .m_axi_awvalid(kctl_aw_valid), .m_axi_awready(kctl_aw_ready),
    .m_axi_wdata (kctl_w_data),  .m_axi_wstrb(kctl_w_strb), .m_axi_wvalid(kctl_w_valid), .m_axi_wready(kctl_w_ready),
    .m_axi_bresp (kctl_b_resp),  .m_axi_bvalid(kctl_b_valid), .m_axi_bready(kctl_b_ready),
    .m_axi_araddr(kctl_ar_addr), .m_axi_arprot(), .m_axi_arvalid(kctl_ar_valid), .m_axi_arready(kctl_ar_ready),
    .m_axi_rdata (kctl_r_data),  .m_axi_rresp(kctl_r_resp), .m_axi_rvalid(kctl_r_valid), .m_axi_rready(kctl_r_ready)
  );
```
Then on the kernel: `.s_axi_control_AWADDR(kctl_aw_addr[12:0])`, `.s_axi_control_AWVALID(kctl_aw_valid)`, `.s_axi_control_AWREADY(kctl_aw_ready)`, … (same pattern as today, with `kctl_*` substituted for `ocl_*_pre`; ARADDR also `[12:0]`).

### CDC-2 — kernel memory: 125 MHz -> 250 MHz, AXI4 512-bit (cl_axi_clock_converter)
- **Slave (125 MHz):** the kernel's `m_axi_gmem` nets (`krn_awaddr`, …). Clock = `gen_clk_extra_a1`, reset = `gen_rst_a1_n`.
- **Master (250 MHz):** new nets `kmem_*` that feed crossbar SI0 in place of the raw `krn_*` today. Clock = `clk_main_a0`, reset = `rst_main_n_sync`.
- ID width = 1 on BOTH sides of this CDC (the kernel uses a 1-bit ID; the crossbar widens 1->16 on its SI0 input as it does today via `{15'h0, krn_awid}`). So the crossbar wiring at SI0 changes only the net names: `krn_awid`->`kmem_awid`, `krn_awaddr`->`kmem_awaddr`, etc. The `assign krn_bid = krn_bid_16[0]` / `krn_rid` reduction now applies to `kmem_*` and the response IDs feed back through the CDC to the kernel.
```systemverilog
  // 250 MHz crossbar-side kernel-memory nets (replace the raw krn_* at the crossbar SI0)
  logic [63:0]  kmem_awaddr, kmem_araddr;  logic [7:0] kmem_awlen, kmem_arlen;
  logic  [2:0]  kmem_awsize, kmem_arsize;  logic [1:0] kmem_awburst, kmem_arburst;
  logic  [3:0]  kmem_awcache, kmem_arcache; logic [2:0] kmem_awprot, kmem_arprot;
  logic  [3:0]  kmem_awqos, kmem_arqos;     logic [1:0] kmem_awlock, kmem_arlock;
  logic         kmem_awid, kmem_arid, kmem_bid, kmem_rid;
  logic         kmem_awvalid, kmem_awready, kmem_arvalid, kmem_arready;
  logic [511:0] kmem_wdata, kmem_rdata;     logic [63:0] kmem_wstrb;
  logic         kmem_wlast, kmem_wvalid, kmem_wready;
  logic  [1:0]  kmem_bresp, kmem_rresp;     logic kmem_bvalid, kmem_bready, kmem_rlast, kmem_rvalid, kmem_rready;

  cl_axi_clock_converter GMEM_CDC (
    .s_axi_aclk (gen_clk_extra_a1), .s_axi_aresetn (gen_rst_a1_n),    // 125 MHz slave = kernel m_axi
    .s_axi_awid(krn_awid), .s_axi_awaddr(krn_awaddr), .s_axi_awlen(krn_awlen), .s_axi_awsize(krn_awsize),
    .s_axi_awburst(krn_awburst), .s_axi_awlock(krn_awlock[0]), .s_axi_awcache(krn_awcache), .s_axi_awprot(krn_awprot),
    .s_axi_awqos(krn_awqos), .s_axi_awregion(4'h0), .s_axi_awvalid(krn_awvalid), .s_axi_awready(krn_awready),
    .s_axi_wdata(krn_wdata), .s_axi_wstrb(krn_wstrb), .s_axi_wlast(krn_wlast), .s_axi_wvalid(krn_wvalid), .s_axi_wready(krn_wready),
    .s_axi_bid(krn_bid), .s_axi_bresp(krn_bresp), .s_axi_bvalid(krn_bvalid), .s_axi_bready(krn_bready),
    .s_axi_arid(krn_arid), .s_axi_araddr(krn_araddr), .s_axi_arlen(krn_arlen), .s_axi_arsize(krn_arsize),
    .s_axi_arburst(krn_arburst), .s_axi_arlock(krn_arlock[0]), .s_axi_arcache(krn_arcache), .s_axi_arprot(krn_arprot),
    .s_axi_arqos(krn_arqos), .s_axi_arregion(4'h0), .s_axi_arvalid(krn_arvalid), .s_axi_arready(krn_arready),
    .s_axi_rid(krn_rid), .s_axi_rdata(krn_rdata), .s_axi_rresp(krn_rresp), .s_axi_rlast(krn_rlast), .s_axi_rvalid(krn_rvalid), .s_axi_rready(krn_rready),
    .m_axi_aclk (clk_main_a0), .m_axi_aresetn (rst_main_n_sync),       // 250 MHz master = crossbar SI0
    .m_axi_awid(kmem_awid), .m_axi_awaddr(kmem_awaddr), .m_axi_awlen(kmem_awlen), .m_axi_awsize(kmem_awsize),
    .m_axi_awburst(kmem_awburst), .m_axi_awlock(kmem_awlock[0]), .m_axi_awcache(kmem_awcache), .m_axi_awprot(kmem_awprot),
    .m_axi_awqos(kmem_awqos), .m_axi_awregion(), .m_axi_awvalid(kmem_awvalid), .m_axi_awready(kmem_awready),
    .m_axi_wdata(kmem_wdata), .m_axi_wstrb(kmem_wstrb), .m_axi_wlast(kmem_wlast), .m_axi_wvalid(kmem_wvalid), .m_axi_wready(kmem_wready),
    .m_axi_bid(kmem_bid), .m_axi_bresp(kmem_bresp), .m_axi_bvalid(kmem_bvalid), .m_axi_bready(kmem_bready),
    .m_axi_arid(kmem_arid), .m_axi_araddr(kmem_araddr), .m_axi_arlen(kmem_arlen), .m_axi_arsize(kmem_arsize),
    .m_axi_arburst(kmem_arburst), .m_axi_arlock(kmem_arlock[0]), .m_axi_arcache(kmem_arcache), .m_axi_arprot(kmem_arprot),
    .m_axi_arqos(kmem_arqos), .m_axi_arregion(), .m_axi_arvalid(kmem_arvalid), .m_axi_arready(kmem_arready),
    .m_axi_rid(kmem_rid), .m_axi_rdata(kmem_rdata), .m_axi_rresp(kmem_rresp), .m_axi_rlast(kmem_rlast), .m_axi_rvalid(kmem_rvalid), .m_axi_rready(kmem_rready)
  );
```
- The `interrupt`/`krn_irq` pin stays unconnected as today.
- Crossbar SI0: replace every `krn_*` at SI0 with `kmem_*` (e.g. `s_axi_awid({{15'h0, kmem_awid}, sh_cl_dma_pcis_awid})`, `s_axi_awaddr({kmem_awaddr, sh_cl_dma_pcis_awaddr})`, …). The crossbar's `aclk` stays `clk_main_a0`. The response-ID reduction becomes `assign kmem_bid = krn_bid_16[0]; assign kmem_rid = krn_rid_16[0];` — i.e. the 16-bit crossbar response IDs collapse to the 1-bit ID that the CDC master expects; the CDC carries them back to `krn_bid`/`krn_rid` for the kernel.

**Exact converter config** (must match the actual port lists of the HDK-provided `cl_axi_clock_converter` / `cl_axi_clock_converter_light` — these are the SAME pre-built IPs sh_ddr already depends on and that the synth driver already `read_ip`s; see sec 4). If the HDK xci's are hardcoded to a width/ID that doesn't match (e.g. ID!=1), generate our own copies instead (sec 4) named `s1_axi_cdc` (512b AXI4, ID_WIDTH=1, ADDR=64) and `s1_axil_cdc` (AXI4LITE, 32b, ADDR=32) so the configs are under our control.

---

## 4. gen_ip.tcl additions

The two CDC IPs are needed as elaboratable modules. Two options:

**Option A (preferred — self-owned IP, deterministic config).** Generate our own CDC IP in `gen_ip.tcl`, named to avoid colliding with the HDK's `cl_axi_clock_converter*` (which sh_ddr reads from HDK_IP_SRC_DIR), exactly the way `s1_ocl_slice` was renamed to dodge the `axi_register_slice_light` clash:
```tcl
# AXI4-Lite CDC: OCL control 250->125 MHz (32b). Named s1_axil_cdc (avoid HDK cl_axi_clock_converter_light clash).
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name s1_axil_cdc -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4LITE} CONFIG.DATA_WIDTH {32} CONFIG.ADDR_WIDTH {32} \
  CONFIG.ID_WIDTH {0} CONFIG.SYNCHRONIZATION_STAGES {3}] [get_ips s1_axil_cdc]
# AXI4 CDC: kernel m_axi 125->250 MHz (512b, 64-bit addr, 1-bit ID to match the kernel). Named s1_axi_cdc.
create_ip -name axi_clock_converter -vendor xilinx.com -library ip -module_name s1_axi_cdc -dir $ipdir
set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.DATA_WIDTH {512} CONFIG.ADDR_WIDTH {64} \
  CONFIG.ID_WIDTH {1} CONFIG.SYNCHRONIZATION_STAGES {3} CONFIG.ACLK_RATIO {2:1}] [get_ips s1_axi_cdc]
```
If Option A is used, the CL must instantiate module names `s1_axil_cdc` / `s1_axi_cdc` (not `cl_axi_clock_converter*`); update sec-3 instance module names accordingly and add `synth_ip [get_ips s1_axil_cdc]` / `synth_ip [get_ips s1_axi_cdc]` + `read_ip [glob $CL_DIR/design/ip/s1_axi_cdc/*.xci]` to the synth driver.

**Note on `aws_clk_gen.sv` itself + its MMCM/xbar IP.** `aws_clk_gen.sv` is a developer-kit COMMON source (`$AWS_FPGA/hdk/common/lib/aws_clk_gen.sv`), and it depends on `clk_mmcm_a/b/c/hbm`, `cl_clk_axil_xbar`, `cl_axi_clock_converter_light`, `cl_axi_register_slice_light`. Those are pre-built IP in the HDK (`$HDK_SHELL_DESIGN_DIR/../../ip/cl_ip/...` per cl_mem_perf's synth tcl lines 58-66). **Do NOT instantiate MMCM_A separately** (spec gotcha #5 — aws_clk_gen contains all four MMCM groups). The cleanest path: pass `--aws_clk_gen` to `aws_build_dcp_from_cl.py`, which AWS wires to pull `aws_clk_gen.sv` + its IP into the build automatically (that flag exists precisely for this). The synth driver then only needs to `read_ip` the group-A MMCM + clk xbar if the build flow does not auto-add them — see sec 5.

**`gen_ip.tcl` keeps** the existing `s1_ocl_slice` and `axi_crossbar_ddr` IP unchanged.

---

## 5. Build recipe (--aws_clk_gen --clock_recipe_a A1)

- Run: `./aws_build_dcp_from_cl.py --cl cl_forge_train_step_s1 --aws_clk_gen --clock_recipe_a A1 --no-encrypt`
- `--aws_clk_gen` pulls in `aws_clk_gen.sv` + its MMCM/xbar/converter IP and applies the AWS_CLK_GEN clock constraints; `--clock_recipe_a A1` sets MMCM_A dividers so `clk_extra_a1 = 125 MHz` (and a2=375, a3=500). Without `--clock_recipe_a A1`, P&R would still try to close clk_extra_a1 at its default and the kernel would see the wrong frequency.
- **Build runner bug to FIX (`scripts/run_s1_afi_build.sh`).** Its current logic: `if [ "$CLOCK_RECIPE" = "A1" ]; then CLOCK_ARGS=""; else CLOCK_ARGS="--aws_clk_gen --clock_recipe_a $CLOCK_RECIPE"; fi`. That gates `--aws_clk_gen` OFF for A1 — which was correct in the OLD design (no clk_gen) but is WRONG now: we need `--aws_clk_gen --clock_recipe_a A1`. Change the runner so the clkgen build always sets `CLOCK_ARGS="--aws_clk_gen --clock_recipe_a ${CLOCK_RECIPE}"` (default A1). Concretely, drop the `A1->empty` special-case and set `CLOCK_ARGS="--aws_clk_gen --clock_recipe_a ${CLOCK_RECIPE:-A1}"`.
- Group B/C/HBM recipes: with `CLK_GRP_B/C/HBM_EN=0` they are unused; passing only `--clock_recipe_a A1` is sufficient (the guide says unspecified groups default; unused groups' recipes are ignored). If the tool insists on all four flags, add `--clock_recipe_b B0 --clock_recipe_c C0 --clock_recipe_hbm H0` (any valid value — unused).
- The `create_clock 4.0 ns` note in the runner header refers to clk_main_a0 (250 MHz) and is correct for the shell domain; the kernel's 125 MHz constraint comes from the A1 recipe applied by `--clock_recipe_a A1`, NOT from a hand-edited create_clock. Do not hand-add a kernel create_clock.

### Runtime/software (driver) sequence — REQUIRED, else the kernel never clocks
After AFI load, `SYS_RST_REG=0xFFFF_FFFE` holds `cl_rst_a1_n` asserted and `clk_extra_a1` is static-low until MMCM_A locks. The host MUST, before kicking the kernel:
1. Poll `MMCM_LOCK_REG` (BASE_REG 0x0005_8000 + 0x20) for bit0 (MMCM_A locked); all-locked mask = `0x0000_0151`.
2. Clear the a1 reset bit in `SYS_RST_REG` (0x0005_8000+0x14) — the simplest is calling the SDK `aws_clkgen_deassert_resets(slot_id)`.
These registers are reached over the SDA path (PF1-BAR4 AXI-Lite) now wired to AWS_CLK_GEN. The existing OCL-based control/kick of the kernel (buffer ptr/mode) is unchanged; only this new clkgen bring-up step is added to the runner's host program.

---

## 6. Risks

1. **CDC correctness / deadlock.** `cl_axi_clock_converter` is a standard Xilinx async FIFO bridge; the main risk is a config mismatch (ID width, ACLK_RATIO, or AXI4 vs AXI4LITE) producing missing handshakes -> kernel hang. Mitigation: self-own the IP (Option A) with explicit ID_WIDTH=1 (mem) / AXI4LITE (ctrl); verify against the HDK xci port lists before instantiating. The 2:1 integer ratio (250/125) is the easy CDC case.

2. **Reset sequencing across domains.** Use `gen_rst_a1_n` for everything on clk_extra_a1 (kernel + both CDC kernel-sides), `rst_main_n`/`rst_main_n_sync` for the 250 MHz sides. Do NOT cross `rst_n_local` (a0-sync'd) into the a1 domain. Boot risk: releasing resets before MMCM lock (spec) -> the driver MUST poll MMCM_LOCK_REG and call `aws_clkgen_deassert_resets` (sec 5) or the kernel sees no clock and the AXI bridges never come out of reset.

3. **SASD crossbar now spans domains only via the CDC.** The crossbar itself STAYS fully in the 250 MHz domain — it never sees clk_extra_a1. SI0 sees the 250 MHz CDC master; SI1 is DMA-PCIS (250 MHz) as before. Staging and kernel access remain sequential (host stages -> kicks -> kernel runs), so no concurrent-arbitration hazard is introduced. The only new latency is the CDC depth (~a few cycles each way), absorbed by AXI handshake.

4. **Timing on the CDC + congestion.** The kernel now closes easily at 125 MHz (6.94 ns need vs 8 ns), removing the -2.958 ns WNS at 250 MHz. New timing surfaces: (a) the CDC master/slice on the 250 MHz side must close at 250 MHz — it is a small register-FIFO, low risk; (b) the 512-bit GMEM_CDC adds fanout near the crossbar — watch SLR placement. S1 is ~5x the muzero kernel's LUT/BRAM; AWS_CLK_GEN + 2 CDC add modest area. If P&R misses, the fallback is NOT to slow clk_main_a0 (impossible) but to lower the kernel further with `--clock_recipe_a A2` (clk_extra_a1=15.625 MHz — too slow) or A0 (62.5 MHz) — A0/62.5 is the safe relief knob giving 16 ns. The kernel ABI is unchanged for any recipe.

5. **AFI manifest / clkgen metadata.** With `--aws_clk_gen`, the produced manifest must record the clock recipe so the runtime `fpga_clkgen` library and `aws_clkgen_deassert_resets` see a clkgen-enabled image. `create-fpga-image` should be given the build's manifest as-is (the AWS flow writes the recipe into it). Risk: a manifest built WITHOUT `--aws_clk_gen` (the old default) would not expose the clkgen registers -> runtime lock poll fails. Verify the manifest/`*.tar` from this build references aws_clk_gen before registering the AFI.

6. **SDA wiring assumption.** This plan routes the shell SDA AXI-Lite slave DIRECTLY to AWS_CLK_GEN (no SDA-BRAM, no `cl_sda_axil_xbar`). That works because S1 has no SDA-BRAM and AWS_CLK_GEN self-decodes 0x5_xxxx. Risk: if the shell/`--aws_clk_gen` flow EXPECTS the `cl_sda_axil_xbar` demux structure (cl_mem_perf has it), add the xbar with only the clkgen master leg used and the BRAM leg tied off. Confirm against the HDK's own `--aws_clk_gen` scaffolding once on the build box (the diag step already dumps interfaces).

7. **Group-enable choice.** Setting `CLK_GRP_B/C/HBM_EN=0` saves 3 MMCMs but the `--clock_recipe_b/c/hbm` flags / shell expectations may assume all groups present. Low risk; if the build complains, set all `*_EN=1` (area cost only) and leave b/c/hbm clocks unconnected.
