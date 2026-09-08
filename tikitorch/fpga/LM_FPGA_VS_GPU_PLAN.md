# Plan: fully tropical LM trained + served on FPGA vs dense nanoGPT on GPU

**End goal.** One table: a strictly-closed TikiTorch LM, trained AND served on
an AWS F2 FPGA, against a dense nanoGPT-class baseline trained and served on a
GPU — same data, same token budget, reporting quality (val CE nats + bits per
byte), training throughput (tok/s), serving throughput, and energy (mJ/token,
train and serve), with estimated-vs-measured clearly labeled.

Strictness doctrine: the FPGA side is Level-B closed by construction (the
kernel IS the audit); the host-side reference recipes and all quality
expectations come from manuscript/results/lm_shakespeare/BAKEOFF_SHAKESPEARE.md Round 4.

## What already exists (verified 2026-06-11)

- Silicon-validated on-chip training: `forge_train_step` (1-layer LM,
  D=128 V=128) and `_2layer` (772 tok/s, ~9.0 mJ/tok, DSP=0, params evolved
  on chip; AGFI `agfi-0a845113b088cf62e`). Morph kernels: `_morph` (D=64 L=4)
  and `_morph_ecg` (D=128 L=8, DDR streaming, 1207 tok/s, 5.8 mJ/tok,
  WNS +0.009) — but the morph kernels keep embed+head+loss on the HOST.
- **NEW (this round): balanced LUT-CE gradient on chip** in
  `forge_train_step` (CFG_LOSS_KIND=1): exp2 Q16 ROM (`forge_ce_lut.hpp`),
  shift-normalize, gradient balanced against the post-truncation sum (row
  sums exactly zero — integer-exact analog of the TropicalCrossEntropyFn
  fix; an unbalanced row bias collapses sign-trained models). Bit-exact
  CSIM validation: `scripts/_validate_kernel_lutce.py` (50 cases, perceptron
  path unchanged). Buffer grew by 2 config words appended at the end
  (`OFF_CFG`); zero-filled buffers reproduce the silicon-validated behavior,
  but the new gradient needs a new AFI.
- Quality stakes (Round 4, morph char-LM, hard-from-scratch): perceptron
  3.74 val CE vs balanced LUT-CE 2.44–2.50 — the on-chip loss upgrade is
  worth ~1.2 nats.

## Configurations (in order of execution)

| | arch | dims | tokenizer | quality target (host-measured) | status |
|---|---|---|---|---|---|
| A "pipeline" | 2-layer tropical attention LM | D=128 V=128 | char | ~bigram (2.5–2.7) | kernel exists; port CE block |
| B "headline" | MorphSSM + on-chip head/loss | D=128 L=4–8 V=128 | char | 2.44–2.48 (strict hard) | morph core exists; head+CE block new |
| C "stretch" | MorphSSM | D=256 V=1024 | BPE | ≥3.27 bpb hard / 3.07 annealed | new layout, DDR streaming |

GPU baseline per config: dense nanoGPT (matched params/tokens; the
shape-matched dense-conv twin as secondary reference), AdamW lr 3e-3,
trained + served on a CUDA GPU (`scripts/run_tikitorch_gpu_benchmarks.sh`
infra); MPS numbers are not the baseline.

## Training-on-fabric protocol

- Hard-from-scratch by construction (kernels are tau=0 only). Host-measured
  cost of losing the anneal: +0.21–0.25 nats. Two recipe variants to report:
  1. **full-fabric**: random init, all training on chip (the pure claim);
  2. **warm-handoff**: brief host anneal (tau 0.5→0), export int weights,
     continue on fabric — quality recovery at the cost of a weaker claim.
- Optimizer: plain on-chip SignSGD is adequate — Round 4 measured a
  0.036-nat band across all closed optimizers (no fancy optimizer HLS
  needed). LFSR-noise tempering as anneal substitute: REFUTED by probes
  (noise 0.01 → 2.77, noise 0.1 → 3.26, vs ~2.53 no-noise at 800 steps;
  probe_noise*.jsonl). Do not port to HLS; warm-handoff is the anneal
  mitigation.
- Architecture note for config B: multi-scale morph conv (kernels 3/7/15,
  max-combined) probed ~0.1 nats ahead of single-kernel at 800 steps
  annealed (2.413 vs ~2.52, +7% params; probe_multiscale.jsonl) — full run
  pending; on-chip cost is 3 depthwise add/compare convs per layer, cheap.
  Grouped convs (group_size 4): a wash, drop.
- SignSGD threshold: LUT-CE grads are Q16 (≤65536), not ±1 — calibrate
  mode-2 `threshold` in CSIM before synth (work item 4).

## Measurement protocol

- Same corpus (tiny-shakespeare; char for A/B, BPE-1024 for C), same token
  budgets (3.46M primary; 13.8M for the long run), identical fixed-seed val
  windows; quality = val CE in nats + bpb (cross-tokenizer).
- FPGA energy: prefer measured board power (`fpga-describe-local-image`
  / XRT power rails) over the WNS-derived estimate used for the 9.0 mJ/tok
  number; report both, labeled. Throughput = tokens / wall over a long
  window, PCIe staging included (serve) and excluded (kernel-only), both
  reported.
- GPU energy: nvidia-smi power sampling integrated over the run; same
  tok/s definitions. Report quality-matched comparisons (energy to reach
  CE=X) in addition to budget-matched ones.

## Work items

1. ~~Balanced LUT-CE gradient in 1-layer kernel + bit-exact CSIM validation~~ DONE
2. `_2layer` needs NO kernel change: its loss is host-conditioned by design
   (logits exported to OFF2_LOGITS; host writes the gradient), and the host
   path now inherits the balanced library CE automatically. Run a CSIM
   end-to-end char-LM training to convergence-at-budget; compare vs the
   Python strict-hard reference curve. (On-chip CE only matters for the
   fused train-window mode and config B.)
3. ~~SignSGD threshold calibration for Q16 grads~~ DONE: threshold=2048
   (CSIM sweep over {0, 256, 2048, 8192, 32768};
   manuscript/results/lm_shakespeare/csim_lutce.json). End-to-end CSIM training of the
   REAL 1-layer kernel (scripts/_csim_train_lutce.py, 300 windows = 38.4k
   tokens, ~7k tok/s): **LUT-CE tail acc 16.5% vs perceptron 6.8%** — the
   on-chip loss upgrade is worth 2.4x accuracy at matched budget, measured
   on the kernel itself, not just host-side.
4. ~~Morph LM kernel (config B)~~ DONE at CSIM level:
   `forge_train_step_morph_lm` (+ `forge_buffer_morph_lm.hpp`) — FULL LM on
   chip (embed gather, 4-layer morph core with out-proj bias, max-plus head,
   CFG-selected perceptron/LUT-CE gradient), position-by-position with the
   BS-stream rolling cache doubling as the backward grad accumulator
   (reverse-walk complete-by-read invariant). Validated
   (`scripts/_validate_kernel_morph_lm.py`): 50/50 controlled-logits CE
   gradients bit-exact, rows sum to zero, perceptron path 2-hot; 20-window
   training smoke 0% -> 15-16% acc (2,560 tokens, ~11k tok/s CSIM) — much
   faster learning than the attention kernel, matching the host-side
   architecture finding. Known deviation: conv excludes out-of-range taps
   (Python zero-pads); documented in the buffer header. Longer CSIM run vs
   the Python strict-hard reference (2.44–2.48 nats) still to do.
5. Remote: Vitis HLS C-synth on the F2 part for modified kernels (DSP=0 +
   timing), then AFI builds (Vivado ~2h43m each), then silicon training
   curves + tok/s + power on f2.6xlarge.
6. GPU baseline benchmark runs (CUDA box): train + serve + energy.
7. Comparison report (quality / throughput / energy, budget- and
   quality-matched), with the Round-4 host table as the quality anchor.

## The long-horizon sign-drift attractor (found 2026-06-11, fix in test)

Sign-step training of these LMs has a **constant-predictor attractor** at
long horizons: consistent-sign loss pressure drifts weights monotonically
until the weight-borne component of the residual stream drowns the input
signal; predictions become input-independent (the modal token) and never
recover. Measured in BOTH stacks:
- kernel (int16, ±1 steps, init ±16): collapse by ~2.5k positions; init
  scale (±4..±128) and SignSGD threshold (2k..262k) sweep — every combo
  converges to the bit-identical attractor (acc = modal rate, pred
  mode-frac 1.000).
- host Python (float, PureTropicalSGD lr 2^-9): healthy to step ~1800
  (4.74 nats / 16.6%), monotone decay into the attractor by 4200
  (acc = BPE modal 0.088) — BEFORE the anneal began, counter momentum on.
The healthy horizon scales ~1/lr (lr_shift 10 runs were still healthy at
1500 steps). NOT an HLS bug (kernel bit-exact 50/50); NOT activation
saturation (BS stream mean |8-18|, zero clipping).

**Resolution after exhaustive falsification (10 experiments): from-scratch
integer-lattice training of the hard-tropical LM is an OPEN RESEARCH
PROBLEM, and the comparison plan pivots to warm-handoff as the primary
training story.** The full chain:
- Saturation: refuted (BS stream healthy, zero clipping).
- Recipe envelope (init scale ±4..±1024, threshold 2k..262k, lr 1..8,
  bit-shift lr decay): exhausted — and **step size is a non-knob**: in hard
  max-plus models predictions depend only on weight ORDERINGS, and uniform
  sign steps from one init flip orderings in the same sequence at any lr
  (measured: bit-identical trajectories across lr/threshold/scale).
- Post-residual norm, batch averaging, PureTropicalSGD counter momentum:
  ALL exonerated (Python ablations — every variant trains to 26-32%).
- **The integer lattice is the trigger**: float raw-sign trains (26%);
  ×64 lattice degrades (11%, mode-frac 0.6); ×256/×1024 freeze at a
  constant-predictor init (orderings can't flip; small-std regions round
  to near-constant rows). Dichotomy: drift-collapse at small scale,
  ordering-freeze at large scale, no good middle found.
- Stochastic rounding (magnitude-proportional ±1 firing, the standard
  quantized-training cure, LFSR-clean on-chip): marginal at ×64
  (mode-frac 0.99→0.92 over 800 steps), frozen at ×256. Not a rescue at
  first calibration; remains the most promising research direction along
  with scale-consistent staging (NOTE: tropical algebra forbids per-region
  weight rescaling — max-plus adds weights to activations, so quantization
  noise hits small-std regions structurally).
- Label smoothing / KL-to-uniform (cross-pollinated from the MuZero
  policy-collapse work; changes the gradient SIGN PATTERN, the one knob
  the ordering argument says matters): softens but does not rescue at
  ×64 (mode-frac 1.00 → ~0.97 at ε=0.1–0.3). Also observed: LUT
  temperature × smoothing interact non-trivially (saturated-temperature
  gradients gave MORE prediction diversity than proper-temperature ones
  at the same scale). The single-knob probes are exhausted; what remains
  is a systematic joint campaign (temperature × smoothing × stochastic
  rounding × scale × horizon) — a research project, not a quick fix.
- The HOST float stack has its own long-horizon attractor (6k run @ lr 2^-9:
  peak at step 1800, constant-predictor by 4200, pre-anneal) — and there
  **bit-shift lr decay CURES it** (float magnitudes break the ordering
  argument). Measured: lr_shift 9 +1 every 1500 steps — drift onset at
  1800-2700 caught by the first halving, then monotone improvement to
  **4.3694 nats = 2.793 bpb / 19.1% acc at 6k steps, still descending**
  (strictH_bpe_anneal_s9decay_6k.jsonl). New best strict-closure LM
  (previous: 3.071 bpb at 1500 steps); strict gap to float-tropical now
  0.30 bpb, to the dense GPT reference 0.52 bpb, at this budget. The
  bit-shift scheduler is REQUIRED equipment for long strict runs
  (`--lr-decay-every`).

**Training story for the comparison, revised:**
1. PRIMARY — warm-handoff: host-train strict (the Round-4 recipe, healthy
   at its budget), stage int weights to fabric, SERVE on fabric, and
   fine-tune on fabric over short horizons (the kernel trains genuinely
   for the first ~2.5k positions before drift — short-horizon adaptation
   is usable today).
2. STRETCH (research) — from-scratch on fabric: needs the integer-lattice
   problem solved (stochastic-rounding calibration sweep, dithered
   thresholds, ordering-aware updates).

## Risks / notes

- Buffer layout change ⇒ existing AGFI cannot exercise the new loss; CFG
  words were appended at the END so old hosts/AFIs are unaffected.
- LUT-CE adds one V-loop + 256-word ROM per position — small vs the V×D
  head loop; BRAM +1.
- The anneal gap (+0.21–0.25) is the biggest known quality risk for the
  full-fabric claim; warm-handoff bounds it from above.
- Round-4 lesson applies on chip too: any future on-chip loss variant must
  keep gradient rows EXACTLY zero-sum in integer arithmetic (post-truncation
  balance), or sign updates will drift into the constant predictor.

## Silicon bring-up status for cl_forge_morph_lm (2026-06-12 session)

**Registered AFI: `agfi-029bc010e061e4626` (cl_forge_morph_lm_v5) — DSP=0,
TIMING MET (+0.007 ns @ the F2's fixed 250 MHz).** The kernel is
silicon-ready; one CL-level integration piece remains before the on-chip
training run completes.

What was proven on silicon this session:
- AFI loads and enumerates; OCL AXI-Lite control path ALIVE (registers
  read/write verified by probe; ocl-slave-timeout = 0).
- Kernel starts and masters PCIM (first weight burst completes: read
  count == WORDS_MH/16 exactly).

The blocking find: **the CL does not back the BAR4 buffer window.** The
bridge stages the int32 buffer via BAR4/DMA-PCIS at 0x01000000 and points
the kernel's m_axi at the same address — but this CL ties off DMA-PCIS
(`unused_dma_pcis_template.inc`), so host staging times out
(dma-pcis-timeout: 128k+ at 0x01000xxx) and the kernel's PCIM reads hit
unrelated host memory. The working v7-era CL evidently contains the missing
block: an on-CL buffer RAM (URAM; ~1.7 MB needed for ML_BUFFER_WORDS)
dual-ported between (a) a DMA-PCIS AXI slave for host staging and (b) the
kernel's gmem master, via a 2x1 AXI interconnect. That v7 CL design was
never carried into this repo — only its AFI survives.

Next session's task (well-scoped): build the buffer-RAM CL block
(HDK interconnect + AXI BRAM controller patterns from cl_mem_perf /
cl_dram_hbm_dma examples), re-run the validated pipeline (12-min P&R
iterations), and execute ~/morph_lm_silicon.py-style training.

### Hard-won F2 HDK build lessons (cost ~6 debug cycles; all now scripted)
1. The 2026 F2 HDK renamed the OCL ports (`sh_ocl_*`->`ocl_cl_*`,
   `ocl_sh_*`->`cl_ocl_*`); May-era CL wrappers synthesize CLEAN with
   floating implicit nets and a dead control plane. ALWAYS audit every
   shell-facing identifier against today's cl_ports.vh.
2. Use the HDK's `create_new_cl.py` scaffold; the flow expects
   build_all.tcl + per-CL aws_build_dcp_from_cl.py (-c CL, no -name, no
   clock recipes without --aws_clk_gen). The encrypt step FLATTENS design/
   (no subdirs).
3. The manifest generator parses LITERAL `CL_SH_ID0/1` defines from
   cl_id_defines.vh (device_vendor packing). Include-redirects break AFI
   registration (MANIFEST_PARSE_FAILED).
4. clk_main_a0 is FIXED 250 MHz on F2 — no slow-clock escape. HLS
   ap_clk estimates of ~11.4 ns on this toolchain are systematically
   pessimistic (the silicon-validated kernel reports the same).
5. **Argmax max-reduce loops do NOT pipeline as written**: UNROLL-8
   compare-select cascades into the running best/argmax registers are a
   ~-4.3 ns OOC path. Restructure as an explicit 8-lane balanced tree
   (feedforward) + ONE carried compare-select: -4.30 -> +0.18 OOC,
   full CL +0.007 MET. Applied to PJ/HEAD/batch-shift in
   forge_train_step_morph_lm.cpp; bit-exact (lowest-index-wins ==
   first-max; integer adds associative).
6. Wedged slots (hung kernel) survive `reboot`; instance STOP/START
   power-cycles the card. /tmp is lost — rebuild bridges.
7. Diagnose with: regprobe (BAR0 readback), fpga-describe-local-image -M
   counters (ocl-slave-timeout = dead control; pcim counts = burst
   progress; dma-pcis-timeout = unbacked BAR4), OOC synth of the
   UNENCRYPTED kernel verilog for named paths (the AWS flow encrypts
   user RTL — post-route reports hide names).
