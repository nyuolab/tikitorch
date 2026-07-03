# G1b — silicon certification of the two-speed learner

**Run 1:** `i-0b6975bb1ca17ae4a` (fresh f2.12xlarge, tag `g1bcert-20260703T092854Z`,
AGFI agfi-0a08046c7d64090ff on both slots, L=2 INPUT+OUTPUT, 500 gated steps + 500-step
control, both EXIT=0, self-terminated; cost ≈ $3). Full logs + audit ledger + trained
masters in this directory.

## ✅ Mechanical certification: PASSED — the ontogenic learner RAN ON REAL SILICON

- **The ε-gated two-speed policy executed on F2 hardware**: gate fire-rate **72.8%**
  (364 fast kicks / 500 stream steps), **136 backward sweeps never issued** — the energy
  saving is physical (skipped BWD = no kick on the wire), 45 sleep replays from a
  364-window surprise-weighted reservoir at lr 3e-4. All mechanisms functioned.
- **Wall-clock savings measured**: two-speed 139.9 s vs control 151.9 s (3.57 vs 3.29
  steps/s) — skipping backward kicks saves real time/energy, bounded by the fire-rate.
- **Kick anchors reproduce exactly**: FWD 53.5/57.6 ms, BWD 91.5/99.1 ms per chip —
  identical to the fabric program's measured anchors on this AGFI. Our own box, same physics.
- **Gate-C clean**: zero weights or gradients crossed chips; only neighbour hidden/error
  messages (15.6 MB total).
- **The audit ledger exists**: `g1b_audit.csv` — 409 applied updates (364 fast + 45 sleep),
  each with step, kind, z-surprise, CE, LRBITS, data offset, seed. This is the
  "certifiable lifelong learning" artifact: every weight change on the deployed device is
  attributable and replayable.
- **Ops path certified end-to-end on first launch**: fresh box → AGFI load (2 slots) →
  source build → both runs → S3 upload → self-terminate. Zero collisions (own tag, own S3
  prefix, own state).

## ⚠️ Honest catch: the quality leg needs the rerun

Both arms (two-speed AND control) diverged identically at this config (bpb ema → ~39.4 in
both; XF_LR=3e-3 with ACCUM=1 at L=2 is too hot — the known-good fabric recipes pair 3e-3
with ACCUM=8). Because both arms show the same trajectory, the divergence is config, not
the patch — the equivalence half of the certification holds, but no quality-vs-twin claim
can be made from run 1. **Run 2 launched immediately** (`i-0f8129d554fe159c5`,
`g1bcert-20260703T094612Z`, XF_LR=1e-3, all else identical) to complete the pre-registered
step-3 quality gate.

## Run 2 (`i-0f8129d554fe159c5`, XF_LR=1e-3): diagnosis complete — it's the recipe, not the LR

Both arms diverged **identically again** at 1e-3 (bpb saturates at the 39.86 ceiling by step
~200; control ≡ twospeed near bit-for-bit through the warmup — the gating arm at 82.2%
fire-rate, 89 BWDs skipped, mechanics again flawless). Two LRs × identical dual-arm
trajectories = the L=2 / **ACCUM=1** recipe itself diverges — precisely the fabric program's
own root-caused finding (effective batch=1 is what in-kernel accumulation was built to fix),
compounded by L=2 being a never-trained topology.

**Fix (v2, better design anyway): window-level gating** — gate entire ACCUM windows instead
of single kicks. A skipped window drops ACCUM BWD kicks *and* ACCUM−1 FWDs (larger energy
multiplier), while a fired window is byte-identical to the serial accumulation path — so the
validated ACCUM=8 recipe carries over unchanged. Run 3 = L=2, ACCUM=8, XF_LR=3e-3,
cosine decay (the known-good pairing) with window-gated TWOSPEED vs control.

## Run 3 (`i-03c6084c50fb8dc4b`, v2 window-gating, ACCUM=8, XF_LR=3e-3, cosine): recipe eliminated — the differential is L=2 itself

Window-gating v2 (commit `d1905ac`) worked exactly as designed on silicon: **window fire-rate
72.6% (363/500 at ACCUM=8) — 137 skipped windows = 1,096 BWD + 959 FWD kicks never issued**,
and gated throughput reached **282 tok/s** (vs ~211 for the always-train pattern) — the
energy/throughput dividend of window-level gating is real and measured. But quality still
saturates (bpb → 39.4, acc 0.002), identically to runs 1–2, now under the exact recipe that
converged at L=3 on this same AGFI (fabric run #3: bpb 3.093, ACCUM=8, 3e-3, cosine).

**Three recipes × identical dual-arm saturation × a known-good-at-L=3 AGFI ⇒ the L=2
topology (INPUT directly feeding OUTPUT, no HID chip — never previously trained) is the
prime suspect.** Free differential test in flight: the CPU bf16 golden at NBLK=2 vs NBLK=3 —
golden-fails-too ⇒ config/topology deficiency; golden-converges ⇒ driver/kernel role-handling
bug at L=2. Next silicon dollar waits on that verdict (if L=3 is required, the quality leg
needs an f2.48xlarge, ~$16/hr).

## CPU golden differential: the topology is EXONERATED — the defect is silicon-runtime-side

A faithful CPU analog of the silicon path (the BF16 golden block with role-parameterized
INPUT/HID/OUT branches mirroring the kernel exactly, chained with the driver's message-relay
order) was built and run locally. **NBLK=2 learns cleanly** — bpb 6.57 → 3.756 over 1,500
steps, tracking NBLK=1 (3.673) and NBLK=3 (3.769) within seed noise; at ACCUM=8 it reaches
batch bpb 3.19 / acc 0.44 and keeps descending. No saturation, either seed.

So: config, recipe, role logic, and the 2-block topology are all sound. The silicon
symptom — bpb pegged at 39.86 with *below-chance* accuracy — is a numerical explosion in the
wide datapath, categorically different from the known benign ~3.8–4.0 batch-1 plateau.
**AGFI verified correct and `loaded ok` on both slots in both runs** (img.txt) — stale-image
ruled out.

**Run 4 (`i-0209bb9dd2eb79c80`) = the localization run** using the driver's built-in probes:
`FIXEDOFF` single-batch overfit (a sound kernel+box must drive one pinned batch toward
bpb≈0; explosion here ⇒ datapath fault) + `ACTSTATS=1` (per-chip wide-R'/logit
min/max/absmean/NaN counts ⇒ which chip explodes first).

## Runs 4–5: the defect is cornered — wrong-direction updates at L=2, silicon-side

- **Run 4** (`i-0209bb9dd2eb79c80`, FIXEDOFF pinned batch, ACCUM=8, 3e-3, ACTSTATS): the box
  **cannot overfit a single batch** (bpb 7.32 → 39.86) — while `ACTSTATS` shows the forward
  is **healthy at init** on both chips (chip-0 wide R′ ±8, zero NaN; chip-1 logits ±6).
- **Run 5** (`i-0876063611fcb7761`, the sign-vs-magnitude bisect: ACCUM=1, **LR 1e-4**,
  pinned batch): the **control leg** (byte-identical to the pre-patch serial driver, no
  gating, no sleep) **still climbs monotonically** — 7.32 → 14.59 @50 → 14.89 @100 → ~17.5
  @199. At 30× smaller LR, loss increases every step on fixed data. **That is the sign-flip
  signature: the backward/update path produces ascent-direction updates at L=2 on silicon.**

### The complete evidence chain

| # | fact | source |
|---|---|---|
| 1 | Two-speed mechanics work on silicon (gate/skip/sleep/audit; +34% tok/s at window gating) | runs 1–3 |
| 2 | L=2 quality fails identically across 3 recipes × both arms | runs 1–3 |
| 3 | The identical 2-block topology/roles/recipe **converge on CPU** (bpb 3.76; role-faithful BF16 analog) | golden differential |
| 4 | Correct AGFI `loaded ok` on both slots, both runs | img.txt |
| 5 | Forward datapath healthy at init | run-4/5 ACTSTATS |
| 6 | Single-batch overfit fails at ANY LR; monotonic climb | runs 4–5 |

### Handoff to the fabric program (their tooling corners this fastest)

Suspects, in order: (a) a backward-path defect expressed only in the never-trained
**L=2 silicon configuration** (INPUT directly adjacent to OUTPUT — the gR seed/sign at that
hop, the INPUT-role EMB/POS wgrad scatter, or an LN-backward boundary term); (b) the
untested **f2.12xlarge** variable (all validated training ran on f2.48xlarge; same VU47P,
never cross-checked). Fastest probes, using their standard machinery: **one-kick
golden↔silicon KAT at L=2** (a wrong-signed gradient shows in a single kick), and **L=3 vs
L=2 on one f2.48xlarge** (isolates topology-vs-boxtype). All five run logs + audit ledgers
are under `s3://tiki-sweep-739966866637/g1bcert-*/results/` and mirrored in this directory;
the v2 driver patch is commits `17b7c26`+`d1905ac` on the detached worktree `/tmp/tt-g1b`.

**G1b standing:** mechanical certification **PASSED** (the pre-registered step-2 mechanics +
the energy ledger); the step-3 quality gate is **blocked on this L=2 silicon defect** —
which is itself a fabric-grade finding produced by the certification process working as
designed. Total spend: 5 boxes, ≈ $12.
