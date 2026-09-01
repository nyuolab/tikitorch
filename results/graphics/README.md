# Historical predictive-coding programme graphics

Status: archived research communication from 2026-07-03, not the current
TikiTorch README, manuscript, or release claim surface. The ontogenic-agent
programme now belongs in a sibling repository; this archive makes no claim that
an agent or cloud instance is currently running. Revalidate every number and
projection against the current claim ledger before reusing a panel publicly.

The underlying depth experiments and theory notes are preserved under
`results/pc_depth_traction/`. Their historical summaries used matched data but
not matched compute and should not be promoted into a systems advantage.

## PC-TikiTorch project graphics

Schematics for the predictive-coding-on-FPGA-fabric program: scaling LLM training across
AWS F2 FPGAs using **predictive coding** as the local learning rule and **tropical/dyadic
(multiply-free) arithmetic** as the compute substrate.

Each figure exists in three forms:

- `NN_name.svg` — the hand-drawn, editable **source of truth**. Every number is real and
  traceable; edit here first.
- `NN_name.png` — a flat render of the SVG (the Nano Banana input / quick preview).
- `NN_name_nb.png` — the **Nano Banana Pro** (gemini-3-pro-image-preview) polished version,
  generated from the PNG + the prompts baked into `nb_generate.py`. These are the
  presentation-grade images.

Numbers were frozen on **2026-07-03** from the then-current predictive-coding
research records. Historical branch and worktree names are not reproduction
instructions. Before reuse, update the SVG text from current, immutable evidence,
re-render, and regenerate.

---

## The figures

### 01 — Big picture (`01_big_picture`)
The one-slide thesis: three pillars (tropical/dyadic arithmetic · predictive coding · FPGA
fabric) converging on what is measured on silicon today, with the 124M north star clearly
labeled as projected. It was the historical opening slide; do not use it as a
current README header without a fresh claim review.

### 02 — Backprop vs predictive coding (`02_backprop_vs_pc`)
Why the learning rule had to change: backprop's global gradient chain / weight transport /
sync barrier on the left, PC's local error units with one-hop x↓/ε↑ messages on the right,
and the punchline that a layer = a chip.

### 03 — The fabric (`03_fabric`)
The machine as built: host + 8 F2 chips, one transformer block per chip, neighbour-only
message arrows, the honest note that today's physical transport is host PCIe relay (TcpLink
next), the 1F1B pipeline staircase, and the measured band (kicks, 258 tok/s, bpb 3.187 /
3.675). The 1F1B grid is data — if regenerating, the exact cell contents are pinned in the
prompt inside `nb_generate.py`.

### 04 — Inside one chip (`04_inside_chip`)
One block, three precision worlds: multiply-free forward (dyad_matmul, int LN, attention),
BF16 backward (the one place multiplies are allowed), dyadic-Adam + in-kernel ACCUM. Plus
the CDC/timing-closure box, the DSP_MULT=0 invariant banner, and the post-P&R utilization
bars (BRAM 97% = the wall that forces DDR streaming at 30M).

### 05 — The ladder to 124M (`05_scaling_ladder`)
The milestone ladder with honest labels: five measured rungs (2-chip backward → 8-chip
Fashion-MNIST → 3-chip transformer → converged 3.187 → 8×8 blocks 3.675), the in-flight
throughput rebuild, and the projected 30M / 124M summit, with the steps/s ladder inset.

### 06 — The realistic path to 100B (`06_path_to_100b`)
The scaling figure, **adversarially fact-checked** (see provenance below). Contents:

- **Target**: D=10240 · L=80 · vocab 50k ≈ 100.7B params — literally YaLM-100B's shape.
  One block per chip → 81 chips = 11 F2 boxes.
- **Memory is not the wall**: per-chip optimizer state 8.8 GB (bf16) fits the 16 GiB HBM;
  fp32 variant fits the 64 GiB DDR4; in-flight activations 7–54 GB go to DDR4 with
  flash-style recompute (+~20% compute). Messages stay negligible → compute-bound.
- **The #1 research risk**: at L=80 the in-flight window is ~160 microbatches, so the
  measured staleness rule (ACCUM ≥ window) floors the batch at ~330k tokens/step — fine by
  100B standards (2–4M is normal) but validated only at 3–8 chips.
- **Four multiplicative axes**: ×L chips (measured), ×S slots (measured→E), ×K MAC width
  (the rate ladder 0.15 [M] → 2.4 [E] → ~150 [P] → ~0.9T [speculative]), ×R replica chains
  (DiLoCo-style outer averaging — no per-step all-reduce; open research at R≥100).
- **The wall, honestly**: wall-clock *and dollars*. One chain: 1B ≈ 9 months ✓, 10B ≈ 50
  years ✗, 100B ≈ 260–1,600 years ✗. Replicas close the time (R≈530–3,200 → ~6 months) but
  rented F2 dollars ($0.4–2.2B) lose to GPUs ($2–10M) by 40–1000× — so the deliverable tier
  is 1–10B, and 100B full pretrain is published as lever math, not a claim.
- **The honest claim ladder**: (1) 1B converged [next], (2) 10B short-token regimes,
  (3) 100B datapath measured one-step (the 124M playbook), (4) 100B pretrain as projection
  only.

**Provenance:** every numeric claim in figure 06 was verified by a 5-lens adversarial
workflow on 2026-07-03 (arithmetic re-derivation · AWS F2/VU47P hardware specs via web ·
consistency with THROUGHPUT_ROADMAP.md/FLAGSHIP_SILICON_RESULT.md · ML-scaling literature
(Chinchilla, GPT-3, DiLoCo arXiv 2311.08105, YaLM-100B) · hostile-reviewer attack pass).
The dollar wall, the 160-deep window risk, the activation row, and the [speculative] tag on
the 0.9 TMAC/s ceiling all came out of that review.

---

## Regenerating

**PNG previews (macOS):**

```sh
cd results/graphics
qlmanage -t -s 1500 -o . NN_name.svg     # writes NN_name.svg.png (square, letterboxed)
mv NN_name.svg.png NN_name.png
sips -c <height> 1500 NN_name.png        # crop letterbox: height = 1500 * viewBox_h / viewBox_w
```

(viewBox sizes: 01/02/04 → 964, 03 → 950, 05 → 900, 06 → 1190.)

**Nano Banana versions:** [`nb_generate.py`](nb_generate.py) (in this directory) holds the
per-figure prompts — including the pinned 1F1B grid data for 03 and the verbatim-transcription
mode for the text-dense 06. It reads `NANOBANANA_API_KEY` / `NANOBANANA_MODEL` from `./.env`
(gitignored) and writes `NN_name_nb.png`:

```sh
python3 nb_generate.py                 # all figures
python3 nb_generate.py 03_fabric       # one figure
```

Generation is nondeterministic — inspect each output; text-dense panels (the 1F1B grid, the
wall table) occasionally need a retry. Keep `.env` out of git (covered by `**/.env` in the
repo `.gitignore`).
