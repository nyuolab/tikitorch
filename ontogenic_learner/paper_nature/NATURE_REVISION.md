# Nature reformat — what changed, and what flagship would still require

This directory now holds two parallel manuscripts:

- `main.tex` — the original ICML two-column version (untouched).
- `main_nature.tex` — a **Nature-flagship reformat** (single column, 193-word
  summary paragraph, flowing main text with run-in headers, Methods, Extended
  Data). Build: `latexmk -pdf main_nature` (12 pp, compiles clean).

Every quantitative figure still obeys the house rule — **numbers come only from
committed jsonls.** Nothing here changed a result; it re-presents them.

## Novelty + impact pass (2026-07-20)

Read-through + expansion focused on capturing novelty and potential impact (no
numbers changed; house rule intact). Three surgical edits to `main_nature.tex`:
- **Contributions paragraph** — sharpened the positive novelty from one sentence
  to a *category* claim: the new object is not a cheaper two-process pipeline but
  the *disappearance* of the second process (serving and learning are the same
  computation, differing only in output-clamping). Framed as a **systems-integration
  first** ("to our knowledge this identity has not before been carried end to end
  ... it is the integrated artifact, not any one ingredient, that we claim"),
  consistent with the honest prior-art positioning (no "first mult-free training").
- **Silicon-energy section** — added a grounded, hedged significance sentence
  ("even as a bound...") tying the ≤1 W floor + low-compute frontier + replayable
  audit to the on-device/edge regime (sensor/field/private data, no training
  cluster).
- **Main-text close** — added a forward-looking impact landing so the paper does
  not end on limitations alone; framed as *opening a design space* (not toppling
  incumbents), with the biological-learning tie kept as analogy.
Build clean: 21 pp, 0 undefined refs, abstract 199 words.

## sn-jnl migration + figure overhaul (2026-07-20)

**Now on the official Springer Nature `sn-jnl` template** (`\documentclass[sn-nature]{sn-jnl}`).
Build: `latexmk -pdf main_nature` (21 pp, 0 undefined refs, one command).

- **Template + deps fetched into the paper dir** (BasicTeX lacked them; `tlmgr` needs admin):
  `sn-jnl.cls`, `sn-nature.bst`, and `cuted`/`stfloats`/`manyfoot`/`nccfoots`/`perpage`/
  `appendix`/`threeparttable`/`breakurl` `.sty`. The manuscript preamble must load
  `xcolor` + `manyfoot` (the class calls their commands at `\AtBeginDocument` but doesn't
  `\RequirePackage` them). The class loads geometry/hyperref/natbib(sn-nature) itself — do
  not re-load, and do **not** add `\bibliographystyle` (the class sets it; a second one
  errors). The old article-styled version is at `_deprecated/main_nature_articlestyle.tex`.
- **Bib change:** the 7 `@inproceedings` entries were converted to `@article` (journal=venue)
  because `naturemag.bst` (sn-nature) drops the venue on `@inproceedings` without `pages`
  ("pop empty literal stack"). No page numbers were fabricated. **TODO (verify):** confirm
  these render acceptably or restore proper proceedings metadata during citation verification.
- **Figure reorganization → 5 dense multi-panel main figures + Box + ED + Supp:**
  - Fig 1 (NEW, Gemini `figs/gen/fig1_hero.jpg`) — paradigm + 6-stage study roadmap.
  - Fig 2 — mechanism (TikZ loop + substrate).  Fig 3 — `fig_pcdepth` (a/b/c).
  - Fig 4 (NEW combined `fig_validation.pdf`, `make_figs_panels.py`) — a: V2 headroom | b: gate
    law | c: Pareto.  Fig 5 (NEW combined `fig_lifelong.pdf`) — a: CE | b: fire | c: probes | d: norm.
  - ED: ladder table, optimizer table (`tab:opt`), `fig_efablation`.
  - Supp: `supp_pc`, `supp_organism`, `supp_depth`, and **NEW Gemini `supp_neuromimetic.jpg`**.
  - The standalone `fig_pareto`/`fig_life`/`fig_gatelaw`/`fig_retention` are superseded (still
    built by `make_figs*.py`); all in-text refs updated to the consolidated Fig 4/Fig 5.
- **Silver & Sutton, "Welcome to the Era of Experience"** (`silver2025experience`) cited in the
  agency section (streams of experience / lifelong learning).
- **AI-figure caveat (per user):** the two Gemini figures (`fig1_hero`, `supp_neuromimetic`) are
  **design drafts**, captioned as such — to be **redrawn as vector before formal submission**
  (Nature bars AI-generated figures). Prompts + variants kept in `figs/gen/`.

## Closed-optimizer update (2026-07-19) — retiring the AdamW waiver

New this pass: a fully **multiply-free, Adam-free** optimizer path, so the loop can
run with *no* audit waiver (the reported lives still used AdamW; this is the closed
alternative, honestly scoped to a shorter gated drift life).

- **`EFTropicalSGD`** (error-feedback) + **adaptive-step / momentum toggles**, and
  **`PrecisionAnchor`** (Fisher-proxy importance from the gradient's frexp tier, not
  Adam `v̂` / not displacement) — all in `tikitorch/optim/__init__.py` +
  `st2_live.py`, wired into `st4_life.py` (`--opt hybrid`, `--anchor {…,precision}`).
  Closure-verified: **0 hardware multiplies** vs AdamW's 4.
- **Result (X1/X1b parity hunt, `v4_life.jsonl` tags `x1*`):** `PureTropicalSGD`
  captures **0.41 headroom = 67% of anchorless AdamW (0.62), 79% of its anchored
  point (0.52)**, no collapse; error feedback **cures the fire-always adversary**
  (`fig_efablation.pdf`). **Parity is a quantified NEGATIVE:** no closed sign-based
  arm beats 0.41; the gap is per-parameter **variance normalization**, which
  sign updates don't replicate. And EF's stability vs plasticity are in tension —
  no single closed arm is both maximally plastic and adversary-immune.
- **In the papers:** `main_nature.tex` — new Methods paragraph "A fully closed
  optimizer", Extended Data Table 2 + Fig 3 (`fig_efablation`), Limitations item
  (vi). `main.tex` — substrate sentence + App. "The closed optimizer".
- **Open (handed to the agent instance):** closing the plasticity gap (a *closed*
  variance normalization) and the retention win (precision-derived consolidation,
  X2/X3 not yet run). Level-2 precision-*weighted* errors (active inference; breaks
  cascade≡BP) reserved for the embodied-agent track.

## What was added / changed

**Figures**
- `figs/fig_pareto.pdf` — **upgraded** to match the D2 artifact: Pareto-frontier
  staircase, low-power-vs-GPU segmentation shading, the "no incumbent cell lies
  here" dominance box, the 3.3×-budget / 2.7×-memory (138 vs 51 MB) callout, and
  de-collided labels. Regenerated by `make_figs_nature.py` (overwrites the old
  bare scatter; the ICML build picks up the better figure too).
- `figs/fig_pcdepth.pdf` — **new** three-panel "predictive coding at depth":
  (a) cascade gradient ≡ backprop, cos = 1.000 over all 282 tensors, rel-L2
  ~1e-7, sync bitwise / async routed-or-not (`g0_gradcheck`, `g1_equivalence`);
  (b) 128-layer residual MNIST — cascade trains to 0.94 where float backprop
  collapses to chance and naive iterative PC never leaves chance
  (`scale_mnist_128layer`); (c) our own MorphSSM LM, cascade ≡ backprop at
  depths 4–32 (`g3_g4_*`). All 3-seed where seeds exist.
- **Fig 1** is inline TikZ in `main_nature.tex` — the one-process serve-and-learn
  loop + the tropical substrate, ported from the artifact's D1 diagram.
- `fig_life`, `fig_gatelaw`, `fig_retention` reused unchanged (already
  data-backed and clean).

**Supplementary conceptual graphics (Gemini / Nano Banana Pro)**
- `figs/gen/supp_pc.jpg` — "Learning reuses the serving computation": the
  winner-take-all forward path (blue) reused backward (purple) for credit, with
  the "no multiply / no second network / no stored activations" badge. This is
  the *conceptual* companion to Fig 1 + Fig 2a.
- `figs/gen/supp_organism.jpg` — "Two ways to keep a model current": the always-on
  living device on its river of experience vs. the conventional GPU-cluster +
  separate-serving split with periodic frozen snapshots.
- `figs/gen/supp_depth.jpg` — "Predictive coding, made stable at depth": naive
  iterative settling explodes with depth (accuracy → chance) vs. the cascade's
  single stable multiply-free sweep accurate to 128 layers. Conceptual companion
  to Fig 2b.
- All three are wired into `main_nature.tex` as **Supplementary Fig. 1 / 2 / 3**
  and *also* into the ICML `main.tex` as an appendix (**Figs 6–8**). Regenerable
  via `gen_gemini_fig.py` (reads `GEMINI_API_KEY` from `.env`,
  `gemini-3-pro-image-preview`, 2K). Prompts + rejected variants
  (`supp_pc_A/B/C`, `supp_org_A`, `supp_depth_A`) are kept in `figs/gen/`. These
  are AI-rendered *conceptual illustrations* (labelled as such in-caption) — they
  carry no data; every quantitative claim remains matplotlib-from-jsonl.

**Text / structure**
- 193-word summary paragraph (Nature ≤ 200, non-specialist, minimal numbers).
- Main text restructured into a science narrative; the V0…V5 ladder mechanics
  moved to Methods + Extended Data Table 1 (the honesty/pre-registration is a
  strength — kept, just relocated).
- **Box 1** — a neuroscience map (predictive coding, no-weight-transport,
  three-factor / neuromodulated plasticity, complementary learning systems,
  ~20 W metabolic efficiency) → mechanism → evidence, with the repo's standing
  guardrail ("a source of hypotheses, not a claim of biological fidelity; every
  claim is settled by the measurements") and two falsifiable predictions.
- `neuro_refs.bib` — 11 added neuroscience references (Friston; Lillicrap et al.
  ×2; Frémaux & Gerstner; McClelland; Kumaran; Attwell & Laughlin; Benna & Fusi;
  Bellec; Payeur; Bartol). **TODO before submission:** run these through the
  repo's citation-verification workflow (as `references.bib`'s header notes were).

## Honest flagship-viability verdict

The reformat makes the paper *structurally* Nature-flagship-ready. It is **not
yet scientifically competitive for the flagship**, for three reasons the
manuscript states plainly and reviewers will press:

1. **Toy scale.** ~1M parameters, BPE-1024. Flagship wants a decisive, general
   advance; a sub-1M-param LM invites "interesting mechanism, prove it matters."
2. **Energy result is a floor, not a win.** V5 is core-rail ≤1 W (below the 1 W
   instrument quantization) with a rails-vs-board GPU caveat. The 79 W GPU
   contrast is real but not a like-for-like board comparison.
3. **Capability ceiling.** A dense transformer reaches matched pre-training
   quality ~7× faster (steps) / ~70× (wall) and generalizes better when frozen;
   we win only the always-on / low-power segment.

Given that, the realistic Nature-family home today is **Nature Machine
Intelligence** (best fit) or **Nature Electronics** (if the silicon story
leads). The current `main_nature.tex` is written to flagship format so it also
drops cleanly into either.

## What flagship acceptance would additionally require (roadmap)

Ranked by leverage. Items 1–2 are the difference between "nice systems paper"
and "flagship."

1. **A non-toy scale result (highest leverage).** Even one convincing run past
   the toy tier — the 30M / BPE-8192 target — showing the fused loop and the
   depth story hold. *Blocker:* the localized silicon backward-path divergence
   (see `memory/s2-30m-*`). Interim credible move: report a converged quality
   number at the **d832 cascade** that already trains (where autograd OOMs at
   44 GB) — turns the memory anecdote into a scaling data point.

2. **Board-level energy.** External 12 V card instrumentation to convert the
   ≤1 W core-rail floor into a real board-power number vs. the 79 W GPU. Without
   it the central "learning is energetically invisible" claim stays a bound.
   (Pre-registered as feeding the ASIC study — that framing is fine, but a
   flagship wants the measured board number.)

3. **A second modality / the agentic result.** The paper's most original claim
   is "experience *is* training." One embodied/control demonstration (the E2
   gate in `ROADMAP_EMBODIED.md`), or any non-language stream, would make the
   thesis general rather than a language-modelling curiosity.

4. **Multi-seed everywhere.** The V4 life is now 3-seed (good); the **incumbent
   Pareto is single-seed** — reviewers will flag it. Re-run V3 at 3 seeds.

5. **Reproducibility as a headline.** Nature requires data + code availability;
   the committed-jsonl house rule + fully-replayable audited update stream is a
   genuine strength — foreground it (already in the availability statements).

6. **Citations + prose polish.** Verify `neuro_refs.bib`; a professional copy
   edit of the reformatted main text.

**Bottom line:** ship the reformat now to NMI/Nature Electronics; items 1 and 2
are the gate to a credible flagship attempt.
