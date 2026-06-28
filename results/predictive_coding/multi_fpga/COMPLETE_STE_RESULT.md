# Complete-STE on the PC-Transformer Cascade — DECISION: NO / PARTIAL (negative lift, diagnosed impl defect)

**Date:** 2026-06-28
**Substrate:** single-block PC transformer cascade (NBLK=1, OUTPUT role), char-LM (tinyshakespeare), CPU golden (`g++ -O2`), D=64, NH=2, SEQ=64, ATTN_DOT=1, HEAD_RESCALE=6, batch1 (ACCUM=1).
**Canonical tree:** `/tmp/tt-pc/tikitorch/hardware/hls/forge_kernels/` and `…/results/.../forge_pc_xformer_mp_golden_block.hpp` — **git-clean, byte-unchanged** (`diff -q` clean). All experiments live in `/tmp/ste_exp/`, `/tmp/ste_verify/`, `/tmp/bar_temp_probe.py`.

---

## DECISION: NO / PARTIAL — wide backward — **DO NOT BUILD off this result**

Complete-STE as implemented **does NOT lift the cascade from ~0.13 to ~0.29**. It is a **negative lift**: it lands **below** the off cap and **barely above chance**. This directly **refutes** the BUILD-GO criterion ("complete STE lifts the cascade to ~the char-LM bar at single AND multi block → cascade de-risked"). The bar IS ~0.29 on this exact config (single AND multi-block), so the implementer's "this config's accuracy ceiling is ~0.15 / the 0.26 bar belongs to a larger char-LM" framing is **wrong**.

```
on_singleblock = 0.091   <   off_cap = 0.127   <   autograd_bar = 0.286
```

Complete-STE closes **0%** of the 0.13→0.29 gap and **regresses ~0.036 acc below the off cap**. Multi-block ON was **never implemented and never measured** (reported as `-1`). The cascade-reaches-the-autograd-bar claim is **unproven**; the no-sync PC transformer cascade is **NOT de-risked**.

---

## 1. The numbers — ALL measured on this exact config (batch1 / ACCUM=1, 1500 steps)

| Config | acc | bpb | seeds | reaches bar? |
|---|---|---|---|---|
| **Autograd bar** — single-block, dot-attn, AdamW, dyad-W | **0.286 / 0.304** | 3.55 / 3.51 | s7 / s2 | — (this IS the bar) |
| Autograd bar — multi-block NBLK=3 H=4 dot | **0.283** | 3.56 | s7 | depth-flat (no depth cap) |
| STE_FULL=off **CAP** (dot) | 0.127 / 0.131 | 6.30 / 6.16 | s7 / s2 | no |
| **COMPLETE-STE ON** (STE_FULL=on, wide-float, dot) | **0.091 / 0.093** | 6.92 / 6.72 | s7 / s2 | **NO — below cap, above chance** |
| Complete-STE ON, L1 variant | 0.109 | 5.65 | s7 | no |
| Complete-STE ON, 5000 steps lr0.005 | 0.091 | 7.14 | s7 | no — **not slow convergence** |

- Chance bpb = 6.0. Complete-STE-ON `bpb = 6.7–7.1` is **above chance** — the model is *worse than guessing* by bits, then training slowly tames it back toward chance from above.
- The autograd bar starts healthy (`bpb ≈ 6.08` at step 0/1) and **drops to ~3.5 by step 400**. The complete-STE wide forward starts **pathologically high — `bpb = 32.7–36.4` at step 1** — and only ever crawls toward chance.
- Determinism: ON reproduced **bit-identically** across recompiles.
- ACCUM>1 cosmetic only: with `ACCUM=4` the bar-witness reaches a trail-100 ~0.15 (`barw_s7_dot.txt`) but still starts at `bpb ≈ 34` and never learns structure; batch1 is the apples-to-apples comparison to the bar and reads 0.091.

**Cross-check vs prior docs (same substrate):** the 0.286/3.55 bar **matches** the documented 0.297/3.55 single-block bar in `STE_CASCADE_RESULT.md` (commit 295bdbf). The 0.127 off cap matches that doc's ~0.135 off cap. So the bar and the cap are stable, independently-confirmed reference points; only the **complete-STE "bar-witness" path** is anomalous.

---

## 2. Forward byte-identical — CONFIRMED (the one hard-constraint that passes)

- FWDCHECK: `ga_hash(fwd1) == ga_hash(after_wide) == ga_hash(fwd2) == 1669636611019089498`, `memcmp = 0` — matches the implementer's reported hash. The wide backward does **not** perturb the forward.
- `STE_FULL=off` is **bit-exact** to the canonical golden (`/tmp/ste_verify/canon_mptest`).
- Canonical `forge_pc_xformer_mp_golden_block.hpp` byte-unchanged; all work isolated in `/tmp/ste_exp` copies.

This is the only piece of the BUILD-GO checklist that holds. The mult-free forward (DSP_MULT=0 on the deploy-weight projections) is intact and unchanged — but a correct forward with a **broken trainer** does not de-risk anything.

---

## 3. Why it is anomalous — the chain-rule guarantee says this MUST be an impl bug

Complete-STE (every forward quantizer = identity in the backward, wide pre-quant activations cached, wide error path) is, by the chain rule, the **exact gradient of the un-quantized forward** = block-local == global == the autograd bar. So a *true* complete-STE **cannot** underperform the autograd bar by construction. The 0.091 ≪ 0.286 divergence is therefore **an implementation defect, not a real STE failure**. Three independent observations localize it:

**(1) The wide forward is numerically pathological at init.** It starts at `bpb = 32.7–36.4` (step 1) vs chance 6.0 and a healthy model ~6.0-and-falling. A correct softmax-CE forward on small random weights cannot produce 32 bpb unless the logits/temperature/scale handling is mis-wired (e.g. the head/Q/K/V `temp_scale` or the logit accumulation in the *wide* path differs from the mult-free path it is supposed to shadow). Training does not learn structure; it slowly tames an explosion toward chance from above.

**(2) The cited "gradcheck proof" never validates the weight gradients — only the input gradient.** `ste_gradcheck.cpp` perturbs residual-stream entries `R[idx]` via `g_ste_dbg_gR` and compares the central difference to `gR[idx]` (line 66 `double an = gR[idx];`, line 73 prints `dL/dR continuous fd-check`). It **never checks** `g.WQ / g.WK / g.WV / g.WO / g.WUP / g.WDN / g.WHEAD` — the *weight* gradients that actually move the model. A correct input-grad path with a wrong **wgrad** path is exactly consistent with the symptom (gradcheck "passes," model won't train). Re-running the gradcheck even for the input path is **not clean**: only **23/29 within 2%** at default `h`, with several `rel = 1.00` misses (e.g. `R[806] fd = -7.35` vs analytic `-0.016`) — the stop-grad temperature-shift discontinuities break even the path that *is* checked.

**(3) The obvious suspect is RULED OUT — it is genuinely the C++ wide path.** I added the bar-witness's exact stop-gradient per-tensor max-shift temperature scaling (Q/K/V + head, `scale = 2^-sh`, detached) to the **python autograd bar** (`/tmp/bar_temp_probe.py`). It **still trains to acc = 0.286 / bpb = 3.547** (reproduced this turn: step 0 `bpb=6.079`, step 1499 `acc_ema=0.286 bpb_ema=3.547`), starting *healthy* at ~6.08 and dropping. So the temperature-scaling design choice is **not** the cause. The defect is in the **C++ `g_forward_wide` / `g_backward_ste`** itself — most likely the wide logit/scale handling (the bpb=32 init) and/or the analytic LayerNorm / attention **weight-gradient** path (untested by the gradcheck).

This **does not contradict** the chain-rule guarantee — it confirms the implementation does not realize complete-STE. A correct complete-STE would, by construction, reproduce the 0.286 bar.

---

## 4. Multi-block ON — NOT IMPLEMENTED, NOT MEASURED (`-1`)

The STE files (`forge_pc_xformer_ste_block.hpp`, `ste_bar_witness.cpp`) are **single-block only** — no NBLK loop, no inter-block wide error path. The task required NBLK=3 multi-block; **it does not exist to test**, and the implementer's report contains no multi-block STE number. (For reference, the autograd bar **is** depth-flat: NBLK=3 batch1 = 0.283 — so the multi-block target is also ~0.28, and there is no depth cap on the bar to "explain away" the single-block miss.)

---

## 5. Wide-vs-mult-free status of the backward

The complete-STE bar-witness backward as built is **WIDE (float / `__int128`)**, not mult-free. The magnitude-aware dyadic mult-free narrowing is the *intended next optimization* — but it is **moot until the wide path first reproduces the bar**. You cannot narrow a backward that does not yet hit the target it is being narrowed against. (Per the earlier `WIDE_BACKWARD_RESULT.md`, widening the backward error path alone — int16band → fix24 → bf16 → double — also **did not** reach the bar; double ≈ int16band. So even setting aside the bar-witness defect, "wide backward → bar" was already independently refuted at single block.)

---

## 6. Honesty caveats

- **Seeds:** small. Bar s7=0.286 / s2=0.304; off cap s7=0.127 / s2=0.131; ON s7=0.091 / s2=0.093. The ordering `on < off < bar` is stable and the ON path is deterministic, so the **sign of the result is robust**, but absolute accuracies carry seed noise of a few pp.
- **Scale:** D=64 single block, char-LM, 1500–5000 steps, CPU golden. This is a *de-risk probe*, not a production run. The 0.286 "bar" is itself a small-model number (batch32 reaches 0.446) — but it is the correct apples-to-apples target for this config, and the cascade misses it by ~0.20 acc.
- **Wide vs mult-free:** the ON path is wide-float, the *most favorable* setting for STE. It still fails — so a mult-free narrowing cannot rescue it.
- The negative is **not** a metric artifact: bpb corroborates acc (ON bpb 6.7–7.1 *above* chance vs bar 3.5).

---

## Files
- `/tmp/ste_exp/forge_pc_xformer_ste_block.hpp` — STE backward (single-block only; `STE_FULL`/`STE_BWD` toggle).
- `/tmp/ste_exp/ste_bar_witness.cpp` — the wide-float bar-witness trainer (pathological bpb=32 init).
- `/tmp/ste_exp/ste_gradcheck.cpp` — fd-check that validates **only** dL/dR, **not** the weight grads.
- `/tmp/ste_verify/ablate_bar.py` — autograd bar (single/multi-block, dot/L1, AdamW/dyadic-Adam).
- `/tmp/bar_temp_probe.py` — proves the stop-grad temperature design is NOT the cause (bar = 0.286 with it).
- Reference docs (same substrate): `STE_CASCADE_RESULT.md` (PARTIAL, 0.135→0.15), `WIDE_BACKWARD_RESULT.md` (NO-GO, widening buys nothing).

---

## What to do next (NOT a build)

1. **Fix the wide path before anything else.** Diagnose the `bpb=32` init: dump `g_forward_wide` logits vs the mult-free forward's logits on identical input (the forward hash matches, but the *wide* forward used for the gradcheck/bar-witness may scale logits differently). Then **add a weight-gradient finite-difference check** (`g.WHEAD`, `g.WQ`, …) — the current gradcheck would pass a wrong wgrad.
2. Only after the wide complete-STE reproduces the **0.286 single-block bar** (and the 0.283 NBLK=3 bar): port to multi-block (NBLK loop + inter-block wide error path) and re-measure.
3. Only after BOTH reach the bar: narrow to magnitude-aware dyadic mult-free, then fold into the canonical golden and write `forge_pc_xformer_mp_block.cpp`.

**Do not** fold complete-STE into the canonical golden, **do not** write `forge_pc_xformer_mp_block.cpp`, and **do not** update `KERNEL_DESIGN_SPEC.md` with a de-risk claim off this result. The de-risk is not achieved.

---

DECISION: NO-PARTIAL — wide (mult-free narrowing is moot until the wide path reaches the bar) — Next action: diagnose the `g_forward_wide` bpb≈32 init and add a finite-difference check on the **weight** gradients (`g.WHEAD`/`g.WQ`/…), because the current gradcheck only validates dL/dR and a wrong wgrad path is exactly consistent with on=0.091 < off=0.127.
