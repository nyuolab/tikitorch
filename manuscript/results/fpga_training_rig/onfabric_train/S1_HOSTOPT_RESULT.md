# S1 — A multiplier-free transformer trains on real F2 (host-optimizer hybrid)

**Date:** 2026-06-26 · **AFI:** agfi-0179e963c146d39f7 (v17) · **Instance:** f2.6xlarge
**Driver commit:** 6710a76 · **Raw log:** [`s1_hostopt_freeze_train.txt`](s1_hostopt_freeze_train.txt)

## Result (measured on silicon)

```
INITIAL  CE = 4.1487   acc = 24/64
step  20: CE = 3.7583   acc = 64/64
step  40: CE = 3.5042   acc = 64/64
step  60: CE = 3.1428   acc = 64/64
step  80: CE = 2.8246   acc = 64/64
step 100: CE = 2.5765   acc = 64/64
step 120: CE = 2.4499   acc = 64/64
step 200: CE = 2.0951   acc = 64/64
FINAL    CE = 2.0951    acc = 64/64

CE 4.1487 -> 2.0951  (drop 2.0536 nats; uniform/collapse floor = 4.1589)   acc 24 -> 64/64
[gates] CE-drops>=0.5: yes   below-floor: yes   acc-improves: yes   no-saturation: yes
-> ON-FABRIC TRAINER CONVERGES (silicon == CSIM)
```

200 steps @ 85.6 ms/step (incl. per-step host round-trips).

> **Reading the gate line honestly:** the binary that produced this log prints the generic gate string
> `ON-FABRIC TRAINER CONVERGES (silicon == CSIM)`. That string is loose and the later driver build softens it
> to `S1 TRAINER (fwd+bwd on F2, host optimizer) CONVERGES (trajectory matches CSIM)`. Two corrections apply:
> (1) the optimizer step is **host-side**, not on-fabric — only the forward and backward run on F2; (2)
> "silicon == CSIM" means the **convergence trajectory matches** the csim recipe, **not** bit-exactness — the
> silicon and csim runs differ by ~0.03 nat (silicon CE0 4.1487 vs csim 4.1627; silicon step-120 2.4499 vs
> csim 2.4761 vs the verifier's real-kernel probe 2.4457). The only **bit-exact** claim is the optimizer VALUE
> math vs the kernel `#else` path (0 mismatch over 8M element-steps).

## What ran where

Per training step:
1. **mode-0 forward on F2** — embed gather → dyadic GEMMs → attention soft-blend → fine-dyadic LayerNorms → max-plus head → CE. Logits written to DDR.
2. **mode-1 backward on F2** — full backprop through the multiplier-free transformer block; writes real gradients `g_EMBED`, `g_W_HEAD` to DDR (measured `max|g_W_HEAD|=516096`, `max|g_EMBED|=54656`).
3. **dyadic-Adam VALUE step on the HOST** — reads back the FPGA-committed grads; updates EMBED + W_HEAD masters/m/v and the live int16 deploy values. Bit-exact to the kernel `#else` value path / TB `ref_dyadic_adam_region` (verified 0 mismatch over 8M element-steps).
4. **BAR4 restage** of the two updated live regions (EMBED + W_HEAD) so the next forward reads the stepped weights (+ read-after-write flush).

The CE is computed from logits **read back from DDR after an FPGA forward** (the host never computes logits); the gradients are **FPGA-computed** (the host never computes gradients). The forward reads the host-updated weights — a flat curve would mean it didn't; the drop is the proof it does.

## Honest scope

- **Hybrid:** forward + backward on the FPGA, optimizer step host-side. The fully-on-fabric single-kick optimizer is blocked by the `do_opt` cosim≠silicon write wall (kernel `do_opt` m_axi writes commit in C/RTL co-sim but never reach board DDR; `do_fwd_loss`/`do_bwd` writes do commit — `do_opt`-specific, below cosim's behavioral-AXI fidelity).
- **FREEZE_CODES:** only the EMBED (input embedding) + W_HEAD (output head) value weights train. The transformer backbone (Wq/Wk/Wv/Wo/W_up/W_down dyadic codes + LN gammas) is frozen at the residual-identity init. Forward + backward flow through the full multiplier-free block on F2 and gradients are computed for it, but the updated parameters are embed + head only — this is **not** a from-scratch dense-backbone train on silicon.
- **Memorization task:** PAT=8 successor, 64 tokens, V=64. acc→64/64 = perfect memorization of the training sequence, not generalization.

## Cross-validation

Silicon step-120 CE `2.4499` ≈ the verifier's independent real-kernel probe `2.4457` (which drove the actual kernel mode-0/mode-1 with this exact host step for 120 steps → CE 4.1627→2.4457, acc 0→64/64). The full-FREEZE g++ csim baseline is CE 4.1627→2.4761.

## Multiplier-free closure

No datapath multiply in the training loop. The one irreducible multiply is the attention soft-blend (`Σ p·V`, on DSPs in the kernel forward/backward). The host dyadic-Adam step is shift-add only. DSP-as-adder/address arithmetic is allowed and used.

## Config

D=64, H=2, DH=32, FFN=256, L_SEQ=64, V=64, N_LAYERS=1, Q_CODE=6. Kernel clock clk_extra_a1 = 62.5 MHz (Recipe A0); shell clk_main_a0 = 250 MHz.
