# Learning a tropical policy prior without collapse (manuscript note)

A self-distillation result for the multiplier-free / zero-DSP setting: how to train a tropical policy head
to imitate a search policy **without** the prior collapsing to a constant/one-hot predictor — and why the
fix stays inside the tropical (compare / add / shift / LUT) semiring. Backed by `board_agent` (the levers
+ `mode_frac` detector), `functional.tropical_cross_entropy` (the balanced gradient), the gradient-balance
regression test in `tests/test_games.py`, and the Connect4 sweep in `CONNECT4_PRIOR_SWEEP.md`.

## Setup

Distill the MCTS visit distribution $\pi$ into logits $z = W h$ with **dyadic** weights $w_{ij}=\pm 2^{k_{ij}}$
(so $z$ is shift+add, no multiply), trained by sign-only SignSGD, $w \leftarrow w - \eta\,\mathrm{sign}(\partial\mathcal L/\partial w)$, $\eta = 2^{-r}$.

## Two collapse mechanisms, two fixes

**(1) Gradient row-bias.** Cross-entropy $\mathcal L=-\sum_a\pi_a\log p_a$ gives $\partial\mathcal L/\partial z_a = p_a-\pi_a$.
The multiplier-free softmax is shift-normalized, $\tilde p_a = e_a\,2^{-\lceil\log_2\sum_b e_b\rceil}$ ($e_a=\mathrm{exp2}(z_a)$ a LUT,
the normalizer a **bit-shift**), so $s\equiv\sum_a\tilde p_a\in[\tfrac12,1)$ and $\sum_a(\tilde p_a-\pi_a)=s-1\neq 0$ — a
**constant per-row bias** that sign updates integrate into a drift → collapse. Fix: scatter the row-sum $s$
(not $1$) at the target,
$$g_a=\tilde p_a - s\,\mathbb 1[a=a^\star],\qquad \textstyle\sum_a g_a = s-s = 0,$$
a **scatter** (data movement + subtraction), not a multiply. (`tropical_cross_entropy`'s gradient; balanced
for exact AND shift LUT — verified by the regression test.)

**(2) Overconfidence.** Sign updates have no magnitude control: a sharp target drives $\|z\|\to\infty$,
$\tilde p\to$ one-hot. Bound it with an **entropy bonus** / **target temperature**:
$$\mathcal L_{\text{tot}}=\mathcal L-\beta H(\tilde p),\;\; H(\tilde p)=-\textstyle\sum_a\tilde p_a\log\tilde p_a;\qquad \pi^{(T)}_a=\frac{\pi_a^{1/T}}{\sum_b\pi_b^{1/T}},$$
pulling $\tilde p$ toward its max-entropy (uniform) point. Monitor the **mode fraction** $m=\mathbb E_x[\max_a\tilde p_a]$;
$m\to 1$ flags collapse hundreds of steps early.

## Why it is purely tropical (zero-DSP)

Readout $Wh$ = sign+shift+add ($W$ dyadic); $\mathrm{exp2}$ = LUT and softmax-normalization = bit-shift;
balanced gradient = subtraction + **scatter** of the scalar $s$; entropy/temperature reuse the same LUT/shift
($T=2^t$ a shift on the log-target); update $=\mathrm{sign}(\partial\mathcal L/\partial z)\oplus\mathrm{sign}(h)$ — a one-bit
sign-product — times a power-of-two $\eta$. **No hardware multiplier anywhere**, so the recipe that *learns*
a useful prior runs on the same zero-DSP fabric as inference.

## Empirical (Connect4, fully-tropical net — clean apples-to-apples run)

The mechanisms above are real and load-bearing in the **LM** setting (the Shakespeare bakeoff: shift-softmax
gradient-bias collapse, found and fixed by the balanced `tropical_cross_entropy`). In the **Connect4 self-play**
setting, a clean, thread-pinned, same-run trained-vs-untrained sweep (7 recipes × [100,200,400] sims × d2/d3,
n=40/cell; `CONNECT4_PRIOR_SWEEP.md`) gives a more measured result:

- **The multiply-free substrate is competitive** — all 7 recipes audit closure `[]` and score 0.6–0.9 vs
  negamax (83% of cells ≥0.6). This holds independently of the prior-learning question.
- **No dramatic collapse here.** Bare perceptron+SignSGD does **not** collapse in the clean run (trained @100 =
  0.975/0.588, the best cell), so the catastrophic 0.025 of the earlier partial run was a seed/run artifact, not
  a stable property. The gradient-bias collapse the balanced loss guards against is the one **observed in the LM
  setting**, where it is decisive; in Connect4 self-play it did not dominate.
- **A learned prior helps at low sims only narrowly:** of 7 recipes, one — soft-CE + entropy 0.1 — shows a real
  6/6-cell trained-over-untrained gain (mean Δ +0.188, largest at sim=100), and even that is partly inflated by a
  weak untrained baseline (true effect ~+0.15–0.19). soft-CE + entropy 0.3 (the partial run's "0.90 winner")
  actually **loses** to its own untrained baseline @100; soft-CE + target-temp 2.0's headline gain is a
  weak-baseline artifact. The trained-vs-untrained ordering is otherwise seed/noise-dominated.

**Takeaway for the manuscript:** the contribution is the *machinery* — a closure-faithful balanced gradient +
entropy/temperature levers + `mode_frac` detector that let you train a tropical prior on the **same zero-DSP
fabric as inference**, with the gradient-bias collapse demonstrated and cured in the LM setting. The Connect4
self-play prior-learning gain is real but recipe-specific and weak (best: soft-CE + entropy 0.1); resolving the
small effect sizes needs a paired same-seed baseline and hundreds of games/cell.
