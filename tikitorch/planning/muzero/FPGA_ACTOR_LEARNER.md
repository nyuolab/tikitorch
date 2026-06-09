# Tropical MuZero — FPGA-Actor / On-Fabric-Learner Architecture

*TikiTorch signature frontier model — code-grounded system spec*

---

## Intro: what this model is and why it ships with TikiTorch

TikiTorch is a tropical-semiring (max-plus / min-plus) deep-learning library: every op in the forward **and** backward pass is comparisons + additions — no multiplications — so the training loop is "algebraically closed" and maps to FPGA fabric with **zero DSP blocks**. That closure property is silicon-proven today for the tropical shift-LM kernel `forge_train_step` (see §11).

**Tropical MuZero** is the library's flagship application of that thesis to *planning*: a learned-model MuZero whose representation / dynamics / prediction cores are tropical, whose MCTS selection (`score = q + u`) and backup (`(max,+)`) are built in the comparison+addition algebra, and whose optimizer (`TropicalSGD` / `SignSGD` at `momentum=0`) is multiply-free. The design bet: a planner whose whole MCTS↔net↔train loop stays multiply-free and on-die wins on **batch-1 latency**, **per-watt**, and **no host↔device round-trip** — the exact regime an FPGA actor/learner serves and a GPU learner does not.

This document is written for an ML+FPGA engineer who must build the system. Every technical claim cites a real symbol (`file:line`, function/class/constant). Silicon numbers are split into **[measured]** (the LM kernel that has bound on AWS F2) and **[projected]** (the MuZero on-fabric binding, which does not yet exist in the kernel cache — see §11/§12). Nothing below invents an API.

> **Status, read first.** The zero-DSP fused training step is **[measured]** for the tropical shift-LM kernel `forge_train_step`, **not** for the MuZero net. `KernelCache.match()` (`tikitorch/fpga/kernels.py:202-217`) routes only `TropicalTransformer` and `MorphSSM`; a `TropicalMuZeroNet` / `DenseMuZeroNet` returns `None` → CPU fallback. The MuZero on-fabric learner is the **[projected]** extension of the same primitive, and §12 is its build spec.

---

## 1. Per-arm summary (the comparison is the point of the model)

| Dimension | **DENSE control arm** | **TROPICAL arm** |
|---|---|---|
| Deployment (locked) | CPU actors + GPU learner | FPGA actors + on-fabric learner |
| Mirrors | open-source SOTA actor/learner split (EfficientZero / LightZero / Ray) — "what production does today" | the TikiTorch zero-DSP thesis (`forge_train_step`) |
| Net | `DenseMuZeroNet` (`networks_dense.py:68`) | `TropicalMuZeroNet` (`networks.py:182`) |
| Cores | `nn.Linear + ReLU` MLP, `hidden_mult=3` param-match (`networks_dense.py:72`) | `ttnn.TropicalProjection` + `ttnn.TropicalLayerNorm` (`networks.py:118-119`) |
| Optimizer (default) | AdamW (the honest production baseline) | `TropicalSGD` (`agent.py:31`), compare+add at `momentum=0` |
| Multiply-free? | No (dense matmul + Adam) | Core yes; declared host boundaries: PUCT `sqrt`/`mul`/`div`, min-max `div`, `nn.Linear` heads |
| On silicon today | n/a (GPU) | **[projected]** — no MuZero `KernelSpec` registered yet; reuses `forge_train_step` closure |
| Defensible win | known-good, expressive | batch-1 latency, per-watt, no host↔device round-trip |
| Domain thesis | wins on **board games** (§2 expressivity gap) | wins on **planning/DP** (max-plus is the right prior) |
| `set_tau` | no-op (`networks_dense.py:83-84`) | soft→hard anneal (`networks.py:197-200`) |

These two arms run the **same** MuZero loop (`agent.py`); the comparison isolates *tropical* from *engineering* because the dense net is param-matched (`networks_dense.py:1-10`, `:72`).

---

## 2. Search — `tikitorch/planning/mcts.py`

`TropicalMCTS(model, MCTSConfig)` (`mcts.py:116`). `MCTSConfig` (`mcts.py:46`) defaults: `num_simulations=50`, `c_puct=1.25`, `gamma=1.0`, **`backup="max"`**, `topk_k=2`, `players=1`, `dirichlet_alpha=0.3`, `root_exploration_frac=0.0`, `seed=0`.

`run(root_state)` (`mcts.py:122`) returns `(visit_policy, root_value)` — the normalised root visit-count distribution over `A` actions and `root.q(cfg.backup, cfg.topk_k)`.

### 2.1 Selection — the load-bearing argmax (`_select_child`, `mcts.py:187`)

```
a* = argmax_a [ q + u ]                                          # score = q + u  (mcts.py:195)
q  = minmax.normalize(_edge_value(child))  if child.N>0 else 0   # mcts.py:193
u  = c_puct * child.prior * sqrtN / (1 + child.N)                # mcts.py:194
```

Children are iterated in `sorted(node.children.items())` for first-index tie-break ("kernel order", `mcts.py:192`). This is the same max-reduction the HLS kernel's **`SCORE_N`** loop is specified to compute (docstring `mcts.py:5-8`).

**Declared host/DSP boundaries** (small, per-node, off the argmax/backup datapath, docstring `mcts.py:22-26`):

* the PUCT `sqrt`: `sqrtN = math.sqrt(max(node.N, 1))` (`mcts.py:189`, "declared host sqrt");
* the PUCT product **and** divide: `c_puct * child.prior * sqrtN / (1 + child.N)` — two multiplies and a divide (`mcts.py:194`);
* the value-normalisation divide: `_MinMaxStats.normalize` returns `(v - lo) / (hi - lo)` (`mcts.py:70-73`) — a *separate* host divide.

These remain host/DSP boundary ops in the design; they are **not** eliminated on fabric.

### 2.2 Backup (`_backup`, `mcts.py:200`)

Composes `g <- node.reward + gamma * sign * g` up the reversed path (`mcts.py:207`), and updates the min-max stats from `node.reward + gamma * sign * node.q(...)` (`mcts.py:206`). `sign = _sign()` is `-1.0` for 2-player zero-sum (negamax) else `+1.0` (`mcts.py:178-181`). `_edge_value` applies the same `sign` to the child value (`mcts.py:184-185`). Three selectable operators (`_Node.q`, `mcts.py:104-113`):

* **`"max"`** — `value_max`, the pure `(max,+)` Bellman step (the tropical bet; optimistic/overestimation-biased on noisy learned leaves).
* **`"mean"`** — `value_sum / N`, MuZero's published visit-average (linear, robust).
* **`"topk"`** — mean of the `topk_k` largest backed-up returns, held in a `k`-element min-heap (`_topk`, recorded at `mcts.py:92`, `:98-102`; averaged at `:111-112`). `k=1` → `max`, large `k` → `mean`. Add/compare + a constant divide.

### 2.3 `_Node` (`mcts.py:76`)

`__slots__ = ("prior", "N", "value_sum", "value_max", "reward", "state", "children", "_topk")` (`mcts.py:77`). Tree statistics live in this Python tree — the CPU reference. The flat-tensor / URAM-resident layout is the **Stage-7 hardware mapping** (docstring `mcts.py:24-26`), a named design target, not an implemented/measured artifact.

---

## 3. Networks — `tikitorch/planning/muzero/networks.py`

`TropicalMuZeroNet(obs_dim, n_actions, latent_dim=64, tau=1.0, value_support=None, reward_support=None)` (`networks.py:182`) bundles three modules over a latent space. `ttnn = tikitorch.nn` (`networks.py:25`).

* **Representation** `TropicalRepresentation` (`networks.py:113`): `obs -> h0`. `ttnn.TropicalProjection` (`:118`) + `ttnn.TropicalLayerNorm` (`:119`), then `_minmax_unit` rescale to `[0,1]` (`return _minmax_unit(self.norm(self.enc(obs)))`, `:125`; `_minmax_unit` at `:36`, one divide). `ttnn.TropicalProjection` is an **alias** of `CompositionalTropicalProjection` (`tikitorch/nn/__init__.py:546`).
* **Dynamics** `TropicalDynamics` (`networks.py:131`): `(h, a) -> (h', r)`. The action enters as a **closed gather** — `nn.Embedding` added into the latent (`:133`), *not* a multiply — then a depth-2 tropical residual block, `_minmax_unit` (`:150`), and a reward head.
* **Prediction** `TropicalPrediction` (`networks.py:157`): `h -> (policy_logits, value_raw)`.

**Heads — declared boundaries (each wraps an `nn.Linear`):**

* The **policy head** is a bare `nn.Linear(latent_dim, n_actions)` — the declared boundary (`networks.py:160`).
* The **value/reward heads** are `ScalarHead` (`networks.py:58`; raw == scalar, MSE loss at `:71-72`) or `CategoricalHead` (`networks.py:75`; logits over an integer support, scalar = `E[support]` under softmax at `:91-92`, loss = cross-entropy to a two-hot target at `:105-106`). Each internally wraps an `nn.Linear` (`ScalarHead.head` `:63`, `CategoricalHead.head` `:86`). `_make_value_head` (`networks.py:109-110`) selects categorical (given a support tuple) vs scalar. Categorical heads are MuZero's standard stability lever over scalar MSE.

> The agent module docstring says value/reward loss is "MSE" (narrower than code); the actual loss is **head-dependent** — MSE for `ScalarHead`, categorical CE for `CategoricalHead`. Follow the code.

**`tau` soft→hard.** Convention from `CompositionalTropicalProjection.forward` (`tikitorch/nn/__init__.py:451-475`): **`tau > 0` = soft path** (`tropical_softmax`/`tropical_softmin`, dense gradients); **`tau = 0` = hard path** (1-of-d argmax via `tropical_dilation`). `set_tau` (`networks.py:197-200`, fanning out via `_set_tau`, `:43-51`) anneals soft→hard. `TropicalMuZeroNet` defaults `tau=1.0` (**soft**, `networks.py:182`). See the deployment protocol in §6.

**Inference interface for MCTS (both `@torch.no_grad`):**

* `initial_inference(obs)` (`networks.py:214-215`) → `Inference(state=h, reward=0.0, policy=softmax(p), value=...)`.
* `recurrent_inference(h, a)` (`networks.py:222-223`) → `(Inference(state=h2, reward, policy, value), done)` with `done` always `False` (latent-space planning to fixed depth).
* Training-side: `initial(obs) -> (h, p, v)` and `recurrent(h, a) -> (h2, r, p, v)` (batched).

`DenseMuZeroNet` (`networks_dense.py:68`) is the param-matched dense control with the identical interface; `hidden_mult=3` (`:72`) widens the MLP cores to param-match the tropical projection; `set_tau` is a no-op (`:83-84`).

---

## 4. Agent loop — `tikitorch/planning/muzero/agent.py`

`TropicalMuZero(env, MuZeroConfig)` (`agent.py:64`). `MuZeroConfig` (`agent.py:28`) wires: `net_kind="tropical"` → `TropicalMuZeroNet` or `"dense"` → `DenseMuZeroNet` (`:69`); **`optimizer="tropicalsgd"`** default (`:31`); **`backup="topk"`, `topk_k=2`** default (`:42-43`); `value_loss_weight=0.25` (`:49`); `tau_start=None`, `tau_end=None` (`:34-35`).

### 4.1 `train_step()` (`agent.py:79`)

K-step unroll over `K = unroll` (default 3). Heads pulled once (`agent.py:84`):

```python
loss  = w * vh.loss(v, vt[:, 0]) + _ce(p, pt[:, 0])               # initial  (agent.py:86)
for k in range(unroll):                                           # recurrent (agent.py:87-90)
    h, r, p, v = net.recurrent(h, acts[:, k])
    loss += w * vh.loss(v, vt[:, k+1]) + rh.loss(r, rt[:, k]) + _ce(p, pt[:, k+1])
loss.backward(); self.opt.step()                                  # agent.py:92-93
```

* `w = value_loss_weight` (default 0.25, `agent.py:49`, `:83`) multiplies the **value term only**; reward (`rh.loss`) and policy (`_ce`) are **unweighted**.
* `vh.loss`/`rh.loss` are MSE (`ScalarHead`) or categorical CE (`CategoricalHead`) per head type (§3).
* `_ce` (`agent.py:199-201`) is cross-entropy to the soft MCTS visit-distribution target.

### 4.2 Optimizer (`_make_optimizer`, `agent.py:188`)

`"adamw"` → `torch.optim.AdamW` (the open baseline); `"signsgd"` → `SignSGD`; otherwise (incl. the default `"tropicalsgd"`) → `TropicalSGD`. Instantiated with `(params, lr)` only (`agent.py:75`), so `momentum` defaults to 0.

> **Closure caveat (hole H7).** `SignSGD`/`TropicalSGD` are zero-DSP **only on the `momentum=0` base path** (`W -= lr*sign(grad)`, `optim/__init__.py:99`, `:150`) — which is how the agent wires them. Their optional `momentum` path uses a float EMA `buf.mul_(momentum)` (`optim/__init__.py:96`, `:147`) — a data-dependent multiply, **not** zero-DSP, explicitly flagged in both docstrings. For strictly multiplier-free momentum use `PureTropicalSGD` / `TropicalLion` (§9).

### 4.3 `run()` and `evaluate()`

`run()` (`agent.py:165`) alternates `self_play -> train_step`; anneals `tau` only when **both** `tau_start` and `tau_end` are set (`agent.py:170-171`) and anneals the visit-sampling temperature; calls `evaluate()` on `eval_every`. `self_play` (`agent.py:107`) runs `play_game` per game (`:111`) and saves the trajectory.

`evaluate()` (`agent.py:127`) scores the greedy policy against an exact **value-iteration oracle** built with `tikitorch.planning.TropicalValueIteration` (`agent.py:122-123`), reporting `agree` (optimal-action agreement over non-terminal states, `:154-156`) and `solve_frac` (`:158-159`).

---

## 5. Replay / targets — `tikitorch/planning/muzero/replay.py`

`Game` (`replay.py:17`) holds parallel lists `obs / actions / rewards / policies / root_values` (`rewards[t]` follows `actions[t]`). `ReplayBuffer.sample_batch(batch, unroll, td, gamma)` (`replay.py:62`) returns, with `B = batch`, `K = unroll`, `A = n_actions`:

| tensor | shape | dtype | source |
|---|---|---|---|
| `obs0` | `(B, obs_dim)` | float | `g.obs[pos]` (`:78`) |
| `acts` | `(B, K)` | long | `g.actions` (`:86`) |
| `value_targets` | `(B, K+1)` | float | n-step bootstrap off stored MCTS root values (`_value_target`, `:48-60`) |
| `reward_targets` | `(B, K)` | float | `g.rewards` (`:83`) |
| `policy_targets` | `(B, K+1, A)` | float | MCTS visit dists, padded uniform past end (`:85`) |

Positions past episode end are padded with an absorbing target (value 0, reward 0, uniform policy, `replay.py:79-85`).

---

## 6. `tau` deployment protocol (soft-train → hard-deploy) — **build-critical**

The on-fabric `SCORE_N` actor and the silicon fused-step datapath assume the **hard** (`tau=0`) 1-of-d argmax (`nn/__init__.py` `forward` else-branch / `tropical_dilation`). The training default is **soft** (`tau=1.0`, `networks.py:182`), and `run()` anneals to hard **only if** `tau_start`/`tau_end` are both set (`agent.py:170-171`) — which default to `None` (`agent.py:34-35`). **Following the default config never reaches `tau=0`.** A soft-trained checkpoint on a hard datapath is a policy mismatch between the CPU reference and the deployed actor.

**Protocol to deploy the tropical arm:**

1. The on-fabric forward is the `tau=0` hard path — set it explicitly. Either set `MuZeroConfig.tau_start`/`tau_end` so `run()` anneals to 0 (`agent.py:170-171`), or call `net.set_tau(0.0)` (`networks.py:197-200`) before binding.
2. **Revalidate** `evaluate()` `agree`/`solve_frac` (`agent.py:154-159`) at `tau=0` — soft→hard changes the policy, so the soft-trained metrics do not transfer.
3. Inference is `@torch.no_grad` (`networks.py:214`, `:222`); no autograd graph is built on the actor path.

---

## 7. Two-player / board-game arm — `tikitorch/planning/muzero/board_agent.py`

The board domain (where dense is expected to win) has a **complete second agent** the engineer must build/deploy — do **not** re-derive it from `play_board_game`/negamax. `board_agent.py` defines:

* **`BoardConfig`** (`board_agent.py:54`): default **`backup="mean"`** (the verified adversarial-correct backup; max/topk are over-optimistic, `:67`), `root_exploration_frac=0.25` self-play / `0.0` at eval (`:188`), `eval_depth=2` negamax opponent (`:85`), `players=2`.
* **`BoardReplay`** (`board_agent.py:89`): a **different** buffer from `ReplayBuffer` — positions labelled with the game **OUTCOME** (`value -> outcome`, `reward -> 0`, `policy -> MCTS visits`, docstring `:4-6`).
* **`BoardMuZero`** (`board_agent.py:134`) with its own `self_play()` (`:149`, parallel workers via `_init_worker`+`mp.Pool`, `:39`/`:165`), `train_step()` (`:170`), `evaluate()` (`:193`), `run()` (`:209`).

Self-play uses `play_board_game` (`selfplay.py:55`), hardcoded `backup="mean"`, two-player negamax (`players=2`), value target = final outcome from each position's player-to-move perspective (`+1/0/-1`, `selfplay.py:80-81`) — a **different target** than the n-step bootstrap in `replay.py`. Skill is validated **not** against `TropicalValueIteration` but against a fixed depth-limited **negamax** opponent via `negamax_agent` + `play_match` (`board_agent.py:196-200`, from `games.py`).

**On-fabric implication.** The negamax sign-flip (`_sign`, `mcts.py:178-181`) is a host-side negate off the `SCORE_N` argmax datapath; board obs is perspective-invariant (`games.py` board obs). The buffer packing differs (outcome targets, §12.4).

---

## 8. Self-play — `tikitorch/planning/muzero/selfplay.py`

`play_game` (`selfplay.py:24`, single-agent) runs MCTS self-play and temperature-samples the visit distribution (`_sample_action`, `:16-21`; `argmax` when `temperature <= 1e-3`). `play_board_game` (`selfplay.py:55`, two-player negamax) is covered in §7.

---

## 9. Optimizers — `tikitorch/optim/__init__.py`

`__all__` (`optim/__init__.py:36-37`): `SignSGD, TropicalSGD, PureTropicalSGD, TropicalLion, TropicalAdam, FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD`.

* **`SignSGD`** (`:53`): `W -= lr * sign(grad)` (`:99`). Base path sign-only; momentum path is a float EMA — **not** zero-DSP (hole H7, docstring `:61-65`, code `:96`).
* **`TropicalSGD`** (`:104`): the semiring view — sign is a comparison, update an addition (`:150`). Same `momentum=0` closure and same H7 caveat (`:112-116`, `:147`).
* **`PureTropicalSGD`** (`:155`): strictly multiplier-free with `adaptive=False` — integer-counter momentum, power-of-2 lr `2^{-lr_shift}` (bit-shift on FPGA), LFSR noise. The `adaptive=True` path uses `median`/divide/`log2` and is **EXPERIMENTAL, NOT closed** (docstring `:168-171`).
* `TropicalLion`, `TropicalAdam`, `FrexpMagnitudeSGD`, `CountAgreementSGD`, `ExponentEMAAdam`, `PerLayerStepSGD` recover per-parameter scaling using sign/compare/power-of-two shift only.

---

## 10. FPGA kernels — `tikitorch/fpga/kernels.py`

`KernelCache` / `KernelSpec`: O(1) dict from `(model_type, config)` to a precompiled fused training-step kernel — **no compilation in the hot path**.

**Mode codes** (`kernels.py:38-40`): `MODE_FWD_LOSS = 0`, `MODE_BWD = 1`, `MODE_SIGNSGD = 2`.

**Two C ABIs** (`KernelABI`, `kernels.py:112-128`):

* `KernelABI.FUSED_5 = KernelABI("fused5", 4, False)` — `(buffer, mode, arg1, arg2, arg3)` (`:127`). Used by `forge_train_step`, `forge_train_step_morph`, `forge_train_step_morph_ecg`.
* `KernelABI.FUSED_6 = KernelABI("fused6", 5, True)` — adds trailing `layer_id` (`:128`). Used by `forge_train_step_2layer` / `forge_train_step_3layer` (the exact extern-C entry symbols).

**`KernelSpec`** (`kernels.py:136`) carries `entry`, `abi`, `source_file`, `buffer_words_const` (the `.hpp` constant for the flat int32 buffer size), kernel-fixed secondary dims (`d_ff/l_max` for LM; `k/l_seq/dil_cap` for morph), `dtype="int16"` / `grad_dtype="int32"`, `dsp = 0` (the closure invariant), and optional silicon operating points `throughput_tok_s` / `energy_mj_tok` / `wns_ns`.

**Registered kernels** (`KernelCache.default()`, `kernels.py:223-267`) — the conservative allow-list:

| entry | model_type | config | ABI | operating points |
|---|---|---|---|---|
| `forge_train_step` | `TropicalTransformer` | `LMKernelConfig(V=128,D=128,n_layers=1)` | FUSED_5 | 1200 tok/s |
| `forge_train_step_2layer` | `TropicalTransformer` | `(V=128,D=128,n_layers=2)` | FUSED_6 | 770 tok/s, 9.0 mJ/tok, WNS −0.197 ns |
| `forge_train_step_3layer` | `TropicalTransformer` | `(V=128,D=128,n_layers=3)` | FUSED_6 | — |
| `forge_train_step_morph` | `MorphSSM` | `MorphKernelConfig(D=64,n_layers=4)` | FUSED_5 | — |
| `forge_train_step_morph_ecg` | `MorphSSM` | `(D=128,n_layers=8)` | FUSED_5 | 1207 tok/s, 5.8 mJ/tok, WNS +0.009 ns |

These operating points are the **LM/morph kernel** metrics transcribed from `hardware/README.md` (docstring `kernels.py:145-146`) — they are **not** MuZero numbers, and not every spec carries them.

**`match(model)`** (`kernels.py:202-217`) routes by class name: `TropicalTransformer` → `LMKernelConfig.from_model` (reads `task_type=="lm"`, an `nn.Embedding embed`, `blocks`, `d_model`; `:47-80`), `MorphSSM` → `MorphKernelConfig.from_model` (reads `layers`, `d_model`; `:93-101`). Any other class returns `None`; ambiguous shapes return `None`. **Nothing binds on-board unless an exact `(model_type, config)` key is registered.**

> **Critical for the MuZero arm. [projected]** Neither `TropicalMuZeroNet` nor `DenseMuZeroNet` is a recognized `model_type`, and `TropicalMuZeroNet` exposes none of `task_type`/`embed`/`blocks`/`d_model`. **`forge_train_step` does not bind the MuZero learner today — `match()` returns `None` and the loop runs on CPU.** Putting the learner on fabric is the build task in §12.

### 10.1 CPU-fallback contract — `accelerate.py` / `backend.py`

`match() -> None` only produces `spec=None`; the fallback behavior lives in the wrapper:

* `FpgaBackend.accelerate(model, ...)` (`backend.py:53-75`) computes `spec = self._cache.match(model)`, sets `native` only when `is_available()` (board present, `backend.py:35-43`), and builds an `AcceleratedModel`. `is_available()` is `False` in this repo/CI (`backend.py:6-8`); the per-op registry `ops()` is intentionally empty (`backend.py:48`).
* **`bound`** is `True` *iff* a native handle **and** a matched `kernel_spec` **and** a registered on-board runner all exist (`accelerate.py:55-67`).
* **Fallback `forward`** (`accelerate.py:81`) runs the inner module unchanged. **Fallback `train_step`** (`accelerate.py:108-130`) runs the inner model's own `compute_loss(x, targets).backward()` + `optimizer.step()/zero_grad()` — **byte-identical to training the unwrapped model** (docstring `:113-116`) — and warns once via `_warn_fallback_once` (`accelerate.py:166`).
* **`strict=True`** (`accelerate.py:47-50`) raises `NativeUnavailable` with one of three reasons (`_unbound_reason`, `accelerate.py:71-77`): no board/native driver; **no kernel of this shape** (when `kernel_spec is None`, `:73-75`); or no on-board runner registered.

**Builder takeaway:** `TropicalMuZeroNet`/`DenseMuZeroNet` wrapped in `.fpga()` today run the identical CPU loop with one warning. Pass `strict=True` in CI to assert no silent fallback (it will currently raise "no kernel of this shape" for MuZero until §12 lands).

---

## 11. On-silicon evidence — scope it honestly

**[measured] (AWS F2)** — the *tropical shift-LM* fused training-step kernel, `results/fpga_silicon/tikitorch_f2.log`:

```
BOUND on silicon: True | kernel: forge_train_step | dsp: 0
params evolved by on-silicon training: 229872 / 231040
ON-SILICON MULTIPLIER-FREE TRAINING ACTIVE: True
```

Provenance: this is a **char-LM** run loading the monolithic `forge_train_step` AGFI `agfi-0a845113b088cf62e` (`cl_forge_train_step_v7_monolithic`, `scripts/run_tikitorch_f2_silicon.sh:2,20`), binding `model.fpga()` and running 50 on-board `train_step()`s. The bound spec is `model_type="TropicalTransformer"`, `LMKernelConfig(V=128,D=128,n_layers=1)`, FUSED_5 (`kernels.py:234-239`).

This is the **existence proof** for the fused-step primitive: a tropical fwd/bwd/SignSGD step binds and trains on F2 fabric with **`dsp: 0`**, multiplier-free. It is the closure invariant the MuZero kernel will inherit.

**[projected] (NOT measured for this model)** — the native Tropical-MuZero MCTS↔net↔train loop on fabric: the `SCORE_N` selection argmax (`mcts.py:5-8`) and the Stage-7 URAM flat-tensor tree layout (`mcts.py:24-26`). The CPU reference (`TropicalMCTS` Python tree, `TropicalMuZeroNet`) is the validated vertical slice; the on-fabric MuZero actor/learner is the design bet, leaning on the F2-demonstrated closure of `forge_train_step` rather than a measured MuZero-on-silicon run. **Do not cite the LM silicon numbers as MuZero-on-silicon results.**

---

## 12. Build spec — the projected MuZero `KernelSpec` and integration

This is THE build task: register a new `(model_type, config)` schema + `KernelSpec` + on-board runner so `match()` binds the MuZero learner. Until it lands, the loop is CPU-only (additive — see §12.2).

### 12.1 `MuZeroKernelConfig` (the match key)

`TropicalMuZeroNet` exposes `n_actions`, `latent_dim`, the `repr/dyn/pred` submodules, and `value_support`/`reward_support` presence — **not** `task_type`/`embed`/`blocks`/`d_model` (LM) nor `layers`/`d_model` (morph), so neither existing `from_model` (`kernels.py:57-80`, `:93-101`) matches. Define a `MuZeroKernelConfig` with fields derivable from the net — `n_actions`, `latent_dim`, `unroll` (= K), and value/reward head kind — plus a `from_model` that **returns `None` on any ambiguity** (mirroring the LM path's safe direction). Register the new class name in `match()` alongside `TropicalTransformer`/`MorphSSM` (`kernels.py:208-214`).

### 12.2 ABI choice and the multi-stage unroll

The MuZero learner is multi-stage (`repr` once, then `K` × `dyn`+`pred`). The natural fit is a `layer_id`-style **FUSED_6** (`kernels.py:128`) or a stage-index variant, driving the fused mode sequence `MODE_FWD_LOSS -> MODE_BWD -> MODE_SIGNSGD` (`kernels.py:38-40`) per unroll step. `dsp = 0` stays the invariant on the new spec.

### 12.3 Declared host boundaries that stay off-fabric

The new kernel covers the tropical cores (`TropicalProjection`/`TropicalLayerNorm`) only. These ops remain host/DSP boundaries: the policy/value/reward `nn.Linear` heads (`networks.py:160`, `:63`, `:86`), the action `nn.Embedding` gather (`networks.py:133`), the per-sample `_minmax_unit` divide (`networks.py:36`), and the PUCT `sqrt`/`mul`/`div` + min-max divide on the search side (`mcts.py:189`, `:194`, `:70-73`).

### 12.4 Buffer packing — two replay schemas

The fused kernel consumes a flat int32 buffer of fixed word count (`buffer_words_const`, `kernels.py:154`; `dtype="int16"` weights / `grad_dtype="int32"`, `:161-162`). The new `KernelSpec` must define the `(B,K,...)` → packed-int32 mapping for **both** target schemas:

* **single-agent** — `ReplayBuffer` n-step bootstrap value targets (`replay.py:48-60`): `obs0 (B,obs_dim)`, `acts (B,K)`, `value_targets (B,K+1)`, `reward_targets (B,K)`, `policy_targets (B,K+1,A)`;
* **board** — `BoardReplay` outcome value targets `+1/0/-1`, `reward -> 0` (`board_agent.py:89`, `selfplay.py:80-81`).

Absorbing-state padding (value 0 / reward 0 / uniform policy, `replay.py:79-85`) must be reproduced in the on-fabric packing. Version the match key with a kernel-revision so a checkpoint can only bind to the matching compiled bitstream.

### 12.5 How it grafts onto `run()` (evolution, not a rewrite)

Three insertion points, only the third missing today:

1. **Net construction** — wrap once: `net = backend.accelerate(net)` at the `net_cls(...)` site (`agent.py:69`). Idempotent (`backend.py:69-71`).
2. **Actor path** — `self_play()` → `play_game` (`agent.py:111`) drives `initial_inference`/`recurrent_inference` on the bound net (already works on CPU; binds when §12.1-12.4 land).
3. **Learner path** — `train_step()`'s `loss.backward(); self.opt.step()` (`agent.py:92-93`) routes through the fused `FWD_LOSS -> BWD -> SIGNSGD` sequence once the MuZero `KernelSpec` is registered. **This is the only missing piece; everything else is additive.**

### 12.6 Error contract

Unknown shape → `match()` returns `None`. In production pass `strict=True` so the wrapper raises `NativeUnavailable` ("no kernel of this shape", `accelerate.py:73-75`) rather than silently degrading; in dev the warn-once CPU fallback (`accelerate.py:166`) runs the byte-identical CPU loop.

---

## 13. Validation gate

The flagship claim is "on-fabric training is multiplier-free **and** correct." Correctness needs:

1. **Single-agent oracle** — `TropicalValueIteration` `agree`/`solve_frac` (`agent.py:122`, `:154-159`).
2. **Board oracle** — depth-limited negamax / minimax via `play_match` (`board_agent.py:196-200`, `games.py`).
3. **CPU-reference-vs-bound equivalence** — same seed/checkpoint **at `tau=0`** (§6), assert `agree`/`solve_frac` and the root `visit_policy` match within `eps`. Gate with `strict=True` (§10.1, §12.6) so unbound CI does not pass silently.
4. **Silicon acceptance** — mirror the LM gate (`results/fpga_silicon/tikitorch_f2.log`): `BOUND==True`, `dsp==0`, and a params-evolved count, once the MuZero `KernelSpec` exists.

---

## 14. Determinism contract — a correctness hazard for parallel actors

**FIXED.** `_add_root_noise` (`mcts.py:170`) now draws the root Dirichlet noise from a **local** `numpy.random.default_rng(self.cfg.seed)` — it does NOT touch the process-global torch RNG. (It previously called `torch.manual_seed(self.cfg.seed)`, a global reseed that both leaked into other code and re-seeded to the *same* fixed value every call, so the noise was identical across moves until `cfg.seed` mutated. `torch.distributions.Dirichlet.sample()` takes no `generator=`, hence numpy.) `play_game` mutates `cfg.seed = seed*100003 + t` per move (`selfplay.py:39`; `play_board_game` likewise `selfplay.py:73`), so noise is per-move distinct; action sampling already used a **local** `Generator` (`_sample_action`, `selfplay.py:21`/`:29`) and the agent a local `self.gen` (`agent.py:79`).

**Contract for parallel actors.** Because the RNG is now local, on an FPGA actor/learner split (and `BoardMuZero`'s `mp.Pool` workers, `board_agent.py`) self-play is reproducible across parallelism — give each parallel actor a **disjoint** seed stream (the per-move `cfg.seed = seed*100003 + t` mutation is per-move, not per-worker) so on-fabric runs match the CPU reference bit-for-bit.

---

## 15. When to pick the tropical arm (domain decision rule)

The §1 thesis ("tropical wins on planning/DP, dense wins on board games") is a design bet; the repo supplies the matched control (`DenseMuZeroNet`, `hidden_mult=3`, `networks_dense.py:72`) and the metrics (`agent.py:162`; board win/draw/loss via `play_match`) but **no measured tropical-vs-dense delta is recorded here**, and the previously-cited `results/backup_sweep/FINDINGS.md` **does not exist** in the repo (extant artifacts: `results/bakeoff_closed_optimizers.{jsonl,md}`, `results/confirm_full250.{jsonl,md}`, `results/confirm_short50.{jsonl,md}`, `results/closed_benchmarks_remote/`, `results/fpga_silicon/`). Treat `backup` as a swept hyperparameter whose defaults differ by entry point: `MCTSConfig=max` (`mcts.py:51`), `MuZeroConfig/play_game=topk,k=2` (`agent.py:42-43`, `selfplay.py:25`), `play_board_game/BoardMuZero=mean` (`selfplay.py:55`, `board_agent.py:68`) — `mean` is the published-MuZero / adversarial-correct safe option, `topk` the wired tropical-leaning default.

**Decision rule for a new domain:**

* **A-priori checklist** — deterministic perfect-information with additive/bottleneck-shaped returns (shortest-path / min-cost = min-plus; reachability = max-plus) favors tropical; stochastic/averaging returns favor dense.
* **A-posteriori gate** — run both `net_kind="tropical"` and `"dense"` (param-matched, `networks_dense.py:72`) on the target env; compare `evaluate()` `agree`/`solve_frac` (`agent.py:162`) for single-agent or `play_match` win-rate (`board_agent.py`) for two-player. Deploy the tropical arm only if it matches dense within a stated tolerance **and** the per-watt / batch-1 latency win justifies it.

State the measured deltas explicitly when they exist; until then, mark the domain split as **not yet measured in this repo**, not as a result.
