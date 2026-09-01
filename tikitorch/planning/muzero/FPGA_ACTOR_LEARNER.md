# Tropical MuZero — FPGA-Actor / On-Fabric-Learner Architecture

*Code-grounded specification for a projected system*

---

## Intro: what this model is and why it ships with TikiTorch

TikiTorch supplies hard tropical operators and routed reverse rules that can be
combined into a declared compare/add/shift recipe. Closure is recipe- and
path-specific: soft routing, dense heads, ordinary losses, host arithmetic, and
optimizer settings can leave that subset. A software closure audit does not
determine FPGA resource use; only a configuration-matched build report can do
that. The archived `forge_train_step` shift-LM run in §11 is evidence for one
exact compiled configuration, not for every TikiTorch model.

**Tropical MuZero** is an experimental application of that idea to *planning*:
its representation, dynamics, and prediction cores use tropical modules, and
selected search and update steps are closure-oriented. The implemented
end-to-end loop is **not** multiplier-free: PUCT, normalization, dense heads,
losses, and the current CPU fallback are explicit boundaries. Putting a
complete actor/learner on fabric and measuring batch-1 latency, full-system
energy, and transfers is the design hypothesis documented here, not a present
result.

This document is written for an ML+FPGA engineer who must build the system.
Symbol names point to the implementation; line numbers are navigation hints and
can move as the source evolves. Silicon numbers are split into **[measured]**
(the LM kernel that bound on AWS F2) and **[projected]** (the MuZero on-fabric
binding, which does not yet exist in the default kernel cache—see §11/§12).

> **Status, read first.** Bound F2 execution and a registered `dsp=0` operating
> point are **[measured/recorded]** for the exact tropical shift-LM
> `forge_train_step` configuration, **not** for MuZero; §11 explains the
> evidence boundary. `KernelCache.match()`
> (`tikitorch/fpga/kernels.py:238-255`) can derive a
> `MuZeroKernelConfig` for `TropicalMuZeroNet`, but
> `KernelCache.default()` (`:262-304`) registers no MuZero `KernelSpec`, so the
> lookup still returns `None`; `DenseMuZeroNet` is not routed. Both therefore
> use CPU fallback. The MuZero on-fabric learner is **[projected]** and must
> earn its own software audit, synthesis report, routed timing report, and
> known-answer test.

---

## 1. Per-arm summary (the comparison is the point of the model)

| Dimension | **DENSE control arm** | **TROPICAL arm** |
|---|---|---|
| Deployment role | CPU actors + GPU learner reference | projected FPGA actors + on-fabric learner |
| Motivating pattern | representative open-source actor/learner split (EfficientZero / LightZero / Ray) | configuration-specific shift-LM evidence (`forge_train_step`) |
| Net | `DenseMuZeroNet` (`networks_dense.py:68`) | `TropicalMuZeroNet` (`networks.py:192`) |
| Cores | `nn.Linear + ReLU` MLP, `hidden_mult=3` param-match (`networks_dense.py:72`) | configurable max-plus or dyadic projections plus `TropicalLayerNorm` (`networks.py:53-60`, `:131-189`) |
| Optimizer (default) | AdamW reference | `TropicalSGD` (`agent.py:32`), with a closure-oriented `momentum=0` path subject to recipe audit |
| Multiply-free? | No (dense matmul + Adam) | Not end-to-end as implemented; a selected hard core/update recipe may be closure-oriented, while PUCT, min-max scaling, losses, and default dense heads remain boundaries |
| On silicon today | n/a (GPU reference) | **[projected]** — no MuZero `KernelSpec`; an LM build does not transfer its resource result |
| Evaluation target | quality and throughput reference | measure batch-1 latency, full-system energy, and host/device transfers |
| Domain hypothesis | evaluate on board games | evaluate the max-plus prior on planning/DP tasks |
| `set_tau` | no-op (`networks_dense.py:83-84`) | soft->hard control (`networks.py:227-230`) |

These two arms run the **same** MuZero loop (`agent.py`); the comparison isolates *tropical* from *engineering* because the dense net is param-matched (`networks_dense.py:1-10`, `:72`).

---

## 2. Search — `tikitorch/planning/mcts.py`

`TropicalMCTS(model, MCTSConfig)` (`mcts.py:116`). `MCTSConfig` (`mcts.py:46`) defaults: `num_simulations=50`, `c_puct=1.25`, `gamma=1.0`, **`backup="max"`**, `topk_k=2`, `players=1`, `dirichlet_alpha=0.3`, `root_exploration_frac=0.0`, `seed=0`.

`run(root_state)` (`mcts.py:122`) returns `(visit_policy, root_value)` — the normalised root visit-count distribution over `A` actions and `root.q(cfg.backup, cfg.topk_k)`.

### 2.1 Selection — the load-bearing argmax (`_select_child`, `mcts.py:192`)

```
a* = argmax_a [ q + u ]                                          # score = q + u  (mcts.py:195)
q  = minmax.normalize(_edge_value(child))  if child.N>0 else 0   # mcts.py:198
u  = c_puct * child.prior * sqrtN / (1 + child.N)                # mcts.py:199
```

Children are iterated in `sorted(node.children.items())` for first-index
tie-break (`mcts.py:197`). This is the max-reduction that the projected HLS
`SCORE_N` loop is specified to compute (docstring `mcts.py:5-8`).

**Declared ordinary-arithmetic host boundaries** (small, per-node, off the
argmax/backup datapath, docstring `mcts.py:22-26`):

* the PUCT `sqrt`: `sqrtN = math.sqrt(max(node.N, 1))` (`mcts.py:194`);
* the PUCT product **and** divide: `c_puct * child.prior * sqrtN / (1 + child.N)`—two multiplies and a divide (`mcts.py:199`);
* the value-normalisation divide: `_MinMaxStats.normalize` returns `(v - lo) / (hi - lo)` (`mcts.py:70-73`) — a *separate* host divide.

These remain host-side ordinary-arithmetic boundaries in the current design.
Whether a future compiled implementation maps any of them to DSP blocks is a
build-report question, not a property of the Python operation.

### 2.2 Backup (`_backup`, `mcts.py:205`)

Composes `g <- node.reward + gamma * sign * g` up the reversed path
(`mcts.py:212`) and updates min-max statistics (`:211`). `sign = _sign()` is
`-1.0` for two-player zero-sum (negamax) and `+1.0` otherwise (`:183-186`).
`_edge_value` applies the same sign to the child value (`:188-190`). Three
selectable operators live in `_Node.q` (`:104-113`):

* **`"max"`** — `value_max`, the pure `(max,+)` Bellman step (the tropical bet; optimistic/overestimation-biased on noisy learned leaves).
* **`"mean"`** — `value_sum / N`, MuZero's published visit-average (linear, robust).
* **`"topk"`** — mean of the `topk_k` largest backed-up returns, held in a `k`-element min-heap (`_topk`, recorded at `mcts.py:92`, `:98-102`; averaged at `:111-112`). `k=1` → `max`, large `k` → `mean`. Add/compare + a constant divide.

### 2.3 `_Node` (`mcts.py:76`)

`__slots__ = ("prior", "N", "value_sum", "value_max", "reward", "state", "children", "_topk")` (`mcts.py:77`). Tree statistics live in this Python tree — the CPU reference. The flat-tensor / URAM-resident layout is the **Stage-7 hardware mapping** (docstring `mcts.py:24-26`), a named design target, not an implemented/measured artifact.

---

## 3. Networks — `tikitorch/planning/muzero/networks.py`

`TropicalMuZeroNet` (`networks.py:192-218`) bundles three modules over a latent
space. Its constructor makes the configuration choices explicit:
`core_kind="maxplus"`, `dyn_kind="maxplus"`, and `dyadic_heads=False` by
default.

* **Representation** `TropicalRepresentation` (`networks.py:131-143`):
  `obs -> h0` through `_make_proj` plus `TropicalLayerNorm`, followed by
  `_minmax_unit` rescaling. `_minmax_unit` (`:36-40`) contains an ordinary
  divide. `_make_proj` (`:53-60`) selects a max-plus projection or the dyadic
  exported-schedule-oriented alternative.
* **Dynamics** `TropicalDynamics` (`networks.py:146-170`): `(h, a) -> (h', r)`.
  The action enters through an indexed `nn.Embedding` gather (`:152`, `:166`),
  then a depth-2 projection/residual stack, `_minmax_unit`, and a reward head.
  The gather can participate in a closure-oriented core recipe; it does not
  make the complete network closed.
* **Prediction** `TropicalPrediction` (`networks.py:173-189`):
  `h -> (policy_logits, value_raw)`.

**Heads are configuration-specific.** `_lin` (`networks.py:68-71`) selects
`nn.Linear` when `dyadic_heads=False` and `DyadicLinear` when it is `True`.
The current `MuZeroConfig` construction does not pass `dyadic_heads`
(`agent.py:70-76`), so the agent's policy, value, and reward heads use the
default dense boundary. Direct construction with `dyadic_heads=True` gives
those heads a verifiable shift-add forward export schedule, but it does not by
itself close their weight gradients, head losses, search arithmetic, or a
physical build.

The value/reward head loss is also configuration-specific: `ScalarHead`
(`networks.py:74-89`) uses L1 loss, while `CategoricalHead` (`:92-123`) uses a
two-hot categorical cross-entropy and softmax expectation. Those loss and
scalar-conversion paths must be included in any full-recipe audit.

**`tau` soft->hard.** In `CompositionalTropicalProjection.forward`
(`tikitorch/nn/__init__.py:567-589`), **`tau > 0` selects the soft path** and
**`tau = 0` the hard max/min path**. `TropicalMuZeroNet.set_tau`
(`networks.py:227-230`) fans this setting through the modules; the constructor
defaults to `tau=1.0` (`:203-205`). Dyadic projections accept `tau` for
interface parity but do not use it (`tikitorch/nn/__init__.py:719-728`). See
the deployment protocol in §6.

**Inference interface for MCTS (both `@torch.no_grad`):**

* `initial_inference(obs)` (`networks.py:244-250`) -> `Inference(state=h, reward=0.0, policy=softmax(p), value=...)`.
* `recurrent_inference(h, a)` (`networks.py:252-257`) -> `(Inference(state=h2, reward, policy, value), done)` with `done` always `False` (latent-space planning to fixed depth).
* Training-side: `initial(obs) -> (h, p, v)` and `recurrent(h, a) -> (h2, r, p, v)` (batched).

`DenseMuZeroNet` (`networks_dense.py:68`) is the param-matched dense control with the identical interface; `hidden_mult=3` (`:72`) widens the MLP cores to param-match the tropical projection; `set_tau` is a no-op (`:83-84`).

---

## 4. Agent loop — `tikitorch/planning/muzero/agent.py`

`TropicalMuZero(env, MuZeroConfig)` (`agent.py:65`). `MuZeroConfig`
(`agent.py:28-62`) wires `net_kind="tropical"` -> `TropicalMuZeroNet` or
`"dense"` -> `DenseMuZeroNet` (`:70`); **`optimizer="tropicalsgd"`** by
default (`:32`); **`backup="topk"`, `topk_k=2`** (`:43-44`);
`value_loss_weight=0.25` (`:50`); and `tau_start=None`, `tau_end=None`
(`:35-36`).

### 4.1 `train_step()` (`agent.py:82`)

K-step unroll over `K = unroll` (default 3). Heads are pulled once
(`agent.py:87`):

```python
loss  = w * vh.loss(v, vt[:, 0]) + _ce(p, pt[:, 0])               # initial  (agent.py:89)
for k in range(unroll):                                           # recurrent (agent.py:90-93)
    h, r, p, v = net.recurrent(h, acts[:, k])
    loss += w * vh.loss(v, vt[:, k+1]) + rh.loss(r, rt[:, k]) + _ce(p, pt[:, k+1])
loss.backward(); self.opt.step()                                  # agent.py:95-96
```

* `w = value_loss_weight` (default 0.25, `agent.py:50`, `:86`) multiplies the **value term only**; reward (`rh.loss`) and policy (`_ce`) are **unweighted**.
* `vh.loss`/`rh.loss` are L1 (`ScalarHead`) or categorical CE (`CategoricalHead`) per head type (§3).
* `_ce` (`agent.py:202-204`) is cross-entropy to the soft MCTS visit-distribution target.

### 4.2 Optimizer (`_make_optimizer`, `agent.py:191`)

`"adamw"` -> `torch.optim.AdamW` (the open baseline); `"signsgd"` ->
`SignSGD`; otherwise (including the default `"tropicalsgd"`) ->
`TropicalSGD`. It is instantiated with `(params, lr)` only (`agent.py:78`), so
`momentum` and weight decay default to zero.

> **Closure caveat (hole H7).** The `momentum=0` base paths of
> `SignSGD`/`TropicalSGD` (`W -= lr*sign(grad)`,
> `optim/__init__.py:104`, `:156`) can participate in a declared
> compare/add/shift recipe when its learning-rate representation is also
> constrained. Their optional momentum path uses a float EMA
> `buf.mul_(momentum)` (`optim/__init__.py:101`, `:153`) and leaves that subset.
> “Zero DSP” is not a Python-optimizer property; it requires an exact compiled
> configuration and matching utilization report. Audit `PureTropicalSGD` or
> `TropicalLion` settings just as specifically (§9).

### 4.3 `run()` and `evaluate()`

`run()` (`agent.py:168`) alternates `self_play -> train_step`; anneals `tau`
only when **both** `tau_start` and `tau_end` are set (`agent.py:173-174`) and
anneals the visit-sampling temperature; it calls `evaluate()` on `eval_every`.
`self_play` (`agent.py:110`) runs `play_game` per game (`:114`) and saves the
trajectory.

`evaluate()` (`agent.py:131`) scores the greedy policy against an exact
**value-iteration oracle** built with
`tikitorch.planning.TropicalValueIteration` (`agent.py:121-127`), reporting
`agree` (optimal-action agreement over non-terminal states, `:150-160`) and
`solve_frac` (`:161-165`).

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

The projected `SCORE_N` actor and proposed hard fused-step datapath use the
**hard** (`tau=0`) 1-of-d route. The training default is **soft** (`tau=1.0`,
`networks.py:203-205`), and `run()` anneals to hard **only if** `tau_start` and
`tau_end` are both set (`agent.py:173-174`); both default to `None`
(`agent.py:35-36`). Following the default configuration therefore never
reaches `tau=0`. A soft-trained checkpoint cannot be treated as validated for a
hard datapath without revalidation.

**Protocol to deploy the tropical arm:**

1. The proposed on-fabric forward is the `tau=0` hard path—set it explicitly. Either set `MuZeroConfig.tau_start`/`tau_end` so `run()` anneals to 0 (`agent.py:173-174`), or call `net.set_tau(0.0)` (`networks.py:227-230`) before a binding attempt.
2. **Revalidate** `evaluate()` `agree`/`solve_frac` (`agent.py:150-165`) at `tau=0`; soft->hard can change the policy, so soft-path metrics do not transfer.
3. Inference is `@torch.no_grad` (`networks.py:244`, `:252`); no autograd graph is built on the actor path.

---

## 7. Two-player / board-game arm — `tikitorch/planning/muzero/board_agent.py`

The board-domain comparison has a **complete second agent** the engineer must
build/deploy—do **not** re-derive it from `play_board_game`/negamax.
`board_agent.py` defines:

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

`__all__` (`optim/__init__.py:39-41`) includes `SignSGD`, `TropicalSGD`,
`PureTropicalSGD`, `TropicalLion`, `TropicalAdam`, `FrexpMagnitudeSGD`,
`CountAgreementSGD`, `ExponentEMAAdam`, `PerLayerStepSGD`, and
`EFTropicalSGD`.

* **`SignSGD`** (`:57`): `W -= lr * sign(grad)` (`:104`). The base path is sign-only; optional momentum is a float EMA and leaves a compare/add/shift recipe (hole H7, docstring `:62-70`, code `:101`). Physical DSP use is not determined by this software path.
* **`TropicalSGD`** (`:109`): the semiring view—sign is a comparison and the update is an addition (`:156`). It has the same `momentum=0` software-closure opportunity and H7 caveat (`:114-122`, `:153`).
* **`PureTropicalSGD`** (`:161`): with `adaptive=False`, it is designed for a multiplier-free exported update recipe using integer-counter momentum, a power-of-two learning rate, and optional LFSR noise. The `adaptive=True` path uses `median`/divide/`log2` and is **EXPERIMENTAL, NOT closed** (`:164-178`). A physical claim still requires the matching compiled report.
* `TropicalLion`, `TropicalAdam`, `FrexpMagnitudeSGD`, `CountAgreementSGD`, `ExponentEMAAdam`, and `PerLayerStepSGD` offer alternative scaling mechanisms. Audit the concrete settings and executed step; a class name alone does not establish full-recipe closure or FPGA utilization.

---

## 10. FPGA kernels — `tikitorch/fpga/kernels.py`

`KernelCache` / `KernelSpec`: O(1) dict from `(model_type, config)` to a precompiled fused training-step kernel — **no compilation in the hot path**.

**Mode codes** (`kernels.py:38-40`): `MODE_FWD_LOSS = 0`, `MODE_BWD = 1`, `MODE_SIGNSGD = 2`.

**Two C ABIs** (`KernelABI`, `kernels.py:148-164`):

* `KernelABI.FUSED_5 = KernelABI("fused5", 4, False)`—`(buffer, mode, arg1, arg2, arg3)` (`:163`). Used by `forge_train_step`, `forge_train_step_morph`, and `forge_train_step_morph_ecg`.
* `KernelABI.FUSED_6 = KernelABI("fused6", 5, True)` adds trailing `layer_id` (`:164`). It is used by `forge_train_step_2layer` / `forge_train_step_3layer`.

**`KernelSpec`** (`kernels.py:172-205`) carries `entry`, `abi`, `source_file`,
`buffer_words_const`, fixed secondary dimensions, `dtype="int16"` /
`grad_dtype="int32"`, recorded `dsp` metadata, and optional operating points.
A `dsp=0` field is not a generic closure invariant and does not replace a
configuration-matched utilization report.

**Registered kernels** (`KernelCache.default()`, `kernels.py:262-304`)—the
conservative default allow-list:

| entry | model_type | config | ABI | historical operating points |
|---|---|---|---|---|
| `forge_train_step` | `TropicalTransformer` | `LMKernelConfig(V=128,D=128,n_layers=1)` | FUSED_5 | 1200 tok/s |
| `forge_train_step_2layer` | `TropicalTransformer` | `(V=128,D=128,n_layers=2)` | FUSED_6 | 770 tok/s, derived 9.0 mJ/tok estimate, WNS −0.197 ns (timing miss) |
| `forge_train_step_3layer` | `TropicalTransformer` | `(V=128,D=128,n_layers=3)` | FUSED_6 | — |
| `forge_train_step_morph` | `MorphSSM` | `MorphKernelConfig(D=64,n_layers=4)` | FUSED_5 | — |
| `forge_train_step_morph_ecg` | `MorphSSM` | `(D=128,n_layers=8)` | FUSED_5 | 1207 tok/s, derived 5.8 mJ/tok estimate, WNS +0.009 ns |

These operating points are **LM/morph-kernel** values transcribed from
`hardware/README.md` (fields at `kernels.py:276-303`). The energy fields are
derived estimates, not measured full-card energy; negative WNS is a timing
miss. They are **not** MuZero numbers, and not every spec carries them.

**`match(model)`** (`kernels.py:238-255`) routes `TropicalTransformer` through
`LMKernelConfig`, `MorphSSM` through `MorphKernelConfig`, and
`TropicalMuZeroNet` through the existing `MuZeroKernelConfig`
(`kernels.py:104-137`). Other classes, ambiguous configurations, and cache
misses return `None`. **Key derivation is not binding:** an exact
full recipe validation, a usable native endpoint, and an explicitly registered
kernel-specific runner must also succeed. Runtime inputs, named marshalling, and
build identity remain runner-owned checks.

> **Critical for the MuZero arm. [projected]** `TropicalMuZeroNet` is recognized
> for key derivation, but no MuZero spec is registered by
> `KernelCache.default()`. `DenseMuZeroNet` is not routed. Thus neither MuZero
> arm binds today, and the unrelated `forge_train_step` LM kernel is not reused.

### 10.1 CPU-fallback contract — `accelerate.py` / `backend.py`

`match() -> None` only produces `spec=None`; the fallback behavior lives in the wrapper:

* `FpgaBackend.accelerate(model, ...)` computes `spec = self._cache.match(model)`, supplies a native handle only when `is_available()` reports a usable endpoint, and builds an `AcceleratedModel`. LOCAL may be a usable host endpoint; availability is not proof of a physical board. The per-op registry `ops()` is intentionally empty.
* **`bound`** is `True` only when a usable native handle, a full model-observable structural match, and an explicitly registered runner all exist. It does not certify the runner-owned runtime, named-buffer, ABI, or build-manifest checks.
* **Fallback `forward`** runs the inner module unchanged. **Fallback `train_step`** runs the inner model's ordinary `compute_loss(x, targets).backward()` + `optimizer.step()/zero_grad()` path and warns once.
* **`strict=True`** raises `NativeUnavailable` when any endpoint, structural-match, or explicit-runner prerequisite is missing. It disables fallback; it does not validate a custom runner or establish physical FPGA execution.

**Builder takeaway:** the ordinary MuZero methods still run on CPU when the
wrapper is unbound. Pass `strict=True` in a binding gate to prevent silent
fallback; the error lists the missing endpoint, full-match, and runner
prerequisites.

---

## 11. On-silicon evidence — scope it honestly

**[measured] (AWS F2)** — the *tropical shift-LM* fused training-step kernel, `results/fpga_silicon/tikitorch_f2.log`:

```
BOUND on silicon: True | kernel: forge_train_step | dsp: 0
params evolved by on-silicon training: 229872 / 231040
ON-SILICON MULTIPLIER-FREE TRAINING ACTIVE: True
```

Provenance: this is a **char-LM** run loading the monolithic
`forge_train_step` AGFI `agfi-0a845113b088cf62e`
(`cl_forge_train_step_v7_monolithic`,
`scripts/run_tikitorch_f2_silicon.sh:2,20`), binding `model.fpga()` and running
50 on-board `train_step()` calls. The registered spec is
`model_type="TropicalTransformer"`,
`LMKernelConfig(V=128,D=128,n_layers=1)`, FUSED_5
(`kernels.py:271-277`).

This log is historical evidence that the retired transport path invoked the LM
kernel on F2 and that buffer values changed. It is not evidence of a faithful
current PyTorch-model binding: the former runner flattened parameters in the
wrong order and omitted B1/B2 synchronization, so it is now disabled. Its
`dsp: 0` line is only registered catalog metadata; by itself the log is not an
HLS operator/resource report. A physical multiplier/DSP claim must bind the
exact source and AFI to matching synthesis and routed reports. No MuZero kernel
can inherit either the software-closure or resource result from this LM build.

**[projected] (NOT measured for this model)** — the native Tropical-MuZero
MCTS↔net↔train loop on fabric: the `SCORE_N` selection argmax (`mcts.py:5-8`)
and the Stage-7 URAM flat-tensor tree layout (`mcts.py:24-26`). The CPU
reference (`TropicalMCTS` Python tree, `TropicalMuZeroNet`) is the current
vertical slice; the on-fabric actor/learner is a design target motivated by a
configuration-specific F2 precedent, not a measured MuZero run. **Do not cite
the LM silicon numbers or DSP metadata as MuZero-on-silicon results.**

---

## 12. Build spec — the projected MuZero `KernelSpec` and integration

The existing `MuZeroKernelConfig` makes the model keyable, but binding still
requires a configuration-complete key, a new `KernelSpec`, a buffer contract,
an explicitly registered runner, and runtime/build-manifest validation. Until
all of those pieces land and pass §13, the loop
is CPU-only.

### 12.1 `MuZeroKernelConfig` (the match key)

`MuZeroKernelConfig` already records `n_actions`, `latent_dim`, and
`core_kind`, and `match()` already routes `TropicalMuZeroNet` through
`from_model` (`kernels.py:104-137`, `:238-255`). This is safe today because no
MuZero spec is registered. Before registration, expand the exact key or a
mandatory bind-time verifier to cover every compiled dimension and recipe
choice: representation input size, unroll length, dynamics kind, head type and
support, `dyadic_heads`, hard-routing state, loss/reverse rule, and kernel
revision. Any ambiguity must return `None`; a key collision must never select a
merely shape-similar bitstream.

### 12.2 ABI choice and the multi-stage unroll

The MuZero learner is multi-stage (`repr` once, then `K` × `dyn`+`pred`). A
candidate fit is a `layer_id`-style **FUSED_6** (`kernels.py:164`) or a
stage-index variant, driving the fused mode sequence `MODE_FWD_LOSS -> MODE_BWD
-> MODE_SIGNSGD` (`kernels.py:38-40`) per unroll step. **DSP = 0 is a design
target, not an inherited invariant**; establish the actual resource mapping for
the exact new configuration from its operator and utilization reports.

### 12.3 Declared host boundaries that stay off-fabric

The proposed first kernel covers selected hard tropical cores only. With the
agent's default `dyadic_heads=False`, policy/value/reward `nn.Linear` heads
remain dense boundaries (`networks.py:68-71`, `:176-182`). With
`dyadic_heads=True`, their exported forward schedules may instead use
shift-add operations, but their weight gradients and the selected head losses
still require explicit reverse/loss recipes. The action gather
(`networks.py:152`, `:166`), `_minmax_unit` divide (`:36-40`), and PUCT
`sqrt`/`mul`/`div` plus search-side min-max divide (`mcts.py:194-200`,
`:70-73`) must likewise be assigned deliberately to host or compiled
implementations. Do not assign any of these boundaries DSP usage until the
concrete implementation is synthesized.

### 12.4 Buffer packing — two replay schemas

The fused kernel consumes a flat int32 buffer of fixed word count
(`buffer_words_const`, `kernels.py:190`; `dtype="int16"` weights /
`grad_dtype="int32"`, `:197-198`). The new `KernelSpec` must define the
`(B,K,...)` -> packed-int32 mapping for **both** target schemas:

* **single-agent** — `ReplayBuffer` n-step bootstrap value targets (`replay.py:48-60`): `obs0 (B,obs_dim)`, `acts (B,K)`, `value_targets (B,K+1)`, `reward_targets (B,K)`, `policy_targets (B,K+1,A)`;
* **board** — `BoardReplay` outcome value targets `+1/0/-1`, `reward -> 0` (`board_agent.py:89`, `selfplay.py:80-81`).

Absorbing-state padding (value 0 / reward 0 / uniform policy, `replay.py:79-85`) must be reproduced in the on-fabric packing. Version the match key with a kernel-revision so a checkpoint can only bind to the matching compiled bitstream.

### 12.5 How it grafts onto `run()` (evolution, not a rewrite)

Three integration surfaces are visible in the current code, but native binding,
packing, runner implementation, and physical validation remain:

1. **Net construction** — wrap once: `net = backend.accelerate(net)` after the `net_cls(...)` site (`agent.py:70-76`). Backend wrapping is idempotent (`backend.py:66-74`).
2. **Actor path** — `self_play()` -> `play_game` (`agent.py:114`) drives
   `initial_inference`/`recurrent_inference` on CPU today; a native actor also
   needs its packing and runner path.
3. **Learner path** — `train_step()`'s `loss.backward(); self.opt.step()`
   (`agent.py:95-96`) could route through a fused `FWD_LOSS -> BWD -> SIGNSGD`
   sequence only after the MuZero spec, loss/reverse recipe, buffer ABI, runner,
   and correctness gates are implemented. This is not a one-field registration
   change.

### 12.6 Error contract

Unknown shape → `match()` returns `None`. In production pass `strict=True` so the wrapper raises `NativeUnavailable` ("no kernel of this shape", `accelerate.py:73-75`) rather than silently degrading; in dev the warn-once CPU fallback (`accelerate.py:166`) runs the byte-identical CPU loop.

---

## 13. Validation gate

For a future exact MuZero configuration, an on-fabric multiplier/resource claim
and a correctness claim require separate evidence:

1. **Single-agent oracle** — `TropicalValueIteration` plus `agree`/`solve_frac` (`agent.py:121-165`).
2. **Board oracle** — depth-limited negamax / minimax via `play_match` (`board_agent.py:196-200`, `games.py`).
3. **CPU-reference-vs-bound equivalence** — same seed/checkpoint **at `tau=0`** (§6), assert `agree`/`solve_frac` and the root `visit_policy` match within `eps`. Gate with `strict=True` (§10.1, §12.6) so unbound CI does not pass silently.
4. **Software recipe audit** — record the concrete hard model, loss, reverse
   rule, optimizer, input path, and representative forward-loss-backward-update
   trace; document every declared boundary.
5. **Physical acceptance** — archive the exact source/build manifest, HLS
   operator and utilization reports, a routed report with WNS >= 0 at the
   stated target, `BOUND==True`, a known-answer comparison, and a
   params-evolved record. If DSP = 0 is claimed, the configuration-matched
   utilization report—not `KernelSpec` metadata—must show it.

---

## 14. Determinism contract — a correctness hazard for parallel actors

**FIXED.** `_add_root_noise` (`mcts.py:170`) now draws the root Dirichlet noise from a **local** `numpy.random.default_rng(self.cfg.seed)` — it does NOT touch the process-global torch RNG. (It previously called `torch.manual_seed(self.cfg.seed)`, a global reseed that both leaked into other code and re-seeded to the *same* fixed value every call, so the noise was identical across moves until `cfg.seed` mutated. `torch.distributions.Dirichlet.sample()` takes no `generator=`, hence numpy.) `play_game` mutates `cfg.seed = seed*100003 + t` per move (`selfplay.py:39`; `play_board_game` likewise `selfplay.py:73`), so noise is per-move distinct; action sampling already used a **local** `Generator` (`_sample_action`, `selfplay.py:21`/`:29`) and the agent a local `self.gen` (`agent.py:79`).

**Contract for parallel actors.** The local RNG removes one process-global
reseed hazard; it does not by itself establish bit-for-bit reproducibility
across worker schedules, numerical backends, or future FPGA execution. Give
each actor a **disjoint** seed stream (the per-move
`cfg.seed = seed*100003 + t` mutation is per-move, not per-worker), record the
assignment, and test trajectories against the CPU reference for the exact
implementation.

---

## 15. When to pick the tropical arm (domain decision rule)

The §1 domain split is a design hypothesis. The repo supplies the matched
control (`DenseMuZeroNet`, `hidden_mult=3`, `networks_dense.py:72`) and the
metrics (`agent.py:131-165`; board win/draw/loss via `play_match`) but **no measured
tropical-vs-dense delta is recorded here**. The previously cited
`results/backup_sweep/FINDINGS.md` **does not exist** in the repo (extant
artifacts: `results/bakeoff_closed_optimizers.{jsonl,md}`,
`results/confirm_full250.{jsonl,md}`, `results/confirm_short50.{jsonl,md}`,
`results/closed_benchmarks_remote/`, `results/fpga_silicon/`). Treat `backup`
as a swept hyperparameter whose defaults differ by entry point:
`MCTSConfig=max` (`mcts.py:51`), `MuZeroConfig/play_game=topk,k=2`
(`agent.py:43-44`, `selfplay.py:25`), and
`play_board_game/BoardMuZero=mean` (`selfplay.py:55`, `board_agent.py:68`).
`mean` is the published-MuZero/adversarial-correct safe option; `topk` is the
wired tropical-leaning default.

**Decision rule for a new domain:**

* **A-priori checklist** — deterministic perfect-information problems with additive/bottleneck-shaped returns (shortest-path / min-cost = min-plus; reachability = max-plus) motivate testing a tropical prior; stochastic/averaging returns motivate a dense control. These are hypotheses, not measured winners.
* **A-posteriori gate** — run both `net_kind="tropical"` and `"dense"` (param-matched, `networks_dense.py:72`) on the target env; compare `evaluate()` `agree`/`solve_frac` (`agent.py:131-165`) for single-agent or `play_match` win-rate (`board_agent.py`) for two-player. Deploy the tropical arm only if it matches dense within a stated tolerance **and** measured batch-1 latency and full-system energy justify it.

State the measured deltas explicitly when they exist; until then, mark the domain split as **not yet measured in this repo**, not as a result.
