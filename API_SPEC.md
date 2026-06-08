# TikiTorch Public API Specification (authoritative contract)

This is the **shared, authoritative contract** for the TikiTorch package. Phase
1 (this document + `functional.py`, the foundational `nn` layers, the package
scaffold) is **done and frozen**. Phase 2 agents (nn higher-level layers, optim,
models) **MUST** produce exactly the public symbols and constructor signatures
specified here, with behavior ported verbatim from the forge sources.

Read this in full before writing any Phase-2 code. If a forge source and this
spec disagree on *behavior of the tropical math*, the forge source wins (the
math is validated science). If they disagree on *naming, packaging, or
constructor shape*, this spec wins (it defines the clean, decoupled API).

---

## 0. Global rules (apply to every Phase-2 file)

1. **No `forge` imports anywhere.** All intra-package imports are
   `from tikitorch...` (or relative). Carve sources:
   - `forge/arithmetic/tropical.py`  -> `tikitorch/functional.py` (done) + `tikitorch/nn`
   - `forge/models/morphological.py` -> `tikitorch/models` + `tikitorch/nn`
   - `forge/models/tropical.py`      -> `tikitorch/models`
   - `forge/models/tropical_transformer.py` -> `tikitorch/models` + `tikitorch/nn`
   - `forge/models/mote.py`          -> `tikitorch/models` + `tikitorch/nn`
   - `forge/training/optimizers.py`  -> `tikitorch/optim`

2. **The tropical/closure MATH is ported EXACTLY.** Do not "improve" the
   autograd forward/backward, the sign/shift updates, or the max-plus/min-plus
   ops. Copy the bodies verbatim, fixing only imports.

3. **Drop `ForgeModel` / abstract base.** Models subclass `torch.nn.Module`
   directly. Replace `from forge.models.base import ForgeModel, HardwareCostProfile`
   with `import torch.nn as nn` and (if needed) `from tikitorch import HardwareCostProfile`.
   - The abstract `@abstractmethod forward` is dropped (Module already requires it).
   - `compute_loss(x, y)` becomes an **optional concrete method** on each model
     (see section 4.0) — provide it because the silicon training loop relies on
     it, but it is not enforced by any base class.
   - `update_step`, `uses_custom_update`, `param_count` from the old base are
     **dropped** unless a specific model overrode `compute_loss`/needs them
     (MorphSSM and MoTEModel override `compute_loss`; keep those overrides).

4. **Drop `register_model` and all `@register_model("...")` decorators.**
   TikiTorch has no global registry.

5. **Constructor shape: keyword args, not a `config` dict.** Every forge model
   `__init__(self, config: dict)` becomes `__init__(self, *, <explicit kwargs>)`
   with the exact defaults taken from the `config.get(key, default)` calls in
   the source. Section 4 lists the precise signatures. The body that read
   `model_cfg.get("k", d)` is replaced by the parameter `k=d`. Where forge read
   the same value from either `model` or `data` sub-configs, expose a single
   keyword.

6. **Drop optional research dependencies.** Remove anything importing
   `forge.training.losses` (GenerationLoss), `forge.models.generation`
   (TropicalState/Frontend/Backend), `forge.arithmetic.lns`,
   `forge.arithmetic.mitchell`, `forge.arithmetic.ternary`, codecs,
   semantic_distillation. Drop the methods that depend on them
   (`init_state`, `step`, frontend/backend paths, `_load_pretrained_backbone`,
   `_generation_loss`, the LNS/hybrid optimizers — see sections 3 and 4 for
   exactly what to keep vs. drop). Keep only the core
   classification / sequence / LM forward + the closure machinery.

7. **Naming + aliasing.** Expose research names AND clean PyTorch-idiomatic
   aliases. Both names point to the same class object (`CleanName = ResearchName`).
   Section 2/3 give the required alias for each symbol.

8. **`hardware_cost(...)` is optional and cheap.** Port the
   `get_hardware_cost_profile` bodies as a method named **`hardware_cost`**
   (returning `tikitorch.HardwareCostProfile`) only where the forge model had
   one and it is self-contained. Do not make it abstract or required. Keep
   `get_hardware_cost_profile` as an alias to `hardware_cost` for compatibility.

---

## 1. Module layout (final)

```
tikitorch/
  __init__.py            __version__, re-exports: functional, nn, optim, models, HardwareCostProfile   [DONE]
  functional.py          stateless tropical ops + autograd Functions                                   [DONE]
  _hardware.py           HardwareCostProfile dataclass                                                  [DONE]
  nn/__init__.py         tropical layer Modules; Phase-1 core done, Phase-2 layers appended below marker
  optim/__init__.py      SignSGD, TropicalSGD, PureTropicalSGD                                          [Phase 2]
  models/__init__.py     TropicalMLP, TropicalSequenceModel, MorphSSM, MorphMLP,
                         TropicalTransformer, MoTEModel(+alias MoTE)                                    [Phase 2]
  hardware/              HLS kernels + BUILD_GUIDE.md                                                   [DONE]
examples/                runnable usage scripts                                                         [Phase 3]
tests/                   pytest suite                                                                   [Phase 3]
```

Idiomatic usage the whole package must support:

```python
import tikitorch as tt
import tikitorch.nn as nn
import tikitorch.optim as optim
model = tt.models.MorphSSM(d_model=128, n_layers=4, task_type="lm", vocab_size=128)
opt = optim.PureTropicalSGD(model.parameters())
loss = model.compute_loss(x, y)   # or cross_entropy on model(x)
loss.backward(); opt.step()
```

---

## 2. `tikitorch.functional` — DONE (frozen)

Public symbols (all importable from `tikitorch.functional`). Phase-2 code
imports tropical ops from here; do **not** redefine them.

```python
# min-plus (erosion) linear
class TropicalLinearFn(torch.autograd.Function)        # forward(ctx, x, weight) -> (batch,out); sparse argmin backward
tropical_linear = TropicalLinearFn.apply               # (x:(B,in), weight:(out,in)) -> (B,out)

# max-plus (dilation) linear
class TropicalDilationFn(torch.autograd.Function)      # argmax dual of the above
tropical_dilation = TropicalDilationFn.apply

# soft relaxations
def tropical_softmin(z, tau=1.0, dim=-1) -> Tensor     # -tau * logsumexp(-z/tau, dim)
def tropical_softmax(z, tau=1.0, dim=-1) -> Tensor     #  tau * logsumexp( z/tau, dim)

# tropical matmul
def tropical_matmul(A, B) -> Tensor                    # C[i,j] = min_k(A[i,k] + B[k,j]); A:(...,m,k) B:(...,k,n)

# attention: dot-product scoring (max-plus inner product)
class TropicalDotProductFn(torch.autograd.Function)
tropical_dot_product = TropicalDotProductFn.apply      # (Q:(...,Lq,D), K:(...,Lk,D)) -> (...,Lq,Lk)
def tropical_softmax_dot_product(Q, K, tau=1.0) -> Tensor

# attention: L_inf scoring (symmetric in Q,K)
class TropicalLInfScoreFn(torch.autograd.Function)
tropical_l_inf_score = TropicalLInfScoreFn.apply       # -max_d|Q-K|
def tropical_softmax_l_inf_score(Q, K, tau=1.0) -> Tensor

# attention: value pull (per-dim max-plus matvec)
class TropicalValuePullFn(torch.autograd.Function)
tropical_value_pull = TropicalValuePullFn.apply        # (V:(...,Lv,D), scores:(...,Lq,Lv)) -> (...,Lq,D)
def tropical_softmax_value_pull(V, scores, tau=1.0) -> Tensor
```

---

## 3. `tikitorch.nn`

### 3.0 Phase-1 core layers — DONE (frozen)

```python
class TropicalSoftMin(nn.Module):  __init__(self, tau: float = 1.0); forward(x, weight)
class TropicalSoftMax(nn.Module):  __init__(self, tau: float = 1.0); forward(x, weight)

class TropicalDilationModule(nn.Module):
    __init__(self, in_features, out_features, use_softmax=False, tau=1.0)
    forward(x) -> y ; forward_with_indices(x) -> (y, argmax_indices) ; set_tau(tau)
class TropicalLinearModule(nn.Module):
    __init__(self, in_features, out_features, use_softmin=False, tau=1.0)
    forward(x) -> y ; set_tau(tau)
class CompositionalTropicalProjection(nn.Module):
    __init__(self, in_features, out_features, d_intermediate=None, tau=0.0)
    forward(x) -> y ; set_tau(tau)
class TropicalTopKModule(nn.Module):
    __init__(self, in_features, out_features, k=4, tau=0.0)
    forward(x) -> y ; set_tau(tau)

# aliases (already defined):
TropicalLinear     = TropicalLinearModule
TropicalDilation   = TropicalDilationModule
TropicalProjection = CompositionalTropicalProjection
TropicalTopK       = TropicalTopKModule
```

### 3.1 Phase-2 layers — APPEND below the `# --- higher-level layers added in Phase 2 ---` marker

Port these `nn.Module` building blocks verbatim from the forge sources, fixing
only imports (`from tikitorch.functional import ...`, and intra-`nn` references).
Add each new public name to `nn.__all__`.

From `forge/models/morphological.py`:

```python
class TropicalBatchShift(nn.Module):
    # no __init__ args; forward(x) -> x - x.mean(dim=0, keepdim=True)
    # clean alias: BatchShift = TropicalBatchShift

class DepthwiseMorphConv1d(nn.Module):
    __init__(self, channels, kernel_size, dilation=1, mode="dilation",
             use_soft=True, tau=1.0, group_size=1)
    forward(x: (B,C,L)) -> (B,C,L)
    step(x_t: (B,C), buffer) -> (y_t, buffer)     # keep; pure tropical, no deps
    # clean alias: MorphConv1d = DepthwiseMorphConv1d

class MultiScaleMorphConv1d(nn.Module):
    __init__(self, channels, kernel_sizes=(3,7,15), dilation=1, mode="dilation",
             use_soft=True, tau=1.0)
    forward(x: (B,C,L)) -> (B,C,L)
    # clean alias: MultiScaleMorphConv1d (no rename needed)

class MorphSSMLayer(nn.Module):
    __init__(self, d_model, kernel_size=7, dilation=1, mode="dilation",
             dropout=0.1, use_soft=True, tau=1.0, tropical_proj=False,
             norm_type="layernorm", group_size=1, residual_scale=1.0,
             multi_scale=False, kernel_sizes=(3,7,15), out_proj_type=None)
    forward(x: (B,L,D)) -> (B,L,D)
    step(x_t: (B,D), buffer) -> (y_t, buffer)     # keep; pure tropical, no deps
    # NOTE: the internal `from forge.arithmetic.tropical import TropicalDilationModule`
    #       inside __init__ becomes `from tikitorch.nn import TropicalDilationModule`
    #       (or a module-level import). Behavior identical.
```

From `forge/models/tropical_transformer.py`:

```python
class TropicalAttention(nn.Module):
    __init__(self, d_model, n_heads, dropout=0.0, tau=0.0, use_standard_proj=False,
             score_variant="dot_product", proj_variant="dilation",
             proj_d_intermediate=None, proj_topk_k=4)
    forward(x: (B,L,D), causal_mask=False) -> (B,L,D) ; set_tau(tau)

class TropicalShiftAttention(nn.Module):
    __init__(self, d_model, n_heads, max_seq_len, tau=0.0,
             score_variant="dot_product", shift_mode="free")
    forward(x: (B,L,D), causal_mask=False) -> (B,L,D) ; set_tau(tau)

class TropicalTransformerBlock(nn.Module):
    __init__(self, d_model, n_heads, d_ff, dropout=0.1, tau=0.0,
             use_shift_attn=False, max_seq_len=None,
             score_variant="dot_product", shift_mode="free")
    forward(x: (B,L,D), causal_mask=False) -> (B,L,D) ; set_tau(tau)
```

From `forge/models/mote.py`:

```python
class TropicalRouter(nn.Module):
    __init__(self, d_model, d_router, n_experts)
    forward(x: (T,D)) -> (expert_ids:(T,), aux_loss: scalar)

class MoTEBlock(nn.Module):
    __init__(self, d_model, n_experts, expert_type="transformer", expert_layers=1,
             d_ff=512, n_heads=4, kernel_size=7, dropout=0.1, d_router=32,
             use_soft=True, tau=1.0, capacity_factor=1.5, use_shift_attn=False,
             max_seq_len=None, score_variant="dot_product")
    forward(x: (B,L,D)) -> (out:(B,L,D), aux_loss: scalar)
```

Required Phase-2 aliases to add: `BatchShift`, `MorphConv1d` (and keep all
research names exported). Update `nn.__all__` accordingly.

---

## 4. `tikitorch.models`

### 4.0 The `task_type` / loss convention (replaces ForgeModel)

Every model takes a `task_type` keyword: `"classification"`, `"lm"`, or
`"regression"` (subset per model — see each). Each model exposes a concrete
`compute_loss(self, x, y) -> Tensor` method:

```python
def compute_loss(self, x, y):
    logits = self.forward(x)
    if self.task_type in ("classification", "lm"):
        if logits.dim() == 3:
            logits = logits.reshape(-1, logits.size(-1)); y = y.reshape(-1)
        return torch.nn.functional.cross_entropy(logits, y)
    else:  # regression
        if logits.shape == y.shape:
            return torch.nn.functional.mse_loss(logits, y.float())
        return torch.nn.functional.mse_loss(logits.squeeze(-1), y.float())
```

This is the carved body of `ForgeModel.compute_loss` (no `GenerationLoss`).
MorphSSM and MoTEModel had richer overrides tied to `GenerationLoss` /
`aux_loss` — for those, **drop the GenerationLoss branch** (regression uses
plain MSE as above) but **keep the MoTE aux-loss term** (see 4.6).

Also provide a `param_count` property on each model
(`sum(p.numel() for p in self.parameters() if p.requires_grad)`), as it is
cheap and convenient.

### 4.1 `TropicalMLP`  (from `forge/models/tropical.py`)

```python
class TropicalMLP(nn.Module):
    __init__(self, *, d_model=512, n_layers=4, dropout=0.1, temperature=1.0,
             use_softmin=True, tau_min=0.01, input_dim=784, num_classes=10,
             task_type="classification")
    forward(x) -> (B, num_classes)
    set_temperature(tau)            # keep
    switch_to_hard_min()            # keep
    switch_to_soft_min()            # keep
    get_gradient_sparsity() -> dict          # keep
    get_gradient_sparsity_summary() -> dict  # keep
    hardware_cost(batch_size=1, seq_len=1) -> HardwareCostProfile   # port get_hardware_cost_profile
    # get_hardware_cost_profile = alias of hardware_cost
```
Internals: builds `TropicalLinearModule` layers + `nn.Linear` head, exactly as
in the source. `_tropical_activation` private method kept verbatim.

### 4.2 `TropicalSequenceModel`  (from `forge/models/tropical.py`)

```python
class TropicalSequenceModel(nn.Module):
    __init__(self, *, d_model=256, n_layers=2, dropout=0.1, temperature=1.0,
             use_softmin=True, tau_min=0.01, input_dim=1, seq_len=784,
             num_classes=10, task_type="classification")
    forward(x) -> (B, num_classes)
    set_temperature(tau)                     # keep
    get_gradient_sparsity() -> dict          # keep
    get_gradient_sparsity_summary() -> dict  # keep
    hardware_cost(batch_size=1, seq_len=None) -> HardwareCostProfile
    # get_hardware_cost_profile = alias
```

### 4.3 `MorphSSM`  (from `forge/models/morphological.py`)

```python
class MorphSSM(nn.Module):
    __init__(self, *, d_model=256, n_layers=8, kernel_size=7, dropout=0.1,
             use_soft=True, tau=1.0, tropical_proj=False, norm_type="layernorm",
             group_size=1, max_dilation=2048, multi_scale=False,
             kernel_sizes=(3,7,15), residual_scaling=False,
             gradient_checkpointing=False,
             task_type="classification",
             # task-dependent:
             vocab_size=32000,          # used when task_type == "lm"
             input_dim=1, num_classes=10)   # used otherwise
    forward(x) -> logits   # dispatches on task_type via _forward_lm/_forward_classification/_forward_regression
    compute_loss(x, y)     # CE for lm/classification, MSE for regression (NO GenerationLoss)
    hardware_cost(...) -> HardwareCostProfile ; get_hardware_cost_profile = alias
    # DROP: init_state, step (depend on forge.models.generation.TropicalState),
    #       _generation_loss / GenerationLoss branch of compute_loss.
    #       Keep _forward_regression (plain) — it has no GenerationLoss dependency
    #       in the forward path; only compute_loss used GenerationLoss, which is dropped.
```
Keep the GPT-2 residual init loop (`nn.init.normal_(layer.out_proj.weight, ...)`)
exactly. `_run_layers` keeps gradient-checkpointing support.
NOTE the forge default `n_layers` for MorphSSM is 8; preserve it.
Internal `from forge.arithmetic.tropical import TropicalDilationModule` (used in
hardware_cost and MorphSSMLayer) -> `from tikitorch.nn import TropicalDilationModule`.

### 4.4 `MorphMLP`  (from `forge/models/morphological.py`)

```python
class MorphMLP(nn.Module):
    __init__(self, *, d_model=128, n_layers=4, dropout=0.0, use_soft=True,
             tau=1.0, input_dim=784, num_classes=10, task_type="classification")
    forward(x) -> (B, num_classes)
    hardware_cost(batch_size=1, seq_len=1) -> HardwareCostProfile
    # get_hardware_cost_profile = alias
```
Alternating `TropicalDilationModule` / `TropicalLinearModule` + `TropicalBatchShift`
norms + `nn.Linear` head, verbatim.

### 4.5 `TropicalTransformer`  (from `forge/models/tropical_transformer.py`)

```python
class TropicalTransformer(nn.Module):
    __init__(self, *, d_model=256, n_heads=4, n_layers=6, d_ff=None, dropout=0.1,
             tau=0.0, use_shift_attn=False, max_seq_len=1024,
             score_variant="dot_product", shift_mode="free",
             task_type="classification",
             vocab_size=32000,           # task_type == "lm"
             input_dim=1, num_classes=10)
    forward(x) -> logits   # classification: GAP + head ; lm: causal + head
    set_tau(tau)           # cascades to all blocks
    # d_ff default: if None -> d_model * 4 (matches forge `d_model*4`)
    # NO hardware_cost in forge source -> do not add one (keep it optional/absent).
```

### 4.6 `MoTEModel`  (alias `MoTE`)  (from `forge/models/mote.py`)

```python
class MoTEModel(nn.Module):
    __init__(self, *, d_model=128, n_experts=8, expert_type="transformer",
             expert_layers=1, d_ff=None, n_heads=4, router_dim=32,
             n_shared_pre=4, n_shared_post=2, kernel_size=7, use_soft=True,
             tau=1.0, dropout=0.1, max_dilation=2048, capacity_factor=1.5,
             use_shift_attn=False, score_variant="dot_product",
             out_proj_type="linear", max_seq_len=1024, aux_loss_weight=0.01,
             task_type="classification",
             vocab_size=32000, input_dim=1, num_classes=10)
    forward(x) -> logits
    compute_loss(x, y)     # task loss + aux_loss_weight * aux_loss (KEEP aux term; DROP GenerationLoss)
    hardware_cost(...) -> HardwareCostProfile ; get_hardware_cost_profile = alias
    get_expert_utilization() -> dict   # keep (returns {})
    # d_ff default: if None -> d_model * 4
    # DROP: frontend/backend (TropicalFrontend/Backend), init_state, step,
    #       _load_pretrained_backbone, pretrained_backbone arg, _generation_loss.
    #       The `frontend`/`backend` config branches and the regression-with-frontend
    #       path are removed; plain regression (embed -> core -> head) is kept.
MoTE = MoTEModel   # required alias
```

`models.__all__` =
`["TropicalMLP", "TropicalSequenceModel", "MorphSSM", "MorphMLP",
  "TropicalTransformer", "MoTEModel", "MoTE"]`.

---

## 5. `tikitorch.optim`  (from `forge/training/optimizers.py`)

Port these three optimizer classes **verbatim** (they are `torch.optim.Optimizer`
subclasses; the step() bodies are validated). Fix only imports. Drop
`LNSOptimizer`, `HybridTernaryOptimizer`, `build_optimizer`,
`_build_hybrid_ternary_optimizer` (they pull `forge.arithmetic.lns/mitchell/ternary`).

```python
import torch.optim as toptim

class SignSGD(toptim.Optimizer):
    __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0)
    step(closure=None)   # W -= lr * sign(grad), optional momentum-on-sign

class TropicalSGD(toptim.Optimizer):
    __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0)
    step(closure=None)   # semiring view of SignSGD (comparison + addition)

class PureTropicalSGD(toptim.Optimizer):
    __init__(self, params, lr_shift=10, momentum_bits=4, decay_every=8,
             noise_scale=0.0, adaptive=False, adaptive_bits=3, weight_decay_shift=0)
    step(closure=None)   # strictly multiplier-free: power-of-2 lr (bit-shift),
                         # int-counter momentum, LFSR noise, sign-L1 adaptive
```

`optim.__all__ = ["SignSGD", "TropicalSGD", "PureTropicalSGD"]`.

Do NOT add a `build_optimizer` factory — TikiTorch is config-free; users
instantiate optimizers directly (PyTorch-idiomatic).

---

## 6. Phase-3 deliverables (for the integration agent)

- Finalize `tikitorch/__init__.py` re-exports once optim/models are real
  (the guards already make `import tikitorch` succeed; once Phase 2 lands,
  `tt.optim` and `tt.models` resolve to the real modules automatically).
- `examples/`: small runnable scripts — at minimum
  `train_morphssm_lm.py`, `train_tropical_mlp_mnist.py`, and a
  `soft_to_hard_anneal.py` demonstrating the `tau -> 0` curriculum.
- `tests/`: pytest suite covering (a) gradcheck of each autograd Function
  against its soft relaxation, (b) forward/backward shape + finiteness for
  every nn layer and every model and every task_type, (c) each optimizer
  reduces a toy loss, (d) `import tikitorch` and `tt.models.MorphSSM(...)`
  smoke path, (e) hard vs. soft equivalence as `tau -> 0`.

---

## 7. Decoupling checklist (every Phase-2 file must pass)

- [ ] No `from forge...` / `import forge` anywhere.
- [ ] No `@register_model(...)`; no `ForgeModel`; no abstract base.
- [ ] No `config: dict` constructor — explicit keyword args with source defaults.
- [ ] No imports of generation / losses / lns / mitchell / ternary / codecs.
- [ ] Tropical math bodies copied verbatim (autograd, sign/shift updates, ops).
- [ ] Research names exported + clean aliases added; `__all__` updated.
- [ ] `hardware_cost` (if present) returns `tikitorch.HardwareCostProfile`,
      is optional, and has a `get_hardware_cost_profile` alias.
- [ ] `python -c "import tikitorch; tikitorch.<thing>(...)"` runs without error.
```
