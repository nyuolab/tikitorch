# TikiTorch Backend Strategy

*An investigation and design recommendation for making TikiTorch a standalone,
easy-to-run-off-GPU deep-learning library — with FPGA as the headline backend,
"easy for others to build on" and "as performant as possible."*

> **Status:** strategy / design artifact. No code is proposed for immediate
> landing; this document is the decision record the TikiTorch and MuZero
> instances execute against. It references and partially supersedes the
> originating vision note (`tikitorch.fpga` Backend Developer Specification,
> provided to this instance — not a committed repo file) — specifically, it
> corrects that note's *greenfield Allo bootstrap* assumption to account for the
> silicon-validated C++ pipeline that already exists in this repo.

---

## 1. Executive summary & recommendation

TikiTorch's entire workload collapses to **one primitive** — max-plus / min-plus
GEMM + top-k — and is multiplier-free by construction (`audit.assert_closed`).
That is an unusually clean target for an accelerator backend. But two facts shape
everything below:

1. **TikiTorch has no device/backend seam today.** Tensors are raw
   `torch.Tensor`, every op is a `torch.autograd.Function`, layers are plain
   `torch.nn.Module`, and all execution routes to PyTorch/CPU. There is nothing
   to plug a backend *into* yet.
2. **A silicon-validated FPGA pipeline already exists — but as separate C++.**
   The HLS kernels and the XRT host driver train tropical models end-to-end on
   AWS F2 at **770–1,207 tok/s, DSP = 0**. There is **no Python↔FPGA bridge**.

**Recommendation:**

- **Build one general, pure-Python backend seam** (dispatch registry + device
  handle + transparent CPU fallback). FPGA is its *first consumer*, but the seam
  is backend-agnostic so a torch-free C/Cython CPU backend, WASM, or other
  accelerators can register into it later. This is the "easy for others" payload.
- **Make the FPGA backend two-tier**, because the silicon is a *fused
  whole-training-step monolith*, not a per-op dispatcher (see §3):
  - **Tier 1 (works now):** Python-bind the existing C++ driver. `model.to('fpga')`
    runs the *whole* fused training step on board for shapes that match a
    compiled kernel; everything else falls back to CPU. Reuses validated assets;
    the fastest route to a real `.to('fpga')`.
  - **Tier 2 (the differentiator, F-gated):** Allo + PYNQ as a *measurement
    bootstrap* on arbitrary shapes, then custom tropical codegen + a MemoryPlanner
    that unlocks true op-level dispatch and arbitrary model shapes.
- **Borrow the runtime forever** (PYNQ → pyXRT). **Own the kernel and the memory
  layout** — that is the only place the tropical advantage actually lives.

The de-risk order is load-bearing: Tier 1 ships a working backend on existing
silicon in weeks; Tier 2's expensive build is gated behind a *measured* F-factor
so we know it's justified before committing months.

---

## 2. Where TikiTorch is today (the seams)

Any backend must plug into the existing structure without breaking the frozen
public API. The relevant seams:

**Ops are `autograd.Function` classes with thin `.apply` aliases.** In
`tikitorch/functional.py`, every tropical primitive is a custom autograd Function
exposed as a module-level callable:

```python
# tikitorch/functional.py
class TropicalLinearFn(torch.autograd.Function): ...   # :87
tropical_linear   = TropicalLinearFn.apply             # :129
class TropicalDilationFn(torch.autograd.Function): ... # :167
tropical_dilation = TropicalDilationFn.apply           # :202
tropical_dot_product = TropicalDotProductFn.apply      # :268
tropical_l_inf_score = TropicalLInfScoreFn.apply       # :328
tropical_hilbert_score = TropicalHilbertScoreFn.apply  # :393
tropical_value_pull  = TropicalValuePullFn.apply       # :481
# + tropical_conv2d (:585/655), perceptron/CE losses (:793/:829), activations
```

These ~10 alias lines are the **dispatch seam**: each is the single point through
which a device could route to a backend implementation. The backward pass already
routes gradients through the saved argmin/argmax index (sparse one-hot scatter,
coefficient 1) — exactly the form an accelerator wants.

**Layers/models are plain `torch.nn.Module`** (no base class), and the public API
is **frozen** by `tikitorch/API_SPEC.md` ("authoritative contract"; §2
`tikitorch.functional` marked *DONE (frozen)*, §3 core layers *frozen*). The
backend abstraction **must be additive** — it may not rename or remove any public
symbol, and the CPU path must remain the default and unchanged.

**The closure contract is `audit.assert_closed`** (`tikitorch/audit.py`). It walks
the module tree and flags any multiply/divide-bearing standard module. Any backend
must *preserve* this guarantee: a model that passes `assert_closed` on CPU must
remain multiplier-free on the backend. This is the property the hardware monetizes
(DSP = 0).

**The op set a backend must cover is small and closed:** the one primitive
(max/min-plus GEMM + top-k), a handful of cheap element-wise ops (`tropical_relu`,
`tropical_leaky_relu` with power-of-2 shift slope, `tropical_hardtanh`), norms
(subtract-mean + power-of-2 rescale), losses (perceptron 2-hot; tropical CE with
exact or shift-LUT softmax), and integer optimizers — **SignSGD**,
**PureTropicalSGD**, **TropicalLion** — in `tikitorch/optim/__init__.py`. No new
math is needed; a backend re-implements this fixed set in its datapath.

---

## 3. The load-bearing insight: the silicon is a fused monolith, not a dispatcher

The originating dev doc imagines per-op dispatch — `tropical_attention`,
`tropical_matmul`, etc. each lowered to a kernel and selected at runtime. **That
does not match the silicon that exists.** The existing host driver runs the
*entire training step* as one fused loop:

```cpp
// forge/hardware/host_training/host_training.hpp:43
long train_step(const int64_t* tokens, int L, int lr, int threshold);
//  ^ forward -> perceptron loss -> backward -> sign-SGD, fused, one call
```

The XRT (on-board) mode swaps only the *kernel-call implementation* — the
orchestration is identical (`forge/hardware/host_training/host_training_xrt.hpp:44`,
`class XrtBackend`). Kernels are dispatched internally by a `mode` scalar
(0 = FWD_LOSS, 1 = BWD, 2 = SIGNSGD) plus a `layer_id`, with **INT16 weights /
INT32 grad accumulators** and **fixed shapes** baked into a `Config{V, D, D_FF,
L_max, n_layers}` (`host_training.hpp:27`). And a hard constraint from prior
silicon bring-up: **there is no multi-kernel mux on F2** — the proven flow is a
single monolithic kernel, not a graph of per-op kernels.

**Consequence:** "change `.to('fpga')` and every op dispatches to its own kernel"
is a *Tier 2* capability that requires codegen we have not built. What we *can*
deliver immediately is **model-level** binding: a TikiTorch model whose shape
matches a compiled kernel runs its *whole* fused training step on board.

The doc's job is to make this two-tier reality explicit rather than promise
per-op dispatch the hardware cannot yet honor:

| | **Tier 1 — model-level (now)** | **Tier 2 — op-level (future, F-gated)** |
|---|---|---|
| Unit of offload | whole fused training step | individual tropical op / layer-chain |
| Shapes | only compiled `Config`s | arbitrary (via codegen) |
| Mechanism | bind existing C++ driver | Allo measure → custom codegen + MemoryPlanner |
| `.to('fpga')` means | "run this supported model on board" | "run any op on board, fall back per-op" |
| Status | assets exist & validated | requires the expensive build |

---

## 4. The general backend abstraction (FPGA-first, but pluggable)

Design only — pure Python, additive to the frozen API, CPU stays the default.

**4.1 Dispatch registry.** A small registry maps `(op_name, device_type)` to an
implementation, with CPU pre-registered to the existing `.apply` callables:

```python
# tikitorch/backend/__init__.py  (illustrative — not landed)
register_backend("cpu")                      # default; existing torch path
register_op("tropical_linear", "cpu", TropicalLinearFn.apply)
register_op("tropical_linear", "fpga", fpga_tropical_linear)   # Tier 2

def dispatch(op_name, device_type):
    return _registry.get((op_name, device_type)) or _fallback(op_name)
```

The functional aliases (§2) become `dispatch("tropical_linear", x.device.type)(...)`.
Because CPU is registered to today's behavior, **nothing changes for existing
users** — the seam is invisible until a non-CPU device is present.

**4.2 Device handle.** Expose `tt.device('fpga'[:idx])` and let models move with
the *inherited* `nn.Module.to(dev)` — do **not** fork PyTorch's device machinery.
For Tier 1, `.to('fpga')` attaches a backend handle to the model (binding it to a
compiled kernel `Config`); per-tensor movement reuses PyTorch device propagation.

**4.3 Transparent fallback (the "easy" guarantee).** Any op/shape/model not
registered for a device **silently runs the CPU reference with a one-time perf
warning**. *Correctness never breaks; performance degrades gracefully.* This is
the single property that makes the backend safe to adopt — an unsupported shape is
a slowdown, never an error.

**4.4 Autograd stays on host.** Forward runs on the backend and retains the argmax
map; backward is a gather. TikiTorch's existing autograd engine is untouched — the
backend integrates as a registered op, never as a fork of the autograd core. (For
Tier 1, forward+backward+update are *all* on board inside the fused step; the host
autograd graph sees one opaque "train step" node.)

**4.5 Closure preservation.** The registry carries the `assert_closed` contract:
a backend op registered for a model that passes `assert_closed` on CPU must be
multiplier-free. CI runs `assert_closed` on the CPU reference and (Tier 1) checks
the bound kernel's synthesis report shows `DSP = 0`.

---

## 5. FPGA backend — Tier 1 (works now): bind the existing driver

The fastest path to a real `.to('fpga')`, reusing silicon-validated assets.

- **Python binding (pybind11, ctypes fallback)** over the two existing modes:
  - `TropicalShiftLM` **LOCAL** mode (`host_training.hpp`) — pure host g++, *no
    board required*. This is the CI/dev path: bit-exact against the Python
    reference, runnable anywhere.
  - `XrtBackend` **XRT** mode (`host_training_xrt.hpp`) — real F2 execution.
  Expose the existing surface: `train_step / evaluate / save / load`.
- **`KernelCache` keyed by `(model_type, Config)` → precompiled xclbin region.**
  Selection is a **lookup — zero hot-path compile**. A TikiTorch model whose shape
  matches a compiled kernel (e.g. the D=128/V=128 tropical shift-LM, or the
  MorphSSM ECG config) maps onto the fused training step; INT16 weights sync in,
  trained weights sync out via the driver's `save`/`load`.
- **Locate bundled kernels** via the existing helpers in
  `tikitorch/hardware/__init__.py` (`hls_dir()`, `build_guide_path()`); the HLS
  sources and build guide already ship as package data.
- **DX & honest scoping:**
  ```python
  import tikitorch as tt, tikitorch.fpga      # registers backend, detects board
  model = tt.models.MorphSSM(...).to(tt.device('fpga'))  # bound to ECG kernel
  model.train_step(tokens)                    # whole fused step on board
  # shape with no compiled kernel -> CPU fallback + perf warning
  ```
  Tier 1 is **model-level** binding (one fused step), *not* per-op dispatch — state
  this plainly in the README so the capability is not oversold.

**Reference operating points (sourced from `tikitorch/hardware/README.md`):**

| Kernel | Config | Throughput | Energy | DSP | WNS |
|---|---|---|---|---|---|
| `forge_train_step` (1-layer LM) | D=128, V=128, L=128 | ~1,200 tok/s | — | 0 | closed |
| `forge_train_step_2layer` | 2-layer, D=128, V=128 | ~770 tok/s | ~9.0 mJ/token | 0 | −0.197 ns |
| **`forge_train_step_morph_ecg`** | D=128, L=8, K=7, L_SEQ=5000 | **1,207 tok/s** | **5.8 mJ/token** | **0** | **+0.009 ns** |

(3-layer LM is silicon-validated but depth-saturated at this width.) Every entry
is **DSP = 0** — the multiplier-free closure proven physically in fabric.

---

## 6. FPGA backend — Tier 2 (the differentiator): codegen + MemoryPlanner

What unlocks op-level `.to('fpga')` and *arbitrary* shapes. Expensive; **gated
behind a measured F-factor.**

- **Allo + PYNQ as a measurement bootstrap.** Per the dev doc's Phase 1, stand up
  a max-plus GEMM in Allo with a PYNQ runtime to measure throughput / energy /
  latency and the **F-factor** (op-efficiency) on shapes the existing fixed
  kernels don't cover. Allo is a *measurement tool here*, not the production
  kernel path — if F is small, we learn it in weeks and Tier 2 stops.
- **Custom tropical codegen** for the closed op set, implementing the DL-specific
  optimizations no general ADL gives for free:
  - *Idempotent narrow datapath* — max-plus never widens; integer-exact, no
    rounding/overflow ⇒ more model on-chip.
  - *Argmax-tag co-generation* — forward emits the winning index inline; that *is*
    the backward (pure gather, no multiplies).
  - *II=1 layer chaining* — dataflow the network on-chip, no per-layer HBM
    round-trip (the structural win over GPU kernel-launch + HBM bounce).
- **MemoryPlanner (the centerpiece).** URAM weight residency across the whole
  forward pass; deliberate HBM channel assignment to spread the crossbar load and
  feed the comparator-tree PEs every cycle. This is the optimization target where
  the tropical advantage concentrates.

Only after this does the registry's per-op `(op_name, "fpga")` path become real
for arbitrary shapes; until then those ops resolve to the §4.3 fallback.

---

## 7. Beyond FPGA — why the seam is general

The same registry admits backends with no FPGA at all. Because the op set is
multiplier-free and integer-exact, it is unusually portable:

- **Torch-free C / Cython CPU backend** — a standalone tropical runtime for edge
  deployment without the PyTorch dependency (the LOCAL-mode C++ driver is already
  most of this).
- **WASM** — run tropical inference in the browser / sandboxed edge.
- **Other accelerators** (e.g. integer-only NPUs) — register their ops; fall back
  per-op otherwise.

Keep this scope secondary. FPGA is the priority; the point is that building the
seam *once*, generally, is what makes "standalone, easy for others" true rather
than a one-off FPGA hook.

---

## 8. Staged roadmap & gates

| Phase | Scope | Gate / output | Rough effort |
|---|---|---|---|
| **0** | CPU/GPU tropical reference + sparse argmax-backward autograd | Already in TikiTorch (frozen, 100+ tests) | done |
| **1** | General backend seam (§4) + Tier-1 binding (§5): pybind11 over LOCAL+XRT, `KernelCache`, `.to('fpga')`, transparent fallback | A supported model trains on F2 via `.to('fpga')`; CPU fallback verified; `assert_closed` + `DSP=0` checked in CI | weeks (binding + seam; LOCAL mode needs no board) |
| **2** | Allo measurement bootstrap → **F-factor decision** → custom codegen + MemoryPlanner (§6) | **F gates the build.** If F justifies it: op-level dispatch for arbitrary shapes | months (F-gated) |
| **3** | On-device backward / full-model dataflow; tropical-attention dynamics for latent planning | Frontier | months+ |

De-risk order: Phase 1 delivers a working backend on *existing* silicon
immediately; Phase 2's cost is never incurred until Phase 1's measurements prove
it pays.

---

## 9. Risks & mitigations

- **Fused-monolith vs per-op mismatch (the §3 insight).** The biggest expectation
  risk. *Mitigation:* ship Tier 1 as explicitly model-level; reserve per-op
  dispatch language for Tier 2.
- **No multi-kernel mux on F2.** Constrains Tier 1 to single monolithic kernels.
  *Mitigation:* `KernelCache` selects one fused kernel per `(model_type, Config)`;
  do not attempt a kernel graph on the current flow.
- **Timing closure is the dominant (generic, not tropical) cost.** Prior builds
  landed at WNS −0.197 ns (needed a rebuild) up to +0.009 ns. *Mitigation:* the PE
  is a short add+compare path, which helps closure; budget AFI rebuilds explicitly
  (the OFFGPU_LEVERS Tier C estimate is ~$10–30 on EC2).
- **HBM crossbar bottleneck** (32 bandwidth-capped pseudo-channels). *Mitigation:*
  the MemoryPlanner's channel-spreading is real Tier-2 work, not free bandwidth.
- **Host↔device overhead on small ops.** *Mitigation:* layer-chaining amortizes
  transfer over the whole model; keep weights resident.
- **F-factor unproven — the load-bearing bet.** *Mitigation:* Phase 1's Allo
  measurement gates Phase 2 precisely to test it before the expensive build.

---

## 10. Verification of this document

This is a design artifact, so "verification" is accuracy + actionability:

- **File references resolve.** Every cited symbol was checked against
  `host_training.hpp` (`train_step` :43, `Config` :27), `host_training_xrt.hpp`
  (`XrtBackend` :44), `functional.py` (`.apply` aliases :129/:202/:268/…),
  `audit.py` (`assert_closed`), `optim/__init__.py` (SignSGD / PureTropicalSGD /
  TropicalLion), and `hardware/__init__.py` (`hls_dir`).
- **Additive to the frozen API.** Re-checked against `API_SPEC.md`: the §4
  abstraction renames/removes nothing public; CPU path and `assert_closed`
  semantics are unchanged.
- **Silicon numbers are sourced**, not invented — they match
  `tikitorch/hardware/README.md` (the §5 table) and the `tikitorch.hardware`
  package docstring.
- **Staging is internally consistent** — the Tier 1 / Tier 2 split in §3–§6 is
  carried through the §8 roadmap and §9 risks.
