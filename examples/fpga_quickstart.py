"""FPGA quickstart: one line to target the board, your training loop unchanged.

``model = tt.fpga.accelerate(model)`` is the headline — analogous to wrapping a
model in ``DistributedDataParallel``. On a machine with an F2 board (and a built
native driver) the whole fused training step runs on silicon; everywhere else it
is a transparent CPU fallback, so this script runs anywhere.

Run:  python examples/fpga_quickstart.py
"""
import warnings

import torch

import tikitorch as tt
import tikitorch.fpga as fpga
import tikitorch.optim as optim

# A model whose shape matches a silicon-validated kernel (2-layer tropical LM,
# V=128 D=128). On a board this binds to `forge_train_step_2layer`.
model = tt.models.TropicalTransformer(
    d_model=128, n_heads=4, n_layers=2, task_type="lm", vocab_size=128,
)
opt = optim.PureTropicalSGD(model.parameters())

print("fpga.is_available() :", fpga.is_available(), "  (False without a board)")
print("device_count()      :", fpga.device_count())

# The kernel catalog already knows this shape (lookup, no hot-path compile):
spec = fpga.KernelCache.default().match(model)
print("matched kernel      :", None if spec is None else f"{spec.entry} (ABI {spec.abi.name}, DSP={spec.dsp})")

# One line — wrap the model. Bound on a board; CPU fallback otherwise.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", tt.backend.BackendFallbackWarning)
    model = fpga.accelerate(model, optimizer=opt)
print("accelerated         :", repr(model))
print("running on silicon  :", model.bound, "  (CPU fallback when False)\n")

# Your ordinary tropical training loop — UNCHANGED whether on CPU or FPGA.
x = torch.randint(0, 128, (4, 32))
y = torch.randint(0, 128, (4, 32))
for step in range(3):
    loss = model.compute_loss(x, y)   # delegates to the wrapped model
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f"step {step}  loss={loss.item():.4f}")

# Or the fused convenience (drives the board when bound, CPU fallback otherwise):
with warnings.catch_warnings():
    warnings.simplefilter("ignore", tt.backend.BackendFallbackWarning)
    fused_loss = model.train_step(x, y, optimizer=opt)
print(f"\ntrain_step (fused/fallback) loss={fused_loss:.4f}")
print("\nSame code, change one line. On an F2 board the loop trains on silicon")
print("at DSP=0; see tikitorch/fpga/BUILD_NATIVE.md to build the native driver.")
