"""FPGA API quickstart using a local reference bridge or compatible native binding.

No board is needed to exercise the interface. Build the local host-reference
bridge, then run this script:

    eval "$(bash scripts/build_fpga_local.sh --export)"
    python examples/fpga_quickstart.py

Without the bridge the script runs the software audit and prints how to enable
the native path. A local bridge is not FPGA execution or synthesis evidence.
Compatible AWS F2 runners use the same high-level API; see
`tikitorch/fpga/BUILD_NATIVE.md` for backend-specific requirements.
"""
import torch

import tikitorch as tt
import tikitorch.fpga as fpga
from tikitorch.audit import assert_closed, find_runtime_multiplies

# A model whose shape matches the bundled 1-layer LM kernel (V=128, D=128,
# n_layers=1) — the kernel that ships a reference on-board runner, so the model
# actually BINDS (the 2-/3-layer + morph kernels need a runner you supply).
model = tt.models.TropicalTransformer(d_model=128, n_heads=4, n_layers=1, d_ff=512,
                                      max_seq_len=128, task_type="lm", vocab_size=128)

# (1) Properties you can verify with NO hardware ------------------------------
assert_closed(model)                                                   # structural module audit
muls = find_runtime_multiplies(model, torch.randint(0, 128, (1, 64)))  # dynamic: trace a real forward
spec = fpga.KernelCache.default().match(model)                         # the compiled kernel this model maps to
print("closure audit :  assert_closed PASS")
print(f"runtime multiplies in the traced step :  {muls}")
print(f"maps to kernel :  {spec.entry!r}   dsp={spec.dsp}   (d_ff={spec.d_ff}, l_max={spec.l_max})")

# (2) Train on the kernel — LOCAL host-faithful bridge or real F2 silicon ------
# `model.fpga()` is the headline: one line, analogous to DistributedDataParallel.
accel = model.fpga()
if not accel.bound:
    print("\n[CPU fallback] no native bridge/board, so the model runs on CPU.")
    print("To exercise the local host-reference bridge, run:")
    print('    eval "$(bash scripts/build_fpga_local.sh --export)"')
    print("    python examples/fpga_quickstart.py")
    raise SystemExit(0)

print(f"\nbound on kernel {spec.entry!r}  (fpga.is_available()={fpga.is_available()}). training 5 steps:")
flat = lambda: torch.cat([p.detach().reshape(-1) for p in model.parameters()])
before = flat().clone()
tokens = torch.randint(0, 128, (64,))
targets = torch.randint(0, 128, (64,))
for i in range(5):
    loss = accel.train_step(tokens, targets=targets, lr=2, threshold=0)  # the fused fwd+bwd+SignSGD kernel step
    print(f"  step {i}:  kernel loss-proxy = {loss:.4f}")
changed = int((before != flat()).sum())
print(f"\nparams changed through the native binding: {changed}/{before.numel()}")
print("This run does not by itself establish FPGA execution or a physical resource claim.")
