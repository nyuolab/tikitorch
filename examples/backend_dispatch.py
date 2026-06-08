"""The TikiTorch backend seam: logical devices, dispatch, transparent fallback.

Shows the *general* (FPGA-first, but backend-agnostic) machinery that lets
TikiTorch run off-GPU. Runs anywhere — CPU only, no board needed.

Run:  python examples/backend_dispatch.py
"""
import warnings

import torch

import tikitorch as tt
import tikitorch.functional as F

# 1) Which backends are registered, and which can run right now?
print("registered backends :", tt.backend.list_backends())     # ['cpu', 'fpga']
print("available backends  :", tt.backend.available_backends())  # ['cpu'] (no board)

# 2) Logical devices spelled like torch.device — but they never move a tensor;
#    they select which TikiTorch backend an op/model dispatches to.
cpu = tt.device("cpu")
fpga = tt.device("fpga:0")
print("device(cpu)         :", repr(cpu), "| is_cpu:", cpu.is_cpu)
print("device(fpga:0)      :", repr(fpga), "| str:", str(fpga))

# 3) The CPU implementation of every tropical op is auto-registered as the SAME
#    callable functional exports — dispatch on CPU is byte-identical, zero overhead.
linear_cpu = tt.backend.dispatch("tropical_linear", "cpu")
assert linear_cpu is F.tropical_linear
print(f"\n#cpu ops registered : {len(tt.backend.list_ops('cpu'))}")
print("dispatch('tropical_linear','cpu') is functional.tropical_linear:", linear_cpu is F.tropical_linear)

# 4) Transparent fallback: ask for an op on a device with no kernel. You get the
#    correct CPU implementation back, plus a ONE-TIME perf warning. Never an error.
tt.backend._reset_fallback_warnings()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    linear_fpga = tt.backend.dispatch("tropical_linear", "fpga")
    tt.backend.dispatch("tropical_linear", "fpga")  # second time: no new warning
fallbacks = [w for w in caught if issubclass(w.category, tt.backend.BackendFallbackWarning)]
print("\nfallback returns the CPU impl :", linear_fpga is F.tropical_linear)
print("fallback warned exactly once  :", len(fallbacks) == 1)

# 5) is_op_supported is a pure query (no warning, no fallback).
print("is_op_supported(tropical_linear, cpu) :", tt.backend.is_op_supported("tropical_linear", "cpu"))
print("is_op_supported(tropical_linear, fpga):", tt.backend.is_op_supported("tropical_linear", "fpga"))

# 6) Run a real tropical op through the dispatched CPU impl — identical result.
x = torch.randn(4, 8)
w = torch.randn(6, 8)
print("\ntropical_linear via dispatch == direct call :",
      torch.equal(linear_cpu(x, w), F.tropical_linear(x, w)))

print("\nThe seam is general: a new backend (FPGA, a torch-free C/CPU runtime, WASM)")
print("registers ops/models the same way. FPGA is simply the first consumer.")
