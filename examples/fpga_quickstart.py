"""Inspect the FPGA catalog and exercise the fail-closed PyTorch fallback.

This example does not require a board or native library, and it does not claim
native training. Even with a LOCAL host-reference library loaded, no safe runner
is bundled or auto-registered. See ``tikitorch/fpga/BUILD_NATIVE.md`` before
supplying a custom runner. From a source checkout, run it as
``python -m examples.fpga_quickstart`` so imports resolve to that checkout.
"""

import torch

import tikitorch as tt
import tikitorch.fpga as fpga


cache = fpga.KernelCache.default()
key = fpga.LMKernelConfig(V=128, D=128, n_layers=1)
record = cache.lookup("TropicalTransformer", key)

print("Catalog inspection (not a compatibility check):")
if record is None:
    print("  no one-layer LM record")
else:
    print(f"  entry={record.entry!r}, d_ff={record.d_ff}, l_max={record.l_max}")
    print(f"  recorded build-specific DSP field={record.dsp}")

# This stock model shares the compact catalog key but not the full HLS recipe.
# match() checks all model-observable structural properties and returns None on
# ambiguity; the retired one-layer flat-parameter runner is also never eligible.
model = tt.models.TropicalTransformer(
    d_model=128,
    n_heads=4,
    n_layers=1,
    d_ff=512,
    max_seq_len=128,
    use_shift_attn=True,
    dropout=0.0,
    task_type="lm",
    vocab_size=128,
)
matched = cache.match(model)
print(f"full structural match: {matched.entry if matched is not None else None}")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
accelerated = fpga.accelerate(model, optimizer=optimizer)
print(f"native endpoint usable: {fpga.is_available()}")
print(f"native runner bound: {accelerated.bound}")

# Because this model is unbound, train_step uses the inner model's PyTorch path
# and emits one BackendFallbackWarning. It cannot call a native kernel here.
tokens = torch.randint(0, 128, (2, 16))
targets = torch.randint(0, 128, (2, 16))
loss = accelerated.train_step(tokens, targets)
print(f"PyTorch fallback loss: {loss:.6f}")

try:
    fpga.accelerate(model, strict=True)
except fpga.NativeUnavailable as exc:
    print(f"strict mode rejected the incomplete binding: {exc}")

print("\nA custom native binding additionally requires:")
print("  1. a full structural match (catalog lookup alone is insufficient);")
print("  2. runtime shape/dtype/range and named-buffer validation;")
print("  3. native symbol/ABI/buffer and build-manifest verification;")
print("  4. an explicitly registered kernel-specific runner.")
print("LOCAL satisfies only the host-endpoint part of that contract, not board execution.")
