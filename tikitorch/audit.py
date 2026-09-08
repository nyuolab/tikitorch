"""Software audits for multiply-bearing modules and executed PyTorch operations.

TikiTorch distinguishes three boundaries. :func:`assert_closed` is a structural
deny-list scan. :func:`assert_closed_runtime` observes one executed forward path.
:func:`assert_training_closed` observes one concrete forward, loss, backward, and
optimizer step. The latter two classify intercepted eager PyTorch operations; they
do not cover unexecuted branches, compiler rewrites, external kernels, or a routed
hardware implementation.

:func:`assert_closed` walks the module tree and raises if it finds a standard
``torch.nn`` layer known to contain a data-dependent multiply or divide
(``nn.Linear``, ``nn.Conv2d``, ``nn.LayerNorm``, ``nn.MultiheadAttention``, the
cuDNN RNNs, ...). A clean structural scan does not certify custom arithmetic,
the backward pass, or physical zero-DSP behavior.

**Scope (necessary, not sufficient).** This is a *structural* check: it scans for
registered standard ``torch.nn`` modules that contain a data-dependent multiply
or divide. It does NOT trace the forward graph, so it cannot catch a raw ``*`` /
``/`` / ``torch.exp`` / ``@`` / ``einsum`` written directly in a custom module's
``forward`` (a TikiTorch tropical layer must be closed by construction and by its
own unit tests). The denylist covers the common offenders — Linear/Conv/Norm,
the cuDNN RNNs, ``MultiheadAttention``, gated/parametric activations
(``GLU``/``PReLU``), ``EmbeddingBag`` (mean/per-sample-weights), and LRN — but a
clean ``assert_closed`` is a *necessary* condition for zero-DSP, not a proof of
it. For an executed-path check use :func:`assert_closed_runtime`, which
traces a real forward pass (via ``TorchDispatchMode``) and applies a conservative
registry to each eager ATen op — catching common raw ``*`` / ``@`` / ``torch.exp``
arithmetic in a custom ``forward`` that the structural scan cannot see. It treats
literal power-of-two scalars and explicitly traced dyadic tensor operands as
bit-shifts; arbitrary runtime tensors do not become trusted constants merely
because their observed values happen to be powers of two. Operations absent from
the versioned registry are reported as ``unknown-operation`` and make the audit
inconclusive; they are never silently certified. The trace still covers only the
executed eager path, not every possible branch or compiled kernel.

Usage::

    import tikitorch as tt
    from tikitorch.audit import assert_closed, find_closure_violations

    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    assert_closed(cnn)                       # no registered deny-list module

    # A model with a standard nn.Linear head is a *declared* DSP boundary:
    cnn2 = tt.models.TropicalCNN(head_type="linear")
    assert_closed(cnn2, allow={"head"})      # explicitly allow that one module

Note: ``nn.Embedding`` is a pure gather (no multiply) and is NOT flagged.
Dropout / Identity / containers are inert and not flagged. The soft (``tau>0``)
tropical relaxations use ``logsumexp`` (exp) and are training-time aids, not the
deploy path — they are layer *modules* in TikiTorch (not ``torch.nn`` types) so
they pass here; harden with ``tau -> 0`` for the closed deploy graph.

**The ``_zero_dsp_layer`` marker is FORWARD-scoped.** A module may set
``_zero_dsp_layer = True`` to tell the *structural* check to skip it — used by
:class:`tikitorch.nn.DyadicLinear`, whose weights are signed powers of two, so the
forward ``W·x`` has a sign+shift+add export schedule, checked *statically* by
:func:`tikitorch.fpga.dyadic_export.verify_schedule`. Synthesis reports establish
physical resource use. Two things this marker does
*not* claim, and which you must not read into it:

1. It is NOT visible to an eager trace. On a CPU/GPU host the dyadic forward runs a
   real BLAS ``addmm``, so :func:`find_runtime_multiplies` will (correctly, by its
   own rules) report it as ``dense-multiply``. Dyadic forward-closure is the static
   shift-schedule, not the eager op stream — the marker only stops the *structural*
   scan from false-flagging the module.
2. It says NOTHING about the backward. The dyadic *input*-gradient
   ``dL/dx = Wᵀ·dL/dy`` is shift+add (closed), but the *weight*-gradient
   ``dL/dW = dL/dy ⊗ x`` multiplies two arbitrary tensors — a genuine multiply that
   :func:`find_training_multiplies` *does* report. A dyadic layer is therefore
   **forward-zero-DSP but not backward-zero-DSP under autograd**. Backward-zero-DSP
   requires a *training rule* that needs only ``sign(dL/dW) = sign(dL/dy) ⊕ sign(x)``
   (SignSGD's sign-product, no multiply) and updates in the (sign, exponent) domain so
   weights stay powers of two — check that recipe with :func:`assert_training_closed`,
   never infer it from the marker.
"""
import copy
import math
from contextlib import contextmanager

import torch
import torch.nn as nn

__all__ = ["assert_closed", "find_closure_violations", "MULTIPLY_MODULES",
           "find_runtime_multiplies", "assert_closed_runtime",
           "find_training_multiplies", "assert_training_closed", "closure_report",
           "assert_no_multiply", "closed", "ClosureViolation",
           "AUDIT_SEMANTICS_VERSION", "AUDIT_REPORT_SCHEMA_VERSION"]


# These identifiers are part of the evidence surface.  Increment the semantics
# version whenever an operation can move between certified, rejected, and
# inconclusive; increment the report schema when the serialized shape changes.
AUDIT_SEMANTICS_VERSION = "strict-eager-v3"
AUDIT_REPORT_SCHEMA_VERSION = "tikitorch.closure-report.v2"
_AUDIT_COVERAGE_SCHEMA_VERSION = "tikitorch.audit-coverage.v1"

_SAFE = "safe"
_UNSAFE = "unsafe"
_UNKNOWN = "unknown"
_PENDING = "pending"

# Standard torch modules that contain a data-dependent hardware multiply/divide.
MULTIPLY_MODULES = tuple(
    m for m in (
        getattr(nn, "Linear", None), getattr(nn, "LazyLinear", None),
        getattr(nn, "Bilinear", None),
        getattr(nn, "Conv1d", None), getattr(nn, "Conv2d", None), getattr(nn, "Conv3d", None),
        getattr(nn, "ConvTranspose1d", None), getattr(nn, "ConvTranspose2d", None),
        getattr(nn, "ConvTranspose3d", None),
        getattr(nn, "LazyConv1d", None), getattr(nn, "LazyConv2d", None), getattr(nn, "LazyConv3d", None),
        getattr(nn, "LayerNorm", None), getattr(nn, "RMSNorm", None),
        getattr(nn, "BatchNorm1d", None), getattr(nn, "BatchNorm2d", None),
        getattr(nn, "BatchNorm3d", None), getattr(nn, "GroupNorm", None),
        getattr(nn, "InstanceNorm1d", None), getattr(nn, "InstanceNorm2d", None),
        getattr(nn, "InstanceNorm3d", None),
        getattr(nn, "LazyBatchNorm1d", None), getattr(nn, "LazyBatchNorm2d", None),
        getattr(nn, "LazyBatchNorm3d", None),
        getattr(nn, "LazyInstanceNorm1d", None), getattr(nn, "LazyInstanceNorm2d", None),
        getattr(nn, "LazyInstanceNorm3d", None),
        getattr(nn, "LocalResponseNorm", None), getattr(nn, "CrossMapLRN2d", None),
        # gated/parametric activations: a * sigmoid(b) / weight * x (data-dep multiply)
        getattr(nn, "GLU", None), getattr(nn, "PReLU", None),
        # embedding-bag mean / per-sample-weights multiply (mode='sum' is a pure gather,
        # but we cannot see the mode from the type, so flag it conservatively)
        getattr(nn, "EmbeddingBag", None),
        getattr(nn, "MultiheadAttention", None),
        getattr(nn, "RNN", None), getattr(nn, "LSTM", None), getattr(nn, "GRU", None),
        getattr(nn, "RNNCell", None), getattr(nn, "LSTMCell", None), getattr(nn, "GRUCell", None),
    )
    if m is not None
)


def find_closure_violations(model: nn.Module):
    """Return ``[(module_path, module_type), ...]`` for every multiply-bearing module.

    An empty list means no registered deny-list module was found. It does not
    inspect raw custom arithmetic, a backward pass, or a physical build.
    """
    out = []
    for name, m in model.named_modules():
        # _zero_dsp_layer is FORWARD-scoped (DyadicLinear: pow2 weights -> shift+add
        # forward, proven by fpga.dyadic_export.verify_schedule). It does NOT assert a
        # multiply-free backward — that is a training-rule property checked by
        # assert_training_closed. See the module docstring.
        if isinstance(m, MULTIPLY_MODULES) and not getattr(m, "_zero_dsp_layer", False):
            out.append((name or "<root>", type(m).__name__))
    return out


# ATen operations whose implementation necessarily contains multiplication,
# division, or a transcendental.  This list is intentionally broader than the
# structural module deny-list: functional and fused kernels otherwise disappear
# behind a single dispatch event (for example ``F.cross_entropy`` becomes
# ``_log_softmax`` + ``nll_loss_forward``).
_DENSE_MUL = {
    "mm", "bmm", "addmm", "matmul", "linear", "einsum", "addmv", "addr",
    "mv", "dot", "vdot", "inner", "outer", "ger", "tensordot",
    "convolution", "convolution_backward", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "addbmm",
    "baddbmm", "chain_matmul", "linalg_multi_dot",
}
_TRANSCENDENTAL = {
    "exp", "expm1", "log", "log10", "log1p", "pow", "float_power",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh",
    "tanh", "sigmoid", "erf", "erfc", "erfinv", "sqrt", "rsqrt",
    "reciprocal", "logaddexp", "logaddexp2", "logsumexp", "lgamma", "digamma",
    "polygamma", "mvlgamma", "igamma", "igammac",
}
_SOFTMAX_OPS = {
    "softmax", "_softmax", "_safe_softmax", "special_softmax",
    "log_softmax", "_log_softmax", "special_log_softmax",
    "_softmax_backward_data", "_log_softmax_backward_data",
}
_FUSED_LOSS_OPS = {
    "cross_entropy_loss", "nll_loss", "nll_loss2d", "nll_loss_forward",
    "nll_loss2d_forward", "nll_loss_backward", "nll_loss2d_backward",
    "mse_loss", "mse_loss_backward", "binary_cross_entropy",
    "binary_cross_entropy_backward", "binary_cross_entropy_with_logits",
    "smooth_l1_loss", "smooth_l1_loss_backward", "huber_loss",
    "huber_loss_backward", "kl_div", "poisson_nll_loss", "cosine_embedding_loss",
    "hinge_embedding_loss", "margin_ranking_loss", "multi_margin_loss",
    "multilabel_margin_loss", "multilabel_soft_margin_loss", "triplet_margin_loss",
}
_FUSED_NORMALIZATION_OPS = {
    "native_layer_norm", "native_layer_norm_backward", "layer_norm",
    "native_batch_norm", "native_batch_norm_backward", "batch_norm",
    "_native_batch_norm_legit", "_native_batch_norm_legit_no_training",
    "native_group_norm", "native_group_norm_backward", "group_norm",
    "instance_norm", "rms_norm",
}
_FUSED_ACTIVATION_OPS = {
    "gelu", "gelu_backward", "silu", "silu_backward", "mish", "mish_backward",
    "elu", "elu_backward", "celu", "selu",
    "glu", "glu_backward", "prelu", "prelu_backward", "rrelu_with_noise",
    "rrelu_with_noise_backward", "leaky_relu", "leaky_relu_backward",
    "hardsigmoid", "hardsigmoid_backward", "hardswish", "hardswish_backward",
}
_ARITHMETIC_REDUCTIONS = {
    "prod", "cumprod", "var", "var_mean", "std", "std_mean", "norm",
    "frobenius_norm", "nuclear_norm", "linalg_vector_norm", "linalg_matrix_norm",
    "renorm", "dist", "pdist", "cdist", "cosine_similarity",
}
_AVERAGING_OPS = {
    "avg_pool1d", "avg_pool2d", "avg_pool3d", "avg_pool2d_backward",
    "avg_pool3d_backward", "adaptive_avg_pool1d", "adaptive_avg_pool2d",
    "adaptive_avg_pool3d", "adaptive_avg_pool2d_backward",
    "adaptive_avg_pool3d_backward", "fractional_max_pool2d",
}
_FUSED_ATTENTION_OPS = {
    "_native_multi_head_attention", "_transformer_encoder_layer_fwd",
    "_scaled_dot_product_flash_attention", "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_efficient_attention", "_scaled_dot_product_cudnn_attention",
    "_flash_attention_forward", "_efficient_attention_forward",
}
_FUSED_EMBEDDING_OPS = {
    "embedding_bag", "_embedding_bag", "_embedding_bag_forward_only",
    "_embedding_bag_backward", "_embedding_bag_dense_backward",
    "_embedding_bag_per_sample_weights_backward",
}
_INTERPOLATION_ARITHMETIC = {
    "upsample_linear1d", "upsample_linear1d_backward",
    "upsample_bilinear2d", "upsample_bilinear2d_backward",
    "_upsample_bilinear2d_aa", "_upsample_bilinear2d_aa_backward",
    "upsample_trilinear3d", "upsample_trilinear3d_backward",
    "upsample_bicubic2d", "upsample_bicubic2d_backward",
    "_upsample_bicubic2d_aa", "_upsample_bicubic2d_aa_backward",
}
_LINALG_ARITHMETIC = {
    "solve", "solve_ex", "triangular_solve", "lu_solve", "cholesky_solve",
    "inverse", "pinverse", "matrix_exp", "matrix_power",
}
_CLAMP_OPS = {
    "clamp", "clamp_", "clamp_min", "clamp_min_", "clamp_max", "clamp_max_",
}
_PROVENANCE_PASSTHROUGH = {
    "alias", "detach", "clone", "contiguous", "view", "reshape", "_reshape_alias",
    "squeeze", "squeeze_", "unsqueeze", "unsqueeze_", "flatten", "permute",
    "transpose", "t", "mT", "mH", "expand", "expand_as", "repeat", "tile",
    "slice", "select", "narrow", "index", "gather", "take", "take_along_dim",
    "to", "_to_copy", "type_as", "neg", "negative",
}

# Operations admitted by the eager software certificate.  This is deliberately
# an allow-list rather than the complement of the arithmetic deny-lists above:
# a newly introduced ATen, special, linalg, or fused operator must remain
# inconclusive until its semantics have been reviewed.  The entries below are
# limited to data movement/construction, selection, addition/reduction,
# comparison, integer/bit logic, or route-based reverse operations.
_SAFE_ATEN_OPS = _PROVENANCE_PASSTHROUGH | _CLAMP_OPS | {
    # Tensor construction and state/data movement.
    "empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided",
    "zeros", "zeros_like", "new_zeros", "ones", "ones_like", "new_ones",
    "full", "full_like", "new_full", "scalar_tensor", "lift", "lift_fresh",
    "eye", "arange", "rand", "rand_like", "randint", "randint_like", "randperm",
    "bernoulli", "bernoulli_", "copy", "copy_", "fill", "fill_", "zero", "zero_",
    "resize", "resize_", "resize_as_", "set_", "as_strided", "as_strided_",
    "_unsafe_view", "cat", "concat", "concatenate", "stack", "hstack", "vstack",
    "dstack", "split", "split_with_sizes", "tensor_split", "chunk", "unbind",
    "unfold", "roll", "flip", "fliplr", "flipud", "rot90", "diagonal", "diag",
    "diag_embed", "triu", "tril", "constant_pad_nd", "reflection_pad1d",
    "reflection_pad2d", "reflection_pad3d", "replication_pad1d", "replication_pad2d",
    "replication_pad3d", "circular_pad_nd", "im2col", "col2im", "pixel_shuffle",
    "pixel_unshuffle", "channel_shuffle",
    # Addressed reads/writes and route replay.
    "index", "_unsafe_index", "index_select", "index_put", "index_put_",
    "_index_put_impl_", "index_copy", "index_copy_", "index_fill", "index_fill_",
    "gather", "take", "take_along_dim", "put_",
    "masked_fill", "masked_fill_", "masked_scatter", "masked_scatter_",
    "masked_select", "scatter_add", "scatter_add_",
    "slice_scatter", "select_scatter", "diagonal_scatter", "as_strided_scatter",
    "slice_backward", "select_backward", "diagonal_backward", "as_strided_backward",
    "nonzero", "nonzero_static", "argwhere", "embedding",
    # Comparison, selection, hard activations, and discrete route construction.
    "eq", "ne", "lt", "le", "gt", "ge", "equal", "where", "maximum",
    "minimum", "fmax", "fmin", "relu", "relu_", "threshold", "threshold_backward",
    "hardtanh", "hardtanh_", "hardtanh_backward", "abs", "absolute", "sign",
    "sgn", "signbit", "frexp", "round", "round_", "floor", "floor_", "ceil", "ceil_",
    "trunc", "trunc_", "fix", "logical_and", "logical_or", "logical_xor",
    "logical_not", "bitwise_and", "bitwise_and_", "bitwise_or", "bitwise_or_",
    "bitwise_xor", "bitwise_xor_", "bitwise_not", "bitwise_not_",
    "bitwise_left_shift", "bitwise_left_shift_", "bitwise_right_shift",
    "bitwise_right_shift_", "isfinite", "isinf", "isnan", "isneginf", "isposinf",
    # Additive and extremum reductions.  Product/statistical/norm reductions are
    # intentionally absent and are rejected above.
    "sum", "nansum", "cumsum", "max", "min", "amax", "amin", "aminmax",
    "argmax", "argmin", "topk", "sort", "argsort", "msort", "kthvalue",
    "median", "nanmedian", "mode", "cummax", "cummin", "any", "all",
    "count_nonzero", "add", "add_", "sub", "sub_", "rsub",
    # Max pooling is compare/route; average pooling remains rejected above.
    "max_pool1d", "max_pool2d", "max_pool3d", "max_pool1d_with_indices",
    "max_pool2d_with_indices", "max_pool3d_with_indices",
    "max_pool1d_with_indices_backward", "max_pool2d_with_indices_backward",
    "max_pool3d_with_indices_backward", "adaptive_max_pool1d",
    "adaptive_max_pool2d", "adaptive_max_pool3d", "adaptive_max_pool2d_backward",
    "adaptive_max_pool3d_backward",
}


def _decision(status, category, name):
    """Return the internal tri-state classification tuple."""
    return status, category, name


def _is_pow2(x) -> bool:
    try:
        ax = abs(float(x))
    except (TypeError, ValueError):
        return False
    if ax == 0.0:
        return True
    if not math.isfinite(ax):
        return False
    lg = math.log2(ax)
    return abs(lg - round(lg)) < 1e-6


def _is_nonzero_pow2(x) -> bool:
    """Whether ``x`` is a finite, nonzero signed power of two."""
    try:
        return float(x) != 0.0 and _is_pow2(x)
    except (TypeError, ValueError):
        return False


def _all_pow2_tensor(t, max_elems: int = 1 << 16) -> bool:
    """Validate a tensor known *by provenance* to be a static literal.

    This helper must never be used to bless an arbitrary runtime tensor.  It is
    only called for ``lift_fresh`` outputs -- tensors visibly constructed inside
    the audited block -- so a value check augments explicit static provenance
    rather than guessing provenance from coincidental runtime values.
    """
    if not isinstance(t, torch.Tensor) or t.dtype == torch.bool:
        return False
    if t.numel() == 0 or t.numel() > max_elems:
        return False
    a = t.detach().abs().to(torch.float64)
    nz = a[a > 0]
    if nz.numel() == 0:
        return True
    lg = torch.log2(nz)
    return bool((lg - lg.round()).abs().max() < 1e-5)


def _literal_dyadic_provenance(t):
    """Describe a freshly lifted dyadic literal without blessing zero division."""
    if not _all_pow2_tensor(t):
        return None
    values = t.detach()
    if bool((values == 0).all()):
        return "zero"
    if bool((values == 0).any()):
        return "dyadic-with-zero"
    return "shift"


def _scaled_dyadic_provenance(meta):
    """Conservatively compose a trusted factor with a literal dyadic scale."""
    if meta in {"gate", "sign", "dyadic-with-zero"}:
        return "dyadic-with-zero"
    if meta == "zero":
        return "zero"
    if meta in {"unit", "shift"}:
        return "shift"
    return None


def _aten_base(func) -> str:
    schema = getattr(func, "_schema", None)
    schema_name = getattr(schema, "name", "")
    if "::" in schema_name:
        return schema_name.split("::", 1)[1]
    name = str(func)
    return name[len("aten."):].split(".")[0] if name.startswith("aten.") else name


def _operator_namespace(func) -> str:
    """Return an operator's dispatcher namespace without trusting its basename."""
    namespace = getattr(func, "namespace", None)
    if isinstance(namespace, str):
        return namespace
    schema_name = getattr(getattr(func, "_schema", None), "name", "")
    if "::" in schema_name:
        return schema_name.split("::", 1)[0]
    name = str(func)
    return "aten" if name.startswith("aten.") else ""


def _aten_overload(func) -> str:
    return str(getattr(getattr(func, "_schema", None), "overload_name", ""))


def _argument(args, kwargs, position, name, default=None):
    """Read a dispatcher argument in either positional or keyword form."""
    if len(args) > position:
        return args[position]
    return (kwargs or {}).get(name, default)


def _walk_values(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _value_tensor_metadata(*values):
    """Collect dtype/device strings from arbitrarily nested tensor values."""
    tensors = [
        value
        for value in _walk_values(values)
        if isinstance(value, torch.Tensor)
    ]
    return {
        "dtypes": sorted({str(tensor.dtype) for tensor in tensors}),
        "devices": sorted({str(tensor.device) for tensor in tensors}),
    }


def _numeric_tensor_operands(args, kwargs) -> bool:
    values = _walk_values((args, kwargs or {}))
    return any(isinstance(v, torch.Tensor) and v.dtype != torch.bool for v in values)


def _normalise_dims(dims, ndim):
    if dims is None:
        return tuple(range(ndim))
    if isinstance(dims, int):
        dims = (dims,)
    if len(dims) == 0:
        return tuple(range(ndim))
    return tuple(sorted({int(d) % ndim for d in dims}))


def _mean_is_static_shift(args, kwargs) -> bool:
    """Whether an ATen mean divides by a compile-time power-of-two extent."""
    if not args or not isinstance(args[0], torch.Tensor):
        return False
    tensor = args[0]
    dims = None
    if len(args) > 1 and isinstance(args[1], (int, tuple, list)):
        dims = args[1]
    elif kwargs and "dim" in kwargs:
        dims = kwargs["dim"]
    try:
        count = math.prod(tensor.shape[d] for d in _normalise_dims(dims, tensor.dim()))
    except (IndexError, TypeError, ValueError):
        return False
    return count > 0 and _is_pow2(count)


def _scalar_shift(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and _is_pow2(value)


def _scalar_nonzero_shift(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and _is_nonzero_pow2(value)
    )


def _scalar_integer(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value)) and float(value).is_integer()
    except (TypeError, ValueError):
        return False


def _clamp_bounds(base, args, kwargs):
    """Return scalar clamp bounds, using ``None`` for an absent side."""
    kwargs = kwargs or {}
    if base in {"clamp", "clamp_"}:
        lower = args[1] if len(args) > 1 else kwargs.get("min")
        upper = args[2] if len(args) > 2 else kwargs.get("max")
    elif base in {"clamp_min", "clamp_min_"}:
        lower = args[1] if len(args) > 1 else kwargs.get("min")
        upper = None
    else:
        lower = None
        upper = args[1] if len(args) > 1 else kwargs.get("max")
    return lower, upper


def _classify_aten(func, args=(), kwargs=None, provenance=None):
    """Classify one dispatched call as safe, unsafe, pending, or unknown.

    ``provenance`` is supplied by :func:`_make_catcher`.  Tensor-by-tensor
    multiply/divide is a violation unless one operand was *traced* to a trusted
    sign/gate/power-of-two constructor.  In particular, inspecting an arbitrary
    tensor and noticing that today's values happen to be powers of two is never
    sufficient.  Unregistered operations are ``unknown`` rather than implicitly
    safe, so a closure certificate fails closed as the PyTorch operator surface
    evolves.
    """
    kwargs = kwargs or {}
    provenance = provenance or (lambda _tensor: None)
    name = str(func)
    if name.startswith("profiler."):
        return _decision(_SAFE, "instrumentation", name)
    # A safe ATen basename does not confer safety on a custom operator with the
    # same name: its implementation is opaque at this interception boundary.
    if _operator_namespace(func) != "aten":
        return _decision(_UNKNOWN, "unknown-operation", name)
    base = _aten_base(func)
    overload = _aten_overload(func)
    numeric_ops = _numeric_tensor_operands(args, kwargs)

    if base in _DENSE_MUL or base.startswith(("convolution", "cudnn_convolution")):
        return _decision(_UNSAFE, "dense-multiply", name)
    if (base in _FUSED_ATTENTION_OPS or "scaled_dot_product" in base or
            "flash_attention" in base or "efficient_attention" in base):
        return _decision(_UNSAFE, "dense-multiply", name)
    if base in _FUSED_EMBEDDING_OPS or "embedding_bag" in base:
        # Even mode='sum' can take per-sample weights. Treat this opaque fused
        # boundary conservatively instead of inferring safety from one call.
        return _decision(_UNSAFE, "fused-embedding-arithmetic", name)
    if base in _INTERPOLATION_ARITHMETIC:
        return _decision(_UNSAFE, "interpolation-arithmetic", name)
    if base in _LINALG_ARITHMETIC or base.startswith(("linalg_", "_linalg_")):
        return _decision(_UNSAFE, "linalg-arithmetic", name)
    if base in {"native_dropout", "_fused_dropout", "feature_dropout"}:
        return _decision(_UNSAFE, "fused-multiply", name)
    if base in {"randn", "randn_like", "normal", "normal_"}:
        # Gaussian generation is an opaque arithmetic boundary (commonly using
        # transcendental transforms), even when its requested std is dyadic.
        return _decision(_UNSAFE, "random-generation-arithmetic", name)
    if base in {"uniform", "uniform_"}:
        lower = _argument(args, kwargs, 1, "from", 0.0)
        upper = _argument(args, kwargs, 2, "to", 1.0)
        try:
            width = float(upper) - float(lower)
        except (TypeError, ValueError):
            return _decision(_UNKNOWN, "unknown-operation", name)
        if math.isfinite(width) and _is_pow2(width):
            return _decision(_SAFE, "uniform-rng-dyadic-map", name)
        return _decision(_UNSAFE, "random-generation-arithmetic", name)
    if base in _SOFTMAX_OPS or base in _TRANSCENDENTAL:
        return _decision(_UNSAFE, "transcendental", name)
    if base in {"softplus", "softplus_backward"}:
        return _decision(_UNSAFE, "transcendental", name)
    if base == "log2":
        # Deferred by the catcher: round(log2(x)) is an exponent/leading-bit
        # extraction, while a standalone log2 remains a transcendental.
        return _decision(_PENDING, "pending-log2", name)
    if base == "exp2":
        exponent = args[0] if args else None
        if isinstance(exponent, torch.Tensor):
            exponent_meta = provenance(exponent)
            if exponent_meta == "integer" or (
                    isinstance(exponent_meta, tuple) and exponent_meta[0] == "rounded"):
                return _decision(_SAFE, "dyadic-shift", name)
        return _decision(_UNSAFE, "transcendental", name)
    if base in _FUSED_LOSS_OPS:
        return _decision(_UNSAFE, "fused-loss-arithmetic", name)
    if base in _FUSED_NORMALIZATION_OPS:
        return _decision(_UNSAFE, "fused-normalization-arithmetic", name)
    if base in _FUSED_ACTIVATION_OPS:
        return _decision(_UNSAFE, "fused-activation-arithmetic", name)
    if base in _ARITHMETIC_REDUCTIONS or base in _AVERAGING_OPS:
        return _decision(_UNSAFE, "reduction-arithmetic", name)
    if base in {"mean", "nanmean"}:
        if _mean_is_static_shift(args, kwargs):
            return _decision(_SAFE, "static-shift-reduction", name)
        return _decision(_UNSAFE, "elementwise-divide", name)
    if base in {"lerp", "addcmul", "addcdiv", "addmv", "addr"}:
        return _decision(_UNSAFE, "fused-multiply", name)

    # These otherwise additive route operations contain optional arithmetic
    # modes/scales, so their schemas must be inspected before admitting them.
    if base in {"index_add", "index_add_"}:
        alpha = kwargs.get("alpha", 1)
        if _scalar_shift(alpha):
            return _decision(_SAFE, "indexed-add-dyadic-scale", name)
        return _decision(_UNSAFE, "scalar-multiply", name)

    if base in {"scatter", "scatter_", "scatter_reduce", "scatter_reduce_"}:
        reducer = kwargs.get("reduce")
        has_reducer = (
            "reduce" in overload
            or reducer is not None
            or base.startswith("scatter_reduce")
        )
        if base.startswith("scatter_reduce") and reducer is None:
            reducer = _argument(args, kwargs, 4, "reduce")
        if not has_reducer:
            return _decision(_SAFE, "indexed-write", name)
        if reducer in {"add", "sum", "amax", "amin", "max", "min"}:
            return _decision(_SAFE, "indexed-add-or-extremum", name)
        if reducer in {"multiply", "prod"}:
            return _decision(_UNSAFE, "elementwise-multiply", name)
        if reducer == "mean":
            return _decision(_UNSAFE, "elementwise-divide", name)
        return _decision(_UNKNOWN, "unknown-operation", name)

    if base in {"embedding_dense_backward", "embedding_sparse_backward"}:
        scale_by_frequency = _argument(
            args, kwargs, 4, "scale_grad_by_freq", default=None
        )
        if scale_by_frequency is False:
            return _decision(_SAFE, "indexed-add", name)
        if scale_by_frequency is True:
            return _decision(_UNSAFE, "elementwise-divide", name)
        return _decision(_UNKNOWN, "unknown-operation", name)

    # add/sub's ``alpha`` is an implicit multiply.  A literal dyadic alpha is a
    # compile-time shift; arbitrary alpha (including tensor alpha) is not.
    if base in {"add", "add_", "sub", "sub_", "rsub"}:
        alpha = kwargs.get("alpha", 1)
        non_unit_alpha = isinstance(alpha, torch.Tensor) or alpha != 1
        if non_unit_alpha and numeric_ops and not _scalar_shift(alpha):
            return _decision(_UNSAFE, "scalar-multiply", name)

    if "mul" in base:
        if not numeric_ops:
            return _decision(_SAFE, "boolean-logic", name)
        operands = [a for a in _walk_values(args) if isinstance(a, torch.Tensor) or
                    (isinstance(a, (int, float)) and not isinstance(a, bool))]
        if any(_scalar_shift(a) for a in operands):
            return _decision(_SAFE, "dyadic-shift", name)
        tensor_operands = [a for a in operands if isinstance(a, torch.Tensor)]
        trusted = {"shift", "dyadic-with-zero", "zero", "sign", "gate", "unit"}
        if any(provenance(t) in trusted for t in tensor_operands):
            return _decision(_SAFE, "trusted-gate-or-shift", name)
        category = "scalar-multiply" if len(tensor_operands) < 2 else "elementwise-multiply"
        return _decision(_UNSAFE, category, name)

    if "div" in base and numeric_ops:
        divisor = args[1] if len(args) > 1 else kwargs.get("other")
        # Zero is a legal multiplier gate but never a legal divisor. Tensor
        # provenance must prove that every divisor is nonzero.
        if _scalar_nonzero_shift(divisor):
            return _decision(_SAFE, "dyadic-shift", name)
        if isinstance(divisor, torch.Tensor) and provenance(divisor) in {"shift", "unit"}:
            return _decision(_SAFE, "trusted-shift-divisor", name)
        return _decision(_UNSAFE, "elementwise-divide", name)

    if base in _SAFE_ATEN_OPS:
        return _decision(_SAFE, "registered-safe", name)
    return _decision(_UNKNOWN, "unknown-operation", name)


def _dedupe_hits(hits):
    seen, out = set(), []
    for hit in hits:
        if hit not in seen:
            seen.add(hit)
            out.append(hit)
    return out


def _make_catcher(hits, records=None):
    """Return a dispatch mode applying :func:`_classify_aten` to every operation.

    The mode carries only operation-derived provenance.  Arbitrary input values
    are never examined to decide that a tensor multiply is a shift. ``hits``
    receives every unsafe or unknown blocker in the legacy ``(category, op)``
    shape.  When supplied, ``records`` receives one tri-state record per executed
    operation for coverage reporting.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class _Catcher(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self._meta = {}
            self._pending_log2 = {}
            self._next_pending = 0

        def _set_meta(self, value, meta):
            if isinstance(value, torch.Tensor):
                self._meta[id(value)] = (value, meta)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    self._set_meta(item, meta)

        def _clear_meta(self, value):
            if isinstance(value, torch.Tensor):
                entry = self._meta.get(id(value))
                if entry is not None and entry[0] is value:
                    del self._meta[id(value)]
            elif isinstance(value, (tuple, list)):
                for item in value:
                    self._clear_meta(item)

        def _get_meta(self, value):
            if not isinstance(value, torch.Tensor):
                return None
            entry = self._meta.get(id(value))
            return entry[1] if entry is not None and entry[0] is value else None

        def _pending_ids(self, value):
            meta = self._get_meta(value)
            if isinstance(meta, tuple) and meta and meta[0] == "pending-log2":
                return {meta[1]}
            return set()

        def _note_pending_uses(self, base, args, kwargs):
            pending = set()
            for value in tuple(args) + tuple((kwargs or {}).values()):
                pending.update(self._pending_ids(value))
            if not pending:
                return
            # Clamp is comparison-only and may safely sit between log2 and the
            # integerising round that resolves an exponent extraction. Its
            # output provenance is handled more narrowly below.
            safe_passthrough = base in _PROVENANCE_PASSTHROUGH or base in _CLAMP_OPS
            integerise = base in {"round", "floor", "ceil", "trunc", "fix"}
            for ident in pending:
                record = self._pending_log2[ident]
                if integerise:
                    record["resolved"] = True
                elif not safe_passthrough:
                    record["unsafe"] = True

        def _record_provenance(self, base, args, kwargs, result):
            first = args[0] if args else None
            first_meta = self._get_meta(first)
            tensor_metas = [
                self._get_meta(value)
                for value in args
                if isinstance(value, torch.Tensor)
            ]
            # In-place operations return the input object. Clear its old tag
            # before recording the result so provenance-changing operations such
            # as clamp_ cannot retain a stale shift certificate.
            self._clear_meta(result)

            if base == "log2":
                ident = self._next_pending
                self._next_pending += 1
                self._pending_log2[ident] = {
                    "op": ("transcendental", "aten.log2.default"),
                    "resolved": False,
                    "unsafe": False,
                }
                self._set_meta(result, ("pending-log2", ident))
                return
            if base in {"round", "floor", "ceil", "trunc", "fix"}:
                # Record the source identity too, so the standard straight-through
                # expression x + (round(x) - x).detach() is recognised exactly.
                self._set_meta(result, ("rounded", id(first)))
                return
            if base == "frexp" and isinstance(result, (tuple, list)) and len(result) == 2:
                # IEEE mantissa/exponent decomposition maps to field extraction.
                # Only the exponent is granted integer provenance; the mantissa
                # remains an ordinary runtime tensor and cannot bless a later
                # tensor multiply as a shift.
                self._set_meta(result[1], "integer")
                return
            if base in _CLAMP_OPS:
                lower, upper = _clamp_bounds(base, args, kwargs)
                bounds = [bound for bound in (lower, upper) if bound is not None]
                if (isinstance(first_meta, tuple) and first_meta and
                        first_meta[0] == "pending-log2"):
                    self._set_meta(result, first_meta)
                elif (first_meta == "integer" or (
                        isinstance(first_meta, tuple) and first_meta[0] == "rounded")):
                    if bounds and all(_scalar_integer(bound) for bound in bounds):
                        self._set_meta(result, "integer")
                elif first_meta in {"shift", "dyadic-with-zero", "zero"}:
                    if bounds and all(_scalar_shift(bound) for bound in bounds):
                        meta = first_meta
                        if any(float(bound) == 0.0 for bound in bounds):
                            meta = "dyadic-with-zero"
                        self._set_meta(result, meta)
                return
            if base in _PROVENANCE_PASSTHROUGH:
                self._set_meta(result, first_meta)
                return
            if base in {"ge", "gt", "le", "lt", "eq", "ne", "logical_and",
                        "logical_or", "logical_not", "bernoulli", "bernoulli_"}:
                self._set_meta(result, "gate")
                return
            if base in {"sign", "sgn", "signbit"}:
                self._set_meta(result, "sign")
                return
            if base in {"ones", "ones_like"}:
                self._set_meta(result, "unit")
                return
            if base == "one_hot":
                self._set_meta(result, "gate")
                return
            if base in {"zeros", "zeros_like"}:
                self._set_meta(result, "zero")
                return
            if base == "lift_fresh" and isinstance(first, torch.Tensor):
                meta = _literal_dyadic_provenance(first)
                if meta is not None:
                    self._set_meta(result, meta)
                return
            if base == "full":
                fill = args[1] if len(args) > 1 else kwargs.get("fill_value")
                if _scalar_nonzero_shift(fill):
                    self._set_meta(result, "shift")
                elif isinstance(fill, (int, float)) and float(fill) == 0.0:
                    self._set_meta(result, "zero")
                return
            if base in {"mul", "mul_", "multiply", "multiply_"}:
                scalars = [
                    value for value in args
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                if len(tensor_metas) == 1 and scalars and _scalar_shift(scalars[0]):
                    meta = _scaled_dyadic_provenance(tensor_metas[0])
                    if meta is not None:
                        self._set_meta(result, "zero" if float(scalars[0]) == 0.0 else meta)
                return
            if base in {"div", "div_", "true_divide", "true_divide_"}:
                divisor = args[1] if len(args) > 1 else kwargs.get("other")
                if len(tensor_metas) == 1 and _scalar_nonzero_shift(divisor):
                    meta = _scaled_dyadic_provenance(tensor_metas[0])
                    if meta is not None:
                        self._set_meta(result, meta)
                return
            if base in {"index", "select", "slice", "gather", "take", "take_along_dim"}:
                self._set_meta(result, first_meta)
                return
            if base == "sub" and len(args) > 1:
                lhs_meta = self._get_meta(args[0])
                if (isinstance(lhs_meta, tuple) and lhs_meta[0] == "rounded" and
                        lhs_meta[1] == id(args[1])):
                    self._set_meta(result, ("round-delta", id(args[1])))
                    return
            if base == "add" and len(args) > 1:
                lhs, rhs = args[:2]
                lhs_meta, rhs_meta = self._get_meta(lhs), self._get_meta(rhs)
                if (isinstance(rhs_meta, tuple) and rhs_meta == ("round-delta", id(lhs))) or (
                        isinstance(lhs_meta, tuple) and lhs_meta == ("round-delta", id(rhs))):
                    self._set_meta(result, "integer")
                    return
            if base == "exp2" and (first_meta == "integer" or (
                    isinstance(first_meta, tuple) and first_meta[0] == "rounded")):
                self._set_meta(result, "shift")

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            base = _aten_base(func)
            self._note_pending_uses(base, args, kwargs)
            status, category, op_name = _classify_aten(
                func, args, kwargs, self._get_meta
            )
            record = {"status": status, "category": category, "op": op_name}
            result = func(*args, **kwargs)
            record.update(_value_tensor_metadata(args, kwargs, result))
            self._record_provenance(base, args, kwargs, result)
            if status == _PENDING:
                meta = self._get_meta(result)
                if isinstance(meta, tuple) and meta and meta[0] == "pending-log2":
                    self._pending_log2[meta[1]]["coverage_record"] = record
            elif status in {_UNSAFE, _UNKNOWN}:
                hits.append((category, op_name))
            if records is not None:
                records.append(record)
            return result

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                for record in self._pending_log2.values():
                    if record["unsafe"] or not record["resolved"]:
                        hits.append(record["op"])
                        coverage_record = record.get("coverage_record")
                        if coverage_record is not None:
                            coverage_record.update(
                                status=_UNSAFE, category="transcendental"
                            )
                    else:
                        coverage_record = record.get("coverage_record")
                        if coverage_record is not None:
                            coverage_record.update(
                                status=_SAFE, category="integer-exponent-extraction"
                            )
            finally:
                return super().__exit__(exc_type, exc_value, traceback)

    return _Catcher()


_MODULE_INFRASTRUCTURE_ATTRS = {
    "training", "_parameters", "_buffers", "_non_persistent_buffers_set",
    "_modules", "_backward_pre_hooks", "_backward_hooks",
    "_is_full_backward_hook", "_forward_hooks", "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called", "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs", "_state_dict_hooks",
    "_state_dict_pre_hooks", "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
}


def _is_plain_data(value):
    if isinstance(value, (type(None), bool, int, float, complex, str, bytes,
                          torch.dtype, torch.device)):
        return True
    if isinstance(value, (tuple, list, set, frozenset)):
        return all(_is_plain_data(item) for item in value)
    if isinstance(value, dict):
        return all(_is_plain_data(key) and _is_plain_data(item)
                   for key, item in value.items())
    return isinstance(value, bytearray)


def _snapshot_attribute(value):
    """Capture common mutable configuration/cache values without copying modules."""
    if isinstance(value, torch.Tensor):
        try:
            return "tensor", value, value.detach().clone(memory_format=torch.preserve_format)
        except (RuntimeError, ValueError):
            return "reference", value, None
    if isinstance(value, (dict, list, set, bytearray)) and _is_plain_data(value):
        try:
            return type(value).__name__, value, copy.deepcopy(value)
        except (RuntimeError, TypeError, ValueError):
            return "reference", value, None
    return "reference", value, None


def _snapshot_attributes(owner, excluded=()):
    original_keys = set(owner.__dict__)
    values = {
        key: _snapshot_attribute(value)
        for key, value in owner.__dict__.items()
        if key not in excluded
    }
    return original_keys, values


def _restore_attributes(owner, snapshot):
    original_keys, values = snapshot
    for key in set(owner.__dict__) - original_keys:
        del owner.__dict__[key]
    for key, (kind, original, saved) in values.items():
        if kind == "tensor":
            try:
                _restore_tensor_value(original, saved)
                restored = original
            except (RuntimeError, ValueError):
                restored = saved
        elif kind == "dict":
            original.clear()
            original.update(copy.deepcopy(saved))
            restored = original
        elif kind == "list":
            original[:] = copy.deepcopy(saved)
            restored = original
        elif kind == "set":
            original.clear()
            original.update(copy.deepcopy(saved))
            restored = original
        elif kind == "bytearray":
            original[:] = saved
            restored = original
        else:
            restored = original
        owner.__dict__[key] = restored


def _snapshot_module_registries(module):
    """Capture registration mappings while retaining their object identities."""
    snapshots = []
    for key in ("_parameters", "_buffers", "_modules"):
        registry = module.__dict__.get(key)
        if registry is not None:
            snapshots.append((key, registry, dict(registry)))
    nonpersistent = module.__dict__.get("_non_persistent_buffers_set")
    if nonpersistent is not None:
        snapshots.append(("_non_persistent_buffers_set", nonpersistent,
                          set(nonpersistent)))
    return snapshots


def _restore_module_registries(module, snapshots):
    for key, original, saved in snapshots:
        original.clear()
        original.update(saved)
        module.__dict__[key] = original


def _tensor_metadata(tensor):
    stride = tensor.stride() if tensor.layout == torch.strided else None
    return tensor.shape, stride, tensor.dtype, tensor.device, tensor.layout


def _restore_tensor_value(tensor, saved):
    """Restore values and shape while retaining a parameter/buffer Python object."""
    if _tensor_metadata(tensor) == _tensor_metadata(saved):
        tensor.copy_(saved)
    else:
        # Resizing via ``set_`` keeps the Parameter/Tensor object held by an
        # optimizer or caller, while replacing storage restores shape/strides.
        try:
            tensor.set_(saved)
        except RuntimeError:
            tensor.data = saved


def _uninitialized_state_names(model, parameters, buffers):
    from torch.nn.parameter import is_lazy

    names = []
    parameter_names = {
        id(value): f"parameter {name!r}"
        for name, value in model.named_parameters()
    } if hasattr(model, "named_parameters") else {}
    buffer_names = {
        id(value): f"buffer {name!r}"
        for name, value in model.named_buffers()
    } if hasattr(model, "named_buffers") else {}
    for value in parameters:
        if is_lazy(value):
            names.append(parameter_names.get(id(value), "an optimizer parameter"))
    for value in buffers:
        if is_lazy(value):
            names.append(buffer_names.get(id(value), "an uninitialized buffer"))
    return names


@contextmanager
def _preserve_audit_state(model, optimizer=None, extra_tensors=()):
    """Rollback observable state touched by an executed-path audit."""
    modules = list(model.modules()) if hasattr(model, "modules") else []
    modes = [(module, module.training) for module in modules]

    parameters = list(model.parameters()) if hasattr(model, "parameters") else []
    if optimizer is not None:
        for group in optimizer.param_groups:
            parameters.extend(group.get("params", ()))
    parameters = list({id(parameter): parameter for parameter in parameters}.values())
    buffers = list(model.buffers()) if hasattr(model, "buffers") else []
    lazy_names = _uninitialized_state_names(model, parameters, buffers)
    if lazy_names:
        details = ", ".join(lazy_names)
        raise RuntimeError(
            "Executed closure audits cannot safely materialize lazy modules. "
            f"Initialize them with an ordinary forward pass first ({details})."
        )

    module_attributes = [
        (module, _snapshot_attributes(module, _MODULE_INFRASTRUCTURE_ATTRS))
        for module in modules
    ]
    module_registries = [
        (module, _snapshot_module_registries(module))
        for module in modules
    ]

    parameter_state = []
    for parameter in parameters:
        value = parameter.detach().clone(memory_format=torch.preserve_format)
        grad = parameter.grad
        grad_value = grad.detach().clone() if grad is not None else None
        parameter_state.append((parameter, value, grad, grad_value))

    buffer_state = [
        (buffer, buffer.detach().clone(memory_format=torch.preserve_format))
        for buffer in buffers
    ]
    managed_ids = {id(value) for value in parameters + buffers}
    extra_tensor_state = []
    seen_extra = set()
    for tensor in _walk_values(extra_tensors):
        if (not isinstance(tensor, torch.Tensor) or id(tensor) in managed_ids or
                id(tensor) in seen_extra):
            continue
        seen_extra.add(id(tensor))
        value = tensor.detach().clone(memory_format=torch.preserve_format)
        tracks_grad = tensor.is_leaf or (
            tensor.requires_grad and getattr(tensor, "retains_grad", False)
        )
        grad = tensor.grad if tracks_grad else None
        grad_value = grad.detach().clone() if grad is not None else None
        extra_tensor_state.append((tensor, value, tracks_grad, grad, grad_value))

    optimizer_state = copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
    optimizer_attributes = (
        _snapshot_attributes(optimizer, {"state", "param_groups"})
        if optimizer is not None else None
    )

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        with torch.no_grad():
            for parameter, value, original_grad, grad_value in parameter_state:
                _restore_tensor_value(parameter, value)
                parameter.grad = original_grad
                if original_grad is not None and grad_value is not None:
                    original_grad.copy_(grad_value)
            for buffer, value in buffer_state:
                _restore_tensor_value(buffer, value)
            for tensor, value, tracks_grad, original_grad, grad_value in extra_tensor_state:
                _restore_tensor_value(tensor, value)
                if tracks_grad:
                    tensor.grad = original_grad
                    if original_grad is not None and grad_value is not None:
                        original_grad.copy_(grad_value)
            for module, snapshot in module_attributes:
                _restore_attributes(module, snapshot)
            for module, snapshots in module_registries:
                _restore_module_registries(module, snapshots)
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
            _restore_attributes(optimizer, optimizer_attributes)
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _execution_tensor_metadata(model, *values):
    """Return stable dtype/device coverage for an executed audit boundary."""
    tensors = []
    seen = set()

    def add(value):
        if isinstance(value, torch.Tensor) and id(value) not in seen:
            seen.add(id(value))
            tensors.append(value)

    for value in _walk_values(values):
        add(value)
    try:
        for value in model.parameters():
            add(value)
        for value in model.buffers():
            add(value)
    except (AttributeError, RuntimeError, ValueError):
        # The execution path will still raise for unsafe lazy/unmaterialized
        # models. Metadata collection must not mask that more useful error.
        pass
    return {
        "dtypes": sorted({str(t.dtype) for t in tensors}),
        "devices": sorted({str(t.device) for t in tensors}),
    }


def _coverage_summary(records, model, values, *, mode, harden_tau):
    unsafe = sorted({r["op"] for r in records if r["status"] == _UNSAFE})
    unknown = sorted({r["op"] for r in records if r["status"] == _UNKNOWN})
    safe = sorted({r["op"] for r in records if r["status"] == _SAFE})
    if unsafe:
        status = "violated"
    elif unknown:
        status = "inconclusive"
    else:
        status = "closed"
    counts = {
        key: sum(r["status"] == key for r in records)
        for key in (_SAFE, _UNSAFE, _UNKNOWN)
    }
    tensor_metadata = _execution_tensor_metadata(model, values)
    observed_dtypes = {
        dtype for record in records for dtype in record.get("dtypes", ())
    }
    observed_devices = {
        device for record in records for device in record.get("devices", ())
    }
    return {
        "schema_version": _AUDIT_COVERAGE_SCHEMA_VERSION,
        "audit_semantics": AUDIT_SEMANTICS_VERSION,
        "scope": "executed-eager-path",
        "mode": mode,
        "harden_tau": bool(harden_tau),
        "harden_tau_applied": bool(harden_tau and hasattr(model, "set_tau")),
        "dtypes": sorted(set(tensor_metadata["dtypes"]) | observed_dtypes),
        "devices": sorted(set(tensor_metadata["devices"]) | observed_devices),
        "status": status,
        "certified": status == "closed",
        "observed_operator_calls": len(records),
        "unique_operators": len({r["op"] for r in records}),
        "safe_operator_calls": counts[_SAFE],
        "unsafe_operator_calls": counts[_UNSAFE],
        "unknown_operator_calls": counts[_UNKNOWN],
        "safe_operators": safe,
        "unsafe_operators": unsafe,
        "unknown_operators": unknown,
    }


def _runtime_audit(model, example_inputs, *, run_eval=True, harden_tau=True):
    hits, records = [], []
    with _preserve_audit_state(model, extra_tensors=example_inputs):
        if run_eval and hasattr(model, "eval"):
            model.eval()
        if harden_tau and hasattr(model, "set_tau"):
            model.set_tau(0.0)
        with torch.no_grad(), _make_catcher(hits, records):
            model(*example_inputs)
    coverage = _coverage_summary(
        records, model, example_inputs,
        mode="eval" if run_eval else "current", harden_tau=harden_tau,
    )
    return _dedupe_hits(hits), coverage


def find_runtime_multiplies(model, *example_inputs, run_eval=True, harden_tau=True):
    """Trace a forward pass and return the multiply/transcendental aten ops it runs.

    The static :func:`assert_closed` only inspects *registered modules*, so it
    cannot see a raw ``*`` / ``@`` / ``torch.exp`` written in a custom ``forward``
    (the gap that let a ``rope_relative`` shift hide a real multiply). This runs
    the model on ``example_inputs`` under a ``TorchDispatchMode`` and classifies
    recognized ATen ops as ``dense-multiply`` (mm/conv/...),
    ``elementwise-multiply`` (tensor*tensor), ``elementwise-divide``,
    ``scalar-multiply`` (flagged only if the scalar is NOT a power of two -> a true
    multiply, vs a shift), or ``transcendental`` (exp/log/... -- the soft tau>0
    relaxations; absent once annealed to tau=0).

    By default the model is put in ``eval()`` and annealed to a hard ``tau=0``
    candidate path first (``harden_tau``). Returns ``[(category, op_name), ...]``.
    For compatibility the function retains its historical name and tuple shape,
    but the result now includes ``("unknown-operation", op_name)`` for any
    unregistered dispatched operation. An empty result means every observed
    operation was explicitly classified safe for these inputs.
    """
    hits, _coverage = _runtime_audit(
        model, example_inputs, run_eval=run_eval, harden_tau=harden_tau
    )
    return hits


def assert_closed_runtime(model, *example_inputs, allow_transcendental=False,
                          harden_tau=True):
    """Require no classified multiply-like op on one executed forward path.

    Complements :func:`assert_closed` (structural) with a graph-level trace via
    :func:`find_runtime_multiplies`. Raises ``AssertionError`` naming every
    multiply/divide op actually executed. ``allow_transcendental=True`` permits
    the soft ``tau>0`` exp/log relaxations (set it when checking a soft model);
    by default the model is hardened to ``tau=0`` first, where none should appear.
    Returns the model on success. Passing is scoped to the supplied inputs and
    eager execution; it is not proof about other paths or synthesized resources.
    """
    hits = find_runtime_multiplies(model, *example_inputs, harden_tau=harden_tau)
    if allow_transcendental:
        hits = [(c, op) for c, op in hits if c != "transcendental"]
    if hits:
        lines = "\n".join(f"    {cat}: {op}" for cat, op in hits)
        raise AssertionError(
            "Model is not certified closed on this executed path. Known arithmetic "
            "violates closure and unknown operations make the result inconclusive:\n" +
            lines + "\n\nRewrite known violations as add/compare/shift (or anneal "
            "tau->0 for supported soft relaxations), and classify or remove every "
            "unknown operation before making a closure claim."
        )
    return model


def _training_audit(model, example_inputs, targets, loss_fn, optimizer=None,
                    harden_tau=True):
    if not isinstance(example_inputs, (tuple, list)):
        example_inputs = (example_inputs,)
    else:
        example_inputs = tuple(example_inputs)
    hits, records = [], []
    with _preserve_audit_state(model, optimizer, (example_inputs, targets)):
        if hasattr(model, "train"):
            model.train()
        if harden_tau and hasattr(model, "set_tau"):
            model.set_tau(0.0)
        if optimizer is not None:
            optimizer.zero_grad()
        with _make_catcher(hits, records):
            pred = model(*example_inputs)
            loss = loss_fn(pred, targets)
            loss.backward()
            if optimizer is not None:
                optimizer.step()
    coverage = _coverage_summary(
        records, model, (example_inputs, targets), mode="train",
        harden_tau=harden_tau,
    )
    return _dedupe_hits(hits), coverage


def find_training_multiplies(model, example_inputs, targets, loss_fn, optimizer=None,
                             harden_tau=True):
    """Trace one training step for classified multiply-like operations.

    This is the **Level-B** software check: it observes forward, loss, backward,
    and (optionally) optimizer execution for the supplied example. A soft
    ``tau>0`` relaxation (``exp``), a
    non-closure-faithful loss (cross-entropy = ``softmax``+``log``), or an adaptive
    optimizer (Adam's moments / divide / sqrt) all show up here even though the
    deployed forward is closed. The closure-faithful recipe -- hard tropical
    forward, perceptron / L1 loss (±1 gradients), a sign-only optimizer
    (``SignSGD``/``PureTropicalSGD`` with a power-of-two lr), and power-of-two
    reduction extents -- traces clean.

    ``example_inputs`` is a tuple/list of model args; ``targets`` is passed to
    ``loss_fn(predictions, targets)``. If ``optimizer`` is given, ``.step()`` is
    traced too. The historical tuple shape is retained, but unknown dispatched
    operations are returned as ``("unknown-operation", op_name)`` and prevent
    certification. An empty result means every observed operation was explicitly
    classified safe on this run; it does not predict compiler or synthesis
    resource use.
    """
    hits, _coverage = _training_audit(
        model, example_inputs, targets, loss_fn, optimizer=optimizer,
        harden_tau=harden_tau,
    )
    return hits


def assert_training_closed(model, example_inputs, targets, loss_fn, optimizer=None,
                           harden_tau=True):
    """Require no classified multiply-like op in one full-step eager trace.

    See :func:`find_training_multiplies`. Raises ``AssertionError`` naming every
    observed multiply across forward+loss+backward(+optimizer); returns the model
    on success. A pass is a recipe- and input-scoped software result, not a
    physical zero-DSP certificate.
    """
    hits = find_training_multiplies(model, example_inputs, targets, loss_fn,
                                    optimizer=optimizer, harden_tau=harden_tau)
    if hits:
        lines = "\n".join(f"    {cat}: {op}" for cat, op in hits)
        raise AssertionError(
            "TRAINING step is not certified multiplier-free (Level-B closure is "
            "violated or inconclusive):\n" + lines +
            "\n\nUse a hard tropical forward (no logsumexp/tau>0), a closure-faithful "
            "loss (tropical_perceptron_loss or L1 -- not cross-entropy), and a sign-only "
            "optimizer (SignSGD/PureTropicalSGD with a power-of-two lr); classify or "
            "remove every unknown operation before making a closure claim."
        )
    return model


def closure_report(model, example_inputs=None, targets=None, loss_fn=None,
                   optimizer=None, harden_tau=True):
    """Collect structural and executed-path software audit results.

    Returns a versioned dict with ``structural`` (module-level
    :func:`find_closure_violations`), ``inference`` (executed-forward blockers),
    and ``training`` (full-step blockers). The legacy booleans
    ``inference_closed`` / ``training_closed`` / ``training_zero_dsp`` remain,
    but are true only when every observed operation is explicitly classified
    safe. The corresponding ``*_status`` is tri-state: ``"closed"``,
    ``"violated"``, or ``"inconclusive"``. Unknown operations are exposed under
    ``*_unknown`` and in the versioned ``coverage`` metadata. It does not report
    synthesized DSP utilization.
    This is the "double-check TikiTorch" one-liner::

        tt.closure_report(model, x)                       # structural + inference
        tt.closure_report(model, x, y, loss_fn, opt)      # + Level-B training

    Reminder: training closure requires the *closed recipe* -- hard tropical
    (``tau=0``; the soft logsumexp path is the non-closed trainable relaxation), a
    closure-faithful loss
    (``tropical_perceptron_loss``/L1, not cross-entropy), a sign-only optimizer
    (``PureTropicalSGD``), closure-aware gating such as ``TropicalDropout``, and
    power-of-two reduction extents (including ``d_model`` for normalization and
    batch/token counts for mean-reduced losses). Results remain scoped to the
    executed configuration and inputs.
    """
    structural = find_closure_violations(model)
    structural_tensors = _execution_tensor_metadata(model)
    rep = {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "audit_semantics": AUDIT_SEMANTICS_VERSION,
        "structural": structural,
        "coverage": {
            "structural": {
                "schema_version": _AUDIT_COVERAGE_SCHEMA_VERSION,
                "audit_semantics": AUDIT_SEMANTICS_VERSION,
                "scope": "registered-module-tree",
                "mode": "structural",
                "harden_tau": False,
                "harden_tau_applied": False,
                "dtypes": structural_tensors["dtypes"],
                "devices": structural_tensors["devices"],
                "status": "violated" if structural else "scan-clean",
                "certified": False,
            }
        },
    }
    if example_inputs is not None:
        if not isinstance(example_inputs, (tuple, list)):
            example_inputs = (example_inputs,)
        else:
            example_inputs = tuple(example_inputs)
        inference, inference_coverage = _runtime_audit(
            model, example_inputs, harden_tau=harden_tau
        )
        rep["inference"] = inference
        rep["inference_violations"] = [
            hit for hit in inference if hit[0] != "unknown-operation"
        ]
        rep["inference_unknown"] = [
            hit for hit in inference if hit[0] == "unknown-operation"
        ]
        rep["inference_status"] = inference_coverage["status"]
        rep["inference_inconclusive"] = bool(rep["inference_unknown"])
        rep["inference_closed"] = inference_coverage["certified"]
        rep["coverage"]["inference"] = inference_coverage
    if targets is not None and loss_fn is not None and example_inputs is not None:
        tr, training_coverage = _training_audit(
            model, example_inputs, targets, loss_fn,
            optimizer=optimizer, harden_tau=harden_tau,
        )
        rep["training"] = tr
        rep["training_violations"] = [
            hit for hit in tr if hit[0] != "unknown-operation"
        ]
        rep["training_unknown"] = [
            hit for hit in tr if hit[0] == "unknown-operation"
        ]
        rep["training_status"] = training_coverage["status"]
        rep["training_inconclusive"] = bool(rep["training_unknown"])
        rep["training_closed"] = training_coverage["certified"]
        # A divide is arithmetic too.  Earlier versions silently exempted every
        # elementwise divide here, allowing genuinely non-closed recipes to earn a
        # zero-DSP badge.  Hardware-specific host exclusions belong in an explicit
        # certificate boundary, not this eager whole-step boolean.
        rep["training_zero_dsp"] = training_coverage["certified"]
        rep["coverage"]["training"] = training_coverage
    return rep


class ClosureViolation(AssertionError):
    """Raised when an unsafe or unclassified operation executes in a guard."""


class assert_no_multiply:
    """Guard that rejects unsafe or unclassified ops in a block of YOUR code.

    The fail-fast, wrap-your-own-loop closure check — ``torch.no_grad()``-shaped::

        import tikitorch as tt
        with tt.assert_no_multiply():
            loss = model.compute_loss(x, y)
            loss.backward()
            opt.step()
        # raises ClosureViolation listing unsafe or unknown ATen ops.

    Unlike :func:`assert_closed` (which only inspects *registered modules*), this traces
    eager ATen stream via the same :func:`_make_catcher` ``TorchDispatchMode`` the audit
    helpers use, so it catches common raw ``*`` / ``@`` / ``torch.exp`` arithmetic
    written in custom ``forward`` or training code. An unregistered operation also
    blocks the guard rather than passing by omission. Passing remains scoped to the
    executed path and the versioned auditor registry. Literal power-of-two scalars
    and tensor operands with traced dyadic/sign/gate provenance pass. Arbitrary
    runtime tensor values are never trusted merely because they currently look
    dyadic; dynamic integer multiply/divide is arithmetic too. Boolean logic passes.
    ``allow_transcendental=True`` permits the soft
    ``tau>0`` ``exp``/``log`` relaxations. Tracing has per-op overhead — use it as an
    opt-in guard / test / CI gate, not in the hot training loop.

    Also usable as a decorator via :func:`closed` (preferred) or ``@assert_no_multiply()``.
    """

    def __init__(self, allow_transcendental: bool = False, label: str = None):
        self.allow_transcendental = allow_transcendental
        self.label = label
        self._hits = []
        self._mode = None

    def __enter__(self):
        self._hits = []
        self._mode = _make_catcher(self._hits)
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._mode.__exit__(exc_type, exc, tb)
        if exc_type is not None:
            return False                       # let the user's own exception propagate
        hits = self._hits
        if self.allow_transcendental:
            hits = [(c, op) for c, op in hits if c != "transcendental"]
        seen, uniq = set(), []
        for h in hits:
            if h not in seen:
                seen.add(h)
                uniq.append(h)
        if uniq:
            where = f" in {self.label}" if self.label else ""
            lines = "\n".join(f"    {c}: {op}" for c, op in uniq)
            raise ClosureViolation(
                f"Closure is violated or inconclusive{where}:\n" + lines +
                "\n\nRewrite known arithmetic as add/compare/shift (powers of two are "
                "bit-shifts and pass); anneal tau->0 for supported soft exp/log "
                "relaxations, or classify/remove unknown operations. "
                "allow_transcendental=True waives only registered transcendental ops.")
        return False

    def __call__(self, fn):                    # @assert_no_multiply() decorator form
        import functools

        @functools.wraps(fn)
        def wrapped(*a, **k):
            with assert_no_multiply(self.allow_transcendental, self.label or fn.__name__):
                return fn(*a, **k)
        return wrapped


def closed(fn=None, *, allow_transcendental=False):
    """Decorator requiring the wrapped function's executed ops to certify closed::

        @tt.closed
        def train_step(x, y): ...

        @tt.closed(allow_transcendental=True)     # permit soft tau>0 exp/log
        def soft_forward(x): ...

    Thin wrapper over :class:`assert_no_multiply`; raises :class:`ClosureViolation`
    (naming each blocker) if the call executes registered unsafe or unknown arithmetic.
    """
    if fn is None:
        return lambda f: closed(f, allow_transcendental=allow_transcendental)
    import functools

    @functools.wraps(fn)
    def wrapped(*a, **k):
        with assert_no_multiply(allow_transcendental=allow_transcendental, label=fn.__name__):
            return fn(*a, **k)
    return wrapped


def assert_closed(model: nn.Module, allow=()):
    """Require ``model`` to contain no registered multiply-bearing module.

    Args:
        model: the module to audit.
        allow (Iterable[str]): names (module paths, e.g. ``"head"``) or type names
            (e.g. ``"Linear"``) to treat as *declared* DSP boundaries and skip.

    Returns the model on success (so it can be chained). Raises ``AssertionError``
    naming every violation otherwise. This structural result is necessary but not
    sufficient for an executed-recipe or physical closure claim.
    """
    allow = set(allow)

    def _allowed(path, module):
        if path in allow:
            return True
        # Match allow against the whole MRO so e.g. allow={"Linear"} also skips
        # MultiheadAttention's internal NonDynamicallyQuantizableLinear (a Linear
        # subclass), not just the exact leaf type name.
        return any(base.__name__ in allow for base in type(module).__mro__)

    violations = [
        (path, typ) for path, mod, typ in (
            (p, m, type(m).__name__)
            for p, m in model.named_modules()
            if isinstance(m, MULTIPLY_MODULES) and not getattr(m, "_zero_dsp_layer", False))
        if not _allowed(path, mod)
    ]
    if violations:
        lines = "\n".join(f"    {path}: {typ}" for path, typ in violations)
        raise AssertionError(
            "Model is NOT multiplier-free — closure violated by these standard "
            "torch modules (each contains a hardware multiply/divide):\n" + lines +
            "\n\nReplace with TikiTorch tropical equivalents (TropicalConv2d, "
            "TropicalLayerNorm/TropicalRMSNorm, TropicalDilationModule head, "
            "TropicalAttention, ...), or pass allow={...} to declare a DSP boundary."
        )
    return model
