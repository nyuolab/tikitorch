"""TikiTorch — PyTorch-shaped tools for tropical deep learning.

TikiTorch packages the tropical (max-plus / min-plus) closure machinery as a
clean, installable, PyTorch-idiomatic library.  Certified model and optimizer
recipes can close their complete learning loop over comparisons, additions,
integer shifts, routing and data movement.  Ordinary PyTorch boundaries remain
ordinary unless a closure audit explicitly certifies them.

Idiomatic usage::

    import tikitorch as tt
    import tikitorch.nn as nn
    import tikitorch.optim as optim

    model = tt.models.MorphSSM(d_model=128, n_layers=4, task_type="lm", vocab_size=128)
    opt = optim.PureTropicalSGD(model.parameters())
    loss = model.compute_loss(x, y) if hasattr(model, "compute_loss") else loss_fn(model(x), y)
    loss.backward()
    opt.step()

Submodules:
    tikitorch.functional  — stateless tropical ops + autograd Functions
    tikitorch.nn          — tropical layer Modules (a la torch.nn)
    tikitorch.optim       — sign-, shift-, and counter-oriented optimizer variants
    tikitorch.models      — ready-to-use tropical architectures
    tikitorch.autodiff    — operator-scoped forward-route certificates and replay VJPs
    tikitorch.hardware    — HLS kernels + an AFI/silicon build guide
    tikitorch.backend     — pluggable execution-backend seam (logical devices + dispatch)
    tikitorch.fpga        — the FPGA backend (DDP-style fpga.accelerate(model))
"""

__version__ = "0.2.0"

# functional and nn are foundational and always importable.
from tikitorch import functional  # noqa: F401
from tikitorch import nn  # noqa: F401
from tikitorch import autodiff  # noqa: F401
from tikitorch._hardware import HardwareCostProfile  # noqa: F401

# The pluggable backend seam (depends only on functional + torch + stdlib). It
# registers the CPU backend and auto-binds the functional ops as the default
# implementation, so CPU dispatch returns the existing reference callables.
from tikitorch import backend  # noqa: F401
from tikitorch.backend import device, to  # noqa: F401  (tt.device(...) / tt.to(...))

# Core public namespaces fail loudly if their Python imports are broken. Optional
# native FPGA availability is handled inside ``tikitorch.fpga`` and is not a reason
# to hide a package error behind ``None``.
from tikitorch import optim  # noqa: F401
from tikitorch import models  # noqa: F401
from tikitorch import fpga  # noqa: F401

# Closure verification, tropical-geometry tools, classical tropical ML, and
# structured pruning (always importable; depend only on functional/nn/torch).
from tikitorch import audit  # noqa: F401
from tikitorch import geometry  # noqa: F401
from tikitorch import classical  # noqa: F401
from tikitorch import prune  # noqa: F401
from tikitorch import algorithmic  # noqa: F401
from tikitorch.audit import (  # noqa: F401
    assert_closed, assert_closed_runtime, assert_training_closed,
    find_runtime_multiplies, find_training_multiplies, closure_report,
    assert_no_multiply, closed, ClosureViolation,
)

# Model (de)serialization and the closure/hardware profiler depend only on
# audit/torch and degrade gracefully if a downstream import is mid-assembly.
from tikitorch import serialization  # noqa: F401
from tikitorch.serialization import save, load  # noqa: F401
from tikitorch import profile  # noqa: F401
from tikitorch.profile import profile_model, format_report  # noqa: F401

# Reference trainer and local-learning namespace are part of the public Python API.
from tikitorch import train  # noqa: F401
from tikitorch.train import Trainer  # noqa: F401
from tikitorch import forward_forward  # noqa: F401
from tikitorch.forward_forward import (  # noqa: F401
    TropicalFFLayer, TropicalFFNet, TropicalFFTrainer,
)

# Top-level convenience re-exports: architectures and optimizers are reachable
# directly (``tt.MorphSSM``, ``tt.PureTropicalSGD``) as well as via submodule
# (``tt.models.MorphSSM``, ``tt.optim.PureTropicalSGD``) -- the way ``torch``
# exposes ``torch.Tensor`` while ``torch.nn`` / ``torch.optim`` also exist.
_top = ["__version__", "functional", "nn", "autodiff", "optim", "models", "HardwareCostProfile",
        "audit", "geometry", "classical", "prune", "algorithmic", "assert_closed",
        "assert_closed_runtime", "assert_training_closed", "find_runtime_multiplies",
        "find_training_multiplies", "closure_report",
        "assert_no_multiply", "closed", "ClosureViolation",
        "backend", "device", "to", "fpga", "train", "Trainer",
        "forward_forward", "TropicalFFLayer", "TropicalFFNet", "TropicalFFTrainer",
        "serialization", "save", "load", "profile", "profile_model", "format_report"]
from tikitorch.models import (  # noqa: F401
    MorphSSM, TropicalTransformer, MoTE, MoTEModel,
    TropicalMLP, TropicalSequenceModel, MorphMLP,
    TropicalForwardForward, DyadicForwardForward,
)
_top += ["MorphSSM", "TropicalTransformer", "MoTE", "MoTEModel",
         "TropicalMLP", "TropicalSequenceModel", "MorphMLP",
         "TropicalForwardForward", "DyadicForwardForward"]
from tikitorch.optim import (  # noqa: F401
    SignSGD, TropicalSGD, PureTropicalSGD, TropicalLion, TropicalAdam,
    FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD,
)
_top += ["SignSGD", "TropicalSGD", "PureTropicalSGD", "TropicalLion", "TropicalAdam",
         "FrexpMagnitudeSGD", "CountAgreementSGD", "ExponentEMAAdam", "PerLayerStepSGD"]

__all__ = _top
