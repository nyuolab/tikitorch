"""TikiTorch — a tropical-semiring deep-learning library, modeled on PyTorch.

TikiTorch packages the tropical (max-plus / min-plus) closure machinery as a
clean, installable, PyTorch-idiomatic library.
Every forward and backward op reduces to comparisons and additions — no
multiplications — which is what makes the whole learning loop FPGA-native.

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
    tikitorch.optim       — multiplier-free optimizers (a la torch.optim)
    tikitorch.models      — ready-to-use tropical architectures
    tikitorch.hardware    — HLS kernels + an AFI/silicon build guide
    tikitorch.backend     — pluggable execution-backend seam (logical devices + dispatch)
    tikitorch.fpga        — the FPGA backend (DDP-style fpga.accelerate(model))
"""

__version__ = "0.1.0"

# functional and nn are foundational and always importable.
from tikitorch import functional  # noqa: F401
from tikitorch import nn  # noqa: F401
from tikitorch._hardware import HardwareCostProfile  # noqa: F401

# The pluggable backend seam (depends only on functional + torch + stdlib). It
# registers the CPU backend and auto-binds the functional ops as the default
# implementation, so the existing CPU path stays byte-identical and zero-overhead.
from tikitorch import backend  # noqa: F401
from tikitorch.backend import device, to  # noqa: F401  (tt.device(...) / tt.to(...))

# optim and models are guarded so that
# `import tikitorch` never hard-fails while the package is being assembled.
try:  # pragma: no cover - import guard
    from tikitorch import optim  # noqa: F401
except Exception:  # noqa: BLE001
    optim = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from tikitorch import models  # noqa: F401
except Exception:  # noqa: BLE001
    models = None  # type: ignore[assignment]

# The FPGA backend self-registers into the seam on import. Guarded like
# optim/models so a missing native layer never breaks `import tikitorch`.
try:  # pragma: no cover - import guard
    from tikitorch import fpga  # noqa: F401
except Exception:  # noqa: BLE001
    fpga = None  # type: ignore[assignment]

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

# The closure-aware Trainer is finalized like optim/models; guard so a missing
# dependency never hard-fails ``import tikitorch``.
try:  # pragma: no cover - import guard
    from tikitorch import train  # noqa: F401
    from tikitorch.train import Trainer  # noqa: F401
except Exception:  # noqa: BLE001
    train = None  # type: ignore[assignment]
    Trainer = None  # type: ignore[assignment]

# Top-level convenience re-exports: architectures and optimizers are reachable
# directly (``tt.MorphSSM``, ``tt.PureTropicalSGD``) as well as via submodule
# (``tt.models.MorphSSM``, ``tt.optim.PureTropicalSGD``) -- the way ``torch``
# exposes ``torch.Tensor`` while ``torch.nn`` / ``torch.optim`` also exist.
_top = ["__version__", "functional", "nn", "optim", "models", "HardwareCostProfile",
        "audit", "geometry", "classical", "prune", "algorithmic", "assert_closed",
        "assert_closed_runtime", "assert_training_closed", "find_runtime_multiplies",
        "find_training_multiplies", "closure_report",
        "assert_no_multiply", "closed", "ClosureViolation",
        "backend", "device", "to",
        "serialization", "save", "load", "profile", "profile_model", "format_report"]
if fpga is not None:
    _top += ["fpga"]
if Trainer is not None:
    _top += ["train", "Trainer"]
if models is not None:
    from tikitorch.models import (  # noqa: F401
        MorphSSM, TropicalTransformer, MoTE, MoTEModel,
        TropicalMLP, TropicalSequenceModel, MorphMLP,
    )
    _top += ["MorphSSM", "TropicalTransformer", "MoTE", "MoTEModel",
             "TropicalMLP", "TropicalSequenceModel", "MorphMLP"]
if optim is not None:
    from tikitorch.optim import (  # noqa: F401
        SignSGD, TropicalSGD, PureTropicalSGD, TropicalLion, TropicalAdam,
        FrexpMagnitudeSGD, CountAgreementSGD, ExponentEMAAdam, PerLayerStepSGD,
    )
    _top += ["SignSGD", "TropicalSGD", "PureTropicalSGD", "TropicalLion", "TropicalAdam",
             "FrexpMagnitudeSGD", "CountAgreementSGD", "ExponentEMAAdam", "PerLayerStepSGD"]

__all__ = _top
