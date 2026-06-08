"""tikitorch.profile — a one-call closure / hardware report for a model.

This stitches the two closure checkers in :mod:`tikitorch.audit` together with
the optional :class:`~tikitorch._hardware.HardwareCostProfile` into a single
human-readable report. It answers the paper's headline question for any model
you paste in: *how many hardware multiplies does this thing actually do?*

  * **Structural closure** — :func:`~tikitorch.audit.find_closure_violations`
    scans the module tree for standard ``torch.nn`` layers that carry a
    data-dependent multiply/divide (``nn.Linear``, ``nn.Conv2d``, ...).
  * **Runtime closure** — :func:`~tikitorch.audit.find_runtime_multiplies`
    (only when ``example_inputs`` is given) traces a real forward pass and flags
    every multiply/transcendental ATen op a custom ``forward`` actually runs
    (power-of-two multiplies/divides are bit-shifts, not multipliers).
  * **Param count** — trainable parameters.
  * **Hardware estimate** — if the model exposes ``hardware_cost(...)`` (the
    optional forge-style accounting) its DSP/LUT/BRAM numbers are folded in.

The whole point is the one line a reader looks for: for a closed tropical model
the report prints ``multiplications: 0``.

Usage::

    import tikitorch as tt
    from tikitorch.profile import profile_model, format_report

    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    print(format_report(profile_model(cnn, torch.rand(1, 1, 28, 28))))

or, for a named built-in with a synthesized input, from the command line::

    python -m tikitorch TropicalCNN
    python -m tikitorch MorphSSM
"""
import torch
import torch.nn as nn

from tikitorch.audit import find_closure_violations, find_runtime_multiplies

__all__ = ["profile_model", "format_report"]


def _param_count(model: nn.Module):
    """``(trainable, total)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _hardware_cost(model: nn.Module):
    """Call ``model.hardware_cost()`` if exposed, else ``None``.

    The method is optional (see :mod:`tikitorch._hardware`); models that do not
    expose it simply omit the FPGA estimate from the report. We call it with no
    arguments (the models default ``batch_size=1``) and swallow any failure so a
    quirky signature never breaks the profile.
    """
    fn = getattr(model, "hardware_cost", None)
    if not callable(fn):
        return None
    try:
        cost = fn()
    except Exception:  # noqa: BLE001 - the estimate is best-effort, never fatal
        return None
    if hasattr(cost, "summary"):
        return cost
    return None


def profile_model(model: nn.Module, example_inputs=None) -> dict:
    """Profile ``model`` for closure + hardware cost; return a structured report.

    Args:
        model: any ``torch.nn.Module`` (a TikiTorch tropical model or a plain one).
        example_inputs: optional input(s) for the runtime trace. Pass a single
            tensor, or a tuple/list of positional args, to additionally run
            :func:`~tikitorch.audit.find_runtime_multiplies` (the graph-level
            check that catches a raw ``*`` / ``@`` / ``exp`` in a custom forward).
            If ``None`` only the structural scan + param/hardware report run.

    Returns a dict with keys:
        ``multiplications`` — the headline integer: number of hardware multiply
            sources detected. ``0`` iff the model is closed (no structural
            violation and, when traced, no runtime multiply).
        ``closed`` — bool, ``multiplications == 0``.
        ``structural_violations`` — ``[(module_path, type_name), ...]``.
        ``runtime_multiplies`` — ``[(category, op_name), ...]`` (only the genuine
            multiply categories; transcendental soft-relaxation ops are reported
            separately under ``runtime_transcendental``). Empty list if traced and
            clean; ``None`` if no ``example_inputs`` were given.
        ``runtime_transcendental`` — ``[(category, op_name), ...]`` soft tau>0
            exp/log ops (training-time aids, anneal to tau=0 for deploy), or
            ``None`` if untraced.
        ``param_count`` / ``trainable_params`` — parameter counts.
        ``hardware`` — the ``HardwareCostProfile.summary()`` dict, or ``None``.
    """
    structural = find_closure_violations(model)

    runtime = None
    transcendental = None
    if example_inputs is not None:
        if isinstance(example_inputs, (tuple, list)):
            args = tuple(example_inputs)
        else:
            args = (example_inputs,)
        hits = find_runtime_multiplies(model, *args)
        runtime = [(c, op) for c, op in hits if c != "transcendental"]
        transcendental = [(c, op) for c, op in hits if c == "transcendental"]

    # Headline: a model is "closed" iff the structural scan is clean AND, where we
    # traced it, the runtime trace found no genuine multiply. The integer is the
    # count of distinct multiply sources (structural modules + runtime ops);
    # transcendental soft-relaxation ops are reported but do NOT count as a
    # hardware multiply for the headline (they vanish at tau=0).
    mult = len(structural) + (len(runtime) if runtime is not None else 0)

    trainable, total = _param_count(model)
    cost = _hardware_cost(model)

    return {
        "model": type(model).__name__,
        "multiplications": mult,
        "closed": mult == 0,
        "structural_violations": structural,
        "runtime_multiplies": runtime,
        "runtime_transcendental": transcendental,
        "param_count": total,
        "trainable_params": trainable,
        "hardware": cost.summary() if cost is not None else None,
    }


def format_report(report: dict) -> str:
    """Pretty-print a :func:`profile_model` report as a multi-line string.

    The headline line always reads exactly ``multiplications: <n>`` (``0`` for a
    closed model) so it is greppable and matches the paper's claim.
    """
    lines = []
    lines.append(f"=== tikitorch closure profile: {report['model']} ===")
    # The headline, exactly as the paper quotes it.
    lines.append(f"multiplications: {report['multiplications']}")
    lines.append(f"closed: {report['closed']}")
    lines.append(
        f"params: {report['param_count']:,} "
        f"({report['trainable_params']:,} trainable)"
    )

    struct = report["structural_violations"]
    if struct:
        lines.append(f"structural violations ({len(struct)}):")
        for path, typ in struct:
            lines.append(f"    {path}: {typ}")
    else:
        lines.append("structural violations: none (no multiply-bearing torch.nn modules)")

    rt = report["runtime_multiplies"]
    if rt is None:
        lines.append("runtime trace: skipped (no example_inputs)")
    elif rt:
        lines.append(f"runtime multiplies ({len(rt)}):")
        for cat, op in rt:
            lines.append(f"    {cat}: {op}")
    else:
        lines.append("runtime multiplies: none (traced forward is multiplier-free)")

    trans = report["runtime_transcendental"]
    if trans:
        lines.append(
            f"runtime transcendental ({len(trans)}, soft tau>0 — anneal to tau=0 for deploy):"
        )
        for cat, op in trans:
            lines.append(f"    {cat}: {op}")

    hw = report["hardware"]
    if hw is not None:
        lines.append("hardware estimate (forge HardwareCostProfile):")
        lines.append(f"    dsp_blocks: {hw['dsp_blocks']}")
        lines.append(f"    lut_estimate: {hw['lut_estimate']:,}")
        lines.append(f"    bram_kb: {hw['bram_kb']:,}")
        lines.append(f"    fwd_multiplications: {hw['fwd_multiplications']:,}")
        lines.append(f"    fwd_comparisons: {hw['fwd_comparisons']:,}")
        lines.append(f"    fwd_additions: {hw['fwd_additions']:,}")
    else:
        lines.append("hardware estimate: not exposed (model has no hardware_cost())")

    return "\n".join(lines)
