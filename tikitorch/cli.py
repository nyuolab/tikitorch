"""``tt-closure`` — command-line closure auditor.

Loads a model by dotted spec and applies TikiTorch's structural software-closure
audit, reusing :mod:`tikitorch.audit`. Exit code 0 = no registered structural
violation after applying ``--allow``, 1 = a multiply-bearing registered module
was found (or 2 for a bad spec) — so it composes as a CI gate. ``--report`` adds
the full diagnostic report without changing that exit-status policy.

    tt-closure tikitorch.models:TropicalForwardForward
    tt-closure my_pkg.models:build_net --allow head
    tt-closure tikitorch.models:MorphSSM --report
"""
from __future__ import annotations

import argparse
import importlib
import sys


def _load(spec: str):
    """Load an ``nn.Module`` from a ``'package.module:factory'`` spec.

    ``factory`` may be an ``nn.Module`` instance or a zero-arg callable returning one
    (a class or factory function); callables are invoked.
    """
    from torch import nn

    mod_name, sep, attr = spec.rpartition(":")
    if not sep or not mod_name or not attr:
        raise ValueError(f"expected 'package.module:factory', got {spec!r}")
    try:
        obj = getattr(importlib.import_module(mod_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"could not load {spec!r}: {exc}") from exc
    # Modules are themselves callable: invoking an existing instance would run
    # forward() with no input instead of loading the target for structural audit.
    if isinstance(obj, nn.Module):
        return obj
    if not callable(obj):
        raise ValueError(f"{spec!r} must name an nn.Module or a zero-argument factory")
    model = obj()
    if not isinstance(model, nn.Module):
        raise ValueError(
            f"factory {spec!r} returned {type(model).__name__}, expected nn.Module"
        )
    return model


def main(argv=None) -> int:
    from tikitorch.audit import assert_closed, closure_report

    parser = argparse.ArgumentParser(
        prog="tt-closure",
        description="Apply TikiTorch's structural software-closure audit to a model.",
    )
    parser.add_argument(
        "target", help="dotted spec 'pkg.module:factory' returning an nn.Module")
    parser.add_argument(
        "--allow", nargs="*", default=(), metavar="NAME",
        help="module paths or type names to treat as declared software boundaries "
             "(e.g. --allow head Linear)")
    parser.add_argument(
        "--report", action="store_true",
        help="also print the full closure_report(); the audit exit status is unchanged")
    args = parser.parse_args(argv)

    try:
        model = _load(args.target)
    except Exception as exc:
        # An invalid import, constructor, or factory is a target-spec error. Keep
        # this boundary limited to loading: auditor errors must still propagate.
        parser.error(f"could not prepare target {args.target!r}: {exc}")

    if args.report:
        import pprint
        pprint.pprint(closure_report(model))

    try:
        assert_closed(model, allow=set(args.allow))
    except AssertionError as exc:   # ClosureViolation subclasses AssertionError
        print(f"NOT CLOSED: {args.target}\n{exc}", file=sys.stderr)
        return 1
    boundary_note = f"; except declared boundaries: {', '.join(args.allow)}" if args.allow else ""
    print(f"SOFTWARE-CLOSED (STRUCTURAL): {args.target}{boundary_note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
