"""``tt-closure`` — command-line closure auditor.

Loads a model by dotted spec and applies TikiTorch's structural software-closure
audit, reusing :mod:`tikitorch.audit`. Exit code 0 = no registered structural
violation, 1 = a multiply-bearing registered module was found
(or 2 for a bad spec) — so it composes as a CI gate.

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
    mod_name, sep, attr = spec.rpartition(":")
    if not sep or not mod_name:
        raise SystemExit(f"expected 'package.module:factory', got {spec!r}")
    try:
        obj = getattr(importlib.import_module(mod_name), attr)
    except (ImportError, AttributeError) as exc:
        raise SystemExit(f"could not load {spec!r}: {exc}")
    return obj() if callable(obj) else obj


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
        help="module paths or type names to treat as declared DSP boundaries "
             "(e.g. --allow head Linear)")
    parser.add_argument(
        "--report", action="store_true",
        help="print the full closure_report() instead of a bare pass/fail")
    args = parser.parse_args(argv)

    model = _load(args.target)

    if args.report:
        import pprint
        pprint.pprint(closure_report(model))
        return 0

    try:
        assert_closed(model, allow=set(args.allow))
    except AssertionError as exc:   # ClosureViolation subclasses AssertionError
        print(f"NOT CLOSED: {args.target}\n{exc}", file=sys.stderr)
        return 1
    print(f"SOFTWARE-CLOSED (STRUCTURAL): {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
