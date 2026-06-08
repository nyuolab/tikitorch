"""Import-safety test: the public package surface must come up fully wired.

``tikitorch/__init__.py`` deliberately guards the ``optim`` / ``models`` / ``fpga``
imports (see the ``try/except -> None`` blocks there) so that ``import tikitorch``
never hard-fails while the package is being assembled. That guard is convenient,
but it also means a *silently broken* submodule degrades to ``None`` instead of
raising — a regression that the rest of the suite (which skip-guards on those
symbols) would happily step around.

This test is the backstop: in a fully-built tree every guarded submodule must be
present (not ``None``), and the closure-checker entrypoints must import. It does
NOT skip — if a submodule is None here, that is a real, reportable failure.
"""
import tikitorch as tt


def test_import_tikitorch_succeeds():
    # The bare import comes up and exposes its version string.
    assert tt.__version__


def test_guarded_submodules_are_not_none():
    # The __init__ swallows import failures into None; catch a silently-broken
    # submodule by asserting each guarded one actually loaded.
    assert tt.optim is not None, "tikitorch.optim degraded to None (import failure swallowed)"
    assert tt.models is not None, "tikitorch.models degraded to None (import failure swallowed)"
    assert tt.fpga is not None, "tikitorch.fpga degraded to None (import failure swallowed)"


def test_closure_checkers_import():
    # The multiplier-free closure audit entrypoints must be on the public surface.
    from tikitorch import assert_closed, assert_closed_runtime
    assert callable(assert_closed)
    assert callable(assert_closed_runtime)
    # ...and reachable as attributes of the top-level package too.
    assert tt.assert_closed is assert_closed
    assert tt.assert_closed_runtime is assert_closed_runtime
