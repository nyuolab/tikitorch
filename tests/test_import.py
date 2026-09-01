"""Import-safety test: the public package surface must come up fully wired.

Core public namespaces import eagerly and fail loudly if the package is broken.
Optional native FPGA *availability* may be false, but the Python namespace itself
must always be present.
"""
import tikitorch as tt


def test_import_tikitorch_succeeds():
    # The bare import comes up and exposes its version string.
    assert tt.__version__


def test_public_submodules_import_eagerly():
    assert tt.optim.__name__ == "tikitorch.optim"
    assert tt.models.__name__ == "tikitorch.models"
    assert tt.fpga.__name__ == "tikitorch.fpga"
    assert tt.train.__name__ == "tikitorch.train"
    assert tt.forward_forward.__name__ == "tikitorch.forward_forward"


def test_closure_checkers_import():
    # The multiplier-free closure audit entrypoints must be on the public surface.
    from tikitorch import assert_closed, assert_closed_runtime
    assert callable(assert_closed)
    assert callable(assert_closed_runtime)
    # ...and reachable as attributes of the top-level package too.
    assert tt.assert_closed is assert_closed
    assert tt.assert_closed_runtime is assert_closed_runtime
