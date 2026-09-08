"""Tests for the closure GUARD (tt.assert_no_multiply / @tt.closed) and the
closure-clean activation/dropout modules.

The guard is the fail-fast, wrap-your-own-code version of the audit layer: it traces
the actual ATen ops in a block and raises ClosureViolation on any hardware
multiply/divide/transcendental — catching a raw ``*`` / ``@`` / ``torch.exp`` that the
structural assert_closed cannot see. Powers of two are bit-shifts and pass.
"""
import pytest
import torch
import torch.nn as nn

import tikitorch as tt
import tikitorch.nn as ttnn


# ---------------------------------------------------------------------------
# assert_no_multiply context manager
# ---------------------------------------------------------------------------
def test_guard_catches_matmul():
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            a, b = torch.randn(8, 8), torch.randn(8, 8)
            (a @ b).sum()


def test_guard_catches_nonpow2_scalar_and_elementwise_multiply():
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            (torch.randn(8) * 0.1).sum()              # non-pow2 scalar multiply
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            (torch.randn(8) * torch.randn(8)).sum()   # elementwise tensor multiply


def test_guard_catches_nonpow2_divide():
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            (torch.randn(8) / 3.0).sum()


def test_guard_passes_closed_ops():
    # max / add / clamp + power-of-two scale (a bit-shift) are all closure-clean
    x = torch.randn(8, 8)
    with tt.assert_no_multiply():
        y = torch.clamp_min(x, 0.0) + (x * 0.25) - (x / 8.0)   # relu, *2^-2, /2^3
        torch.maximum(y, x).sum()


def test_guard_transcendental_flag():
    x = torch.randn(8)
    # exp is flagged by default...
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            x.exp().sum()
    # ...but allowed with allow_transcendental=True (the soft tau>0 relaxations)
    with tt.assert_no_multiply(allow_transcendental=True):
        x.exp().sum()


def test_guard_propagates_user_exception():
    # a real error inside the block must surface (not be masked by the closure check)
    with pytest.raises(ValueError):
        with tt.assert_no_multiply():
            raise ValueError("boom")


# ---------------------------------------------------------------------------
# @closed decorator
# ---------------------------------------------------------------------------
def test_closed_decorator_bare_and_parametrized():
    @tt.closed
    def good(x):
        return torch.clamp_min(x, 0.0).sum()

    good(torch.randn(4))                              # passes

    @tt.closed
    def bad(x):
        return (x * 0.1).sum()

    with pytest.raises(tt.ClosureViolation):
        bad(torch.randn(4))

    @tt.closed(allow_transcendental=True)
    def soft(x):
        return x.exp().sum()

    soft(torch.randn(4))                              # transcendental allowed


# ---------------------------------------------------------------------------
# Activation + dropout modules
# ---------------------------------------------------------------------------
def test_activation_modules_values():
    x = torch.tensor([-4.0, -1.0, 0.0, 2.0])
    assert torch.equal(ttnn.TropicalReLU()(x), torch.tensor([0.0, 0.0, 0.0, 2.0]))
    # leaky: negatives scaled by 2**-shift (here 2**-2 = 0.25)
    assert torch.allclose(ttnn.TropicalLeakyReLU(shift=2)(x),
                          torch.tensor([-1.0, -0.25, 0.0, 2.0]))
    assert torch.equal(ttnn.TropicalHardtanh(-1.0, 1.0)(x), torch.tensor([-1.0, -1.0, 0.0, 1.0]))


def test_activations_closure_clean():
    seq = nn.Sequential(ttnn.TropicalReLU(), ttnn.TropicalLeakyReLU(3), ttnn.TropicalHardtanh())
    x = torch.randn(16, 32)
    with tt.assert_no_multiply():
        seq(x)


def test_tropical_dropout_eval_is_identity():
    d = ttnn.TropicalDropout(0.5).eval()
    x = torch.randn(64)
    assert torch.equal(d(x), x)


def test_tropical_dropout_train_masks_without_rescale():
    torch.manual_seed(0)
    d = ttnn.TropicalDropout(0.5).train()
    x = torch.ones(10000)
    y = d(x)
    # survivors are UNSCALED (==1.0, not 1/(1-p)=2.0); dropped are 0.0
    uniq = set(round(v, 4) for v in y.unique().tolist())
    assert uniq <= {0.0, 1.0}, uniq
    assert 0.0 in uniq and 1.0 in uniq                # some dropped, some kept


def test_tropical_dropout_is_closure_clean_in_training():
    # THE fix: standard nn.Dropout's 1/(1-p) survivor rescale breaks training closure
    # for any p where 1/(1-p) is NOT a power of two; TropicalDropout (a {0,1} gate with
    # no rescale) is closure-clean for ALL p.
    x = torch.randn(256)
    for p in (0.3, 0.5, 0.7):
        d = ttnn.TropicalDropout(p).train()
        with tt.assert_no_multiply():
            d(x)

    # contrast: nn.Dropout(0.3) rescales by 1/0.7 (non-pow2) -> a real divide/multiply
    nd = nn.Dropout(0.3).train()
    with pytest.raises(tt.ClosureViolation):
        with tt.assert_no_multiply():
            nd(x)

    # nuance: nn.Dropout(0.5) happens to be clean — 1/(1-0.5) = 2 is a power of two (a shift)
    with tt.assert_no_multiply():
        nn.Dropout(0.5).train()(x)


def test_tropical_dropout_rejects_bad_p():
    with pytest.raises(ValueError):
        ttnn.TropicalDropout(1.0)
    with pytest.raises(ValueError):
        ttnn.TropicalDropout(-0.1)
