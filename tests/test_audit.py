"""Tests for tikitorch.audit — the multiplier-free closure checker."""
import pytest
import torch
import torch.nn as tnn

from tikitorch.audit import assert_closed, find_closure_violations


def test_flags_standard_torch_modules():
    model = tnn.Sequential(tnn.Linear(4, 4), tnn.ReLU(), tnn.LayerNorm(4))
    types = [t for _, t in find_closure_violations(model)]
    assert "Linear" in types and "LayerNorm" in types
    with pytest.raises(AssertionError):
        assert_closed(model)


def test_embedding_and_inert_modules_pass():
    # Embedding is a pure gather (no multiply); dropout/relu are inert.
    model = tnn.Sequential(tnn.Embedding(10, 4), tnn.Dropout(0.1), tnn.ReLU())
    assert find_closure_violations(model) == []
    assert_closed(model) is model


def test_allow_declares_boundary():
    model = tnn.Linear(4, 4)
    with pytest.raises(AssertionError):
        assert_closed(model)
    # allow by type name
    assert_closed(model, allow={"Linear"}) is model


def test_tropical_modules_are_closed():
    import tikitorch.nn as nn
    closed = tnn.Sequential(
        nn.TropicalConv2d(1, 4, 3, padding=1),
        nn.TropicalLayerNorm2d(4),
        nn.TropicalMaxPool2d(2),
    )
    assert find_closure_violations(closed) == []
    assert_closed(closed)


def test_gated_and_parametric_activations_flagged():
    # Soundness fix: GLU/PReLU carry data-dependent multiplies and must be caught.
    import torch.nn as tnn
    for layer in (tnn.GLU(), tnn.PReLU()):
        from tikitorch.audit import find_closure_violations
        assert find_closure_violations(layer), f"{type(layer).__name__} should be flagged"


def test_allow_is_mro_aware():
    # MultiheadAttention's internal out_proj is a Linear subclass; allow={'Linear'}
    # plus the MHA type should let an MHA-containing model pass.
    import torch.nn as tnn
    from tikitorch.audit import assert_closed
    m = tnn.MultiheadAttention(8, 2, batch_first=True)
    assert_closed(m, allow={"MultiheadAttention", "Linear"})


# --- runtime closure checker (traces actual aten ops; catches raw * / @ / exp) ---

def test_runtime_checker_passes_closed_models():
    import tikitorch as tt
    from tikitorch.audit import find_runtime_multiplies
    cnn = tt.models.TropicalCNN(in_channels=1, num_classes=10)
    assert find_runtime_multiplies(cnn, torch.rand(1, 1, 28, 28)) == []   # pow2 rescales = shifts
    mssm = tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm", vocab_size=128)
    assert find_runtime_multiplies(mssm, torch.randint(0, 128, (2, 16))) == []


def test_runtime_checker_passes_rope_and_iter_topk_reasoner():
    import tikitorch.algorithmic as A
    from tikitorch.audit import assert_closed_runtime
    m = A.build_reasoner("running_max", d_model=32, n_layers=2, attention="tropical",
                         agg_variant="iter_topk", agg_topk_k=3, pos_mode="rope_shift")
    x, _ = A.gen_running_max(2, 20, n_symbols=20, seed=0)
    assert_closed_runtime(m, x)                                  # closure holds at runtime


def test_runtime_checker_catches_real_multiply():
    import pytest
    from tikitorch.audit import assert_closed_runtime, find_runtime_multiplies

    class Bad(torch.nn.Module):           # a non-pow2 scalar multiply (real DSP op)
        def forward(self, x):
            return x * 1.3
    with pytest.raises(AssertionError):
        assert_closed_runtime(Bad(), torch.randn(3))

    # an nn.Linear (dense matmul) is caught too
    lin = torch.nn.Sequential(torch.nn.Linear(4, 4))
    hits = find_runtime_multiplies(lin, torch.randn(2, 4))
    assert any(c == "dense-multiply" for c, _ in hits)


def test_runtime_checker_pow2_multiply_is_a_shift():
    from tikitorch.audit import find_runtime_multiplies

    class Pow2(torch.nn.Module):          # x * 0.25 and x / 8 are shifts, not multiplies
        def forward(self, x):
            return (x * 0.25) / 8.0
    assert find_runtime_multiplies(Pow2(), torch.randn(3)) == []


# --- Level-B training-closure check (forward + loss + backward + optimizer) ---

def _running_max_setup(d_model=24):
    import tikitorch.algorithmic as A
    import tikitorch.functional as F
    x, y = A.gen_running_max(6, 12, n_symbols=12, seed=0)
    m = A.build_reasoner("running_max", d_model=d_model, n_layers=1,
                         attention="tropical", agg_variant="argmax", pos_mode="none")
    C = A.get_task("running_max").n_symbols
    loss_fn = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, C), yy.reshape(-1))
    return m, x, y, loss_fn


def test_training_closure_perceptron_puretropicalsgd_has_no_multiply():
    # The closed recipe's weight UPDATE is multiplier-free / zero-DSP: no dense or
    # elementwise multiply, no transcendental. (The float loss VALUE carries a
    # batch-mean ``/n`` divide that is sign-invariant for sign-SGD and is NOT
    # computed on-chip -- the silicon emits the ±1 gradient directly -- so we exempt
    # ``elementwise-divide`` here; the DSP-critical check is the multiply/exp one.)
    import tikitorch.optim as optim
    from tikitorch.audit import find_training_multiplies
    m, x, y, loss_fn = _running_max_setup(d_model=32)   # pow2 d_model: norm mean = shift
    opt = optim.PureTropicalSGD(m.parameters(), lr_shift=4)
    hits = find_training_multiplies(m, (x,), y, loss_fn, optimizer=opt)
    bad = [(c, op) for c, op in hits if c != "elementwise-divide"]
    assert not bad, f"closed recipe has DSP-critical ops: {bad}"


def test_training_closure_catches_adam_and_cross_entropy():
    import pytest
    import tikitorch.algorithmic as A
    from tikitorch.audit import assert_training_closed, find_training_multiplies
    m, x, y, _ = _running_max_setup()
    C = A.get_task("running_max").n_symbols
    ce = lambda p, yy: torch.nn.functional.cross_entropy(p.reshape(-1, C), yy.reshape(-1))
    adam = torch.optim.Adam(m.parameters(), lr=1e-3)
    hits = find_training_multiplies(m, (x,), y, ce, optimizer=adam)
    assert hits, "Adam + cross-entropy must be flagged as non-closed training"
    with pytest.raises(AssertionError):
        assert_training_closed(m, (x,), y, ce, optimizer=adam)


def test_training_closure_catches_signsgd_float_momentum():
    # SignSGD's float momentum does buf.mul_(0.9) -- a real multiply, so it is NOT
    # Level-B closed (this is the bug the checker is designed to surface).
    import tikitorch.optim as optim
    from tikitorch.audit import find_training_multiplies
    m, x, y, loss_fn = _running_max_setup()
    opt = optim.SignSGD(m.parameters(), lr=2 ** -4, momentum=0.9)
    find_training_multiplies(m, (x,), y, loss_fn, optimizer=opt)   # step 1 builds buffer
    hits = find_training_multiplies(m, (x,), y, loss_fn, optimizer=opt)
    assert any("mul" in op for _, op in hits), "float-momentum SignSGD should not be closed"


# --- closed-by-construction CONTRACT: the closed recipe is verified Level-B ---

def test_closure_report_one_call_inference_and_training():
    import tikitorch as tt
    import tikitorch.functional as F
    import tikitorch.optim as optim
    m = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1,
                                      task_type="lm", vocab_size=32, dropout=0.0)
    x = torch.randint(0, 32, (2, 16)); y = torch.randint(0, 32, (2, 16))
    perc = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, p.shape[-1]), yy.reshape(-1))
    rep = tt.closure_report(m, x, y, perc, optimizer=optim.PureTropicalSGD(m.parameters(), lr_shift=4))
    assert rep["inference_closed"] is True
    assert rep["training_zero_dsp"] is True          # closed recipe -> zero-DSP training


def test_inference_closed_by_construction_for_tropical_models():
    # Every standard tropical model is closed at INFERENCE by construction.
    import tikitorch as tt
    cases = [
        (tt.models.TropicalCNN(in_channels=1, num_classes=8), torch.rand(2, 1, 16, 16)),
        (tt.models.TropicalMLP(d_model=32, n_layers=2, input_dim=16, num_classes=8), torch.rand(2, 16)),
        (tt.models.MorphSSM(d_model=64, n_layers=2, task_type="lm", vocab_size=32), torch.randint(0, 32, (2, 16))),
        (tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1, task_type="lm", vocab_size=32), torch.randint(0, 32, (2, 16))),
    ]
    for m, x in cases:
        assert tt.closure_report(m, x)["inference_closed"], f"{type(m).__name__} not inference-closed"


def test_dropout_breaks_training_closure_off_switch():
    # Dropout is a train-time mask multiply -> NOT closed; dropout=0 is the off-switch.
    import tikitorch as tt
    import tikitorch.functional as F
    import tikitorch.optim as optim
    x = torch.randint(0, 32, (2, 16)); y = torch.randint(0, 32, (2, 16))
    perc = lambda p, yy: F.tropical_perceptron_loss(p.reshape(-1, p.shape[-1]), yy.reshape(-1))
    m = tt.models.TropicalTransformer(d_model=64, n_heads=4, n_layers=1,
                                      task_type="lm", vocab_size=32, dropout=0.1)
    rep = tt.closure_report(m, x, y, perc, optimizer=optim.PureTropicalSGD(m.parameters(), lr_shift=4))
    assert rep["training_zero_dsp"] is False         # dropout>0 -> training not zero-DSP
