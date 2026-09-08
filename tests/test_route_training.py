"""Small deterministic learning/trajectory gates, not benchmark quality claims."""
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from tikitorch.autodiff import morphssm_forward_tape, morphssm_vjp, route_logits
from tikitorch.models import MorphSSM


def small_model(dtype=torch.float64, device="cpu"):
    torch.manual_seed(12)
    model = MorphSSM(d_model=8, n_layers=1, vocab_size=7, kernel_size=3,
                     dropout=0.0, task_type="lm")
    model.set_tau(0)
    return model.to(device=device, dtype=dtype).eval()


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("ties", [True, False])
def test_route_autograd_and_external_cotangent_parity(packed, ties):
    model = small_model()
    if ties:
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
    ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    cotangent = torch.randn(2, 4, 7, dtype=torch.float64)
    expected_logits = model(ids)
    expected = torch.autograd.grad(expected_logits, tuple(model.parameters()), cotangent)
    logits, tape = morphssm_forward_tape(model, ids, packed=packed)
    assert torch.equal(logits, expected_logits)
    replayed = morphssm_vjp(tape, model, cotangent)
    actual = torch.autograd.grad(route_logits(model, ids, packed=packed),
                                 tuple(model.parameters()), cotangent)
    for (name, _), ref, got in zip(model.named_parameters(), expected, actual):
        torch.testing.assert_close(replayed[name], ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("packed", [True, False])
def test_multistep_learning_trajectory(packed):
    eager = small_model()
    routed = deepcopy(eager)
    ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    targets = (ids + 1) % 7
    opts = [torch.optim.SGD(m.parameters(), lr=0.001) for m in (eager, routed)]
    for _ in range(5):
        for model, opt in zip((eager, routed), opts):
            opt.zero_grad(set_to_none=True)
            logits = model(ids) if model is eager else route_logits(model, ids, packed=packed)
            F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1)).backward()
            opt.step()
        for a, b in zip(eager.parameters(), routed.parameters()):
            torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)


def test_saved_tensors_are_routes_and_hooks_work():
    model = small_model()
    ids = torch.tensor([[0, 1, 2, 3]])
    saved = []
    def pack(t):
        saved.append(t)
        return t.clone()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = route_logits(model, ids)
    assert saved and all(t.dtype == torch.uint8 for t in saved)
    result.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_dropout_routes_match_regular_forward():
    from tikitorch.nn import TropicalDropout
    model = small_model().train()
    model.dropout = TropicalDropout(0.25)
    model.layers[0].dropout = TropicalDropout(0.25)
    ids = torch.tensor([[0, 1, 2, 3]])
    torch.manual_seed(97)
    expected = model(ids)
    torch.manual_seed(97)
    actual, tape = morphssm_forward_tape(model, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "embed_drop_gate" in tape and "layer0.drop_gate" in tape
    model.dropout.eval()
    model.layers[0].dropout.eval()
    expected = model(ids)
    actual, tape = morphssm_forward_tape(model, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "embed_drop_gate" not in tape and "layer0.drop_gate" not in tape


def test_route_dropout_gate_preserves_public_ieee_semantics():
    from tikitorch.autodiff.morphssm_tape import _dropout_gate
    from tikitorch.nn import TropicalDropout
    x = torch.tensor([float("nan"), float("inf"), -0.0, -2.0, 3.0], requires_grad=True)
    torch.manual_seed(7)
    expected = TropicalDropout(0.75)(x)
    torch.manual_seed(7)
    gate = torch.rand_like(x) >= 0.75
    actual = _dropout_gate(x, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(actual[2:]), torch.signbit(expected[2:]))
    cotangent = torch.tensor([1.0, float("nan"), -0.0, -2.0, 3.0])
    reference = torch.autograd.grad(expected, x, cotangent)[0]
    torch.testing.assert_close(_dropout_gate(cotangent, gate), reference,
                               rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("option", ["padding_idx", "scale_grad_by_freq", "sparse", "max_norm", "dropout", "residual", "alias"])
def test_unsupported_configuration_refused(option):
    model = small_model()
    if option == "dropout":
        model.dropout = torch.nn.Dropout(0.1)
    elif option == "residual":
        model.layers[0].residual_scale = 0.5
    elif option == "alias":
        model.output_head.weight = model.embed.weight
    else:
        setattr(model.embed, option, 0 if option == "padding_idx" else (2.0 if option == "max_norm" else True))
    with pytest.raises(ValueError):
        morphssm_forward_tape(model, torch.tensor([[0, 1, 2, 3]]))


def test_configuration_binding_covers_each_layer_and_parameter_schema():
    from tikitorch.nn import DepthwiseMorphConv1d
    model = MorphSSM(d_model=8, n_layers=2, vocab_size=7, kernel_size=3,
                     task_type="lm", dropout=0.0).double().eval()
    model.set_tau(0)
    logits, tape = morphssm_forward_tape(model, torch.tensor([[0, 1, 2, 3]]))
    conv = model.layers[1].morph_conv
    model.layers[1].morph_conv = DepthwiseMorphConv1d(8, 5, dilation=conv.dilation,
                                                    mode=conv.mode, use_soft=False, tau=0).double()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        morphssm_vjp(tape, model, torch.ones_like(logits))


def test_global_module_hooks_are_rejected():
    model = small_model()
    hook = torch.nn.modules.module.register_module_forward_hook(lambda m, x, y: y)
    try:
        with pytest.raises(ValueError, match="global module hooks"):
            morphssm_forward_tape(model, torch.tensor([[0, 1, 2, 3]]))
    finally:
        hook.remove()


def test_custom_backend_cannot_silently_change_the_recorded_model():
    import tikitorch.backend as backend
    from tikitorch import functional
    backend.register_op("tropical_dilation", "cpu", lambda x, w: functional.tropical_dilation(x, w) + 1)
    try:
        with pytest.raises(ValueError, match="route-derivative contract"):
            morphssm_forward_tape(small_model(), torch.tensor([[0, 1, 2, 3]]))
    finally:
        backend.register_op("tropical_dilation", "cpu", functional.tropical_dilation)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_route_training():
    model = small_model(dtype=torch.float32, device="cuda")
    ids = torch.tensor([[0, 1, 2, 3]], device="cuda")
    cotangent = torch.randn(1, 4, 7, device="cuda")
    ref = torch.autograd.grad(model(ids), tuple(model.parameters()), cotangent)
    got = torch.autograd.grad(route_logits(model, ids), tuple(model.parameters()), cotangent)
    for a, b in zip(ref, got):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
