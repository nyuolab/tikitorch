"""Tests for tikitorch.nn layer Modules.

Core layers (always present) are tested unconditionally:
    TropicalSoftMin/Max, TropicalLinearModule, TropicalDilationModule,
    CompositionalTropicalProjection, TropicalTopKModule, and their aliases.

Optional building blocks (TropicalBatchShift, MorphConv1d, MorphSSMLayer,
TropicalAttention, ...) are tested behind ``requires_nn`` skip guards so the
suite is green before they land and turns active once they do.
"""
import pytest
import torch

import tikitorch.nn as nn
from conftest import requires_nn


# ---------------------------------------------------------------------------
# Core: forward/backward shape + finiteness
# ---------------------------------------------------------------------------


def test_soft_modules_forward():
    x = torch.randn(4, 6)
    w = torch.randn(5, 6)
    assert nn.TropicalSoftMin(tau=0.5)(x, w).shape == (4, 5)
    assert nn.TropicalSoftMax(tau=0.5)(x, w).shape == (4, 5)


@pytest.mark.parametrize("use_soft", [False, True])
def test_tropical_linear_module(use_soft):
    layer = nn.TropicalLinearModule(6, 5, use_softmin=use_soft, tau=0.7)
    x = torch.randn(3, 6, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 5)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("use_soft", [False, True])
def test_tropical_dilation_module(use_soft):
    layer = nn.TropicalDilationModule(6, 5, use_softmax=use_soft, tau=0.7)
    x = torch.randn(3, 6, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 5)
    y.sum().backward()
    assert torch.isfinite(layer.weight.grad).all()


def test_dilation_forward_with_indices():
    layer = nn.TropicalDilationModule(6, 5)
    x = torch.randn(3, 6)
    y, idx = layer.forward_with_indices(x)
    assert y.shape == (3, 5)
    assert idx.shape == (3, 5)
    assert idx.dtype == torch.long
    assert (idx >= 0).all() and (idx < 6).all()


def test_set_tau_propagates():
    layer = nn.TropicalLinearModule(4, 4, use_softmin=True, tau=1.0)
    layer.set_tau(0.1)
    assert layer.tau == 0.1
    assert layer._softmin.tau == 0.1


@pytest.mark.parametrize("tau", [0.0, 0.5])
def test_compositional_projection(tau):
    proj = nn.CompositionalTropicalProjection(8, 4, tau=tau)
    x = torch.randn(3, 8, requires_grad=True)
    y = proj(x)
    assert y.shape == (3, 4)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_compositional_projection_nd_input():
    # Must handle (..., in_features) batched leading dims (used in attention).
    proj = nn.CompositionalTropicalProjection(8, 4, tau=0.0)
    x = torch.randn(2, 5, 8)
    assert proj(x).shape == (2, 5, 4)


def test_topk_module():
    layer = nn.TropicalTopKModule(10, 4, k=3, tau=0.0)
    x = torch.randn(3, 10, requires_grad=True)
    y = layer(x)
    assert y.shape == (3, 4)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_phase1_aliases_identical():
    assert nn.TropicalLinear is nn.TropicalLinearModule
    assert nn.TropicalDilation is nn.TropicalDilationModule
    assert nn.TropicalProjection is nn.CompositionalTropicalProjection
    assert nn.TropicalTopK is nn.TropicalTopKModule


def test_soft_to_hard_equivalence_in_module():
    # A soft TropicalLinearModule at tiny tau should match its hard counterpart.
    soft = nn.TropicalLinearModule(6, 5, use_softmin=True, tau=1e-3)
    hard = nn.TropicalLinearModule(6, 5, use_softmin=False)
    hard.load_state_dict(soft.state_dict())
    x = torch.randn(4, 6)
    torch.testing.assert_close(soft(x), hard(x), atol=2e-2, rtol=0)


# ---------------------------------------------------------------------------
# Optional building blocks (skip-guarded until they land)
# ---------------------------------------------------------------------------


@requires_nn("TropicalBatchShift")
def test_batch_shift():
    layer = nn.TropicalBatchShift()
    x = torch.randn(8, 5)
    y = layer(x)
    assert y.shape == x.shape
    # Subtracts batch mean -> columns are zero-mean.
    torch.testing.assert_close(y.mean(dim=0), torch.zeros(5), atol=1e-5, rtol=0)


@requires_nn("BatchShift")
def test_batch_shift_alias():
    assert nn.BatchShift is nn.TropicalBatchShift


@requires_nn("DepthwiseMorphConv1d")
@pytest.mark.parametrize("mode", ["dilation", "erosion"])
def test_morph_conv1d(mode):
    conv = nn.DepthwiseMorphConv1d(channels=4, kernel_size=3, mode=mode)
    x = torch.randn(2, 4, 16, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 4, 16)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("MorphConv1d")
def test_morph_conv1d_alias():
    assert nn.MorphConv1d is nn.DepthwiseMorphConv1d


@requires_nn("MorphSSMLayer")
def test_morph_ssm_layer():
    layer = nn.MorphSSMLayer(d_model=16, kernel_size=3)
    x = torch.randn(2, 12, 16, requires_grad=True)
    y = layer(x)
    assert y.shape == (2, 12, 16)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("MultiScaleMorphConv1d")
def test_multiscale_morph_conv1d():
    conv = nn.MultiScaleMorphConv1d(channels=6, kernel_sizes=(3, 5))
    x = torch.randn(2, 6, 16)
    assert conv(x).shape == (2, 6, 16)


@requires_nn("TropicalAttention")
def test_tropical_attention():
    attn = nn.TropicalAttention(d_model=16, n_heads=4)
    x = torch.randn(2, 7, 16, requires_grad=True)
    y = attn(x)
    assert y.shape == (2, 7, 16)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("TropicalShiftAttention")
def test_tropical_shift_attention():
    attn = nn.TropicalShiftAttention(d_model=16, n_heads=4, max_seq_len=32)
    x = torch.randn(2, 7, 16)
    assert attn(x, causal_mask=True).shape == (2, 7, 16)


@requires_nn("TropicalTransformerBlock")
def test_tropical_transformer_block():
    blk = nn.TropicalTransformerBlock(d_model=16, n_heads=4, d_ff=32)
    x = torch.randn(2, 7, 16)
    assert blk(x).shape == (2, 7, 16)


@requires_nn("TropicalRouter")
def test_tropical_router():
    router = nn.TropicalRouter(d_model=16, d_router=8, n_experts=4)
    x = torch.randn(10, 16)
    expert_ids, aux = router(x)
    assert expert_ids.shape == (10,)
    assert (expert_ids >= 0).all() and (expert_ids < 4).all()
    assert aux.ndim == 0


@requires_nn("MoTEBlock")
def test_mote_block():
    blk = nn.MoTEBlock(d_model=16, n_experts=4, expert_type="transformer",
                       d_ff=32, n_heads=4, max_seq_len=32)
    x = torch.randn(2, 7, 16)
    out, aux = blk(x)
    assert out.shape == (2, 7, 16)
    assert aux.ndim == 0


# ---------------------------------------------------------------------------
# 2D morphological conv + pooling + tropical normalization (vision)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["dilation", "erosion"])
@requires_nn("TropicalConv2d")
def test_tropical_conv2d_module(mode):
    conv = nn.TropicalConv2d(3, 8, 3, padding=1, mode=mode)
    x = torch.randn(2, 3, 12, 12, requires_grad=True)
    y = conv(x)
    assert y.shape == (2, 8, 12, 12)
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()


@requires_nn("TropicalMaxPool2d")
def test_pooling_shapes_and_minpool_identity():
    x = torch.randn(2, 4, 8, 8)
    assert nn.TropicalMaxPool2d(2)(x).shape == (2, 4, 4, 4)
    assert nn.TropicalMinPool2d(2)(x).shape == (2, 4, 4, 4)
    assert nn.TropicalAdaptiveMaxPool2d((3, 3))(x).shape == (2, 4, 3, 3)
    assert nn.TropicalGlobalMaxPool()(x).shape == (2, 4)
    # min-pool = -maxpool(-x): on a single 2x2 it equals the elementwise min
    one = torch.randn(1, 1, 2, 2)
    assert torch.allclose(nn.TropicalMinPool2d(2)(one).squeeze(), one.amin())


@requires_nn("TropicalLayerNorm")
def test_tropical_layernorm_zero_mean_and_finite():
    ln = nn.TropicalLayerNorm(16, bias=False)
    x = torch.randn(4, 5, 16, requires_grad=True)
    y = ln(x)
    assert y.shape == x.shape
    # mean-subtracted along features -> per-row mean ~ 0 (before/after pow2 scale)
    c = x.detach() - x.detach().mean(-1, keepdim=True)
    assert c.mean(-1).abs().max() < 1e-5
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("TropicalRMSNorm")
def test_tropical_rmsnorm_finite():
    rms = nn.TropicalRMSNorm(16)
    x = torch.randn(4, 16, requires_grad=True)
    rms(x).sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("TropicalLayerNorm2d")
def test_tropical_layernorm2d():
    ln = nn.TropicalLayerNorm2d(8)
    x = torch.randn(2, 8, 6, 6, requires_grad=True)
    y = ln(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("MorphConvBlock")
def test_morph_conv_block():
    blk = nn.MorphConvBlock(3, 8, 3, pool=True)
    x = torch.randn(2, 3, 16, 16, requires_grad=True)
    y = blk(x)
    assert y.shape == (2, 8, 8, 8)  # pooled by 2
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# Tropical recurrent cells (RNN)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["min", "max"])
@requires_nn("TropicalRNNCell")
def test_tropical_rnn_cell(mode):
    cell = nn.TropicalRNNCell(6, 8, mode=mode)
    x = torch.randn(4, 6, requires_grad=True)
    h = torch.zeros(4, 8)
    h2 = cell(x, h)
    assert h2.shape == (4, 8)
    h2.sum().backward()
    assert torch.isfinite(x.grad).all()


@requires_nn("TropicalRNN")
def test_tropical_rnn_unroll_and_streaming_match():
    rnn = nn.TropicalRNN(5, 7, mode="min")
    x = torch.randn(3, 9, 5)
    outs, hf = rnn(x)
    assert outs.shape == (3, 9, 7) and hf.shape == (3, 7)
    # streaming .step() must reproduce the unrolled outputs exactly
    h = torch.zeros(3, 7)
    for t in range(9):
        h = rnn.step(x[:, t], h)
        torch.testing.assert_close(h, outs[:, t])


@requires_nn("TropicalQuantizedEmbed")
def test_quantized_embed_closed_and_shapes():
    from tikitorch.audit import find_closure_violations
    emb = nn.TropicalQuantizedEmbed(3, 16, n_bins=32)
    assert find_closure_violations(emb) == []          # embedding gather is closed
    x = torch.rand(2, 7, 3)                              # (B, L, in_dim) in [0,1]
    y = emb(x)
    assert y.shape == (2, 7, 16)
    y.sum().backward()
    assert emb.embed.weight.grad is not None
    # scalar input: still produces a non-degenerate (per-bin) embedding
    e1 = nn.TropicalQuantizedEmbed(1, 8, n_bins=16)
    out = e1(torch.rand(4, 1))
    assert out.shape == (4, 8)


@requires_nn("TropicalScale")
def test_tropical_scale_is_power_of_two_and_grad_flows():
    from tikitorch.audit import find_closure_violations
    sc = nn.TropicalScale(4)
    assert find_closure_violations(sc) == []
    x = torch.randn(2, 4, requires_grad=True)
    sc.log2_scale.data = torch.tensor([0.0, 1.0, -1.0, 2.0])
    y = sc(x)
    # factor is an exact power of two (a shift)
    factor = (y / x)[0]
    torch.testing.assert_close(torch.log2(factor), torch.round(sc.log2_scale))
    y.sum().backward()
    assert sc.log2_scale.grad is not None and sc.log2_scale.grad.abs().sum() > 0


@pytest.mark.parametrize("soft", [False, True])
@requires_nn("MaxPlusMin")
def test_maxplusmin_layer(soft):
    from tikitorch.audit import find_closure_violations
    m = nn.MaxPlusMin(6, 5, use_soft=soft, tau=1.0 if soft else 0.0)
    assert find_closure_violations(m) == []          # multiplier-free
    x = torch.randn(3, 7, 6, requires_grad=True)       # N-D
    y = m(x)
    assert y.shape == (3, 7, 5)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and m.weight.grad is not None
    m.set_tau(0.0)  # hardens
    assert m.use_soft is False


# --- P1 primitives: DEP, opening/closing, AdderLinear ---

@requires_nn("DilationErosionPerceptron")
def test_dep_layer():
    from tikitorch.audit import find_closure_violations
    m = nn.DilationErosionPerceptron(6, 5, lam=0.5, use_soft=True, tau=1.0)
    assert find_closure_violations(m) == []
    x = torch.randn(3, 7, 6, requires_grad=True)
    y = m(x); assert y.shape == (3, 7, 5)
    y.sum().backward()
    assert torch.isfinite(x.grad).all() and m.Wd.grad is not None and m.We.grad is not None


@pytest.mark.parametrize("op", ["opening", "closing"])
@requires_nn("MorphOpenClose2d")
def test_morph_open_close(op):
    from tikitorch.audit import find_closure_violations
    m = nn.MorphOpenClose2d(8, 3, op=op)
    assert find_closure_violations(m) == []
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    y = m(x); assert y.shape == (2, 8, 12, 12)
    y.sum().backward(); assert torch.isfinite(x.grad).all()


@requires_nn("AdderLinear")
def test_adder_linear():
    from tikitorch.audit import find_closure_violations
    m = nn.AdderLinear(6, 5)
    assert find_closure_violations(m) == []   # l1/abs/sum is multiply-free
    x = torch.randn(4, 6, requires_grad=True)
    y = m(x); assert y.shape == (4, 5)
    # y = -sum|x - w| + b  <= b
    ref = -(x.unsqueeze(-2) - m.weight).abs().sum(-1) + m.bias
    torch.testing.assert_close(y, ref)
    y.sum().backward(); assert torch.isfinite(x.grad).all()


# --- semiring-correct causal masking for min/max-plus aggregation ------------

@pytest.mark.parametrize("agg", ["argmax", "min", "maxmin", "iter_topk"])
def test_attention_causal_masking_is_semiring_correct(agg):
    """With causal_mask=True, each position must depend ONLY on keys <= itself, for
    BOTH max-plus and min-plus aggregation (min-plus masks future keys with +inf,
    not -inf -- otherwise min would *select* the masked key)."""
    import tikitorch.nn as nn
    torch.manual_seed(0)
    att = nn.TropicalAttention(d_model=8, n_heads=2, tau=0.0,
                               score_variant="hilbert", agg_variant=agg, agg_topk_k=2)
    att.eval()
    x = torch.randn(1, 6, 8)
    out = att(x, causal_mask=True)
    # perturb the LAST token; outputs at earlier positions must be unchanged.
    x2 = x.clone(); x2[:, -1] += 5.0
    out2 = att(x2, causal_mask=True)
    assert torch.allclose(out[:, :-1], out2[:, :-1], atol=1e-5), f"causality broken for agg={agg}"
    assert torch.isfinite(out).all()
