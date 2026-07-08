"""Tests for the real transformer-block fabric stage.

The load-bearing check is the gradient: the block's local backward (the PC cascade through
attention + FFN + LN) must match numerical gradients — otherwise the distributed-training
conclusions are meaningless. Plus an end-to-end check that the char-LM pipeline learns.
"""
import numpy as np
import pytest

from tikitorch.fabric import PipelineRunner, make_thread_links
from tikitorch.fabric.transformer_stage import TransformerBlock, make_charlm_pipeline


def test_transformer_block_gradcheck():
    rng = np.random.default_rng(0)
    d, f, seq = 8, 16, 5
    blk = TransformerBlock(d, f, seed=1)
    x = rng.standard_normal((seq, d))
    dy = rng.standard_normal((seq, d))               # loss = sum(y*dy) -> dL/dy = dy
    y, cache = blk.forward(x)
    grads, dx = blk.backward(dy, cache)
    eps = 1e-6

    def loss():
        yy, _ = blk.forward(x); return np.sum(yy * dy)

    worst = 0.0
    for p, g in zip(blk.params, grads):
        flat = p.ravel(); gn = np.zeros_like(flat)
        for i in range(flat.size):
            o = flat[i]; flat[i] = o + eps; lp = loss(); flat[i] = o - eps; lm = loss(); flat[i] = o
            gn[i] = (lp - lm) / (2 * eps)
        worst = max(worst, np.max(np.abs(g - gn.reshape(p.shape))) / (np.max(np.abs(gn)) + 1e-12))
    # input grad
    flat = x.ravel(); gx = np.zeros_like(flat)
    for i in range(flat.size):
        o = flat[i]; flat[i] = o + eps; lp = loss(); flat[i] = o - eps; lm = loss(); flat[i] = o
        gx[i] = (lp - lm) / (2 * eps)
    worst = max(worst, np.max(np.abs(dx - gx.reshape(x.shape))) / (np.max(np.abs(gx)) + 1e-12))
    assert worst < 1e-5, f"transformer block gradcheck failed: worst rel {worst:.2e}"


def test_charlm_pipeline_learns():
    """A real char-LM (embed + 2 blocks + head) trained under the fabric pipeline must
    drive bpb well below the uniform-prior bits."""
    text = "to be or not to be that is the question " * 6
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
    toks = np.array([stoi[c] for c in text]); seq = 16
    M = (len(toks) - 1) // seq
    inp = [toks[k * seq:(k + 1) * seq] for k in range(M)]
    tgt = [toks[k * seq + 1:(k + 1) * seq + 1] for k in range(M)]
    stages = make_charlm_pipeline(V, d=32, seq=seq, n_blocks=2, seed=0)
    S = len(stages)
    first = PipelineRunner(stages, make_thread_links(S), lr=0.05, window=M,
                           schedule="1f1b").run(inp, tgt).mean_loss
    last = first
    for _ in range(60):
        last = PipelineRunner(stages, make_thread_links(S), lr=0.05, window=M,
                              schedule="1f1b").run(inp, tgt).mean_loss
    assert last < 0.5 * first, f"char-LM did not learn: bpb {first:.2f} -> {last:.2f}"
    assert last < np.log2(V), "should beat the uniform prior"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
