"""Pure FP64 exact-CE fitting helpers, not a corpus experiment or native model.

With real mean centering, all tropical operations are positively homogeneous:
u=beta*W gives logits(u)=beta*logits(W). These dimensionless logits enter
ordinary softmax directly; a second beta or the old raw-gradient divisor would
change the objective/update units. Hard selectors retain first-index ties.

The GRU adds a learned correction to an externally supplied frozen prior.
Neither this module nor its constructors load data or run an optimizer step.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from manuscript.tools import fpga_lm_float_reference as reference
from manuscript.tools import fpga_lm_trajectory_arithmetic as arithmetic

SCHEMA = "tikitorch.fpga_lm_exact_fit_helper.v1"
BETA, WEIGHTS = reference.BETA, reference.WEIGHTS
ATOL, RTOL = 1e-9, 1e-11
TRACE_KEYS = ("logits", "centering_offsets", "AD_HEAD", "PROJ_IDX", "TAP_IDX")


def source_identity():
    path = Path(__file__).resolve()
    return {**reference.source_identity(), **arithmetic.source_identity(),
            path.relative_to(path.parents[2]).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()}


def _finite(value, label, *, dtype=torch.float64, detached=False):
    if (not isinstance(value, torch.Tensor) or value.dtype != dtype
            or value.device.type != "cpu" or value.layout != torch.strided
            or not value.numel() or not bool(torch.isfinite(value).all())
            or (detached and value.requires_grad)):
        raise ValueError(label + " requires nonempty finite CPU " + str(dtype) + " values")
    return value


def validate_weights(weights):
    """Detached natural five-region tensors; no integer/hardware range claim."""
    return arithmetic._mapping(weights, "dimensionless tropical parameters")


def parameter_digest(weights):
    validate_weights(weights)
    digest = hashlib.sha256()
    for key in WEIGHTS:
        value = weights[key]
        digest.update(json.dumps([key, list(value.shape), "float64"], separators=(",", ":")).encode())
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def dimensionless(weights):
    """Fresh detached u=beta*W; source W need not be integer-valued."""
    validate_weights(weights)
    result = {key: weights[key] * BETA for key in WEIGHTS}
    validate_weights(result)
    return result


def _tokens(tokens, vocab, ndim=1):
    _finite(tokens, "token IDs", dtype=torch.int64, detached=True)
    if tokens.ndim != ndim or bool(((tokens < 0) | (tokens >= vocab)).any()):
        raise ValueError("complete in-vocabulary token IDs with the required rank are needed")
    return tokens


def _targets(targets, length, vocab):
    if isinstance(targets, (list, tuple)):
        if len(targets) != length or any(type(value) is not int or not 0 <= value < vocab for value in targets):
            raise ValueError("literal in-vocabulary target IDs required")
        targets = torch.tensor(targets, dtype=torch.int64)
    _tokens(targets, vocab)
    if targets.shape != (length,):
        raise ValueError("one complete target ID per position is required")
    return targets


@torch.no_grad()
def _independent_forward(weights, tokens, context):
    """Full-matrix selection independent of the producer's chunked selector."""
    geometry = reference._geometry(weights, tokens)
    d, n, k, length = (geometry[key] for key in ("D", "N", "K", "T"))
    history = [[] for _ in range(n)]
    taps, projections = ([[] for _ in range(n)] for _ in range(2))
    logits, heads = [], []
    for position in range(length):
        hidden = weights["EMBED"][int(tokens[position])]
        for layer in range(n):
            centered = _finite(hidden - hidden.mean(), "independent real centering")
            history[layer].append(centered)
            dilation = 1 << min(layer, 6)
            count = min(k, position // dilation + 1) if context else 1
            candidates = torch.stack([history[layer][position - tap * dilation]
                                      + weights["MH"][layer, :, tap] for tap in range(count)])
            conv, tap = candidates.max(dim=0) if layer % 2 == 0 else candidates.min(dim=0)
            _finite(conv, "independent temporal selection")
            projected, projection = (weights["MP"][layer] + conv[None, :]).max(dim=1)
            projected = _finite(projected + weights["MB"][layer], "independent projection plus bias")
            hidden = _finite(hidden + projected, "independent residual")
            taps[layer].append(tap)
            projections[layer].append(projection)
        scores, head = (weights["W_HEAD"] + hidden[None, :]).max(dim=1)
        logits.append(_finite(scores, "independent head"))
        heads.append(head)
    return {"schema": reference.SCHEMA, "mode": "smooth", "geometry": geometry,
            "tokens": tokens.detach().clone(), "use_context": context,
            "logits": torch.stack(logits), "centering_offsets": torch.zeros((n, length), dtype=torch.float64),
            "routes": {"AD_HEAD": torch.stack(heads),
                       "PROJ_IDX": torch.stack([torch.stack(rows) for rows in projections]),
                       "TAP_IDX": torch.stack([torch.stack(rows) for rows in taps])}}


def forward(weights_u, tokens, context=True, independent=False):
    """Finite real-centered dimensionless logits and complete first-tie tape.

    Both paths are detached. The independent path selects from full matrices;
    no saved route is reused to determine the dynamic forward.
    """
    validate_weights(weights_u)
    _tokens(tokens, weights_u["EMBED"].shape[0])
    if type(context) is not bool or type(independent) is not bool:
        raise ValueError("context and independent must be literal booleans")
    with torch.no_grad():
        result = (_independent_forward(weights_u, tokens, context) if independent
                  else reference.forward(weights_u, tokens, mode="smooth", use_context=context))
    return {**result, "helper_schema": SCHEMA, "units": "dimensionless",
            "parameter_sha256": parameter_digest(weights_u), "native_execution": False,
            "independent_forward": independent}


def trace(base):
    if (not isinstance(base, dict) or base.get("helper_schema") != SCHEMA
            or base.get("units") != "dimensionless" or base.get("mode") != "smooth"):
        raise ValueError("an exact-fit dimensionless smooth base is required")
    geometry = base.get("geometry")
    if (not isinstance(geometry, dict) or set(geometry) != {"D", "V", "N", "K", "T"}
            or any(type(value) is not int or value <= 0 for value in geometry.values())):
        raise ValueError("complete positive literal base geometry is required")
    reference._base(base, geometry, base.get("tokens"), base.get("use_context"))
    logits = _finite(base.get("logits"), "base logits", detached=True)
    if logits.shape != (base["geometry"]["T"], base["geometry"]["V"]):
        raise ValueError("base logits differ from the complete geometry")
    return {"logits": logits, "centering_offsets": base["centering_offsets"], **base["routes"]}


def gradient(weights_u, tokens, targets, base, divisor_positions, independent=False):
    """True mean-CE selected derivative in u units, without a second beta.

    divisor_positions is the complete minibatch position count, not necessarily
    this window's length. Manual reverse and fixed-route autograd use the same
    detached exact-softmax cotangent. At ties this is the declared first-index
    selected local reverse rule, with no global Clarke-subgradient claim or
    differentiability assertion across route boundaries.
    """
    validate_weights(weights_u)
    geometry = reference._geometry(weights_u, tokens)
    _tokens(tokens, geometry["V"])
    ids = _targets(targets, geometry["T"], geometry["V"])
    if type(divisor_positions) is not int or divisor_positions <= 0:
        raise ValueError("divisor_positions must be a positive literal integer")
    if type(independent) is not bool:
        raise ValueError("independent must be a literal boolean")
    trace(base)
    reference._base(base, geometry, tokens, base["use_context"])
    if base.get("parameter_sha256") != parameter_digest(weights_u):
        raise ValueError("base belongs to different parameter bytes")
    cotangent = (base["logits"].softmax(dim=-1)
                 - F.one_hot(ids, geometry["V"]).to(torch.float64)) / divisor_positions
    _finite(cotangent, "exact-CE cotangent", detached=True)
    if independent:
        leaves = {key: weights_u[key].detach().clone().requires_grad_(True) for key in WEIGHTS}
        with torch.enable_grad():
            replay = reference.forward(leaves, tokens, mode="fixed_routes", base=base)
            if not torch.allclose(replay["logits"].detach(), base["logits"], atol=ATOL, rtol=RTOL):
                raise ValueError("fixed-route source-point logits differ")
            values = torch.autograd.grad(replay["logits"], tuple(leaves.values()), grad_outputs=cotangent)
        result = {key: value.detach() for key, value in zip(WEIGHTS, values)}
    else:
        result = reference.manual_vjp(weights_u, tokens, cotangent, base)
    arithmetic._mapping(result, "complete exact-CE gradient", weights_u)
    return result


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(label + " must be a positive literal integer")


class FittedPriorGRU(nn.Module):
    """Conventional GRU correction with an external, never-trainable prior.

    A call resets hidden state per window. context=False additionally resets
    it at every position. independent=True expands PyTorch's r/z/n equations
    directly and never calls the built-in GRU forward.
    """

    def __init__(self, vocab=128, embedding=128, hidden=128):
        super().__init__()
        for value, label in ((vocab, "vocab"), (embedding, "embedding"), (hidden, "hidden")):
            _positive_int(value, label)
        self.vocab, self.embedding_size, self.hidden_size = vocab, embedding, hidden
        self.embedding = nn.Embedding(vocab, embedding, device="cpu", dtype=torch.float64)
        self.gru = nn.GRU(embedding, hidden, num_layers=1, batch_first=True,
                          dropout=0.0, bidirectional=False, device="cpu", dtype=torch.float64)
        self.head = nn.Linear(hidden, vocab, device="cpu", dtype=torch.float64)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _validate(self):
        v, e, h = self.vocab, self.embedding_size, self.hidden_size
        expected = {"embedding.weight": (v, e), "gru.weight_ih_l0": (3 * h, e),
                    "gru.weight_hh_l0": (3 * h, h), "gru.bias_ih_l0": (3 * h,),
                    "gru.bias_hh_l0": (3 * h,), "head.weight": (v, h), "head.bias": (v,)}
        parameters = dict(self.named_parameters())
        if set(parameters) != set(expected):
            raise ValueError("complete single-layer GRU parameter set required")
        for name, shape in expected.items():
            value = _finite(parameters[name], "GRU " + name)
            if value.shape != shape:
                raise ValueError("inconsistent GRU parameter geometry: " + name)

    def forward(self, tokens, prior_logits, context=True, independent=False):
        self._validate()
        _tokens(tokens, self.vocab, ndim=2)
        _finite(prior_logits, "frozen prior logits", detached=True)
        if prior_logits.shape != (self.vocab, self.vocab):
            raise ValueError("frozen prior must map every current token to every class")
        if type(context) is not bool or type(independent) is not bool:
            raise ValueError("context and independent must be literal booleans")
        embedded = self.embedding(tokens)
        batch, length = tokens.shape
        if independent:
            hidden = embedded.new_zeros((batch, self.hidden_size))
            states = []
            for position in range(length):
                if not context:
                    hidden = torch.zeros_like(hidden)
                inputs = F.linear(embedded[:, position], self.gru.weight_ih_l0, self.gru.bias_ih_l0)
                recurrent = F.linear(hidden, self.gru.weight_hh_l0, self.gru.bias_hh_l0)
                ir, iz, inn = inputs.chunk(3, dim=-1)
                hr, hz, hn = recurrent.chunk(3, dim=-1)
                reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
                candidate = (inn + reset * hn).tanh()
                hidden = (1 - update) * candidate + update * hidden
                states.append(_finite(hidden, "independent GRU hidden state"))
            features = torch.stack(states, dim=1)
        elif context:
            features, _ = self.gru(embedded)
        else:
            features, _ = self.gru(embedded.reshape(batch * length, 1, self.embedding_size))
            features = features.reshape(batch, length, self.hidden_size)
        return _finite(prior_logits[tokens] + self.head(features), "GRU corrected logits")


def make_gru(seed=0, vocab=128, embedding=128, hidden=128):
    """Isolated CPU RNG initialization; zero correction gives the prior exactly."""
    if type(seed) is not int or not 0 <= seed < (1 << 63):
        raise ValueError("seed must be a nonnegative literal signed-64-bit integer")
    for value, label in ((vocab, "vocab"), (embedding, "embedding"), (hidden, "hidden")):
        _positive_int(value, label)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model = FittedPriorGRU(vocab, embedding, hidden)
    return model


def make_optimizer(parameters):
    """Construct fixed eager Adam; this does NOT guard or transact its step.

    Detached leaf tensors with manually assigned .grad are allowed, as are
    ordinary nn.Parameters. The invoking runner must validate all gradients
    before each step and all parameters/moments afterward, retaining failures.
    """
    if isinstance(parameters, (torch.Tensor, dict, str, bytes)):
        raise ValueError("an iterable of unique parameter tensors is required")
    try:
        values = list(parameters)
    except TypeError as error:
        raise ValueError("an iterable of parameter tensors is required") from error
    if not values or len({id(value) for value in values}) != len(values):
        raise ValueError("nonempty unique parameter tensors are required")
    for value in values:
        _finite(value, "Adam parameter")
        if not value.is_leaf:
            raise ValueError("Adam requires leaf parameter tensors")
    return torch.optim.Adam(values, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0.0, amsgrad=False, maximize=False,
                            foreach=False, fused=False, capturable=False, differentiable=False)


def empirical_bigram_floor(tokens, targets, vocab):
    """Empirical unsmoothed H(Y|current token), in nats per supplied pair.

    Counts come only from the explicitly aligned B-by-T pairs, without adding
    cross-window adjacency. Beating this training floor requires extra input
    information, potentially position; it is not a generalization claim.
    """
    _positive_int(vocab, "vocab")
    _tokens(tokens, vocab, ndim=2)
    _tokens(targets, vocab, ndim=2)
    if tokens.shape != targets.shape:
        raise ValueError("complete aligned B-by-T token and target arrays required")
    counts = {}
    totals = {}
    for token, target in zip(tokens.reshape(-1).tolist(), targets.reshape(-1).tolist()):
        counts[(token, target)] = counts.get((token, target), 0) + 1
        totals[token] = totals.get(token, 0) + 1
    positions = tokens.numel()
    result = math.fsum(count * math.log(totals[token] / count)
                       for (token, _), count in counts.items()) / positions
    if not math.isfinite(result) or result < 0:
        raise ValueError("invalid empirical conditional entropy")
    return result
