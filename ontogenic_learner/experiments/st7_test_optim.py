"""E6 optimizer unit battery (house rule: green before anything touches
st7_life). Run: python st7_test_optim.py"""
import copy
import torch

from st7_optim import IDBDAdam, IDBDDyadic, IDBDSGD, build_life_opt, _AnchorOpt


def _quad_problem(seed=0):
    torch.manual_seed(seed)
    w = torch.randn(64, requires_grad=True)
    target = torch.randn(64)
    return w, target


def test_theta0_adamw_parity():
    """theta_meta=0 IDBDAdam == torch.optim.AdamW over 200 steps."""
    wa, target = _quad_problem()
    wb = wa.detach().clone().requires_grad_(True)
    oa = torch.optim.AdamW([wa], lr=1e-2, weight_decay=0.01)
    ob = IDBDAdam([wb], lr=1e-2, weight_decay=0.01, theta_meta=0.0)
    for _ in range(200):
        for w, o in ((wa, oa), (wb, ob)):
            o.zero_grad()
            ((w - target) ** 2).sum().backward()
            o.step()
    assert torch.allclose(wa, wb, atol=1e-6), \
        f"parity broke: max diff {(wa - wb).abs().max():.2e}"
    print("PASS theta0 AdamW parity")


def test_sign_direction():
    """Constant gradient -> beta rises to beta_max; alternating -> beta falls.
    Also: every meta step bounded by theta_meta."""
    for name, cls in (("idbd-adam", IDBDAdam), ("idbd-sgd", IDBDSGD)):
        w = torch.zeros(8, requires_grad=True)
        o = cls([w], lr=1e-3, theta_meta=0.05, weight_decay=0.0)
        prev = None
        for i in range(3000):
            o.zero_grad()
            w.grad = torch.full_like(w, 1.0)          # constant descent
            o.step()
            b = o.state[w]["beta"]
            if prev is not None:
                assert float((b - prev).abs().max()) <= 0.05 + 1e-6, \
                    f"{name}: meta step exceeded theta_meta"
            prev = b.clone()
        assert float(prev.mean()) > 1.5, f"{name}: beta did not rise ({float(prev.mean()):.2f})"
        w2 = torch.zeros(8, requires_grad=True)
        o2 = cls([w2], lr=1e-3, theta_meta=0.05, weight_decay=0.0)
        for i in range(3000):
            o2.zero_grad()
            w2.grad = torch.full_like(w2, 1.0 if i % 2 == 0 else -1.0)
            o2.step()
        assert float(o2.state[w2]["beta"].mean()) < -1.0, \
            f"{name}: beta did not fall under alternation"
    print("PASS sign direction + meta-step bound")


def test_dyadic_hysteresis():
    """beta oscillating INSIDE the hysteresis band never moves q; a single
    clean band-crossing moves it exactly once."""
    w = torch.zeros(4, requires_grad=True)
    o = IDBDDyadic([w], lr=1e-3, theta_meta=0.0, weight_decay=0.0,
                   hysteresis=0.25)
    o.zero_grad(); w.grad = torch.ones_like(w); o.step()   # init state
    st = o.state[w]

    def drive(vals):
        moves, lastq = 0, st["q"].clone()
        for v in vals:
            st["beta"].fill_(v)
            o.zero_grad(); w.grad = torch.ones_like(w) * 1e-12; o.step()
            if not torch.equal(st["q"], lastq):
                moves += 1
                lastq = st["q"].clone()
        return moves

    # in-band oscillation around the q=0 -> 1 boundary (up-threshold 0.75)
    assert drive([0.5 + (0.2 if i % 2 == 0 else -0.2) for i in range(200)]) \
        == 0, "q moved inside the hysteresis band"
    # one clean crossing
    assert drive([0.9]) == 1, "q failed to cross"
    # settle just under the new down-threshold region: no further motion
    assert drive([0.65 + (0.1 if i % 2 == 0 else -0.1) for i in range(200)]) \
        == 0, "q flapped after crossing"
    print("PASS dyadic hysteresis")


def test_burst_stability():
    """Tropical-like gradients: 95% exact zeros + heavy-tailed spikes.
    Everything stays finite and in bounds; IDBDSGD h decays through a
    2k-step drought."""
    torch.manual_seed(1)
    for cls in (IDBDAdam, IDBDDyadic, IDBDSGD):
        w = torch.randn(256, requires_grad=True)
        o = cls([w], lr=1e-3, theta_meta=0.05)
        for i in range(10_000):
            o.zero_grad()
            mask = (torch.rand(256) < 0.05).float()
            spike = torch.distributions.StudentT(2.0).sample((256,))
            w.grad = mask * spike
            o.step()
        st = o.state[w]
        assert torch.isfinite(w).all() and torch.isfinite(st["h"]).all()
        assert float(st["beta"].max()) <= 2.0 and float(st["beta"].min()) >= -6.0
    w = torch.randn(16, requires_grad=True)
    o = IDBDSGD([w], lr=1e-3, theta_meta=0.05)
    o.zero_grad(); w.grad = torch.ones_like(w); o.step()
    h0 = float(o.state[w]["h"].abs().max())
    for _ in range(2000):
        o.zero_grad(); w.grad = torch.zeros_like(w); o.step()
    h1 = float(o.state[w]["h"].abs().max())
    assert h1 < 0.2 * h0, f"h-leak failed through drought ({h0:.2e} -> {h1:.2e})"
    print("PASS burst stability + h-leak")


def test_state_roundtrip():
    """50 steps -> save/load -> 50 more == 100 uninterrupted, bitwise;
    int8 q survives the base-class float cast."""
    def run(cls, n, sd=None):
        torch.manual_seed(7)
        w = torch.randn(32, requires_grad=True)
        target = torch.randn(32)
        o = cls([w], lr=1e-2, theta_meta=0.05)
        if sd is not None:
            w.data.copy_(sd.pop("_w"))
            o.load_state_dict(sd)
        for _ in range(n):
            o.zero_grad()
            ((w - target) ** 2).sum().backward()
            o.step()
        return w, o
    for cls in (IDBDAdam, IDBDDyadic, IDBDSGD):
        w100, _ = run(cls, 100)
        w50, o50 = run(cls, 50)
        sd = copy.deepcopy(o50.state_dict())
        sd["_w"] = w50.detach().clone()
        wr, orx = run(cls, 50, sd=sd)
        assert torch.equal(w100, wr), f"{cls.__name__} roundtrip not bitwise"
        if cls is IDBDDyadic:
            for st in orx.state.values():
                assert st["q"].dtype == torch.int8, "q dtype lost on load"
    print("PASS state_dict roundtrip (incl. int8 q)")


def test_anchor_opt():
    """_AnchorOpt: .step() adds lam*vhat*(w-theta) to .grad before stepping;
    param_groups/state_dict write through."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from st2_live import Anchor
    torch.manual_seed(3)
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)
    # warm Adam state so exp_avg_sq exists
    for _ in range(5):
        opt.zero_grad()
        model(torch.randn(16, 8)).sum().backward()
        opt.step()
    anchor = Anchor(model, lam=0.01)
    aopt = _AnchorOpt(opt, anchor, model)
    # displace weights from theta; with zero task grads, pull() must add
    # exactly lam*vhat*(w-theta) to .grad (the st2_live contract)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    aopt.zero_grad()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)
    expected = {}
    for n, p in model.named_parameters():
        v = opt.state[p]["exp_avg_sq"]
        vhat = (v / v.mean().clamp_min(1e-12)).clamp(max=100.0)
        expected[n] = anchor.lam * vhat * (p.detach() - anchor.theta[n])
    w_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    anchor.pull(model, opt)
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, expected[n], atol=1e-7), \
            f"pull wrong on {n}"
    opt.step()
    assert any(not torch.equal(p.detach(), w_before[n])
               for n, p in model.named_parameters()), "step inert"
    assert aopt.param_groups is opt.param_groups
    sd = aopt.state_dict()
    aopt.load_state_dict(sd)
    print("PASS _AnchorOpt pull direction + passthrough")


def test_builder():
    cfg = dict(lr=1e-3)
    assert isinstance(build_life_opt([torch.nn.Parameter(torch.randn(2))],
                                     cfg), torch.optim.AdamW)
    for k, cls in (("idbd-adam", IDBDAdam), ("idbd-dyadic", IDBDDyadic),
                   ("idbd-sgd", IDBDSGD)):
        o = build_life_opt([torch.nn.Parameter(torch.randn(2))],
                           dict(lr=1e-3, optimizer=k))
        assert type(o) is cls
    print("PASS builder")


if __name__ == "__main__":
    test_theta0_adamw_parity()
    test_sign_direction()
    test_dyadic_hysteresis()
    test_burst_stability()
    test_state_roundtrip()
    test_anchor_opt()
    test_builder()
    print("ALL OPTIM UNITS GREEN")


def test_action_uniform_pull():
    """Engine E6 field: with action_loss set, action_uniform_pull (a) leaves
    the LM head's grads bitwise unchanged, (b) adds the softmax-minus-uniform
    term to the action head's grad, (c) repeated kicks on a saturated head
    raise its softmax entropy."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "scripts"))
    from pc_convrnn_charlm import MorphPCEngine, build_model
    from tikitorch.nn import TropicalTopKModule
    import argparse
    torch.manual_seed(11)
    V, D, K, B, T = 24, 32, 7, 1, 16
    ns = argparse.Namespace(d_model=D, n_layers=2, kernel=7, dropout=0.0,
                            tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", V, ns)
    model.action_head = TropicalTopKModule(D, K, k=8)
    with torch.no_grad():                       # saturate the head
        model.action_head.weight.mul_(0.0).add_(
            torch.randn_like(model.action_head.weight) * 0.02)
        model.action_head.weight[3].add_(1.0)   # one dominant row
    model.set_tau(0.0); model.eval()
    eng = MorphPCEngine(model)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    amask = torch.zeros(B, T, dtype=torch.bool); amask[0, ::4] = True
    atgt = torch.randint(0, K, (B, T))
    asgn = torch.ones(B, T, dtype=torch.int8)
    akex = torch.zeros(B, T, dtype=torch.long)

    def kick(pull):
        eng.action_loss = (amask, atgt, asgn, akex)
        eng.action_uniform_pull = pull
        opt = torch.optim.SGD(model.parameters(), lr=0.0)  # grads only
        eng.serve_and_learn(x, y, opt=opt, fire=lambda *a, **k: True)
        eng.action_loss = None; eng.action_uniform_pull = None
        return {n: (p.grad.clone() if p.grad is not None else None)
                for n, p in model.named_parameters()}

    g_off = kick(None)
    g_on = kick((-2, 0, K))
    assert torch.equal(g_off["output_head.weight"], g_on["output_head.weight"]), \
        "LM head grads changed by action pull"
    assert not torch.equal(g_off["action_head.weight"],
                           g_on["action_head.weight"]), "pull inert on head"
    # (c) entropy rises under repeated pulled kicks with a real optimizer
    def head_entropy():
        with torch.no_grad():
            lg = model.action_head(torch.randn(64, D) * 0.1)
            pi = torch.softmax(lg.float(), -1)
            return float(-(pi * pi.clamp_min(1e-9).log()).sum(-1).mean())
    h0 = head_entropy()
    opt = torch.optim.SGD(model.parameters(), lr=5e-2)
    for _ in range(200):
        eng.action_loss = (amask, atgt, asgn * 0, akex)  # sgn=0: advantage off,
        eng.action_uniform_pull = (2, 0, K)              # pull alone drives it
        eng.serve_and_learn(x, y, opt=opt, fire=lambda *a, **k: True)
    eng.action_loss = None; eng.action_uniform_pull = None
    h1 = head_entropy()
    assert h1 > h0 + 0.1, f"entropy did not rise ({h0:.3f} -> {h1:.3f})"
    print("PASS action_uniform_pull (LM untouched, head moved, entropy up)")


if __name__ == "__main__":
    test_action_uniform_pull()
    print("ALL ENGINE UNITS GREEN")
