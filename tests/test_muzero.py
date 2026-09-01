"""Stage-4 tests: the Tropical MuZero networks (representation/dynamics/prediction).

Covers shapes + the MuZero inference interface, plugging the learned net into
TropicalMCTS (the same engine validated against value iteration in test_mcts.py),
trainability (a tiny value-fit overfit reduces loss), and set_tau.
"""
import pytest
import torch
import torch.nn.functional as F

import tikitorch.planning as P
from tikitorch.planning.muzero import GridWorldNav, TropicalMuZeroNet

GRID = ["S...", "....", "....", "...G"]

# Both core kinds run the identical interface/MCTS/training surface: "maxplus" is the
# axis-aligned tropical core; "dyadic" is the multiplier-free OBLIQUE core (the §2a fix).
CORE_KINDS = ["maxplus", "dyadic"]


def _net(latent=16, tau=1.0, core_kind="maxplus"):
    env = GridWorldNav(GRID)
    obs_dim = 3 * env.H * env.W
    return env, TropicalMuZeroNet(obs_dim, env.n_actions, latent_dim=latent, tau=tau,
                                  core_kind=core_kind)


@pytest.mark.parametrize("core_kind", CORE_KINDS)
def test_inference_shapes_and_interface(core_kind):
    env, net = _net(core_kind=core_kind)
    obs = env.obs(env.reset()).flatten()
    inf = net.initial_inference(obs)
    assert inf.state.shape == (1, 16)
    assert inf.policy.shape == (env.n_actions,)
    assert abs(float(inf.policy.sum()) - 1.0) < 1e-4
    assert isinstance(inf.value, float)
    inf2, done = net.recurrent_inference(inf.state, 1)
    assert inf2.state.shape == (1, 16)
    assert isinstance(inf2.reward, float) and done is False
    assert abs(float(inf2.policy.sum()) - 1.0) < 1e-4


def test_batched_forward_shapes():
    env, net = _net(latent=24)
    B, obs_dim = 5, 3 * env.H * env.W
    x = torch.rand(B, obs_dim)
    h, p, v = net.initial(x)
    assert h.shape == (B, 24) and p.shape == (B, env.n_actions) and v.shape == (B,)
    h2, r, p2, v2 = net.recurrent(h, torch.randint(0, env.n_actions, (B,)))
    assert h2.shape == (B, 24) and r.shape == (B,) and v2.shape == (B,)


@pytest.mark.parametrize("core_kind", CORE_KINDS)
def test_plugs_into_mcts(core_kind):
    env, net = _net(core_kind=core_kind)
    obs = env.obs(env.reset()).flatten()
    for spec in [dict(backup="mean"), dict(backup="topk", topk_k=2), dict(backup="max")]:
        mcts = P.TropicalMCTS(net, P.MCTSConfig(num_simulations=16, root_exploration_frac=0.0, **spec))
        vp, rv = mcts.run(obs)
        assert vp.shape == (env.n_actions,)
        assert abs(float(vp.sum()) - 1.0) < 1e-4
        assert isinstance(rv, float)


@pytest.mark.parametrize("core_kind", CORE_KINDS)
def test_value_fit_trains(core_kind):
    env, net = _net(latent=32, tau=1.0, core_kind=core_kind)
    R, succ, _ = env.as_mdp()
    V, _ = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0)(
        torch.zeros(env.n_states), R, succ)
    X = torch.stack([env.obs(s).flatten() for s in range(env.n_states)])
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(120):
        v = net.initial(X)[2]
        loss = F.mse_loss(v, V)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.6 * losses[0]   # learns to fit V* substantially


def test_set_tau_runs():
    env, net = _net()
    net.set_tau(0.2)
    obs = env.obs(env.reset()).flatten()
    inf = net.initial_inference(obs)   # still produces valid output after annealing
    assert inf.policy.shape == (env.n_actions,)


# ---------------------------------------------------------------------------
# Stage 5: replay buffer targets, self-play, full loop
# ---------------------------------------------------------------------------


def test_replay_nstep_value_target():
    from tikitorch.planning.muzero import Game, ReplayBuffer
    A = 4
    g = Game()
    # 3-step episode, rewards -1 each, root_values arbitrary; check n-step target math.
    for t in range(3):
        g.obs.append(torch.zeros(2)); g.actions.append(0); g.rewards.append(-1.0)
        g.policies.append(torch.full((A,), 1.0 / A)); g.root_values.append(-5.0)
    buf = ReplayBuffer(n_actions=A); buf.save(g)
    # td=1 from pos 0: r0 + gamma*root_value[1] = -1 + (-5) = -6
    assert abs(buf._value_target(g, 0, td=1, gamma=1.0) - (-6.0)) < 1e-6
    # td=5 (beyond end) from pos 0: just sum rewards to end = -3, no bootstrap
    assert abs(buf._value_target(g, 0, td=5, gamma=1.0) - (-3.0)) < 1e-6
    # past terminal -> 0
    assert buf._value_target(g, 3, td=5, gamma=1.0) == 0.0


def test_selfplay_and_batch_shapes():
    from tikitorch.planning.muzero import play_game, ReplayBuffer
    env, net = _net()
    g = play_game(net, env, num_simulations=8, backup="topk", topk_k=2, seed=1)
    assert len(g) == len(g.obs) == len(g.rewards) == len(g.policies) == len(g.root_values)
    buf = ReplayBuffer(n_actions=env.n_actions); buf.save(g)
    obs0, acts, vt, rt, pt = buf.sample_batch(6, unroll=3, td=4, gamma=1.0)
    assert obs0.shape[0] == 6 and acts.shape == (6, 3) and vt.shape == (6, 4)
    assert rt.shape == (6, 3) and pt.shape == (6, 4, env.n_actions)


@pytest.mark.parametrize("core_kind", CORE_KINDS)
def test_agent_loop_runs(core_kind):
    from tikitorch.planning.muzero import GridWorldNav, TropicalMuZero, MuZeroConfig
    env = GridWorldNav(GRID)
    cfg = MuZeroConfig(core_kind=core_kind, latent_dim=16, num_simulations=8, iters=2,
                       games_per_iter=2, train_steps_per_iter=4, unroll=2, td_steps=3,
                       batch=16, eval_every=1)
    hist = TropicalMuZero(env, cfg).run(log=lambda m: None)
    assert len(hist) >= 1
    assert "return_" in hist[-1] and "solved" in hist[-1]


def test_dyadic_core_is_zero_dsp_and_oblique():
    """Audit dyadic cores, verify their exported power-of-two schedule, and check obliqueness.

    These are software/export properties, not a physical zero-DSP result; synthesis is separate.
    """
    import tikitorch as tt
    fpga = pytest.importorskip("tikitorch.fpga")
    env, net = _net(latent=16, core_kind="dyadic")
    # structural: cores are not flagged; only the nn.Linear heads are the declared boundary
    fcv = tt.audit.find_closure_violations
    assert fcv(net) == fcv(_net(core_kind="maxplus")[1])
    tt.assert_closed(net, allow={"Linear"})
    # The exported dyadic weights are exact powers of two and exactly reconstructable.
    sched = fpga.export_net(net)
    assert sched and fpga.verify_schedule(net, sched) == []
    # oblique: the dyadic repr encoder routes gradient to many inputs (vs maxplus ~1)
    x = torch.rand(1, 3 * env.H * env.W, requires_grad=True)
    net.repr.enc(x).sum().backward()
    assert int((x.grad.abs() > 1e-9).sum()) > 1


def test_dense_control_matches_interface_and_runs():
    # The matched dense MuZero control: same inference interface, plugs into MCTS,
    # and runs through the identical agent loop via net_kind="dense".
    from tikitorch.planning.muzero import (GridWorldNav, DenseMuZeroNet,
                                           TropicalMuZero, MuZeroConfig)
    import tikitorch.planning as P
    env = GridWorldNav(GRID)
    net = DenseMuZeroNet(3 * env.H * env.W, env.n_actions, latent_dim=16)
    obs = env.obs(env.reset()).flatten()
    inf = net.initial_inference(obs)
    assert inf.state.shape == (1, 16) and inf.policy.shape == (env.n_actions,)
    inf2, done = net.recurrent_inference(inf.state, 1)
    assert isinstance(inf2.reward, float) and done is False
    vp, _ = P.TropicalMCTS(net, P.MCTSConfig(num_simulations=8, backup="mean",
                                             root_exploration_frac=0.0)).run(obs)
    assert vp.shape == (env.n_actions,)
    cfg = MuZeroConfig(net_kind="dense", latent_dim=16, num_simulations=8, iters=2,
                       games_per_iter=2, train_steps_per_iter=4, batch=16, eval_every=1)
    agent = TropicalMuZero(env, cfg)
    assert type(agent.net).__name__ == "DenseMuZeroNet"
    hist = agent.run(log=lambda m: None)
    assert "agree" in hist[-1]
