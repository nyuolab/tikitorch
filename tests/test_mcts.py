"""Stage-3 tests: tropical MCTS validated against exact value iteration.

A known (perfect) model + a value-iteration oracle should make MCTS selection +
max backup pick optimal actions and plan optimal paths -- isolating the search
machinery from any model learning. Also checks determinism and that the engine
runs with the mean-backup variant (for the max-vs-mean ablation).
"""
import pytest
import torch

import tikitorch.planning as P
from tikitorch.planning.muzero.env import GridWorldNav, KnownModel

# Connected grids (every free cell can reach the goal), so V* is finite everywhere.
GRID_OPEN = ["S...", "....", "....", "...G"]
GRID_WALL = ["S..", ".#.", "..G"]


def _vi_oracle(env, sweeps=None):
    R, succ, _ = env.as_mdp()
    vi = P.TropicalValueIteration(env.n_states, env.n_actions, gamma=1.0, n_sweeps=sweeps)
    V, policy = vi(torch.zeros(env.n_states), R, succ)
    return V, policy, R, succ


def _optimal_actions(s, V, R, succ, tol=1e-5):
    """All actions achieving the Bellman optimum at state s."""
    best = V[s].item()
    return {a for a in range(R.shape[1])
            if abs((R[s, a] + V[succ[s, a]]).item() - best) < tol}


@pytest.mark.parametrize("grid", [GRID_OPEN, GRID_WALL])
def test_value_iteration_recovers_cost_to_go(grid):
    env = GridWorldNav(grid)
    V, _, _, _ = _vi_oracle(env)
    # Reward is -1/step, so V[s] = -(shortest-path length to goal): goal is 0,
    # all others strictly negative.
    assert abs(V[env.goal_state].item()) < 1e-6
    for s in range(env.n_states):
        assert V[s].item() <= 1e-6
        if s != env.goal_state:
            assert V[s].item() <= -1 + 1e-6


@pytest.mark.parametrize("grid", [GRID_OPEN, GRID_WALL])
def test_mcts_with_oracle_selects_optimal_action(grid):
    env = GridWorldNav(grid)
    V, _, R, succ = _vi_oracle(env)
    model = KnownModel(env, value_fn=lambda s: V[s].item(), gamma=1.0)
    cfg = P.MCTSConfig(num_simulations=64, backup="max", root_exploration_frac=0.0)
    mcts = P.TropicalMCTS(model, cfg)

    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        visit_policy, root_value = mcts.run(s)
        chosen = int(visit_policy.argmax())
        assert chosen in _optimal_actions(s, V, R, succ), f"state {s}: chose {chosen}"
        # backed-up root value matches the optimum
        assert abs(root_value - V[s].item()) < 1e-4


def test_mcts_greedy_episode_is_optimal_length():
    env = GridWorldNav(GRID_OPEN)
    V, _, _, _ = _vi_oracle(env)
    model = KnownModel(env, value_fn=lambda s: V[s].item(), gamma=1.0)
    mcts = P.TropicalMCTS(model, P.MCTSConfig(num_simulations=64, backup="max"))

    start = env.reset()
    optimal_len = int(round(-V[start].item()))
    steps = 0
    s = start
    done = False
    while not done and steps < env.n_states + 1:
        visit_policy, _ = mcts.run(s)
        a = int(visit_policy.argmax())
        s, _, done = env.step_state(s, a)
        steps += 1
    assert done and steps == optimal_len


def test_mcts_deterministic_without_noise():
    env = GridWorldNav(GRID_OPEN)
    V, _, _, _ = _vi_oracle(env)
    model = KnownModel(env, value_fn=lambda s: V[s].item(), gamma=1.0)
    cfg = P.MCTSConfig(num_simulations=32, backup="max", root_exploration_frac=0.0)
    p1, v1 = P.TropicalMCTS(model, cfg).run(env.reset())
    p2, v2 = P.TropicalMCTS(model, cfg).run(env.reset())
    torch.testing.assert_close(p1, p2)
    assert v1 == v2


def test_mean_backup_runs_and_is_no_better_than_max():
    # With the value oracle both should pick optimal actions; this exercises the
    # mean-backup path and confirms max is at least as good (the ablation knob).
    env = GridWorldNav(GRID_WALL)
    V, _, R, succ = _vi_oracle(env)
    model = KnownModel(env, value_fn=lambda s: V[s].item(), gamma=1.0)

    def agreement(backup):
        mcts = P.TropicalMCTS(model, P.MCTSConfig(num_simulations=64, backup=backup,
                                                  root_exploration_frac=0.0))
        hits = 0
        total = 0
        for s in range(env.n_states):
            if env.is_terminal_state(s):
                continue
            total += 1
            vp, _ = mcts.run(s)
            if int(vp.argmax()) in _optimal_actions(s, V, R, succ):
                hits += 1
        return hits / total

    assert agreement("max") >= agreement("mean")
    assert agreement("max") == 1.0


# ---------------------------------------------------------------------------
# top-k softened backup: interpolation limits (k=1 == max, large k == mean)
# ---------------------------------------------------------------------------


def _run_from_all(env, value_fn, cfg):
    mcts = P.TropicalMCTS(KnownModel(env, value_fn=value_fn, gamma=1.0), cfg)
    pols, vals = [], []
    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        vp, rv = mcts.run(s)
        pols.append(vp)
        vals.append(rv)
    return pols, vals


def test_topk_k1_equals_max():
    env = GridWorldNav(GRID_WALL)
    V, _, _, _ = _vi_oracle(env)
    vfn = lambda s: V[s].item()
    sims = 48
    pols_max, vals_max = _run_from_all(
        env, vfn, P.MCTSConfig(num_simulations=sims, backup="max", root_exploration_frac=0.0))
    pols_topk, vals_topk = _run_from_all(
        env, vfn, P.MCTSConfig(num_simulations=sims, backup="topk", topk_k=1,
                               root_exploration_frac=0.0))
    for pm, pk in zip(pols_max, pols_topk):
        torch.testing.assert_close(pm, pk)
    assert vals_max == vals_topk


def test_topk_large_k_approaches_mean():
    env = GridWorldNav(GRID_WALL)
    V, _, _, _ = _vi_oracle(env)
    vfn = lambda s: V[s].item()
    sims = 48
    # k >= num_simulations -> the heap holds every return -> mean.
    _, vals_mean = _run_from_all(
        env, vfn, P.MCTSConfig(num_simulations=sims, backup="mean", root_exploration_frac=0.0))
    _, vals_topk = _run_from_all(
        env, vfn, P.MCTSConfig(num_simulations=sims, backup="topk", topk_k=10_000,
                               root_exploration_frac=0.0))
    for vm, vk in zip(vals_mean, vals_topk):
        assert abs(vm - vk) < 1e-6


def test_diagonal_env_has_eight_actions_and_mcts_optimal():
    # 8-connected grid (branching factor A=8) for the k~=A coupling study.
    env = GridWorldNav(GRID_OPEN, diagonal=True)
    assert env.n_actions == 8
    R, succ, _ = env.as_mdp()
    assert R.shape == (env.n_states, 8) and succ.shape == (env.n_states, 8)
    V, _, R2, succ2 = _vi_oracle(env)
    vfn = lambda s: V[s].item()
    mcts = P.TropicalMCTS(KnownModel(env, value_fn=vfn, gamma=1.0),
                          P.MCTSConfig(num_simulations=96, backup="topk", topk_k=2,
                                       root_exploration_frac=0.0))
    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        vp, _ = mcts.run(s)
        assert int(vp.argmax()) in _optimal_actions(s, V, R2, succ2)


def test_topk_runs_under_oracle_and_is_optimal():
    env = GridWorldNav(GRID_WALL)
    V, R, succ = _vi_oracle(env)[0], *_vi_oracle(env)[2:]
    vfn = lambda s: V[s].item()
    mcts = P.TropicalMCTS(KnownModel(env, value_fn=vfn, gamma=1.0),
                          P.MCTSConfig(num_simulations=64, backup="topk", topk_k=4,
                                       root_exploration_frac=0.0))
    for s in range(env.n_states):
        if env.is_terminal_state(s):
            continue
        vp, _ = mcts.run(s)
        assert int(vp.argmax()) in _optimal_actions(s, V, R, succ)


def test_root_noise_is_seeded_and_global_rng_isolated():
    """Root Dirichlet noise (root_exploration_frac>0) is reproducible from cfg.seed and does
    NOT mutate the process-global torch RNG — the parallel-actor / repro fix (mcts.py:170)."""
    env = GridWorldNav(GRID_OPEN)
    m = KnownModel(env)

    def run(seed):
        cfg = P.MCTSConfig(num_simulations=12, root_exploration_frac=0.25,
                           dirichlet_alpha=0.3, seed=seed)
        return P.TropicalMCTS(m, cfg).run(0)[0]      # state 0; noise is applied at the root

    torch.manual_seed(999); before = torch.randn(2)
    vp1, vp2 = run(3), run(3)
    torch.manual_seed(999); after = torch.randn(2)
    assert torch.equal(before, after)                # MCTS noise left the global torch RNG untouched
    assert torch.allclose(vp1, vp2)                  # same seed -> reproducible visit policy
    assert not torch.allclose(vp1, run(4))           # different seed -> different noise
