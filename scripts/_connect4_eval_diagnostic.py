"""Diagnose the Connect4 'plateau': is it a real learning ceiling or a deterministic-EVAL artifact?

The trained-net gauntlet plays a deterministic MCTS agent (root noise off) vs deterministic negamax,
so agent-as-X is the SAME game every time -> the 30-game gauntlet is effectively 2 games, score
quantized to {0,0.5,1.0} and blind to gradual skill (explains the invariant 0.5/0.0/0.0 across all
prior runs). This re-measures a properly-trained dyadic net under a STOCHASTIC, random-opening eval
(many distinct games, both colors) vs random / negamax-d1/d2/d3, against an untrained control.

    PYTHONPATH=<repo> python3 scripts/_connect4_eval_diagnostic.py
"""
import json
import os

import torch

torch.set_num_threads(12)

from tikitorch.planning.muzero.games import Connect4, negamax_agent  # noqa: E402
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "manuscript", "results", "board_scaling")


def random_opening(seed, n_plies):
    g = torch.Generator().manual_seed(seed)
    s = Connect4.initial_state()
    for _ in range(n_plies):
        if Connect4.is_terminal(s[0]):
            break
        b, tp = s
        la = Connect4.legal_actions(b)
        mv = la[int(torch.randint(0, len(la), (1,), generator=g).item())]
        s = (Connect4.apply(b, mv, tp), 3 - tp)
    return s


def eval_random_openings(agent, opp, n_games=60):
    """Score = win + 0.5*draw for `agent`, over n_games DISTINCT games from random 4/5-ply openings
    (even/odd opening length alternates which color the agent then plays). agent/opp are state->action."""
    w = d = 0
    for g in range(n_games):
        n_plies = 4 + (g % 2)                       # 4 -> agent plays player 1, 5 -> player 2
        s = random_opening(1000 + g, n_plies)
        agent_id = s[1]                             # agent takes whoever is to move after the opening
        ags = {agent_id: agent, 3 - agent_id: opp}
        while not Connect4.is_terminal(s[0]):
            b, tp = s
            s = (Connect4.apply(b, ags[tp](s), tp), 3 - tp)
        res = Connect4.winner(s[0])
        w += res == agent_id
        d += res == 0
    return round((w + 0.5 * d) / n_games, 3)


def random_agent(seed=1):
    gg = torch.Generator().manual_seed(seed)

    def act(state):
        la = Connect4.legal_actions(state[0])
        return la[int(torch.randint(0, len(la), (1,), generator=gg).item())]
    return act


def main():
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", optimizer="tropicalsgd",
                      latent_dim=64, num_simulations=200, iters=40, games_per_iter=40,
                      train_steps_per_iter=30, batch=128, unroll=3, num_workers=12,
                      eval_every=999, eval_games=2, tau_start=1.0, tau_end=0.1, lr=3e-3, seed=0)
    print("training dyadic+tropicalsgd Connect4 (200 sims, 1600 games)...", flush=True)
    trained = BoardMuZero(Connect4, cfg)
    trained.run(log=lambda s: None)
    untrained = BoardMuZero(Connect4, cfg)             # fresh net = control

    # eval agents at 100 sims (cheaper than training sims; root noise off)
    cfg.num_simulations = 100
    t_agent = trained._mcts_agent()
    u_agent = untrained._mcts_agent()

    opps = {"random": random_agent(7), "negamax-d1": negamax_agent(Connect4, 1),
            "negamax-d2": negamax_agent(Connect4, 2), "negamax-d3": negamax_agent(Connect4, 3)}
    res = {"trained": {}, "untrained": {}}
    for name, opp in opps.items():
        res["trained"][name] = eval_random_openings(t_agent, opp, 60)
        res["untrained"][name] = eval_random_openings(u_agent, opp, 60)
        print(f"  vs {name:12s}: trained {res['trained'][name]:.3f} | untrained {res['untrained'][name]:.3f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "connect4_eval_diagnostic.json"), "w").write(json.dumps(
        {"note": "random-opening stochastic eval (60 distinct games each) vs the deterministic gauntlet",
         "config": "dyadic+tropicalsgd, 200 sims train / 100 sims eval, 1600 self-play games",
         "score": "win + 0.5*draw from agent perspective", "results": res}, indent=2))
    print("\nGauntlet (deterministic, prior runs) said: 0.5 / 0.0 / 0.0 vs d1/d2/d3.")
    print("Random floor (random mover, measured): 0.175 / 0.025 / 0.025 vs d1/d2/d3.")
    print("DONE_DIAG")


if __name__ == "__main__":
    main()
