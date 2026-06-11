"""run_chess_alphazero — does the AlphaZero-style insight (true model in the tree) rescue CHESS?

Pure-MuZero chess lost to a random mover (0.07): the learned-model tree explored the 4096-way action
space blindly. AlphaZeroBoardModel injects the TRUE rules at every node, so the tree branches only on
the ~30 LEGAL moves, with exact terminals — no net training required for that. This probes whether
depth alone (UNTRAINED dyadic net, true model in tree) makes chess play non-trivial: score vs a random
mover and a 1-ply material grabber at increasing sims, with the pure-MuZero tree as the contrast.
Multiplier-free throughout (net = shift+add; python-chess rules = logic).

    PYTHONPATH=<repo> python3 scripts/run_chess_alphazero.py
"""
import json
import os

import torch

torch.set_num_threads(12)

from tikitorch.planning.muzero.chess_game import Chess, random_agent, material_agent  # noqa: E402
from tikitorch.planning.muzero.games import AlphaZeroBoardModel, BoardGameModel, play_match  # noqa: E402
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig  # noqa: E402
from tikitorch.planning.mcts import TropicalMCTS, MCTSConfig  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "board_scaling"))


def agent(net, model_cls, sims):
    mc = TropicalMCTS(model_cls(net, Chess),
                      MCTSConfig(num_simulations=sims, c_puct=1.25, gamma=1.0,
                                 backup="mean", players=2, root_exploration_frac=0.0))
    return lambda s: int(mc.run(s)[0].argmax())


def score_vs(ag, opp_factory, n=12):
    w = d = 0
    for g in range(n):
        opp = opp_factory(g + 1)
        res = play_match(ag, opp, Chess) if g % 2 == 0 else play_match(opp, ag, Chess)
        me = 1 if g % 2 == 0 else 2
        w += res == me
        d += res == 0
    return round((w + 0.5 * d) / n, 3)


def main():
    net = BoardMuZero(Chess, BoardConfig(net_kind="tropical", core_kind="dyadic", latent_dim=64)).net
    net.eval()                                            # UNTRAINED — the search does the work
    rnd = lambda s: random_agent(s)                       # noqa: E731
    mat = lambda s: material_agent()                      # noqa: E731

    res = {"alphazero": {}, "muzero": {}}
    print("AlphaZero-style (TRUE rules in tree) + UNTRAINED dyadic net, chess:", flush=True)
    for sims in (50, 200):
        res["alphazero"][sims] = {"random": score_vs(agent(net, AlphaZeroBoardModel, sims), rnd),
                                  "material": score_vs(agent(net, AlphaZeroBoardModel, sims), mat)}
        print(f"  {sims:3d} sims:  vs random {res['alphazero'][sims]['random']}  "
              f"vs material {res['alphazero'][sims]['material']}", flush=True)
    print("contrast — pure-MuZero (learned model in tree) + same net @ 200 sims:", flush=True)
    res["muzero"][200] = {"random": score_vs(agent(net, BoardGameModel, 200), rnd),
                          "material": score_vs(agent(net, BoardGameModel, 200), mat)}
    print(f"  200 sims:  vs random {res['muzero'][200]['random']}  vs material {res['muzero'][200]['material']}",
          flush=True)

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "chess_alphazero.json"), "w").write(json.dumps(
        {"note": "chess: AlphaZero-style (true rules in tree, ~30 legal branch) vs pure-MuZero (4096-way "
                 "learned tree); UNTRAINED dyadic net, depth only. score=win+0.5draw, 12 games. "
                 "Pure-MuZero trained chess agent previously scored 0.07-0.14 vs random.",
         "results": res}, indent=2))
    print("\n[saved] chess_alphazero.json")
    print("DONE_CAZ")


if __name__ == "__main__":
    main()
