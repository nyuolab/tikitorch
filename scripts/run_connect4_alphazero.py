"""run_connect4_alphazero — the competitive Connect4 path: AlphaZero-style MCTS (TRUE game model in
the tree) + the zero-DSP dyadic net as prior/leaf-value. Quantifies (a) that depth beats negamax-d2/d3
and (b) that TRAINING the net buys sim-efficiency — the trained net reaches a given strength at FEWER
simulations than an untrained one (the net's real job when the simulator does the lookahead).

All multiplier-free: the net is the shift+add kernel; the tree is integer game-rule logic.

    PYTHONPATH=<repo> python3 scripts/run_connect4_alphazero.py
"""
import json
import os

import torch

torch.set_num_threads(12)

from tikitorch.planning.muzero.games import Connect4, negamax_agent, random_opening, play_match  # noqa: E402
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig  # noqa: E402
from tikitorch.planning.mcts import TropicalMCTS, MCTSConfig  # noqa: E402
from tikitorch.planning.muzero.games import AlphaZeroBoardModel  # noqa: E402

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "board_scaling"))


def az_agent(net, game, sims):
    mc = TropicalMCTS(AlphaZeroBoardModel(net, game),
                      MCTSConfig(num_simulations=sims, c_puct=1.25, gamma=1.0,
                                 backup="mean", players=2, root_exploration_frac=0.0))
    return lambda s: int(mc.run(s)[0].argmax())


def score_vs(net, game, sims, depth, n=30):
    ag = az_agent(net, game, sims)
    opp = negamax_agent(game, depth)
    rng = torch.Generator().manual_seed(31 + depth)
    w = d = 0
    for g in range(n):
        start = random_opening(game, rng, 4)
        me = 1 if g % 2 == 0 else 2
        ags = {me: ag, 3 - me: opp}
        r = play_match(ags[1], ags[2], game, start_state=start)
        w += r == me
        d += r == 0
    return round((w + 0.5 * d) / n, 3)


def main():
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", optimizer="tropicalsgd",
                      tree_model="alphazero", latent_dim=64, num_simulations=120, iters=30,
                      games_per_iter=24, train_steps_per_iter=30, batch=128, unroll=3,
                      num_workers=12, tau_start=1.0, tau_end=0.1, lr=3e-3, seed=0,
                      eval_every=999, eval_games=2)
    print("training dyadic net via AlphaZero-style self-play (120 sims, 720 games)...", flush=True)
    tnet = BoardMuZero(Connect4, cfg)
    tnet.run(log=lambda s: None)                     # trains tnet.net in place (best snapshot loaded)
    unet = BoardMuZero(Connect4, cfg)                # fresh = untrained control

    sims_grid = [50, 100, 200]
    res = {"trained": {}, "untrained": {}}
    for label, mz in (("trained", tnet), ("untrained", unet)):
        for sims in sims_grid:
            res[label][sims] = {d: score_vs(mz.net, Connect4, sims, d) for d in (2, 3)}
            print(f"  {label:9s} @ {sims:3d} sims:  d2={res[label][sims][2]}  d3={res[label][sims][3]}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "connect4_alphazero.json"), "w").write(json.dumps(
        {"note": "AlphaZero-style (true model in tree) + dyadic zero-DSP net; score=win+0.5draw, "
                 "random openings, 30 games. Depth beats d2/d3; this measures what (light) training "
                 "buys on top (sim-efficiency, if any) — interpret vs n=30 noise (SE~0.09).",
         "results": res}, indent=2))
    md = ["# Connect4 — AlphaZero-style (true model in tree) + zero-DSP dyadic net", "",
          "_Score = win + 0.5*draw vs depth-limited negamax, random openings (30 games). The TRUE game "
          "model in the MCTS tree (exact transitions/terminals/legality) + the dyadic net as prior/leaf "
          "value. Still multiplier-free (net = shift+add; rules = integer logic). Shows (a) depth beats "
          "the negamax-d2/d3 wall that pure-MuZero couldn't; (b) what light training buys on top "
          "(sim-efficiency, if any) — read vs n=30 noise._", "",
          "| sims | trained vs d2 | untrained vs d2 | trained vs d3 | untrained vs d3 |",
          "|---|---|---|---|---|"]
    for sims in sims_grid:
        t, u = res["trained"][sims], res["untrained"][sims]
        md.append(f"| {sims} | {t[2]} | {u[2]} | {t[3]} | {u[3]} |")
    open(os.path.join(OUT, "CONNECT4_ALPHAZERO.md"), "w").write("\n".join(md))
    print("\n[saved] CONNECT4_ALPHAZERO.md")
    print("DONE_AZ")


if __name__ == "__main__":
    main()
