"""run_chess_strength — lift the multiply-free chess agent above the random floor.

The chess SCAFFOLD (run_chess_scaffold.py) used the WEAK config: learned-dynamics tree
(tree_model="muzero") + no anti-collapse, and trained pure-MuZero scored 0.071 vs random —
*below* its 0.393 untrained control (loss-down/strength-down decoupling; CHESS_SCAFFOLD.md).
Meanwhile the AlphaZero true-rules tree (legal moves at every node) already hits 0.625 vs random
UNTRAINED (chess_alphazero.json). This script applies the levers that fixed Connect4
(CONNECT4_PRIOR_SWEEP.md) to chess and asks the apples-to-apples question:

    with the STRONG (AlphaZero true-rules) tree, does TRAINING the multiply-free prior beat the
    UNTRAINED control (same tree)? i.e. does the learned prior add to search, instead of collapsing?

Levers: tree_model="alphazero" (biggest), policy_loss="ce" (full-dist soft-CE), entropy bonus +
target-temp (anti-collapse), best-snapshot tracking (MuZero oscillates). Net stays fully multiply-free
(dyadic cores + dyadic heads, closure []). Eval vs random + material(1-ply) (the scaffold's opponents,
so numbers are comparable), trained vs untrained-control, BOTH with the configured tree.

    PYTHONPATH=<repo> python3 scripts/run_chess_strength.py [--tree alphazero --sims 64 --iters 20 \
        --games 24 --entropy 0.1 --target-temp 1.5 --workers N --eval 30 --tag run]
"""
import argparse
import json
import os

import torch

from tikitorch.audit import find_closure_violations
from tikitorch.planning.muzero.chess_game import Chess, random_agent, material_agent
from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig
from tikitorch.planning.muzero.games import play_match

OUT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results", "board_scaling"))


def eval_vs(mz, opp_factory, n, label, tag):
    """Win/draw/loss vs an opponent, alternating colors; score = win + 0.5·draw."""
    me = mz._mcts_agent()
    w = d = ls = 0
    for g in range(n):
        opp = opp_factory(g + 1)
        my_id = 1 if g % 2 == 0 else 2
        agents = {my_id: me, 3 - my_id: opp}
        res = play_match(agents[1], agents[2], Chess)
        w += res == my_id
        d += res == 0
        ls += (res != 0 and res != my_id)
    r = dict(opponent=label, win=round(w / n, 3), draw=round(d / n, 3), loss=round(ls / n, 3),
             score=round((w + 0.5 * d) / n, 3))
    print(f"[{tag}]   vs {label:24s}: win {r['win']:.0%} draw {r['draw']:.0%} loss {r['loss']:.0%} "
          f"| score {r['score']:.2f}", flush=True)
    return r


def main(a):
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    cfg = BoardConfig(net_kind="tropical", core_kind="dyadic", dyadic_heads=True,
                      tree_model=a.tree, policy_loss=a.policy_loss,
                      policy_entropy_coef=a.entropy, target_temp=a.target_temp,
                      optimizer="tropicalsgd", latent_dim=64,
                      num_simulations=a.sims, iters=a.iters, games_per_iter=a.games,
                      train_steps_per_iter=a.steps, batch=128, unroll=3, num_workers=a.workers,
                      tau_start=1.0, tau_end=0.3, lr=a.lr, seed=0)
    mz = BoardMuZero(Chess, cfg)
    viol = find_closure_violations(mz.net)
    print(f"[{a.tag}] closure audit (empty = fully multiply-free): {viol or 'NONE'}", flush=True)
    print(f"[{a.tag}] chess strength: tree={a.tree} loss={a.policy_loss} entropy={a.entropy} "
          f"temp={a.target_temp} | {a.sims} sims, {a.iters}x{a.games} games, {a.workers} workers", flush=True)

    # ---- train with best-snapshot tracking (MuZero oscillates; keep the best-vs-random net) ----
    best_score, best_state, loss_curve = -1.0, None, []
    for it in range(cfg.iters):
        frac = it / max(cfg.iters - 1, 1)
        mz.current_tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac
        mz.net.set_tau(mz.current_tau)
        mz.self_play(temperature=1.0, base_seed=1 + it * 1000)
        losses = [mz.train_step() for _ in range(cfg.train_steps_per_iter)] if len(mz.buffer) else []
        avg = sum(losses) / len(losses) if losses else float("nan")
        loss_curve.append(round(avg, 4))
        mf = getattr(mz, "last_mode_frac", float("nan"))
        if it % a.eval_every == 0 or it == cfg.iters - 1:
            mz.net.eval()
            sc = eval_vs(mz, random_agent, a.eval_track, "random (track)", a.tag)["score"]
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().cpu().clone() for k, v in mz.net.state_dict().items()}
            mz.net.train()
        print(f"[{a.tag}] iter {it:3d}: games={len(mz.buffer)} loss={avg:.4f} mode_frac={mf:.3f} "
              f"best_track={best_score:.2f}", flush=True)
    if best_state is not None:
        mz.net.load_state_dict(best_state)
    mz.net.eval()

    # ---- final eval: trained (best snapshot) vs untrained control, SAME tree, vs random + material ----
    print(f"[{a.tag}] === TRAINED (best snapshot, tree={a.tree}) ===", flush=True)
    tr_rand = eval_vs(mz, random_agent, a.eval, "random", a.tag)
    tr_mat = eval_vs(mz, lambda s: material_agent(), a.eval, "material(1-ply)", a.tag)
    ctrl = BoardMuZero(Chess, cfg)
    ctrl.net.eval()
    print(f"[{a.tag}] === UNTRAINED control (same tree={a.tree}) ===", flush=True)
    un_rand = eval_vs(ctrl, random_agent, a.eval, "random", a.tag)
    un_mat = eval_vs(ctrl, lambda s: material_agent(), a.eval, "material(1-ply)", a.tag)

    res = {"note": "multiply-free chess (dyadic cores+heads, closure []), trained vs untrained control, "
                   "SAME tree. score=win+0.5draw. Compare to scaffold: trained-muzero-tree 0.071, "
                   "untrained-control 0.393, untrained-alphazero-tree 0.625 (chess_alphazero.json).",
           "config": dict(tree=a.tree, policy_loss=a.policy_loss, entropy=a.entropy, target_temp=a.target_temp,
                          sims=a.sims, iters=a.iters, games=a.games, eval=a.eval),
           "closure_violations": viol, "loss_curve": loss_curve, "best_track_score": round(best_score, 3),
           "trained": {"random": tr_rand, "material": tr_mat},
           "untrained": {"random": un_rand, "material": un_mat},
           "delta_vs_random": round(tr_rand["score"] - un_rand["score"], 3)}
    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f"chess_strength_{a.tag}.json")
    open(out, "w").write(json.dumps(res, indent=2))
    print(f"[{a.tag}] trained-vs-untrained Δ(random) = {res['delta_vs_random']:+.3f}  [saved] {out}", flush=True)
    print(f"DONE_CHESS_{a.tag}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tree", default="alphazero", choices=["alphazero", "muzero"])
    p.add_argument("--policy-loss", dest="policy_loss", default="ce", choices=["ce", "perceptron", "tropce"])
    p.add_argument("--entropy", dest="entropy", type=float, default=0.1)
    p.add_argument("--target-temp", dest="target_temp", type=float, default=1.5)
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--eval", type=int, default=30, help="final eval games per opponent")
    p.add_argument("--eval-track", dest="eval_track", type=int, default=12, help="cheap in-loop eval games")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=4)
    p.add_argument("--tag", default="run")
    main(p.parse_args())
