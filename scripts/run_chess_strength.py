"""run_chess_strength — test whether training lifts a dyadic chess agent above random.

The chess SCAFFOLD (run_chess_scaffold.py) used the WEAK config: learned-dynamics tree
(tree_model="muzero") + no anti-collapse, and trained pure-MuZero scored 0.071 vs random —
*below* its 0.393 untrained control (loss-down/strength-down decoupling; CHESS_SCAFFOLD.md).
Meanwhile the AlphaZero true-rules tree (legal moves at every node) already hits 0.625 vs random
UNTRAINED (chess_alphazero.json). This script applies the levers that fixed Connect4
(CONNECT4_PRIOR_SWEEP.md) to chess and asks the apples-to-apples question:

    with the AlphaZero-style explicit-game tree, does TRAINING the dyadic prior beat the
    UNTRAINED control (same tree)? i.e. does the learned prior add to search, instead of collapsing?

Levers: tree_model="alphazero" (biggest), policy_loss="ce" (full-dist soft-CE), entropy bonus +
target-temp (anti-collapse), best-snapshot tracking (MuZero oscillates). The dyadic model reports a
software closure audit, but exact loss/entropy and Python MCTS arithmetic remain outside that audit.
Eval is vs random + material(1-ply), trained vs untrained-control, both with the configured tree.

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


# --- parallel eval (chess eval is the bottleneck: ~84 s/game serial). Fan games across a Pool,
#     rebuilding the net per worker (same pattern as board_agent's self-play workers). ---
_EVAL = {}


def _eval_init(blueprint, state_dict, tree, sims):
    import torch as _t
    _t.set_num_threads(1)
    from tikitorch.planning.muzero.board_agent import build_muzero_net, _robust_mp_sharing
    _robust_mp_sharing()
    net = build_muzero_net(**blueprint)
    net.load_state_dict(state_dict)
    net.eval()
    _EVAL.update(net=net, tree=tree, sims=sims)


def _eval_one(args):
    g, opp_kind = args
    from tikitorch.planning.mcts import TropicalMCTS, MCTSConfig
    from tikitorch.planning.muzero.games import AlphaZeroBoardModel, BoardGameModel, play_match
    model = (AlphaZeroBoardModel(_EVAL["net"], Chess) if _EVAL["tree"] == "alphazero"
             else BoardGameModel(_EVAL["net"], Chess))
    mc = TropicalMCTS(model, MCTSConfig(num_simulations=_EVAL["sims"], c_puct=1.25, gamma=1.0,
                                        backup="mean", players=2, root_exploration_frac=0.0))
    me = lambda s: int(mc.run(s)[0].argmax())
    opp = random_agent(g + 1) if opp_kind == "random" else material_agent()
    my_id = 1 if g % 2 == 0 else 2
    agents = {my_id: me, 3 - my_id: opp}
    res = play_match(agents[1], agents[2], Chess)
    return (res == my_id, res == 0, res != 0 and res != my_id)


def eval_vs_par(mz, opp_kind, n, label, tag, workers):
    """Parallel win/draw/loss vs random|material, alternating colors; score = win + 0.5·draw."""
    import multiprocessing as mp
    from tikitorch.planning.muzero.board_agent import _robust_mp_sharing
    _robust_mp_sharing()
    bp = dict(mz._bp); bp["tau"] = mz.current_tau
    sd = {k: v.detach().cpu() for k, v in mz.net.state_dict().items()}
    w = d = ls = 0
    with mp.Pool(workers, initializer=_eval_init, initargs=(bp, sd, mz.cfg.tree_model, mz.cfg.num_simulations)) as pool:
        for (win, draw, loss) in pool.imap_unordered(_eval_one, [(g, opp_kind) for g in range(n)]):
            w += win; d += draw; ls += loss
    r = dict(opponent=label, win=round(w / n, 3), draw=round(d / n, 3), loss=round(ls / n, 3),
             score=round((w + 0.5 * d) / n, 3))
    print(f"[{tag}]   vs {label:24s}: win {r['win']:.0%} draw {r['draw']:.0%} loss {r['loss']:.0%} "
          f"| score {r['score']:.2f}", flush=True)
    return r


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
                      legal_mask=a.legal_mask,
                      optimizer="tropicalsgd", latent_dim=64,
                      num_simulations=a.sims, iters=a.iters, games_per_iter=a.games,
                      train_steps_per_iter=a.steps, batch=128, unroll=3, num_workers=a.workers,
                      tau_start=1.0, tau_end=0.3, lr=a.lr, seed=0)
    mz = BoardMuZero(Chess, cfg)
    viol = find_closure_violations(mz.net)
    print(f"[{a.tag}] model closure audit (does not cover loss/MCTS): {viol or 'NONE'}", flush=True)
    print(f"[{a.tag}] chess strength: tree={a.tree} loss={a.policy_loss} entropy={a.entropy} "
          f"temp={a.target_temp} | {a.sims} sims, {a.iters}x{a.games} games, {a.workers} workers", flush=True)

    # ---- train with best-snapshot tracking (MuZero oscillates; keep the best-vs-random net) ----
    import time as _time
    ew = a.eval_workers if a.eval_workers > 0 else a.workers
    best_score, best_state, loss_curve = -1.0, None, []
    t_sp = t_ev = 0.0
    for it in range(cfg.iters):
        frac = it / max(cfg.iters - 1, 1)
        mz.current_tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac
        mz.net.set_tau(mz.current_tau)
        _t = _time.time(); mz.self_play(temperature=1.0, base_seed=1 + it * 1000); t_sp += _time.time() - _t
        losses = [mz.train_step() for _ in range(cfg.train_steps_per_iter)] if len(mz.buffer) else []
        avg = sum(losses) / len(losses) if losses else float("nan")
        loss_curve.append(round(avg, 4))
        mf = getattr(mz, "last_mode_frac", float("nan"))
        if it % a.eval_every == 0 or it == cfg.iters - 1:
            mz.net.eval()
            _t = _time.time()
            sc = (eval_vs_par(mz, "random", a.eval_track, "random (track)", a.tag, ew) if ew > 1
                  else eval_vs(mz, random_agent, a.eval_track, "random (track)", a.tag))["score"]
            t_ev += _time.time() - _t
            if sc > best_score:
                best_score = sc
                best_state = {k: v.detach().cpu().clone() for k, v in mz.net.state_dict().items()}
            mz.net.train()
        print(f"[{a.tag}] iter {it:3d}: games={len(mz.buffer)} loss={avg:.4f} mode_frac={mf:.3f} "
              f"best_track={best_score:.2f} | cum self-play={t_sp/60:.1f}m eval={t_ev/60:.1f}m", flush=True)
    if best_state is not None:
        mz.net.load_state_dict(best_state)
    mz.net.eval()

    # ---- final eval: trained (best snapshot) vs untrained control, SAME tree, vs random + material ----
    ew = a.eval_workers if a.eval_workers > 0 else a.workers
    def final_eval(m, who):
        print(f"[{a.tag}] === {who} (tree={a.tree}) ===", flush=True)
        if ew > 1:
            return (eval_vs_par(m, "random", a.eval, "random", a.tag, ew),
                    eval_vs_par(m, "material", a.eval, "material(1-ply)", a.tag, ew))
        return (eval_vs(m, random_agent, a.eval, "random", a.tag),
                eval_vs(m, lambda s: material_agent(), a.eval, "material(1-ply)", a.tag))
    tr_rand, tr_mat = final_eval(mz, "TRAINED (best snapshot)")
    ctrl = BoardMuZero(Chess, cfg); ctrl.net.eval()
    un_rand, un_mat = final_eval(ctrl, "UNTRAINED control")

    res = {"note": "multiply-free chess (dyadic cores+heads, closure []), trained vs untrained control, "
                   "SAME tree. score=win+0.5draw. Compare to scaffold: trained-muzero-tree 0.071, "
                   "untrained-control 0.393, untrained-alphazero-tree 0.625 (chess_alphazero.json).",
           "config": dict(tree=a.tree, policy_loss=a.policy_loss, entropy=a.entropy, target_temp=a.target_temp,
                          legal_mask=a.legal_mask, sims=a.sims, iters=a.iters, games=a.games, eval=a.eval),
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
    p.add_argument("--legal-mask", dest="legal_mask", action="store_true",
                   help="mask illegal-move logits in the policy loss (canonical AlphaZero/MuZero; multiply-free)")
    p.add_argument("--sims", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--eval", type=int, default=30, help="final eval games per opponent")
    p.add_argument("--eval-track", dest="eval_track", type=int, default=12, help="cheap in-loop eval games")
    p.add_argument("--eval-workers", dest="eval_workers", type=int, default=0, help="parallel eval workers (0=serial; chess eval is the bottleneck)")
    p.add_argument("--eval-every", dest="eval_every", type=int, default=4)
    p.add_argument("--tag", default="run")
    main(p.parse_args())
