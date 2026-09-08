"""muzero_train_diagnostics — root-cause WHY tropical MuZero training hurts (chess trained 0.071 <
untrained 0.393; loss falls while strength doesn't). Step 1 of the FPGA roadmap: a CHEAP, CPU-only
test of the binding hypothesis before any further fabric work.

The binding question: is the failure (a) the OPTIMIZER CLASS — closed SignSGD/TropicalSGD is
``W -= lr*sign(grad)``, which DISCARDS gradient magnitude and so may be unable to regress a smooth /
value-like target — or (b) the self-play LOOP DYNAMICS (non-stationary targets, prior collapse), or
(c) genuine compute/data starvation?

This file isolates (a) with a STATIONARY, controlled bisect (no self-play confound), and — crucially —
tests it CONSTRUCTIVELY: alongside the sign-only suspect and the open AdamW ceiling, it runs the
multiply-free MAGNITUDE-AWARE optimizer family (FrexpMagnitudeSGD / TropicalAdam / CountAgreement /
ExponentEMAAdam) that already exists in tikitorch.optim. If sign-only underfits the smooth value target
but a multiply-free magnitude-aware optimizer recovers AdamW-class fit, the root cause is the optimizer
AND a zero-DSP fix exists (no new learning rule needed, fabric program stays alive).

Two sub-tests on the SAME dyadic net (TropicalMuZeroNet, core_kind="dyadic" — the scaffold net), all
runs starting from byte-identical init weights, each optimizer reported at the BEST of a small
hyperparameter sweep (no straw-manning):
  * value-fit   : regress the net's value path to a SMOOTH MC-rollout value in [-1,1] (genuinely
                  value-like / oblique). Metric: L1 + R^2 vs predict-the-mean baseline.
  * policy-fit  : fit the net's policy to a depth-limited negamax move distribution. Metric: CE,
                  top-1 agreement, mode_frac (collapse), entropy.

    PYTHONPATH=. python3 scripts/muzero_train_diagnostics.py optbisect [--positions 512 --steps 800]
"""
import argparse
import copy
import json
import os
import random

import torch
import torch.nn.functional as F

import tikitorch.optim as ttopt
from tikitorch.planning.muzero.games import Connect4
from tikitorch.planning.muzero.networks import TropicalMuZeroNet

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "manuscript", "results", "board_scaling", "diagnostics"))


# --------------------------------------------------------------------------- data
def _mc_value(board, tp, rollouts, rng):
    """Smooth value-like target: mean random-rollout outcome from ``to_play``'s perspective, in [-1,1]."""
    tot = 0.0
    for _ in range(rollouts):
        b, p = board, tp
        while not Connect4.is_terminal(b):
            b = Connect4.apply(b, rng.choice(Connect4.legal_actions(b)), p)
            p = 3 - p
        w = Connect4.winner(b)
        tot += 0.0 if w == 0 else (1.0 if w == tp else -1.0)
    return tot / rollouts


def _negamax_policy(board, tp, depth, temp=0.5):
    """A move distribution over the 7 columns from depth-limited negamax values (mass on legal moves)."""
    from tikitorch.planning.muzero.games import negamax_value
    legal = Connect4.legal_actions(board)
    scores = torch.full((7,), -1e9)
    for a in legal:
        scores[a] = -negamax_value(Connect4, Connect4.apply(board, a, tp), 3 - tp, depth - 1)
    return F.softmax(scores / temp, dim=-1)


def collect_positions(n_pos, rollouts, depth, seed):
    """A FROZEN, stationary supervised set of Connect4 positions: (obs, smooth MC value, negamax policy).
    Positions come from uniform-random self-play (diverse, on-distribution-ish for the board)."""
    rng = random.Random(seed)
    obs, vval, pol = [], [], []
    while len(obs) < n_pos:
        board, tp = Connect4.initial_state()
        states = []
        while not Connect4.is_terminal(board):
            states.append((board, tp))
            board = Connect4.apply(board, rng.choice(Connect4.legal_actions(board)), tp)
            tp = 3 - tp
        for (b, p) in rng.sample(states, min(3, len(states))):
            obs.append(Connect4.obs((b, p)))
            vval.append(_mc_value(b, p, rollouts, rng))
            pol.append(_negamax_policy(b, p, depth))
            if len(obs) >= n_pos:
                break
    return (torch.stack(obs), torch.tensor(vval, dtype=torch.float32), torch.stack(pol))


# --------------------------------------------------------------------------- optimizers
def optimizer_grid():
    """name -> list of (label, builder(params)). Each optimizer gets a small sweep; we report its BEST so
    no optimizer is straw-manned. Tags: 'sign'=sign-only (zero-DSP, the suspect); 'open'=AdamW (the
    multiply ceiling); 'magfree'=multiply-free magnitude-aware (the candidate zero-DSP fix)."""
    A = torch.optim.AdamW
    return {
        "tropicalsgd (sign-only)": ("sign", [
            (f"lr={lr}", (lambda lr: (lambda p: ttopt.TropicalSGD(p, lr=lr)))(lr)) for lr in (1e-3, 3e-3, 1e-2)]),
        "signsgd+mom0.9": ("sign_mom", [    # sign-of-float-EMA: recovers magnitude but NOT zero-DSP (hole H7)
            (f"lr={lr}", (lambda lr: (lambda p: ttopt.SignSGD(p, lr=lr, momentum=0.9)))(lr)) for lr in (1e-3, 3e-3, 1e-2)]),
        "adamw (open ceiling)": ("open", [
            (f"lr={lr}", (lambda lr: (lambda p: A(p, lr=lr)))(lr)) for lr in (3e-3, 1e-2, 3e-2)]),
        "frexpmag (magfree)": ("magfree", [
            (f"bs={bs}", (lambda bs: (lambda p: ttopt.FrexpMagnitudeSGD(p, base_shift=bs)))(bs)) for bs in (6, 8, 10)]),
        "tropicaladam (magfree)": ("magfree", [
            (f"si={si}", (lambda si: (lambda p: ttopt.TropicalAdam(p, step_init=si)))(si)) for si in (0.03125, 0.0625, 0.125)]),
        "countagreement (magfree)": ("magfree", [
            (f"bs={bs}", (lambda bs: (lambda p: ttopt.CountAgreementSGD(p, base_shift=bs)))(bs)) for bs in (6, 8, 10)]),
        "exponentemaadam (magfree)": ("magfree", [
            (f"bs={bs}", (lambda bs: (lambda p: ttopt.ExponentEMAAdam(p, base_shift=bs)))(bs)) for bs in (6, 8, 10)]),
    }


# --------------------------------------------------------------------------- fit loops
def _fresh(net0):
    net = copy.deepcopy(net0)
    net.train()
    return net


def fit_value(net0, obs, target, build_opt, steps):
    """Full-batch regress net value path -> smooth target (L1, matching ScalarHead.loss). Returns best L1."""
    net = _fresh(net0)
    opt = build_opt(net.parameters())
    best = float("inf")
    for _ in range(steps):
        _, _, v = net.initial(obs)
        v = net.value_scalar(v)
        loss = F.l1_loss(v, target)
        opt.zero_grad(); loss.backward(); opt.step()
        best = min(best, float(loss.detach()))
    return best


def fit_policy(net0, obs, target, build_opt, steps):
    """Full-batch fit net policy -> negamax distribution (soft CE). Returns dict of final metrics."""
    net = _fresh(net0)
    opt = build_opt(net.parameters())
    best_ce = float("inf"); snap = None
    for _ in range(steps):
        _, p, _ = net.initial(obs)
        loss = -(target * F.log_softmax(p, -1)).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        ce = float(loss.detach())
        if ce < best_ce:
            best_ce = ce
            with torch.no_grad():
                _, pp, _ = net.initial(obs)
                sm = pp.softmax(-1)
                snap = dict(ce=ce,
                            top1=float((pp.argmax(-1) == target.argmax(-1)).float().mean()),
                            mode_frac=float(sm.max(-1).values.mean()),
                            entropy=float(-(sm * sm.clamp_min(1e-9).log()).sum(-1).mean()))
    return snap


def run_optbisect(a):
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(0)
    print(f"[data] collecting {a.positions} Connect4 positions "
          f"(mc_rollouts={a.rollouts}, negamax_depth={a.depth}) ...", flush=True)
    obs, vtar, ptar = collect_positions(a.positions, a.rollouts, a.depth, seed=1234)
    vmean = float(vtar.mean()); vbase = float((vtar - vtar.mean()).abs().mean())  # predict-the-mean L1
    vvar = float(((vtar - vtar.mean()) ** 2).mean())
    pent_uniform = float(torch.log(torch.tensor(7.0)))
    print(f"[data] value target: mean={vmean:+.3f} std={vtar.std():.3f}  "
          f"predict-mean L1 baseline={vbase:.3f}  |  policy target mean entropy="
          f"{float(-(ptar*ptar.clamp_min(1e-9).log()).sum(-1).mean()):.3f} (uniform={pent_uniform:.3f})", flush=True)

    # ONE seeded init net; every optimizer run starts from a byte-identical copy of it.
    net0 = TropicalMuZeroNet(Connect4.obs_dim, Connect4.n_actions, latent_dim=a.latent, core_kind="dyadic")
    net0.set_tau(1.0)

    grid = optimizer_grid()
    results = []
    for name, (tag, settings) in grid.items():
        best_v = (float("inf"), None)
        best_p = (float("inf"), None)
        for label, build in settings:
            vL1 = fit_value(net0, obs, vtar, build, a.steps)
            if vL1 < best_v[0]:
                best_v = (vL1, label)
            psnap = fit_policy(net0, obs, ptar, build, a.steps)
            if psnap["ce"] < best_p[0]:
                best_p = (psnap["ce"], label, psnap)
        vL1, vlab = best_v
        # value-fit quality as fraction of the predict-the-mean L1 baseline removed (1 - vL1/vbase);
        # higher = better, 0 = no better than predicting the constant mean.
        vr = 1.0 - vL1 / vbase
        ce, plab, ps = best_p[0], best_p[1], best_p[2]
        results.append(dict(optimizer=name, tag=tag,
                            value_L1=round(vL1, 4), value_setting=vlab,
                            value_redux=round(vr, 3),         # fraction of predict-mean L1 removed (higher=better)
                            policy_CE=round(ce, 4), policy_setting=plab,
                            policy_top1=round(ps["top1"], 3), policy_mode_frac=round(ps["mode_frac"], 3),
                            policy_entropy=round(ps["entropy"], 3)))
        print(f"  {name:28s} | valL1 {vL1:.3f} ({vr:+.0%} vs mean, {vlab:>7s}) | "
              f"polCE {ce:.3f} top1 {ps['top1']:.0%} mode {ps['mode_frac']:.0%} ent {ps['entropy']:.2f}", flush=True)

    summary = dict(task="connect4_supervised_stationary",
                   positions=a.positions, rollouts=a.rollouts, depth=a.depth, steps=a.steps,
                   latent=a.latent, value_predict_mean_L1=round(vbase, 4),
                   value_target_std=round(float(vtar.std()), 4), value_target_var=round(vvar, 4),
                   policy_uniform_entropy=round(pent_uniform, 4), results=results)
    path = os.path.join(OUT, "optbisect_connect4.json")
    open(path, "w").write(json.dumps(summary, indent=2))
    print(f"\n[saved] {path}")
    _print_verdict(results, vbase)


def _print_verdict(results, vbase):
    sign = min((r for r in results if r["tag"] == "sign"), key=lambda r: r["value_L1"])
    openr = next(r for r in results if r["tag"] == "open")
    magf = min((r for r in results if r["tag"] == "magfree"), key=lambda r: r["value_L1"])
    print("\n=== READOUT (value-fit = the value-like/smooth-target test) ===")
    print(f"  sign-only best : {sign['optimizer']:28s} L1 {sign['value_L1']:.3f} ({sign['value_redux']:+.0%} vs mean)")
    print(f"  AdamW ceiling  : {openr['optimizer']:28s} L1 {openr['value_L1']:.3f} ({openr['value_redux']:+.0%} vs mean)")
    print(f"  magfree best   : {magf['optimizer']:28s} L1 {magf['value_L1']:.3f} ({magf['value_redux']:+.0%} vs mean)")
    gap_sign = sign["value_L1"] - openr["value_L1"]
    gap_magf = magf["value_L1"] - openr["value_L1"]
    print(f"\n  sign-only vs AdamW value-L1 gap : {gap_sign:+.3f}  -> {'SIGN-ONLY UNDERFITS smooth value' if gap_sign > 0.02 else 'sign-only ~matches Adam (optimizer NOT the value bottleneck)'}")
    print(f"  magfree   vs AdamW value-L1 gap : {gap_magf:+.3f}  -> {'a ZERO-DSP magnitude-aware optimizer recovers Adam-class fit' if gap_magf < gap_sign - 0.01 else 'magfree did not clearly beat sign-only'}")


# --------------------------------------------------------------------------- online (in-situ)
def build_named_optimizer(name, params):
    """Single good setting per optimizer for the in-situ loop (offline bisect tunes the grid)."""
    if name == "adamw":
        return torch.optim.AdamW(params, lr=1e-2)
    if name == "tropicalsgd":
        return ttopt.TropicalSGD(params, lr=3e-3)
    if name == "signsgd_m9":
        return ttopt.SignSGD(params, lr=3e-3, momentum=0.9)
    if name == "tropicaladam":
        return ttopt.TropicalAdam(params, step_init=0.0625)
    if name == "frexpmag":
        return ttopt.FrexpMagnitudeSGD(params, base_shift=8)
    if name == "countagreement":
        return ttopt.CountAgreementSGD(params, base_shift=8)
    raise ValueError(name)


RECIPES = {
    "ce":            dict(policy_loss="ce", policy_entropy_coef=0.0, target_temp=1.0),
    "ce_ent0.1":     dict(policy_loss="ce", policy_entropy_coef=0.1, target_temp=1.0),
    "ce_ent0.1_tt2": dict(policy_loss="ce", policy_entropy_coef=0.1, target_temp=2.0),
}


def _random_agent(game, seed):
    gen = torch.Generator().manual_seed(seed)
    def act(state):
        board, tp = state
        legal = game.legal_actions(board)
        return legal[int(torch.randint(0, len(legal), (1,), generator=gen).item())]
    return act


def _score_vs(agent, opp_factory, game, n, open_plies, seed):
    """Score (win + 0.5 draw)/n for ``agent`` vs a fresh opponent each game, alternating color, from
    random openings (avoids the deterministic-vs-deterministic single-game collapse)."""
    from tikitorch.planning.muzero.games import play_match, random_opening
    rng = torch.Generator().manual_seed(seed)
    w = d = 0
    for g in range(n):
        start = random_opening(game, rng, open_plies) if open_plies else None
        my_id = 1 if g % 2 == 0 else 2
        me, opp = agent, opp_factory(seed + g + 1)
        res = play_match(me, opp, game, start_state=start) if my_id == 1 else play_match(opp, me, game, start_state=start)
        w += (res == my_id); d += (res == 0)
    return (w + 0.5 * d) / n


def _batch_diag(mz):
    """Prior entropy, KL(prior||uniform), value-pred L1 + value-target spread from one sampled batch."""
    c = mz.cfg
    obs0, acts, vt, pt, lm = mz.buffer.sample_batch(c.batch, c.unroll, mz.gen)
    with torch.no_grad():
        _, p, v = mz.net.initial(obs0)
        sm = p.softmax(-1)
        ent = float(-(sm * sm.clamp_min(1e-9).log()).sum(-1).mean())
        kl_unif = float(torch.log(torch.tensor(float(mz.game.n_actions)))) - ent
        vpred = mz.net.value_scalar(v)
        vL1 = float((vpred - vt[:, 0]).abs().mean())
        vspread = float(vt[:, 0].std())
    return dict(prior_entropy=round(ent, 3), kl_to_uniform=round(kl_unif, 3),
                value_pred_L1=round(vL1, 3), value_target_std=round(vspread, 3))


def run_online_cell(game, base_cfg_kw, opt_name, recipe, args):
    from tikitorch.planning.muzero.board_agent import BoardMuZero, BoardConfig
    cfg = BoardConfig(**{**base_cfg_kw, **RECIPES[recipe]})
    mz = BoardMuZero(game, cfg)
    init_sd = {k: v.detach().clone() for k, v in mz.net.state_dict().items()}  # the untrained start
    mz.opt = build_named_optimizer(opt_name, mz.net.parameters())
    iters = args.iters
    log = []
    for it in range(iters):
        frac = it / max(iters - 1, 1)
        if cfg.tau_start is not None and cfg.tau_end is not None:
            mz.current_tau = cfg.tau_start + (cfg.tau_end - cfg.tau_start) * frac
            mz.net.set_tau(mz.current_tau)
        temp = cfg.temp_start + (cfg.temp_end - cfg.temp_start) * frac
        mz.self_play(temperature=temp, base_seed=cfg.seed + it * 1000 + 1)
        mode_fracs, losses = [], []
        if len(mz.buffer):
            for _ in range(cfg.train_steps_per_iter):
                losses.append(mz.train_step())
                mode_fracs.append(mz.last_mode_frac)
        if (it % args.eval_every == 0) or it == iters - 1:
            d = _batch_diag(mz) if len(mz.buffer) else {}
            rec = dict(iter=it, loss=round(sum(losses) / len(losses), 3) if losses else None,
                       mode_frac=round(sum(mode_fracs) / len(mode_fracs), 3) if mode_fracs else None, **d)
            log.append(rec)
            print(f"    [{opt_name}/{recipe}] it{it:2d} loss {rec.get('loss')} "
                  f"mode {rec.get('mode_frac')} ent {d.get('prior_entropy')} "
                  f"vL1 {d.get('value_pred_L1')} vstd {d.get('value_target_std')}", flush=True)

    # ---- strength: trained (soft tau_end) AND trained (hard tau=0) AND untrained control ----
    from tikitorch.planning.muzero.games import negamax_agent
    opp = (lambda s: negamax_agent(game, args.eval_depth)) if args.opponent == "negamax" else (lambda s: _random_agent(game, s))
    mz.net.eval()
    mz.net.set_tau(cfg.tau_end); s_soft = _score_vs(mz._mcts_agent(), opp, game, args.eval_games, args.open_plies, 777)
    mz.net.set_tau(0.0);         s_hard = _score_vs(mz._mcts_agent(), opp, game, args.eval_games, args.open_plies, 777)
    ctrl = BoardMuZero(game, cfg); ctrl.net.load_state_dict(init_sd); ctrl.net.eval()
    ctrl.net.set_tau(cfg.tau_end); s_ctrl = _score_vs(ctrl._mcts_agent(), opp, game, args.eval_games, args.open_plies, 777)
    # HEAD-TO-HEAD: trained vs its own untrained self (the ceiling/floor-free "did training help?" metric)
    mz.net.set_tau(cfg.tau_end)
    h2h = _score_vs(mz._mcts_agent(), lambda s: ctrl._mcts_agent(), game, args.eval_games, args.open_plies, 4242)
    res = dict(net=cfg.net_kind, optimizer=opt_name, recipe=recipe, score_soft=round(s_soft, 3), score_hard=round(s_hard, 3),
               score_untrained=round(s_ctrl, 3), lift=round(s_soft - s_ctrl, 3), h2h_vs_untrained=round(h2h, 3),
               tau_gap=round(s_soft - s_hard, 3), final_mode_frac=log[-1].get("mode_frac") if log else None,
               trace=log)
    print(f"  => {cfg.net_kind}/{opt_name}/{recipe}: soft {s_soft:.3f} untr {s_ctrl:.3f} LIFT {res['lift']:+.3f} | "
          f"HEAD-TO-HEAD trained-vs-untrained {h2h:.3f} ({'HELPS' if h2h>0.55 else 'hurts' if h2h<0.45 else 'tie'})", flush=True)
    return res


def run_online(args):
    os.makedirs(OUT, exist_ok=True)
    if args.game == "connect4":
        game = Connect4
    else:
        from tikitorch.planning.muzero.chess_game import Chess
        game = Chess
    base = dict(net_kind=args.net, core_kind="dyadic", latent_dim=args.latent,
                num_simulations=args.sims, num_workers=args.workers, unroll=3, batch=128,
                games_per_iter=args.games, train_steps_per_iter=args.steps, iters=args.iters,
                tau_start=1.0, tau_end=0.3, temp_start=1.0, temp_end=0.15,
                c_puct=1.25, backup="mean", lr=3e-3, seed=args.seed,
                legal_mask=args.legal_mask, tree_model=args.tree_model)
    opts = args.optimizers.split(",")
    recs = args.recipes.split(",")
    print(f"[online] game={args.game} optimizers={opts} recipes={recs} "
          f"iters={args.iters} sims={args.sims} games/iter={args.games} opponent={args.opponent}", flush=True)
    results = []
    for opt_name in opts:
        for recipe in recs:
            results.append(run_online_cell(game, base, opt_name, recipe, args))
    summary = dict(game=args.game, config=base, opponent=args.opponent, eval_games=args.eval_games,
                   results=[{k: v for k, v in r.items() if k != "trace"} for r in results],
                   traces={f"{r['optimizer']}/{r['recipe']}": r["trace"] for r in results})
    path = os.path.join(OUT, f"online_{args.game}{('_' + args.tag) if args.tag else ''}.json")
    open(path, "w").write(json.dumps(summary, indent=2))
    print(f"\n[saved] {path}")
    print("\n=== ONLINE READOUT (h2h = trained vs its own untrained self; >0.5 means training HELPED) ===")
    for r in sorted(results, key=lambda r: -r["h2h_vs_untrained"]):
        flag = "HELPS" if r["h2h_vs_untrained"] > 0.55 else ("hurts" if r["h2h_vs_untrained"] < 0.45 else "tie")
        print(f"  {r.get('net','?'):8s}/{r['optimizer']:12s}/{r['recipe']:12s} "
              f"H2H {r['h2h_vs_untrained']:.3f} [{flag}] | vs-opp lift {r['lift']:+.3f} "
              f"(soft {r['score_soft']:.2f} untr {r['score_untrained']:.2f}) mode {r['final_mode_frac']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ob = sub.add_parser("optbisect", help="stationary optimizer-class bisect (value-fit + policy-fit)")
    ob.add_argument("--positions", type=int, default=512)
    ob.add_argument("--rollouts", type=int, default=24)
    ob.add_argument("--depth", type=int, default=4)
    ob.add_argument("--steps", type=int, default=800)
    ob.add_argument("--latent", type=int, default=64)
    on = sub.add_parser("online", help="instrumented self-play loop: optimizer x recipe, strength vs untrained control")
    on.add_argument("--game", choices=["connect4", "chess"], default="connect4")
    on.add_argument("--optimizers", default="tropicalsgd,adamw,tropicaladam")
    on.add_argument("--recipes", default="ce,ce_ent0.1")
    on.add_argument("--iters", type=int, default=12)
    on.add_argument("--sims", type=int, default=30)
    on.add_argument("--games", type=int, default=12)
    on.add_argument("--steps", type=int, default=30)
    on.add_argument("--latent", type=int, default=64)
    on.add_argument("--workers", type=int, default=6)
    on.add_argument("--eval-games", dest="eval_games", type=int, default=40)
    on.add_argument("--eval-every", dest="eval_every", type=int, default=3)
    on.add_argument("--eval-depth", dest="eval_depth", type=int, default=2)
    on.add_argument("--open-plies", dest="open_plies", type=int, default=4)
    on.add_argument("--opponent", choices=["negamax", "random"], default="negamax")
    on.add_argument("--tree-model", dest="tree_model", choices=["muzero", "alphazero"], default="muzero")
    on.add_argument("--legal-mask", dest="legal_mask", action="store_true")
    on.add_argument("--seed", type=int, default=0)
    on.add_argument("--net", choices=["tropical", "dense"], default="tropical")
    on.add_argument("--tag", default=None, help="suffix for the output json (avoids clobbering)")
    a = ap.parse_args()
    if a.cmd == "optbisect":
        run_optbisect(a)
    elif a.cmd == "online":
        run_online(a)
