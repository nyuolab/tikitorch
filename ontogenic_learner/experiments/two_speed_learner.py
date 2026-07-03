"""G1 — the two-speed learner: the decisive experiment of the Ontogenic Learning paradigm.

Question (pre-registered in results/experience_paradigm/ONTOGENIC_LEARNING.md): on a strictly
sequential domain stream (Shakespeare -> Dickens -> Austen) from a shared Moby-Dick-pretrained
base, does an eps-gated two-speed learner (fast cascade kicks on surprise + slow consolidation
over a surprise-weighted reservoir) beat rehearsal-backprop on the retention-plasticity Pareto
at matched update budget?

Arms: frozen | cascade | rehearsal | two_speed_casc (sleep=low-LR cascade — schedule ablation)
      | two_speed_iter (sleep=iterative settled PC train_step — the rule ablation)

The learning modes are the repo's measured ones (scripts/pc_continual.py, scripts/pc_charlm.py):
train_step_cascade == backprop (plastic), train_step == settle-then-local-update (stable).

Run (from the tikitorch repo root):
     PYTHONPATH=. python ontogenic_learner/experiments/two_speed_learner.py           (full)
     PYTHONPATH=. python ontogenic_learner/experiments/two_speed_learner.py --smoke   (sanity)

G1.2 arm grammar: rehearsal[50][_bal] (replay fraction 25/50%, domain-balanced sampling),
two_speed_{casc,iter}[_zNNN] (gate-threshold override, NNN = z*100, e.g. _z075 -> 0.75).
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
import torch
import torch.nn.functional as F

from tikitorch.nn.predictive_coding import DyadicPCNet

DATA_DIR = os.path.join(REPO, "data/experience_stream")
OUT_DIR = os.path.join(REPO, "ontogenic_learner/results/g1_two_speed")

# Fully-local experience stream (Gutenberg unreachable from this sandbox; local is also
# the harder, better-ordered shift): Wikipedia prior -> Shakespeare -> tech docs -> code.
BASE_DOMAIN = "wiki"
STREAM = ["shakespeare", "docs", "code"]

# char-64-compatible alphabet (61 symbols incl. code chars); everything else -> space
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,;:!?'\"()-\n_=+*/<>[]{}#"


# ---------------------------------------------------------------- data
def _raw_corpus(name):
    if name == "wiki":
        return open(os.path.join(REPO, "data/text8_20m.txt"),
                    encoding="utf-8", errors="ignore").read()[:5_000_000]
    if name == "shakespeare":
        return open(os.path.join(REPO, "data/tinyshakespeare.txt"),
                    encoding="utf-8", errors="ignore").read()
    if name == "docs":
        paths = sorted(p for p in glob.glob(os.path.join(REPO, "**/*.md"), recursive=True)
                       if ".git" not in p and "ontogenic_learner" not in p)
        return "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in paths)
    if name == "code":
        paths = sorted(glob.glob(os.path.join(REPO, "tikitorch/**/*.py"), recursive=True) +
                       glob.glob(os.path.join(REPO, "scripts/*.py")))
        return "\n\n".join(open(p, encoding="utf-8", errors="ignore").read() for p in paths)
    raise ValueError(name)


def build_corpora():
    os.makedirs(DATA_DIR, exist_ok=True)
    texts = {}
    for name in [BASE_DOMAIN] + STREAM:
        cache = os.path.join(DATA_DIR, f"{name}.txt")
        if not os.path.exists(cache):
            raw = _raw_corpus(name)
            allowed = set(ALPHABET)
            norm = "".join(c if c in allowed else " " for c in raw.lower())
            norm = " ".join(norm.split(" "))  # collapse runs of spaces (keeps \n)
            open(cache, "w").write(norm)
        texts[name] = open(cache).read()
    chars = sorted(set("".join(texts.values())))
    stoi = {c: i for i, c in enumerate(chars)}
    print(f"vocab={len(chars)} " + " ".join(f"{k}:{len(v)//1000}k" for k, v in texts.items()),
          flush=True)
    enc = {k: torch.tensor([stoi[c] for c in v], dtype=torch.long) for k, v in texts.items()}
    return enc, len(chars)


def windows(data, ctx, start, count, stride):
    """count windows of length ctx (+next-char target) from data[start:], given stride."""
    xs, ys = [], []
    i = start
    for _ in range(count):
        if i + ctx + 1 > len(data):
            break
        xs.append(data[i:i + ctx])
        ys.append(data[i + ctx])
        i += stride
    return torch.stack(xs), torch.stack(ys)


def onehot(xi, V):
    return F.one_hot(xi, V).float().reshape(xi.size(0), -1)


# ---------------------------------------------------------------- eval
@torch.no_grad()
def val_bpc(net, Xva, yva, V, batch=4096):
    tot, n = 0.0, 0
    for i in range(0, Xva.size(0), batch):
        xb = onehot(Xva[i:i + batch], V)
        yb = yva[i:i + batch]
        tot += F.cross_entropy(net.predict_logits(xb), yb, reduction="sum").item()
        n += yb.size(0)
    return tot / n / math.log(2)


# ---------------------------------------------------------------- reservoir
class Reservoir:
    """Windows + priorities. Uniform reservoir-sampling insert (rehearsal) or
    priority insert (two-speed); sampling is uniform or priority-weighted."""

    def __init__(self, cap, seed):
        self.cap, self.x, self.y, self.p, self.d = cap, [], [], [], []
        self.seen = 0
        self.gen = torch.Generator().manual_seed(seed)

    def insert(self, xb, yb, prio=None, k=32, dom=""):
        take = torch.randperm(xb.size(0), generator=self.gen)[:k]
        for j in take.tolist():
            self.seen += 1
            item = (xb[j], yb[j], prio if prio is not None else 1.0, dom)
            if len(self.x) < self.cap:
                self.x.append(item[0]); self.y.append(item[1])
                self.p.append(item[2]); self.d.append(item[3])
            else:  # reservoir sampling
                r = torch.randint(self.seen, (1,), generator=self.gen).item()
                if r < self.cap:
                    self.x[r], self.y[r], self.p[r], self.d[r] = item

    def sample(self, n, weighted, balanced=False):
        if not self.x:
            return None, None
        if balanced:                            # equal draw per stored domain
            doms = sorted(set(self.d))
            per = max(1, n // len(doms))
            idx = []
            for dm in doms:
                pool = torch.tensor([i for i, x in enumerate(self.d) if x == dm])
                pick = torch.randint(len(pool), (per,), generator=self.gen)
                idx.extend(pool[pick].tolist())
            idx = torch.tensor(idx[:n])
        elif weighted:
            w = torch.tensor(self.p) + 1e-3
            idx = torch.multinomial(w, min(n, len(self.x)), replacement=True,
                                    generator=self.gen)
        else:
            idx = torch.randint(len(self.x), (n,), generator=self.gen)
        return torch.stack([self.x[i] for i in idx]), torch.stack([self.y[i] for i in idx])


# ---------------------------------------------------------------- arms
class Arm:
    def __init__(self, name, base_net, V, args, seed):
        self.name, self.V, self.args = name, V, args
        self.net = copy.deepcopy(base_net)
        self.opt_fast = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.opt_slow = torch.optim.Adam(self.net.parameters(), lr=args.lr_slow)
        self.res = Reservoir(args.reservoir, seed + hash(name) % 1000)
        self.mu, self.var = None, None          # surprise EMA (mean, var of batch CE)
        self.updates = self.fwd_eq = 0          # energy bookkeeping
        self.gate_hits = self.gate_total = self.diverged = 0
        # --- G1.2 arm grammar ---
        self.replay_frac = 0.5 if name.startswith("rehearsal50") else 0.25
        self.balanced = name.endswith("_bal")
        self.gate_z = args.gate_z
        if "_z" in name:                        # e.g. two_speed_casc_z075 -> 0.75
            self.gate_z = int(name.rsplit("_z", 1)[1]) / 100.0

    def surprise(self, loss):
        if self.mu is None:
            self.mu, self.var = loss, 1e-2
            return 0.0
        z = (loss - self.mu) / math.sqrt(self.var + 1e-8)
        self.mu = 0.99 * self.mu + 0.01 * loss
        self.var = 0.99 * self.var + 0.01 * (loss - self.mu) ** 2
        return z

    def fast_kick(self, xb, yb):
        self.net.train_step_cascade(xb, yb, self.opt_fast, mode="dyadic")
        self.updates += 1
        self.fwd_eq += 3                        # fwd + cascade bwd ~ 3 fwd-equivalents

    def sleep_kick(self, xb, yb, iterative):
        if iterative:
            snap = [p.detach().clone() for p in self.net.parameters()]
            self.net.train_step(xb, yb, self.opt_slow, n_steps=self.args.settle)
            self.fwd_eq += 2 * self.args.settle + 3
            if not all(torch.isfinite(p).all() for p in self.net.parameters()):
                with torch.no_grad():
                    for p, s in zip(self.net.parameters(), snap):
                        p.copy_(s)
                self.diverged += 1
                return                          # divergence guard: restore, don't count
        else:
            self.net.train_step_cascade(xb, yb, self.opt_slow, mode="dyadic")
            self.fwd_eq += 3
        self.updates += 1


def run_arm(arm, enc, V, args, log_rows, seed):
    """Stream all domains sequentially through one arm."""
    ctx, B = args.context, args.batch
    slots = set(torch.linspace(0, args.stream_batches - 1, args.budget).long().tolist())
    val = {d: windows(enc[d], ctx, len(enc[d]) - len(enc[d]) // 10, args.val_windows, 3)
           for d in [BASE_DOMAIN] + STREAM}
    matrix = {}

    def transfer(tag):
        row = {d: round(val_bpc(arm.net, *val[d], V), 4) for d in val}
        matrix[tag] = row
        log_rows.append({"arm": arm.name, "seed": seed, "event": "transfer", "after": tag,
                         **{f"bpc_{d}": v for d, v in row.items()},
                         "updates": arm.updates, "fwd_eq": arm.fwd_eq})
        print(f"  [{arm.name} s{seed}] after {tag}: " +
              " ".join(f"{d}={v:.3f}" for d, v in row.items()) +
              f"  (updates={arm.updates}, fwd_eq={arm.fwd_eq})", flush=True)

    transfer("base")
    for phase, dom in enumerate(STREAM):
        data = enc[dom]
        n_fast = 0
        for step in range(args.stream_batches):
            xi, yi = windows(data, ctx, step * B, B, 1)
            if xi.size(0) < B // 2:
                break
            xb, yb = onehot(xi, V), yi
            with torch.no_grad():
                loss = F.cross_entropy(arm.net.predict_logits(xb), yb).item()
            arm.fwd_eq += 1
            z = arm.surprise(loss)
            if step % args.eval_every == 0:
                log_rows.append({"arm": arm.name, "seed": seed, "event": "stream",
                                 "domain": dom, "step": step,
                                 "loss_bpc": round(loss / math.log(2), 4),
                                 "z": round(z, 3)})

            if arm.name == "frozen":
                continue
            if arm.name == "cascade":
                if step in slots:
                    arm.fast_kick(xb, yb)
            elif arm.name.startswith("rehearsal"):
                arm.res.insert(xi, yi, dom=dom)  # uniform reservoir over everything seen
                if step in slots:
                    r_n = int(B * arm.replay_frac)
                    rx, ry = arm.res.sample(r_n, weighted=False, balanced=arm.balanced)
                    if rx is not None:
                        xb = torch.cat([onehot(xi[:B - rx.size(0)], V), onehot(rx, V)])
                        yb = torch.cat([yi[:B - rx.size(0)], ry])
                    arm.fast_kick(xb, yb)
            else:                               # two_speed_{casc,iter}[_zNNN]
                arm.gate_total += 1
                if z > arm.gate_z and n_fast < args.fast_budget:
                    arm.fast_kick(xb, yb)
                    arm.res.insert(xi, yi, prio=max(z, 0.1), dom=dom)
                    arm.gate_hits += 1
                    n_fast += 1
                if step > 0 and step % args.sleep_every == 0:
                    for _ in range(args.sleep_updates):
                        rx, ry = arm.res.sample(B, weighted=True)
                        if rx is not None:
                            arm.sleep_kick(onehot(rx, V), ry,
                                           iterative="iter" in arm.name)
        transfer(dom)
    return matrix


# ---------------------------------------------------------------- scoring
def score(matrix, arms_matrices):
    """Forgetting / plasticity / Pareto per arm from its transfer matrix."""
    out = {}
    frozen = arms_matrices["frozen"]
    casc = arms_matrices["cascade"]
    for name, m in arms_matrices.items():
        final = m[STREAM[-1]]
        forget = [final[d] - m[d][d] for d in STREAM[:-1]]           # bpc regained = bad
        base_drift = final[BASE_DOMAIN] - m["base"][BASE_DOMAIN]
        # plasticity: bpc improvement on each domain right after learning it,
        # as a fraction of cascade's improvement (vs the frozen floor)
        plast = []
        for d in STREAM:
            gain = frozen[d][d] - m[d][d]
            casc_gain = frozen[d][d] - casc[d][d]
            plast.append(gain / casc_gain if casc_gain > 1e-6 else 0.0)
        out[name] = {
            "final_avg_bpc": round(sum(final[d] for d in STREAM) / len(STREAM), 4),
            "mean_forgetting_bpc": round(sum(forget) / max(len(forget), 1), 4),
            "base_drift_bpc": round(base_drift, 4),
            "mean_plasticity_frac": round(sum(plast) / len(plast), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-slow", type=float, default=3e-4, dest="lr_slow")
    ap.add_argument("--base-steps", type=int, default=1200, dest="base_steps")
    ap.add_argument("--stream-batches", type=int, default=1500, dest="stream_batches")
    ap.add_argument("--budget", type=int, default=600)       # updates per domain, all arms
    ap.add_argument("--fast-budget", type=int, default=450, dest="fast_budget")
    ap.add_argument("--sleep-every", type=int, default=150, dest="sleep_every")
    ap.add_argument("--sleep-updates", type=int, default=15, dest="sleep_updates")
    ap.add_argument("--settle", type=int, default=16)         # n_steps for iterative sleep
    ap.add_argument("--gamma", type=float, default=0.125)     # settle step size
    ap.add_argument("--tag", default="")
    ap.add_argument("--gate-z", type=float, default=0.25, dest="gate_z")
    ap.add_argument("--reservoir", type=int, default=4096)
    ap.add_argument("--val-windows", type=int, default=4096, dest="val_windows")
    ap.add_argument("--eval-every", type=int, default=100, dest="eval_every")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--arms", nargs="+",
                    default=["frozen", "cascade", "rehearsal",
                             "two_speed_casc", "two_speed_iter"])
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.base_steps, args.stream_batches, args.budget = 100, 150, 60
        args.fast_budget, args.sleep_every, args.sleep_updates = 45, 50, 5
        args.val_windows, args.seeds = 1024, [0]

    torch.set_num_threads(max(1, os.cpu_count() - 2))
    enc, V = build_corpora()
    dims = [args.context * V] + [args.hidden] * (args.depth - 1) + [V]
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = ("smoke" if args.smoke else "full") + args.tag
    rows_path = os.path.join(OUT_DIR, f"g1_{tag}.jsonl")
    log_rows = []
    t00 = time.time()

    all_scores = []
    for seed in args.seeds:
        print(f"\n===== seed {seed}: phylogeny pretrain ({args.base_steps} steps on "
              f"{BASE_DOMAIN}) =====", flush=True)
        torch.manual_seed(seed)
        base_net = DyadicPCNet(dims, residual=True, gamma=args.gamma)
        opt = torch.optim.Adam(base_net.parameters(), lr=args.lr)
        data, ctx, B = enc[BASE_DOMAIN], args.context, args.batch
        gen = torch.Generator().manual_seed(seed)
        t0 = time.time()
        train_hi = int(0.9 * len(data)) - ctx - B - 1   # keep off the val tail
        for step in range(args.base_steps):
            i = torch.randint(train_hi, (1,), generator=gen).item()
            xi, yi = windows(data, ctx, i, B, 1)
            base_net.train_step_cascade(onehot(xi, V), yi, opt, mode="dyadic")
            if step % 200 == 0:
                print(f"  base step {step} ({time.time()-t0:.0f}s)", flush=True)

        arms_matrices = {}
        for name in args.arms:
            arm = Arm(name, base_net, V, args, seed)
            # warm the surprise EMA on the tail of base pretraining data
            for w in range(20):
                xi, yi = windows(data, ctx, len(data) - (w + 2) * B * 3, B, 1)
                with torch.no_grad():
                    l = F.cross_entropy(arm.net.predict_logits(onehot(xi, V)), yi).item()
                arm.surprise(l)
            if name != "frozen" and name != "cascade":
                # seed replay with base-domain memories (consolidated prior life)
                for w in range(8):
                    xi, yi = windows(data, ctx, w * B * 40, B, 1)
                    arm.res.insert(xi, yi, prio=1.0, dom=BASE_DOMAIN)
            t0 = time.time()
            arms_matrices[name] = run_arm(arm, enc, V, args, log_rows, seed)
            log_rows.append({"arm": name, "seed": seed, "event": "done",
                             "updates": arm.updates, "fwd_eq": arm.fwd_eq,
                             "gate_hits": arm.gate_hits, "gate_total": arm.gate_total,
                             "diverged": arm.diverged,
                             "wall_s": round(time.time() - t0, 1)})
        sc = score(None, arms_matrices)
        all_scores.append(sc)
        print(f"\n----- seed {seed} scorecard -----", flush=True)
        for name, s in sc.items():
            print(f"  {name:>16}: final_avg={s['final_avg_bpc']:.3f} "
                  f"forget={s['mean_forgetting_bpc']:+.3f} "
                  f"base_drift={s['base_drift_bpc']:+.3f} "
                  f"plasticity={s['mean_plasticity_frac']:.2f}", flush=True)
        with open(rows_path, "w") as f:
            for r in log_rows:
                f.write(json.dumps(r) + "\n")

    # ------------------------------------------------------------ verdict
    def mean(name, key):
        return sum(s[name][key] for s in all_scores) / len(all_scores)

    print("\n================= G1 VERDICT (mean over seeds) =================", flush=True)
    hdr = f"{'arm':>16} {'final_avg':>9} {'forget':>8} {'base_drift':>10} {'plasticity':>10}"
    print(hdr)
    summary = {}
    for name in args.arms:
        summary[name] = {k: round(mean(name, k), 4) for k in
                         ["final_avg_bpc", "mean_forgetting_bpc", "base_drift_bpc",
                          "mean_plasticity_frac"]}
        s = summary[name]
        print(f"{name:>16} {s['final_avg_bpc']:>9.3f} {s['mean_forgetting_bpc']:>+8.3f} "
              f"{s['base_drift_bpc']:>+10.3f} {s['mean_plasticity_frac']:>10.2f}", flush=True)

    controls = [a for a in args.arms if a.startswith("rehearsal")]
    go = []
    for ts in [a for a in args.arms if a.startswith("two_speed")]:
        s = summary[ts]
        beats_all = s["mean_plasticity_frac"] >= 0.7
        for c in controls:
            r = summary[c]
            pareto = (s["mean_forgetting_bpc"] <= r["mean_forgetting_bpc"] and
                      s["final_avg_bpc"] <= r["final_avg_bpc"] + 0.02)
            beats_all &= pareto
            print(f"  {ts} vs {c}: forget {s['mean_forgetting_bpc']:+.3f} vs "
                  f"{r['mean_forgetting_bpc']:+.3f}, final {s['final_avg_bpc']:.3f} vs "
                  f"{r['final_avg_bpc']:.3f} => {'beats' if pareto else 'NO'}", flush=True)
        go.append((ts, beats_all))
        print(f"  {ts}: plasticity {s['mean_plasticity_frac']:.2f} | beats ALL controls: "
              f"{beats_all}\n", flush=True)
    verdict = ("GO" if any(p for _, p in go) else "NO-GO (or PARTIAL — check energy column)")
    print(f"\n  G1 verdict: {verdict}   ({time.time()-t00:.0f}s total)", flush=True)
    with open(os.path.join(OUT_DIR, f"g1_{tag}_summary.json"), "w") as f:
        json.dump({"summary": summary, "verdict": verdict, "args": vars(args)}, f, indent=1)
    print(f"wrote {rows_path} and g1_{tag}_summary.json", flush=True)


if __name__ == "__main__":
    main()
