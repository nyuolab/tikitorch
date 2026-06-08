"""Phase-C.1 -- the closed algorithmic-reasoner model zoo.

Trains one CLOSED (zero-DSP) TropicalReasoner per algorithmic task under the verified
closed recipe (hard tropical forward, perceptron/L1 loss, a multiplier-free optimizer),
then for each:
  * ``tt.save`` it with a fully-reconstructable architecture config + closure manifest,
  * certifies it with ``tt.closure_report`` on a freshly LOADED clone (the report traces
    a real training step and would mutate weights, so we never run it on the artifact
    we keep), asserting ``inference_closed`` and ``training_zero_dsp``,
  * round-trips ``tt.load`` and asserts the reloaded model is BIT-IDENTICAL.

The default optimizer is the Phase-B bake-off winner. Usage:
    python scripts/build_algorithmic_zoo.py [--opt puretropical] [--out zoo/algorithmic]
"""
from __future__ import annotations

import argparse
import json
import os

import torch

import tikitorch as tt
import tikitorch.algorithmic as A
import tikitorch.functional as F
import tikitorch.optim as O

# The five reference tasks (token-seq + binary-selection DPs). Per-task training
# length / steps; the architecture is shared so the zoo is uniform and comparable.
TASKS = ["running_max", "sort", "subset_sum", "knapsack", "min_coin_change"]

ARCH = dict(d_model=64, n_layers=2, n_heads=4, attention="tropical",
            score_variant="hilbert", norm_type="tropical_shift",
            pos_mode="rope_shift", agg_variant="argmax", head="auto")

# evaluate_task kwargs that affect the trained weights (kept in the saved config so a
# loader can rebuild the exact architecture before tt.load).
TRAIN = dict(train_len=16, ood_len=32, train_n=2048, test_n=512, batch_size=64,
             steps=400, accum=4, lr=2 ** -4, anneal=False)

_OPT_CLS = {
    "puretropical": O.PureTropicalSGD, "tropicaladam": O.TropicalAdam,
    "countagree": O.CountAgreementSGD, "perlayer": O.PerLayerStepSGD,
    "frexpmag": O.FrexpMagnitudeSGD, "expadam": O.ExponentEMAAdam,
}


def _reasoner_config(task_name: str) -> dict:
    """Everything needed to rebuild the architecture for tt.load."""
    task = A.get_task(task_name)
    max_len = max(TRAIN["train_len"], TRAIN["ood_len"]) + 1
    return dict(task=task_name, max_len=max_len,
                n_symbols=(task.n_symbols if task.input_kind == "tokens" else None),
                **ARCH)


# Only these keys are TropicalReasoner constructor args; the rest of a saved config
# (optimizer/recipe/metrics) is metadata and must NOT be forwarded to the builder.
_ARCH_KEYS = ("max_len", "d_model", "n_layers", "n_heads", "attention",
              "score_variant", "norm_type", "pos_mode", "agg_variant", "head")


def build_from_config(config: dict):
    """Reconstruct a TropicalReasoner from a saved zoo config (inverse of save).

    ``n_symbols`` lives in the Task (not a builder kwarg); mirror evaluate_task and
    rebuild the Task with the stored vocab for the token tasks.
    """
    task = A.get_task(config["task"])
    if config.get("n_symbols") is not None and task.input_kind == "tokens":
        task = A.Task(**{**task.__dict__, "n_symbols": config["n_symbols"]})
    kw = {k: config[k] for k in _ARCH_KEYS if k in config}
    return A.TropicalReasoner(task, **kw)


def _loss_fn(task):
    C = task.n_symbols if task.metric != "f1" else None
    if task.kind == "class":
        return lambda p, y: F.tropical_perceptron_loss(p.reshape(-1, p.shape[-1]), y.reshape(-1))
    return lambda p, y: (p - y).abs().mean()


def _build_one(args):
    """Train + save + certify + round-trip one task. Top-level (picklable) so the
    five reference models train in parallel across cores.

    Trains BEST-OF-K seeds and keeps the highest-id_acc model. This is a deliberate,
    documented choice: closed tropical training is bimodally unstable on the DP tasks
    (some seeds converge, some collapse near chance -- see the P4 bake-off), so a
    single-seed reference can land on a collapse draw. Best-of-k selects a *converged*
    representative of the closed recipe; ``n_seeds``/``best_seed`` are recorded.
    """
    task, opt_name, out_dir, steps, n_seeds = args
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    train_kw = {**TRAIN, "steps": steps if steps is not None else TRAIN["steps"]}
    arch_kw = {k: v for k, v in ARCH.items()
               if k in ("d_model", "n_layers", "n_heads", "attention", "score_variant",
                        "norm_type", "pos_mode", "agg_variant", "head")}
    best = None  # (id_acc, seed, out, model)
    for seed in range(n_seeds):
        out, model = A.evaluate_task(task, closed=True, closed_opt=opt_name, seed=seed,
                                     device="cpu", return_model=True, **train_kw, **arch_kw)
        if best is None or out["id_acc"] > best[0]:
            best = (out["id_acc"], seed, out, model)
    _, best_seed, out, model = best
    cfg = _reasoner_config(task)
    cfg.update(optimizer=opt_name, recipe="closed", n_seeds=n_seeds, best_seed=best_seed,
               id_acc=round(out["id_acc"], 4), ood_len_acc=round(out["ood_len_acc"], 4))
    path = os.path.join(out_dir, f"{task}.tt")
    tt.save(model, path, config=cfg)

    # --- certify on a freshly LOADED clone (closure_report traces a step + mutates) ---
    task_obj = A.get_task(task)
    clone = build_from_config(cfg)
    tt.load(path, model=clone)
    xid, yid = A._make_batch(task_obj, 32, TRAIN["train_len"], scale=1.0, seed=7, device="cpu")
    rep = tt.closure_report(clone, example_inputs=(xid,), targets=yid,
                            loss_fn=_loss_fn(task_obj),
                            optimizer=_OPT_CLS[opt_name](clone.parameters()))
    inf_ok, train_ok = rep["inference_closed"], rep["training_zero_dsp"]

    # --- round-trip bit-identity (fresh load, no report mutation) ---
    rt = build_from_config(cfg)
    tt.load(path, model=rt)
    model.eval(); rt.eval()
    if hasattr(model, "set_tau"):
        model.set_tau(0.0); rt.set_tau(0.0)
    with torch.no_grad():
        identical = bool(torch.equal(model(xid), rt(xid)))

    assert inf_ok, f"{task}: NOT inference-closed"
    assert train_ok, f"{task}: training not zero-DSP"
    assert identical, f"{task}: round-trip not bit-identical"
    return dict(task=task, path=os.path.basename(path), best_seed=best_seed, n_seeds=n_seeds,
                id_acc=cfg["id_acc"], ood_len_acc=cfg["ood_len_acc"],
                inference_closed=inf_ok, training_zero_dsp=train_ok,
                round_trip_identical=identical)


def build_zoo(opt_name: str, out_dir: str, workers: int = 5, steps: int | None = None,
              n_seeds: int = 4):
    os.makedirs(out_dir, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    summary = []
    with ProcessPoolExecutor(max_workers=min(workers, len(TASKS))) as ex:
        for s in ex.map(_build_one, [(t, opt_name, out_dir, steps, n_seeds) for t in TASKS]):
            print(f"  {s['task']:16s} id={s['id_acc']:.3f} ood={s['ood_len_acc']:.3f} "
                  f"(best of {s['n_seeds']}, seed {s['best_seed']}) "
                  f"inf_closed={s['inference_closed']} train_zero_dsp={s['training_zero_dsp']} "
                  f"roundtrip={s['round_trip_identical']}")
            summary.append(s)
    summary.sort(key=lambda s: TASKS.index(s["task"]))

    with open(os.path.join(out_dir, "zoo_manifest.json"), "w") as fh:
        json.dump(dict(optimizer=opt_name, arch=ARCH, train=TRAIN, models=summary), fh, indent=2)
    print(f"\nzoo manifest -> {os.path.join(out_dir, 'zoo_manifest.json')}")
    print("ALL MODELS: inference-closed + training-zero-DSP + bit-identical round-trip.")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", default="puretropical", choices=list(_OPT_CLS))
    ap.add_argument("--out", default="zoo/algorithmic")
    ap.add_argument("--steps", type=int, default=None, help="override training steps (smoke)")
    ap.add_argument("--seeds", type=int, default=4, help="best-of-k seeds per task (bimodal training)")
    ap.add_argument("--tasks", nargs="*", default=None, help="subset of tasks (smoke)")
    args = ap.parse_args()
    if args.tasks:
        global TASKS
        TASKS = args.tasks
    build_zoo(args.opt, args.out, steps=args.steps, n_seeds=args.seeds)


if __name__ == "__main__":
    main()
