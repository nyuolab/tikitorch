"""Exp-1 item 6 — the saved-state auditor (SAVED_STATE_AUDIT.md protocol,
measured on the ACTUAL MorphSSM at tau=0).

Arms measured on IDENTICAL model/inputs:
  inherited   MorphPCEngine._forward_cache retained tensors (field split;
              the dead x_in reported separately)
  packed      the minimal route tape (logical bits, payload bytes,
              serialized bytes incl. header)
  autograd    torch saved-tensor bytes via saved_tensors_hooks (storage-
              deduped, parameter vs non-parameter split)
  checkpoint  layer-boundary inputs only (the recomputation arm's retained
              set: L x (B,T,D) + embedding output + top boundary)
  boundary    logits (declared boundary state for the packed claim)

Usage:
  python -m scripts.route_certificates.saved_state_audit [--dry-run]
Scales: (64,2,96) / (128,4,96)=the lifetime model / (256,4,192).
CPU-only; seconds per point. Results ->
results/route_certificates/runs/exp1_saved_state/<uuid>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tikitorch.research.pc_convrnn import MorphPCEngine, build_model  # noqa: E402

from tikitorch.autodiff.morphssm_tape import (  # noqa: E402
    pack_engine_cache, serialize)
from scripts.route_certificates.provenance import (  # noqa: E402
    finish, run_record)

SCALES = ((64, 2, 96), (128, 4, 96), (256, 4, 192))
V, B = 38, 1


def tensor_bytes(t) -> int:
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


def audit_point(d_model: int, n_layers: int, T: int, seed: int = 0):
    torch.manual_seed(seed)
    ns = argparse.Namespace(d_model=d_model, n_layers=n_layers, kernel=7,
                            dropout=0.0, tau=0.5, lam_min=2 ** -6)
    model = build_model("morphssm", V, ns)
    model.set_tau(0.0)
    model.eval()
    g = torch.Generator().manual_seed(seed + 1)
    ids = torch.randint(0, V, (B, T), generator=g)
    targets = torch.randint(0, V, (B, T), generator=g)
    n_tok = B * T

    # ---- inherited cache ----------------------------------------------
    engine = MorphPCEngine(model)
    logits, caches, top = engine._forward_cache(ids)
    inherited, dead = {}, 0
    for li, c in enumerate(caches):
        for k, v in c.items():
            nb = tensor_bytes(v)
            if k == "x_in":
                dead += nb
            inherited[k] = inherited.get(k, 0) + nb
    for k, v in top.items():
        inherited[f"top.{k}"] = tensor_bytes(v)
    inherited_total = sum(inherited.values())

    # ---- packed tape ----------------------------------------------------
    tape = pack_engine_cache(caches, top, model, ids)
    blob = serialize(tape, model)
    packed = dict(logical_bits=tape.bit_length,
                  payload_bytes=tape.payload_nbytes,
                  serialized_bytes=len(blob))

    # ---- autograd saved tensors ------------------------------------------
    param_ptrs = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen, saved_param, saved_other = set(), 0, 0

    def pack_hook(t):
        nonlocal saved_param, saved_other
        try:
            ptr = t.untyped_storage().data_ptr()
        except Exception:
            ptr = id(t)
        if ptr not in seen:
            seen.add(ptr)
            nb = t.numel() * t.element_size()
            if ptr in param_ptrs:
                saved_param += nb
            else:
                saved_other += nb
        return t

    for p in model.parameters():
        p.requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        lg = model(ids)
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, V).float(), targets.reshape(-1))
    loss.backward()
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None

    # ---- checkpoint arm (retained set) -----------------------------------
    elem = 4  # fp32
    ckpt_bytes = (n_layers + 1) * B * T * d_model * elem \
        + tensor_bytes(logits)

    boundary = tensor_bytes(logits)
    return dict(
        d_model=d_model, n_layers=n_layers, T=T, vocab=V, n_tok=n_tok,
        inherited_total=inherited_total,
        inherited_per_token=round(inherited_total / n_tok, 1),
        inherited_dead_x_in=dead,
        inherited_fields={k: v for k, v in sorted(inherited.items())},
        packed=packed,
        packed_per_token=round(tape.payload_nbytes / n_tok, 1),
        autograd_saved_param=saved_param,
        autograd_saved_nonparam=saved_other,
        autograd_nonparam_per_token=round(saved_other / n_tok, 1),
        checkpoint_retained=ckpt_bytes,
        boundary_logits=boundary,
        ratio_inherited_over_packed=round(
            inherited_total / tape.payload_nbytes, 2),
        ratio_autograd_nonparam_over_packed=round(
            saved_other / tape.payload_nbytes, 2),
        ratio_checkpoint_over_packed=round(
            ckpt_bytes / tape.payload_nbytes, 2),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmd = "python -m scripts.route_certificates.saved_state_audit"
    if a.dry_run:
        print(json.dumps(dict(would_run=[
            dict(d_model=d, n_layers=l, T=t) for d, l, t in SCALES],
            output="results/route_certificates/runs/exp1_saved_state/<uuid>/",
            est_wall="~1 min CPU total"), indent=1))
        return
    rec = run_record("exp1_saved_state", cmd,
                     config=dict(scales=list(SCALES), vocab=V, batch=B))
    rows = [audit_point(*s) for s in SCALES]
    out_dir = Path(rec["_out_dir"])
    (out_dir / "saved_state.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    for r in rows:
        print(f"d{r['d_model']}/L{r['n_layers']}/T{r['T']}: "
              f"inherited {r['inherited_per_token']} B/tok, packed "
              f"{r['packed_per_token']} B/tok, autograd-nonparam "
              f"{r['autograd_nonparam_per_token']} B/tok | ratios "
              f"inh/packed {r['ratio_inherited_over_packed']}x, "
              f"ag/packed {r['ratio_autograd_nonparam_over_packed']}x, "
              f"ckpt/packed {r['ratio_checkpoint_over_packed']}x")
    path = finish(rec, metrics=dict(
        n_points=len(rows),
        lifetime_model=[r for r in rows if r["d_model"] == 128][0]))
    print("manifest:", path)


if __name__ == "__main__":
    main()
