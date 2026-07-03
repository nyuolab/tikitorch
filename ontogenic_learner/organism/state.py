"""Checkpointing: atomic torch.save/load of EVERYTHING a life needs to resume bit-exactly —
net, both Adam states, reservoir, surprise EMA, cursors, schedule state, all RNG streams,
counters, hashes. Retention: last N local; S3 sync is the daemon's job."""
import glob
import os

import torch


def capture(net, policy, world, sched, gens: dict, life_step: int, counters: dict,
            hashes: dict) -> dict:
    return {
        "net": net.state_dict(),
        "policy": policy.state(),
        "cursors": world.cursor_states(),
        "sched": sched.state(),
        "rng": {"torch": torch.get_rng_state(),
                **{k: g.get_state() for k, g in gens.items()}},
        "life_step": life_step,
        "counters": counters,
        "hashes": hashes,
    }


def restore(blob: dict, net, policy, world, sched, gens: dict):
    net.load_state_dict(blob["net"])
    policy.restore(blob["policy"])
    world.restore_cursors(blob["cursors"])
    sched.restore(blob["sched"])
    torch.set_rng_state(blob["rng"]["torch"])
    for k, g in gens.items():
        if k in blob["rng"]:
            g.set_state(blob["rng"][k])
    return blob["life_step"], blob.get("counters", {}), blob.get("hashes", {})


def save_atomic(blob: dict, ckpt_dir: str, life_step: int, keep: int = 8) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, "ckpt.tmp")
    path = os.path.join(ckpt_dir, f"ckpt-{life_step:012d}.pt")
    with open(tmp, "wb") as f:
        torch.save(blob, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    with open(os.path.join(ckpt_dir, "LATEST"), "w") as f:
        f.write(os.path.basename(path))
    for old in sorted(glob.glob(os.path.join(ckpt_dir, "ckpt-*.pt")))[:-keep]:
        os.remove(old)
    return path


def load_latest(ckpt_dir: str):
    marker = os.path.join(ckpt_dir, "LATEST")
    if os.path.exists(marker):
        p = os.path.join(ckpt_dir, open(marker).read().strip())
        if os.path.exists(p):
            return torch.load(p, weights_only=False)
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "ckpt-*.pt")))
    return torch.load(cands[-1], weights_only=False) if cands else None
