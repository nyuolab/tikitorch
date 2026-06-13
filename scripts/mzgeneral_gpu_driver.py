"""mzgeneral_gpu_driver — run muzero-general's chess training bounded by WALL-CLOCK and record the
self-play + training throughput (the systems half of CHESS_MUZERO_COMPARISON.md, axes 2-3).

muzero-general's CLI is an interactive menu; this drives its API directly. It launches train() (Ray
workers) in a thread, polls shared_storage for the cumulative counters every 15 s into a JSONL, runs
for MZ_MAX_SECONDS, then computes STEADY-STATE rates (skipping a warmup window) and exits hard.

MuZero reports TWO throughputs that map to the FPGA actor/learner split — we report both:
  * actor   : self-play moves/s  = d(num_played_steps)/dt
  * learner : train  steps/s     = d(training_step)/dt   (samples/s = steps/s * batch_size)

Run from inside a muzero-general checkout (with games/chessgame.py present):
    MZ_MAX_SECONDS=900 MZ_NUM_WORKERS=12 python3 mz_driver.py
Env: MZ_MAX_SECONDS (default 900), MZ_NUM_WORKERS (default 12), MZ_BATCH (override batch_size),
     MZ_SELFPLAY_ON_GPU (0/1), MZ_OUT (jsonl path).
"""
import json
import os
import threading
import time

import ray

from muzero import MuZero

MAX_SECONDS = int(os.environ.get("MZ_MAX_SECONDS", "900"))
NUM_WORKERS = int(os.environ.get("MZ_NUM_WORKERS", "12"))
OUT = os.environ.get("MZ_OUT", "mz_throughput.jsonl")
SUMMARY = os.environ.get("MZ_SUMMARY", "mz_summary.json")

overrides = {"num_workers": NUM_WORKERS, "training_steps": 10_000_000}
if os.environ.get("MZ_BATCH"):
    overrides["batch_size"] = int(os.environ["MZ_BATCH"])
if os.environ.get("MZ_SELFPLAY_ON_GPU"):
    overrides["selfplay_on_gpu"] = os.environ["MZ_SELFPLAY_ON_GPU"] == "1"

mz = MuZero("chessgame", overrides)
batch = mz.config.batch_size
sims = mz.config.num_simulations
print(f"[driver] chess MuZero on GPU: workers={NUM_WORKERS} batch={batch} sims={sims} "
      f"net=resnet b{mz.config.blocks}c{mz.config.channels} selfplay_on_gpu={mz.config.selfplay_on_gpu} "
      f"train_on_gpu={mz.config.train_on_gpu} | window={MAX_SECONDS}s", flush=True)

t0 = time.time()
samples = []
stop = {"v": False}


def poll():
    while not stop["v"]:
        ss = getattr(mz, "shared_storage_worker", None)
        if ss is not None:
            try:
                info = ray.get(ss.get_info.remote(
                    ["training_step", "num_played_steps", "num_played_games"]))
                info["elapsed"] = round(time.time() - t0, 1)
                samples.append(info)
                with open(OUT, "w") as f:
                    for s in samples:
                        f.write(json.dumps(s) + "\n")
            except Exception:
                pass
        time.sleep(15)


threading.Thread(target=poll, daemon=True).start()
threading.Thread(target=mz.train, daemon=True).start()
time.sleep(MAX_SECONDS)
stop["v"] = True
time.sleep(3)

# steady-state rates: skip a warmup window (first game + ramp), measure the rest
if len(samples) >= 2:
    warmup = min(180, MAX_SECONDS * 0.25)
    warm = [s for s in samples if s["elapsed"] >= warmup] or samples[-2:]
    lo, hi = warm[0], samples[-1]
    dt = hi["elapsed"] - lo["elapsed"]
    if dt <= 0:
        lo, hi, dt = samples[0], samples[-1], samples[-1]["elapsed"] - samples[0]["elapsed"]
    rate = lambda k: round((hi[k] - lo[k]) / dt, 4) if dt > 0 else 0.0
    summary = {
        "instance_window_s": round(dt, 1),
        "selfplay_moves_per_s": rate("num_played_steps"),
        "train_steps_per_s": rate("training_step"),
        "train_samples_per_s": round(rate("training_step") * batch, 2),
        "games_per_s": rate("num_played_games"),
        "batch_size": batch, "num_simulations": sims, "num_workers": NUM_WORKERS,
        "net": f"resnet_b{mz.config.blocks}_c{mz.config.channels}",
        "final_counts": hi, "warmup_skipped_s": round(warmup, 1),
    }
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print("SUMMARY " + json.dumps(summary), flush=True)
else:
    print(f"[driver] too few samples ({len(samples)}) — buffer/self-play may not have started", flush=True)
print("DONE_MZ_GPU", flush=True)
os._exit(0)
