"""OrganismConfig — every constant of a life, serializable and hashable.

The learning constants are the G1/G3-validated two-speed values; do not tune them here
without a pre-registered reason.
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field


@dataclass
class Segment:
    """One era of the lifetime schedule. mix maps domain -> fraction; blocks alternate
    domains at block_steps granularity (block-scale drift is what the gate is validated on)."""
    start_day: float
    end_day: float
    mix: dict
    block_steps: int = 8000          # ~2h at full pace


@dataclass
class OrganismConfig:
    # --- identity
    life_tag: str = ""               # set at birth (org-YYYYmmddTHHMMSSZ)
    seed: int = 0

    # --- model (the validated scale)
    ctx: int = 24
    hidden: int = 768
    depth: int = 8
    gamma: float = 0.0625

    # --- two-speed policy (G1/G3 constants, verbatim)
    batch: int = 256
    lr_fast: float = 1e-3
    lr_slow: float = 3e-4
    gate_z: float = 0.25
    sleep_every: int = 150
    sleep_updates: int = 15
    reservoir_cap: int = 4096
    fast_rate_cap: float = 0.30      # rolling: fast kicks <= 30% of trailing window
    fast_rate_window: int = 1500
    warm_steps: int = 20             # always-train steps after birth/segment shifts? (birth only)

    # --- phylogeny (birth warm-start)
    birth_steps: int = 6000

    # --- lifetime
    max_life_days: float = 14.0
    steps_per_day: int = 1_100_000   # pacing assumption; schedule is keyed on life_step
    schedule: list = field(default_factory=list)   # list[Segment as dict]

    # --- persistence / observability
    ckpt_every_steps: int = 10_000
    ckpt_every_secs: int = 900
    ckpt_keep_local: int = 8
    s3_prefix: str = ""              # s3://bucket/organism/<life_tag> (set by launcher)
    s3_sync_secs: int = 300
    probe_every_hours: float = 6.0
    inbox_poll_secs: int = 60
    heartbeat_secs: int = 60
    nan_check_every: int = 50

    # --- runtime
    device: str = "cpu"
    threads: int = 0                 # 0 = auto (nproc-1)
    smoke: bool = False

    @property
    def max_life_steps(self) -> int:
        return int(self.max_life_days * self.steps_per_day)

    def seg_bounds(self, seg: dict):
        return (int(seg["start_day"] * self.steps_per_day),
                int(seg["end_day"] * self.steps_per_day))

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=1, sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "OrganismConfig":
        d = json.load(open(path))
        return cls(**d)


def default_schedule() -> list:
    """The 14-day pilot world (plan §lifetime-schedule). Domains: wiki, shakespeare, mnist,
    plus the day-8 novelty life-event (code). docs is held back for life (probe-only)."""
    S = lambda a, b, mix, blk=8000: dataclasses.asdict(
        Segment(start_day=a, end_day=b, mix=mix, block_steps=blk))
    return [
        S(0.0, 2.0, {"wiki": 1.0}),
        S(2.0, 3.0, {"shakespeare": 1.0}),
        S(3.0, 4.0, {"mnist": 1.0}),
        S(4.0, 5.0, {"wiki": 0.7, "shakespeare": 0.3}),
        S(5.0, 6.0, {"mnist": 0.6, "wiki": 0.4}),
        S(6.0, 7.0, {"shakespeare": 1.0}, 16000),
        S(7.0, 8.0, {"mnist": 0.5, "wiki": 0.5}, 16000),
        S(8.0, 8.25, {"code": 1.0}),                      # NOVELTY LIFE-EVENT
        S(8.25, 9.0, {"wiki": 0.75, "code": 0.25}),
        S(9.0, 12.0, {"wiki": 0.4, "shakespeare": 0.3, "mnist": 0.3}, 16000),
        S(12.0, 14.0, {"wiki": 0.5, "shakespeare": 0.25, "mnist": 0.25}),
    ]


def smoke_schedule(steps_per_day: int) -> list:
    """Minutes-scale life for local testing: same shape, tiny durations."""
    S = lambda a, b, mix, blk=120: dataclasses.asdict(
        Segment(start_day=a, end_day=b, mix=mix, block_steps=blk))
    return [
        S(0.0, 0.2, {"wiki": 1.0}),
        S(0.2, 0.4, {"shakespeare": 1.0}),
        S(0.4, 0.6, {"mnist": 1.0}),
        S(0.6, 0.8, {"wiki": 0.5, "mnist": 0.5}),
        S(0.8, 0.9, {"code": 1.0}),                       # smoke novelty event
        S(0.9, 1.0, {"wiki": 0.4, "shakespeare": 0.3, "mnist": 0.3}),
    ]


def make_config(smoke: bool = False, **over) -> OrganismConfig:
    cfg = OrganismConfig(**over)
    if smoke:
        cfg.smoke = True
        cfg.steps_per_day = 3000                 # a "day" = 3k steps
        cfg.max_life_days = 1.0
        cfg.birth_steps = 150
        cfg.ckpt_every_steps = 500
        cfg.ckpt_every_secs = 3600
        cfg.probe_every_hours = 0.03             # ~every 2 min wall
        cfg.inbox_poll_secs = 10
        cfg.hidden, cfg.depth, cfg.ctx = 256, 4, 12
        cfg.sleep_every, cfg.sleep_updates = 60, 5
        cfg.schedule = smoke_schedule(cfg.steps_per_day)
    elif not cfg.schedule:
        cfg.schedule = default_schedule()
    return cfg
