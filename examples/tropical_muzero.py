"""Tropical MuZero research demo with dyadic export.

TikiTorch's signature planning model: MuZero whose representation / dynamics / prediction
cores are tropical and whose MCTS selection and backup use additive/max operations.
With `core_kind="dyadic"` the cores use signed-power-of-two weights that add the oblique reach the
plain max-plus core lacks (see results/capacity_probe/FIX_FINDINGS.md). Same PyTorch-shaped
loop you already know: build, `.run()`, evaluate. The exported schedule is not a
whole-agent closure or physical FPGA claim.

Run:  python examples/tropical_muzero.py
"""
import tikitorch as tt
from tikitorch.planning.muzero import GridWorldNav, TropicalMuZero, MuZeroConfig

# A small, fully-connected GridWorld: V*(s) = -(shortest path to G), a (min,+) cost-to-go.
env = GridWorldNav(["S...", "....", "....", "...G"])

cfg = MuZeroConfig(
    core_kind="dyadic",          # signed-power-of-two cores; "maxplus" = axis-aligned
    optimizer="tropicalsgd",     # sign/shift-oriented optimizer variant
    latent_dim=32, num_simulations=25,
    iters=30, games_per_iter=8, train_steps_per_iter=12, eval_every=5, seed=0,
)
mz = TropicalMuZero(env, cfg)

# Train (self-play MCTS -> replay -> K-step unrolled training), scored vs a value-iteration oracle.
hist = mz.run()                  # prints per-eval: agree (optimal-action %) / solve_frac / SOLVED
# MuZero training oscillates (especially with the sign-only optimizer), so the deployable
# model is the BEST snapshot over training, not necessarily the last (cf. BoardMuZero).
best = max(hist, key=lambda h: (h["agree"], h["solve_frac"]))
print(f"\nbest snapshot: optimal-action agreement {best['agree']:.0%} | "
      f"solved {best['solve_frac']:.0%} of starts")

# Structural audit: only the declared
# nn.Linear value/policy heads are DSP boundaries.
tt.assert_closed(mz.net, allow={"Linear"})
print("closure: structural audit passes with linear heads as declared boundaries")

# Freeze the dyadic weights to a bit-exact (sign, exponent) SHIFT SCHEDULE — the FPGA deploy
# artifact — and verify it reproduces the trained forward exactly.
import tikitorch.fpga as fpga
sched = fpga.export_net(mz.net)
assert fpga.verify_schedule(mz.net, sched) == []
print(f"export: {len(sched)} dyadic layers match the shift-schedule reference")
