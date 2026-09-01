"""Pipeline-parallelism THROUGHPUT MODEL: PC vs backprop — honest version with an
async-backprop comparator and a sensitivity band.

NOT measured wall-clock — a transparent dependency-structure model. The earlier version
of this script compared PC ONLY against STRICTLY SYNCHRONOUS backprop (latency 2L·δ on the
critical path, "stale gradient is wrong"), which an adversarial review correctly called a
strawman: async-SGD / Hogwild! / PipeDream-1F1B already hide pipeline latency and tolerate
bounded-stale gradients for backprop. So this script now models THREE pipelines and
reports the result as a BAND, not a single number:

  - t_bp_sync  : synchronous backprop — backward sweep + latency 2L·δ on crit path + barrier
  - t_bp_async : async/1F1B backprop  — backward sweep, but latency HIDDEN (overlapped),
                 no hard barrier (this is the FAIR comparator)
  - t_pc       : PC error-relaxation  — NO backward sweep (local update τ_u « τ_b), latency
                 also hidden, BUT pays a non-zero share (frac) of message transit and a
                 measured round-tax R (staleness reconvergence; see pc_staleness_frontier).

HONEST READING: vs async-backprop the latency advantage ~vanishes; what survives is ONLY
the missing-backward-sweep term (1+τ_b)/(R·(1+τ_u)), and only on substrates where the
backward datapath itself is the cost. The durable claim is the qualitative direction +
the substrate property (weight-transport-free, credit-local), NOT a crisp multiplier.
tau_f normalized to 1.
"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TAU_F = 1.0; TAU_B = 2.0; TAU_U = 0.3      # forward / backward / local-update per stage
PC_LAT_FRAC = 0.5                          # PC pays this share of message transit (stalls removed, transit not)
# MEASURED (pc_staleness_frontier.py, 2026-06-20): async error-relaxation reaches the EXACT
# backprop gradient (cosine->1.000) but the rounds-to-converge GROW with depth: stale = L,
# partial-50% = 1.5*L, worst(50%+stale+rand) ~ 4-6*L. So the async relaxation costs R(L)
# LOCAL ROUNDS, R proportional to L -- it does NOT hide latency for free. We encode the
# measured partial-50% law R(L) = TAX_PER_L * L and show that this FLIPS the speedup-vs-depth
# curve from "grows" to "decays": the depth-growing throughput claim is FALSIFIED by measurement.
TAX_PER_L = 1.5                            # measured partial-50% round-tax slope (rounds per layer)
def R_of_L(L): return TAX_PER_L * L        # async relaxation rounds, MEASURED to grow with depth


def t_bp_sync(L, B, delta):   # synchronous: backward + latency on crit path (2L hops) + barrier
    return (B + L - 1) * (TAU_F + TAU_B) + 2 * L * delta + delta
def t_bp_async(L, B, delta):  # 1F1B/async: backward sweep, latency hidden, no hard barrier (FAIR baseline)
    return (B + L - 1) * (TAU_F + TAU_B) + PC_LAT_FRAC * L * delta
def t_pc(L, B, delta):        # PC, OPTIMISTIC: round-tax counted once (full pipeline overlap), no backward sweep
    return (B + L - 1) * (TAU_F + TAX_PER_L * TAU_U) + PC_LAT_FRAC * L * delta
def t_pc_measured(L, B, delta):  # PC, REALISTIC: measured R(L)=1.5L relaxation rounds load the update stage
    return (B + L - 1) * TAU_F + R_of_L(L) * TAU_U + PC_LAT_FRAC * L * delta
def thru(t, B): return B / t


B = 32
# honest no-latency floor/ceiling band: speedup vs ASYNC backprop = (1+tau_b)/(R*(1+tau_u))
def nolat_speedup(tb, tu, R): return (1 + tb) / (R * (1 + tu))
band = [nolat_speedup(tb, tu, R)
        for tb in (1.3, 1.5, 2.0) for tu in (0.1, 0.3, 0.5) for R in (1.0, 1.5, 2.0)]
print("NO-LATENCY speedup vs ASYNC backprop (missing-backward-sweep term only), "
      "over tau_b∈[1.3,2.0] tau_u∈[0.1,0.5] round-tax∈[1,2]:")
print(f"  band = {min(band):.2f}x .. {max(band):.2f}x   (nominal tau_b=2,tau_u=0.3,R=1.5: {nolat_speedup(2.0,0.3,1.5):.2f}x)")
print("vs SYNCHRONOUS backprop (the old strawman) at L=32:")
for d in (0, 1, 2):
    print(f"  delta={d}: PC {thru(t_pc(32,B,d),B)/thru(t_bp_sync(32,B,d),B):.2f}x   "
          f"(vs async-bp: {thru(t_pc(32,B,d),B)/thru(t_bp_async(32,B,d),B):.2f}x)")

# (1) throughput vs latency, L=32 — three pipelines
L = 32; deltas = np.linspace(0, 4, 40)
bps = [thru(t_bp_sync(L, B, d), B) for d in deltas]
bpa = [thru(t_bp_async(L, B, d), B) for d in deltas]
pc = [thru(t_pc(L, B, d), B) for d in deltas]
# (2) speedup vs depth — vs async-bp, OPTIMISTIC (round-tax once) vs MEASURED (R=1.5L)
delta = 0.5; Ls = np.arange(4, 129, 4)
spd_async = [t_bp_async(l, B, delta) / t_pc(l, B, delta) for l in Ls]          # optimistic
spd_measured = [t_bp_async(l, B, delta) / t_pc_measured(l, B, delta) for l in Ls]  # measured R(L)
print("speedup vs depth (vs async-bp): the old 'grows with depth' claim is REFUTED — both")
print("curves are FLAT-to-decaying (avoiding tau_b gives a depth-independent ~2x ceiling under")
print("full overlap; measured R(L)=1.5L erodes it, and pessimistic no-overlap pushes it <1x):")
for l in (8, 32, 64, 128):
    print(f"  L={l:>3}: full-overlap {t_bp_async(l,B,delta)/t_pc_measured(l,B,delta):.2f}x   "
          f"no-overlap {t_bp_async(l,B,delta)/((B+l-1)*TAU_F + R_of_L(l)*TAU_U*(B+l-1) + PC_LAT_FRAC*l*delta):.2f}x")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(deltas, bps, label="backprop (synchronous — strawman)", lw=2, ls=":")
ax[0].plot(deltas, bpa, label="backprop (async / 1F1B — fair baseline)", lw=2)
ax[0].plot(deltas, pc, label="PC error-relaxation", lw=2)
ax[0].set_xlabel("inter-stage latency  δ"); ax[0].set_ylabel("throughput (microbatch/time)")
ax[0].set_title("Throughput vs latency (L=32): PC ≈ async-bp in latency"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
spd_nooverlap = [t_bp_async(l, B, delta) / ((B+l-1)*TAU_F + R_of_L(l)*TAU_U*(B+l-1) + PC_LAT_FRAC*l*delta) for l in Ls]
ax[1].fill_between(Ls, spd_nooverlap, spd_measured, color="C0", alpha=0.15, label="honest range (overlap-dependent)")
ax[1].plot(Ls, spd_measured, lw=2, color="C2", label="full relaxation-overlap (~2×, flat in depth)")
ax[1].plot(Ls, spd_nooverlap, lw=2, color="C3", label="no overlap (measured R=1.5L on crit path → <1×)")
ax[1].axhline(1, ls="--", c="gray")
ax[1].set_xlabel("pipeline depth  L"); ax[1].set_ylabel("PC throughput / async-backprop")
ax[1].set_title("Throughput is overlap-dependent (straddles 1×) — NO speedup claimed; not growing with depth")
ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/predictive_coding/throughput_model.png", dpi=85)
print("saved results/predictive_coding/throughput_model.png")
