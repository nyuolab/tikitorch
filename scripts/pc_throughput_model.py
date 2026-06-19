"""Pipeline-parallelism THROUGHPUT MODEL: backprop vs legit-PC (error-relaxation).

NOT measured wall-clock on real hardware — a transparent model from the dependency
structure of each method. The only PC-favoring assumptions are the *actual* structural
differences, each grounded:
  - backprop has a BACKWARD SWEEP (cost tau_b/stage) ; PC has none (local updates).
  - backprop needs a SYNC BARRIER (gradients are exact, all-reduce) ; PC has none.
  - backprop CANNOT use stale gradients so inter-stage latency delta is ON the critical
    path (2L hops) ; PC TOLERATES staleness (MEASURED: async 50%+stale+rand within
    ~0.6pp, SCALE_TESTS.md) so latency is HIDDEN (overlapped), off the critical path.
Standard L-stage / B-microbatch pipeline (GPipe-style). tau_f normalized to 1.
"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
TAU_F=1.0; TAU_B=2.0; TAU_U=0.3      # forward / backward / local-update per stage

def t_bp(L,B,delta):   # forward+backward pipeline + latency on crit path (2L hops) + 1 barrier
    return (B+L-1)*(TAU_F+TAU_B) + 2*L*delta + delta
def t_pc(L,B,delta):   # forward pipeline + local update; NO backward, latency hidden, no barrier
    return (B+L-1)*(TAU_F+TAU_U)
def thru(t,B): return B/t

B=32
# (1) throughput vs inter-stage latency, fixed depth L=32
L=32; deltas=np.linspace(0,4,40)
bp=[thru(t_bp(L,B,d),B) for d in deltas]; pc=[thru(t_pc(L,B,d),B) for d in deltas]
# (2) speedup vs depth, fixed latency
delta=0.5; Ls=np.arange(4,129,4)
spd=[t_bp(l,B,delta)/t_pc(l,B,delta) for l in Ls]

print(f"at L=32, delta=0 (no latency): PC throughput {thru(t_pc(32,B,0),B)/thru(t_bp(32,B,0),B):.2f}x backprop (no backward sweep)")
print(f"at L=32, delta=1.0*tau_f:      PC throughput {thru(t_pc(32,B,1),B)/thru(t_bp(32,B,1),B):.2f}x backprop")
print(f"at L=32, delta=2.0*tau_f:      PC throughput {thru(t_pc(32,B,2),B)/thru(t_bp(32,B,2),B):.2f}x backprop")
print(f"at L=128, delta=0.5*tau_f:     PC throughput {t_bp(128,B,0.5)/t_pc(128,B,0.5):.2f}x backprop")

fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot(deltas,bp,label="backprop (sync backward+barrier)",lw=2)
ax[0].plot(deltas,pc,label="legit-PC (local, stale-tolerant)",lw=2)
ax[0].set_xlabel("inter-stage latency  δ  (units of forward time)"); ax[0].set_ylabel("throughput (microbatches / time)")
ax[0].set_title("Throughput vs communication latency (L=32 stages)"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(Ls,spd,lw=2,color="C2"); ax[1].axhline(1,ls="--",c="gray")
ax[1].set_xlabel("pipeline depth  L  (stages)"); ax[1].set_ylabel("PC throughput / backprop throughput")
ax[1].set_title("Parallel speedup vs depth (δ=0.5·forward)"); ax[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig("results/predictive_coding/throughput_model.png",dpi=85)
print("saved results/predictive_coding/throughput_model.png")
