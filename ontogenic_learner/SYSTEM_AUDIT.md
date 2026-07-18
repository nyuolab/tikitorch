# SYSTEM_AUDIT — the embodied agent vs the thesis (2026-07-18, post-E4d)

*The thesis: a device sits on its stream; every example it serves it may
also learn from, at about the cost of serving it twice, with no extra
silicon, no extra memory class, and no synchronization. This audit maps the
CURRENT agent system onto that claim, component by component. Claim
language per the locked goal-alignment framing.*

## The system (one process, one network, two readouts)

```
      token stream (frames: [ACT] a [RWD] r [OBS] cells...)
            │
            ▼            ┌──────────────────────────────────┐
   ┌─── serve fwd ──────►│ MorphSSM trunk (tropical, d128/4L)│
   │  (teacher-forced,   └───────┬──────────────┬───────────┘
   │   tau=0, 1 fwd)             ▼              ▼
   │                       LM head (V)    action head (K)   ← E4d: two
   │                       obs+reward      policy readout      readouts,
   │                       prediction      (records both envs) one trunk
   │                            │              │
   │   gate: obs-CE margin ─────┤              │
   │   (perception-side,        ▼              ▼
   │    self-quieting)     CE error      advantage-weighted CE
   │                            └──────┬───────┘
   │                                   ▼
   └── acting: 1 fwd probe      ONE fused cascade kick
       (act_token on the        (summed head errors, single
        action head)             ln adjoint, audit-clean)
                                       │
   credit: within-window reward-to-go  │
   + intrinsic surprise (β=0.5)        ▼
   + advantage norm → dyadic sign·2^k  weights update in place;
   [+ PRE5: wm-value tail bootstrap]   next token served by them
```

## Closure ledger (zero-multiply audit status)

| component | status |
|---|---|
| trunk forward/adjoint (morph conv, leaky scan, tropical proj, topk heads) | shift/compare/add — audited (G-gates, `_make_catcher`) |
| dyadic advantage weight (sign·2^k via ldexp + mux) | shift-class ✓ |
| pow-2 LN factors, bit-shift lr machinery | shift-class ✓ (V-program) |
| advantage normalization (√var-EMA) | HOST-side scalar — declared waiver |
| intrinsic credit (β·CE-excess, EMA) | HOST-side scalars — declared waiver |
| baseline EMA, gate EMAs | HOST-side scalars — declared (same status since V2) |
| softmax at acting/serve CE | the sanctioned mult-free-exp rung (S1 recipe) |
| AdamW | NOT closed (dyadic-Adam exists as the closed variant; not the screened config) |

## Cost ledger (measured)

| operation | cost |
|---|---|
| serve (predict the stream) | 1 fwd |
| act | +1 fwd (act_token probe; head arm 0.97× — free) |
| learn (fused kick, fire-always) | +1 reverse sweep ≈ 1 fwd-eq |
| **total, the E5 config** | **≈3× serve-equivalent per lived step** |
| + wm-value tail credit (PRE5, if won) | +1 fwd → ≈4× (declared) |
| context 2.7–5.3× windows | 2.9–3.45× wall (cache-superlinear, measured; retired) |

The thesis says "~cost of serving it twice"; the AGENT pays ~3× because
acting is itself a forward. Honest restatement for the agentic setting:
**serve + act + learn ≈ 3 forwards, still one process, still in place.**
The prediction-only loop keeps its ~2× (V0) and its gated ~1.006× amortized
(V4) numbers — those claims are untouched.

## Thesis scorecard

| leg | status |
|---|---|
| one process, no actor/learner split | ✓ delivered (incl. policy learning) |
| no replay / no extra memory class | ✓ (window buffers are serving state) |
| no synchronization | ✓ single-agent; fleet = V4b/E7 track |
| learns from what it serves, in order, once | ✓ |
| learn ONLY when surprised (efficiency) | ✓ perception (V4 1.006×) · ✗ POLICY (E4c/E4d: fire-always is the price; 3 schedules failed their bars) |
| retention over a life | INSTRUMENTED — E5 tests it (Goal-B posture) |
| capability vs purpose-built RL | 2.6× behind streaming-AC on foraging (algorithmic: TD credit; PRE5 wmv attacks exactly this); ABOVE it on MinAtar late-life (head) |
| closure | audited core + declared host-scalar waivers (above) |

## Open problems (ranked)

1. Sparse policy learning (three failed levers; the one efficiency leg the
   agent lacks). 2. MinAtar cold-start reliability (2/3 sustained under the
   head). 3. The TD-credit gap (PRE5 wmv is the minimality-respecting
   attack). 4. Scale/capacity (PRE5-S resolves at this tier). 5. AdamW
   closure (dyadic-Adam swap unscreened for the agent).

## Claim guardrails (banned/required phrasing)

"Biologically inspired constraints" — never "biologically plausible
learning rule" (cascade ≡ BP by our own proof; transposes; AdamW).
"Systems-integration novelty" with margins stated. Agent cost is "~3×
serve", not "~2×". streaming-AC comparison appears wherever agent
capability is claimed.
