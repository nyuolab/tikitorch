#!/usr/bin/env python3
"""morph_lm_a2_power.py — A2 ISO-MODEL serve-vs-update power harness for the
config-B MorphSSM LM training kernel on the DEPLOYED morph-LM F2 AFI
(cl_forge_morph_lm_v8_aligned, agfi-0433370d93cb58a78; DSP=0; fused mode 3).

THE POINT OF A2: ONE bitstream both serves and updates. On this AFI the SAME
kernel invocation (fused mode 3 = whole-window fwd + balanced-LUT-CE loss +
backward + on-fabric SignSGD) is the ONLY thing that runs; whether the run
LEARNS is decided purely by the SignSGD threshold (do_signsgd steps a weight
only when |grad| > threshold, forge_train_step_morph_lm.cpp:640). So:

    serve  = fused mode-3 kick with a HUGE threshold  -> NO weight moves
             (weights reload/store bit-identical; buffer wcrc == init wcrc)
    update = fused mode-3 kick with the real threshold -> weights evolve (learn)

Serve and update are therefore the IDENTICAL fabric computation; the only
on-fabric difference is the handful of BRAM weight-word writes SignSGD makes when
it steps. That is the measurement: any serve-vs-update Vccint delta is exactly
"the cost of applying the optimizer step", and the A2 claim is that it sits below
the mgmt-tool instrument floor.

!!! RECON-vs-FILES DEVIATION (read SILICON_FINDINGS.md) !!!
The recon described "serve = mode-0 forward sweeps". That is WRONG for this AFI:
the per-position modes 0/1/2 are `#if !SYNTHESIS` HOST-CSIM-only
(forge_train_step_morph_lm.cpp:854-867); on silicon they fall through to
`return 99`. There is NO forward-only serve path synthesized. So "serve" here is
the fused kernel with learning SUPPRESSED (frozen threshold), which INCLUDES the
backward compute — i.e. it is a CONSERVATIVE serve proxy (a true forward-only
serve would draw strictly LESS, shrinking the serve/update gap further, never
widening it). This is disclosed in every emitted line (`serve_is_frozen_mode3`).

HONESTY: the on-box sampler reads the Vccint CORE rail via fpga-describe-local-
image -M, NOT true full-card PMBus. All numbers are core-rail watts; a real
board-watt figure needs external card instrumentation that has never been done
(see SILICON_FINDINGS.md / the muzero power docs). Do not imply full-card power.

Arms (each invocation = ONE arm chunk; the user-data run_timed loop wraps it with
the 1 Hz Vccint sampler and joins by unix timestamp):
    kat     stage fresh init, ONE fused mode-3 kick of A2_KAT_STEPS steps at the
            REAL threshold, assert wcrc/preds16 == golden. Prints KAT_PASS/KAT_FAIL.
    idle    board loaded + buffer staged, NO kicks for `duration` s (power floor)
    serve   loop fused mode-3 kicks at the FROZEN threshold for `duration` s
    update  loop fused mode-3 kicks at the REAL threshold for `duration` s
    gated   serve-frozen kicks with ONE real-threshold (learning) kick every
            A2_GATE_EVERY kicks (serve-mostly, learn-occasionally duty cycle)

Usage:
    morph_lm_a2_power.py <arm> <buffer_init.bin> [duration_seconds]

Env knobs (all optional; defaults chosen to match the golden CE schedule):
    A2_STEPS_PER_KICK  fused steps per serve/update/gated kick   (default 5)
    A2_KAT_STEPS       steps for the KAT golden kick             (default 30)
    A2_GATE_EVERY      kicks between learning kicks in `gated`   (default 20)
    A2_SGD_LR          SignSGD lr (arg2)                         (default 1)
    A2_SGD_THR         REAL learning threshold (arg3, CE Q16)    (default 2048)
    A2_FROZEN_THR      frozen threshold (arg3) -> no step        (default 2**30)
    A2_CHUNK_SEC       fallback duration when arg omitted        (default 20)
    TIKITORCH_FPGA_LIB path to _libtikifpga_pci.so (via morph_lm_silicon)
"""
import ctypes
import json
import os
import struct
import sys
import time

# Reuse morph_lm_silicon's board plumbing VERBATIM so the ctypes signatures can
# never drift (load_lib / run / write_region / read_region / weight_crc, and the
# shared layout module gen_buffer_init as S.L). Importing it has no side effects
# (module-level = constants + defs; main() only runs under __main__).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import morph_lm_silicon as S   # noqa: E402
L = S.L                        # gen_buffer_init layout constants

# ---- golden fused-CE reference (from silicon_fused_ce_traj.jsonl, bit-exact on
# THIS AFI) — the KAT bar. 30-step fused run from the shared init buffer. ----
GOLDEN_STEPS   = 30
GOLDEN_WCRC    = 14650122959732499763
GOLDEN_PREDS16 = [116, 116, 32, 116, 32, 111, 116, 32,
                  116, 111, 116, 116, 32, 116, 116, 32]

# ---- run parameters (env-overridable) ----
STEPS_PER_KICK = int(os.environ.get("A2_STEPS_PER_KICK", "5"))
KAT_STEPS      = int(os.environ.get("A2_KAT_STEPS", str(GOLDEN_STEPS)))
GATE_EVERY     = int(os.environ.get("A2_GATE_EVERY", "20"))
SGD_LR         = int(os.environ.get("A2_SGD_LR", "1"))
SGD_THR        = int(os.environ.get("A2_SGD_THR", "2048"))      # real learning
FROZEN_THR     = int(os.environ.get("A2_FROZEN_THR", str(1 << 30)))  # no step
CHUNK_SEC      = float(os.environ.get("A2_CHUNK_SEC", "20"))

L_SEQ = L.ML_L_SEQ


def load_init(path):
    with open(path, "rb") as f:
        raw = f.read()
    nwords = len(raw) // 4
    if nwords != L.ML_BUFFER_WORDS:
        raise SystemExit("bad init size: %d words, want %d"
                         % (nwords, L.ML_BUFFER_WORDS))
    return list(struct.unpack("<%di" % nwords, raw))


def open_and_stage(lib, init):
    """Open slot 0, alloc the single-instance ML_BUFFER_WORDS buffer, write the
    full shared init buffer once (weights, CFG, tok/tgt, exp2 LUT). Matches
    morph_lm_silicon.main exactly (this AFI is single-instance v8_aligned; the
    N-way v3 CL uses morph_lm_silicon_n4.py — a DIFFERENT bitstream)."""
    h = lib.tt_fpga_open(None)
    if not h:
        raise SystemExit("tt_fpga_open returned NULL")
    dbuf = ctypes.c_void_p(lib.tt_fpga_alloc(h, L.ML_BUFFER_WORDS))
    S.write_region(lib, h, dbuf, 0, init)
    return h, dbuf


def probe(lib, h, dbuf, tgts):
    """Read back the 5 weight regions + PRED; return (wcrc, preds16, acc)."""
    shadow = [0] * L.ML_BUFFER_WORDS
    for off, nw in ((L.OFF_EMBED, L.WORDS_EMBED), (L.OFF_MH, L.WORDS_MH),
                    (L.OFF_MP, L.WORDS_MP), (L.OFF_MB, L.WORDS_MB),
                    (L.OFF_W_HEAD, L.WORDS_W_HEAD)):
        shadow[off:off + nw] = S.read_region(lib, h, dbuf, off, nw)
    wcrc = S.weight_crc(shadow)
    preds = S.read_region(lib, h, dbuf, L.OFF_PRED, L_SEQ)
    correct = sum(1 for m in range(L_SEQ) if preds[m] == tgts[m])
    return wcrc, [int(preds[m]) for m in range(16)], correct / float(L_SEQ)


def fused_kick(lib, h, dbuf, n_steps, thr):
    """ONE fused whole-run kick: mode 3, arg1=n_steps, arg2=lr, arg3=threshold.
    Blocking (tt_fpga_run polls ap_done). Identical call shape to
    morph_lm_silicon's fused path: S.run(lib,h,dbuf,3,n,SGD_LR,thr)."""
    S.run(lib, h, dbuf, 3, n_steps, SGD_LR, thr)


def emit(obj):
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def arm_kat(lib, init, tgts):
    """Golden bit-exactness gate: fresh init -> ONE fused mode-3 kick of
    KAT_STEPS at the REAL threshold -> wcrc/preds16 must equal the golden."""
    h, dbuf = open_and_stage(lib, init)
    t0 = time.time()
    fused_kick(lib, h, dbuf, KAT_STEPS, SGD_THR)
    t1 = time.time()
    wcrc, preds16, acc = probe(lib, h, dbuf, tgts)
    ok = (KAT_STEPS != GOLDEN_STEPS) or (
        wcrc == GOLDEN_WCRC and preds16 == GOLDEN_PREDS16)
    emit({"arm": "kat", "kat_steps": KAT_STEPS, "t_start": t0, "t_end": t1,
          "wall_s": t1 - t0, "wcrc": wcrc, "preds16": preds16, "acc": acc,
          "golden_wcrc": GOLDEN_WCRC, "golden_preds16": GOLDEN_PREDS16,
          "bit_exact": bool(wcrc == GOLDEN_WCRC and preds16 == GOLDEN_PREDS16),
          "kat_pass": bool(ok)})
    sys.stderr.write("KAT_PASS\n" if ok else "KAT_FAIL\n")
    return 0 if ok else 7


def arm_idle(lib, init, tgts, dur):
    """Board loaded + buffer staged, NO kicks for `dur` seconds (power floor)."""
    h, dbuf = open_and_stage(lib, init)
    init_wcrc, _, _ = probe(lib, h, dbuf, tgts)
    t0 = time.time()
    deadline = t0 + dur
    while time.time() < deadline:
        time.sleep(0.5)
    t1 = time.time()
    emit({"arm": "idle", "t_start": t0, "t_end": t1, "dur_s": t1 - t0,
          "kicks": 0, "steps_per_kick": 0, "tokens": 0, "tok_s": 0.0,
          "n_update_kicks": 0, "wcrc": init_wcrc, "init_wcrc": init_wcrc,
          "preds16": [], "serve_is_frozen_mode3": True,
          "note": "core-rail Vccint only; NOT full-card PMBus"})
    return 0


def arm_loop(lib, init, tgts, dur, arm):
    """serve / update / gated: loop fused mode-3 kicks for `dur` seconds.
        serve  -> every kick FROZEN_THR (no weight step)
        update -> every kick SGD_THR   (learn)
        gated  -> FROZEN_THR except every GATE_EVERY-th kick (SGD_THR)
    """
    h, dbuf = open_and_stage(lib, init)
    init_wcrc, _, _ = probe(lib, h, dbuf, tgts)
    kicks = 0
    n_update = 0
    t0 = time.time()
    deadline = t0 + dur
    while time.time() < deadline:
        if arm == "serve":
            thr = FROZEN_THR
        elif arm == "update":
            thr = SGD_THR
            n_update += 1
        else:  # gated
            learn = ((kicks + 1) % GATE_EVERY == 0)
            thr = SGD_THR if learn else FROZEN_THR
            if learn:
                n_update += 1
        fused_kick(lib, h, dbuf, STEPS_PER_KICK, thr)
        kicks += 1
    t1 = time.time()
    wcrc, preds16, acc = probe(lib, h, dbuf, tgts)
    tokens = kicks * STEPS_PER_KICK * L_SEQ
    dt = t1 - t0
    frozen_ok = (arm != "serve") or (wcrc == init_wcrc)
    emit({"arm": arm, "t_start": t0, "t_end": t1, "dur_s": dt,
          "kicks": kicks, "steps_per_kick": STEPS_PER_KICK,
          "tokens": tokens, "tok_s": (tokens / dt) if dt > 0 else 0.0,
          "n_update_kicks": n_update, "gate_every": GATE_EVERY,
          "sgd_thr": SGD_THR, "frozen_thr": FROZEN_THR,
          "wcrc": wcrc, "init_wcrc": init_wcrc, "preds16": preds16, "acc": acc,
          "golden_wcrc": GOLDEN_WCRC, "serve_is_frozen_mode3": True,
          "serve_wcrc_unchanged": bool(frozen_ok),
          "note": "core-rail Vccint only; NOT full-card PMBus"})
    if arm == "serve" and not frozen_ok:
        sys.stderr.write("WARN: serve wcrc moved (%d != init %d) — frozen "
                         "threshold failed to suppress the step\n"
                         % (wcrc, init_wcrc))
    return 0


def main(argv):
    if len(argv) < 3:
        sys.stderr.write(
            "usage: morph_lm_a2_power.py <kat|idle|serve|update|gated> "
            "<buffer_init.bin> [duration_seconds]\n")
        return 2
    arm = argv[1]
    init = load_init(argv[2])
    dur = float(argv[3]) if len(argv) > 3 else CHUNK_SEC
    tgts = [init[L.OFF_TARGETS + m] for m in range(L_SEQ)]

    lib = S.load_lib(S.LIB_PATH)

    if arm == "kat":
        return arm_kat(lib, init, tgts)
    if arm == "idle":
        return arm_idle(lib, init, tgts, dur)
    if arm in ("serve", "update", "gated"):
        return arm_loop(lib, init, tgts, dur, arm)
    sys.stderr.write("unknown arm: %s\n" % arm)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
