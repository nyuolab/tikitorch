#!/usr/bin/env bash
# Silicon forward-faithfulness retest for the s2 PC-transformer: 2-chip matched-init FWD_DUMP on real F2,
# compared per-stage against the local shim reference built at IDENTICAL -D flags (see build_and_run.sh).
# This is the recipe that confirmed the LN hls::log2 root cause and then the fully bit-exact forward.
#
# VERDICT thresholds (measured 2026-07):
#   FIXED     : chip0 X1 (LN1) relerr <= ~1e-5 ; ATTN <= ~1e-3   (== fp/order-noise floor)
#   NOT FIXED : chip0 X1 relerr >> 1e-3 (the old LN bug measured 5.36e-2)
#   (post-bit-exact-fix silicon measured 0.00 / 100% bitmatch at every stage)
set -euo pipefail
export AWS_PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"   # repo root (tikitorch/fpga/silicon/s2_shim -> root)
SHIM="$REPO/tikitorch/fpga/silicon/s2_shim"
AGFI="${AGFI:?set AGFI (the AFI under test)}"

# --- 0) local shim reference at the SAME -D flags as the AFI (K changes reduction order!) ---
#   SHIM_DEFS="-DS2_K=... -DS2_VOCAB=..." FWD_DUMP=1 OUTDIR=$SHIM/fwd_ref bash $SHIM/build_and_run.sh

# --- 1) launch: NL=2, STEPS=1, FWD_DUMP=1, matched init + BYTE-IDENTICAL data window ---
# (the default BINS_S3=pc_enwik8_bins 404s, and s2_bins/train.u16's head != the shim window -> R mismatch)
cd "$REPO"
AGFI="$AGFI" \
ENWIK_S3=s3://tiki-sweep-739966866637/s2_locinit_data/train.u16 \
NSLOTS=2 STEPS=1 ACCUM=1 \
FWD_DUMP=1 S2_DIAG=1 FIXEDOFF=0 \
INIT_S3=s3://tiki-sweep-739966866637/s2_locinit \
LR_DECAY=0 XF_WARMUP=0 PIPE=0 \
  bash scripts/run_s2_shard_f2.sh launch

# --- 2) fetch (after the DONE marker appears in S3) ---
#   source "$REPO/.tiki_s2_shard_state/run.env"   # -> $S3_RESULTS
#   rm -rf "$SHIM/fwd_sil" && mkdir -p "$SHIM/fwd_sil"
#   aws s3 cp "$S3_RESULTS/" "$SHIM/fwd_sil/" --recursive --exclude '*' --include 'fwd_*.f32' --only-show-errors

# --- 3) compare ---
#   python3 "$SHIM/compare_fwd.py" "$SHIM/fwd_sil" "$SHIM/fwd_ref" --chips 0,1
