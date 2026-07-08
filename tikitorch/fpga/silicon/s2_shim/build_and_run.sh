#!/usr/bin/env bash
# Bit-faithful 2-chip (NL=2) one-step reference for the S2 PC-transformer.
# Compiles the REAL kernel forge_pc_xformer_s2_block.cpp locally via the ap_int/hls_math shims and drives it
# as chip0(HID+host-embed) + chip1(OUT) for exactly one training step from the SHARED matched init.
#
# IMPORTANT KNOWN LIMIT (the lesson this harness taught): the _mpshim maps hls::log2/round/exp to EXACT libm,
# but F2 SILICON APPROXIMATES them — a silicon-vs-shim mismatch in an hls:: intrinsic is INVISIBLE here.
# The kernel's dyadic paths are now bit-exact integer ops (silicon == shim by construction); the residual
# shim-blind spot is softmax hls::exp (attention). Use the per-token RATIO analysis (compare_fwd.py + the
# per-row zero-residual-scale test) to localize that class on silicon.
#
# Usage:
#   bash build_and_run.sh                                     # default flags (must match the AFI's -D flags!)
#   SHIM_DEFS="-DS2_K=128 -DS2_VOCAB=8192" bash build_and_run.sh   # matched-flag reference for a new AFI
#   FWD_DUMP=1 OUTDIR=./fwd_ref bash build_and_run.sh         # per-stage forward dumps for compare_fwd.py
# Matched-init data (byte-identical to the silicon FWD_DUMP recipe, see SILICON_CMD.sh):
#   s3://tiki-sweep-739966866637/s2_locinit/master{0,1}.f32   (INIT_DIR)
#   s3://tiki-sweep-739966866637/s2_locinit_data/train.u16    (ENWIK: the 4096-token fixed window)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
K="$(cd "$HERE/../../../hardware/hls/forge_kernels" && pwd)"
DATA="${DATA_DIR:-$HERE/data}"
OUT="${OUTDIR:-$HERE/out}"
mkdir -p "$OUT" "$HERE/build" "$DATA"

# fetch the matched init + fixed data window from S3 if not already local
if [ ! -f "$DATA/master0.f32" ] || [ ! -f "$DATA/train_head.u16" ]; then
  : "${AWS_PROFILE:?matched-init data missing under $DATA — set AWS_PROFILE to fetch from S3}"
  aws s3 cp s3://tiki-sweep-739966866637/s2_locinit/master0.f32 "$DATA/master0.f32" --only-show-errors
  aws s3 cp s3://tiki-sweep-739966866637/s2_locinit/master1.f32 "$DATA/master1.f32" --only-show-errors
  aws s3 cp s3://tiki-sweep-739966866637/s2_locinit_data/train.u16 "$DATA/train_head.u16" --only-show-errors
fi

echo "[build] compiling real kernel + shim (SHIM_DEFS='${SHIM_DEFS:-}' — MUST match the AFI's -D flags)..."
g++ -O2 -std=c++14 -D__AP_INT_SPECIAL_H__ ${SHIM_DEFS:-} \
    -I"$HERE/_apshim" -I"$HERE/_mpshim" -I"$K" \
    "$K/forge_pc_xformer_s2_block.cpp" "$HERE/s2_twochip_shim.cpp" \
    -o "$HERE/build/s2twochip"
echo "[build] OK -> $HERE/build/s2twochip"

echo "[run] one step, shared init, FIXEDOFF=${FIXEDOFF:-0}, ACCUM=${ACCUM:-8}"
INIT_DIR="$DATA" \
ENWIK="$DATA/train_head.u16" \
OUTDIR="$OUT" \
STEPS="${STEPS:-1}" ACCUM="${ACCUM:-8}" XF_LR="${XF_LR:-2.6e-4}" \
XF_WARMUP="${XF_WARMUP:-0}" LR_DECAY="${LR_DECAY:-0}" FIXEDOFF="${FIXEDOFF:-0}" \
FWD_DUMP="${FWD_DUMP:-0}" \
    "$HERE/build/s2twochip"
echo "[run] outputs -> $OUT (shim_master{0,1}.f32; fwd_<chip>_<stage>.f32 when FWD_DUMP=1)"
