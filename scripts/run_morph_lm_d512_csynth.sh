#!/bin/bash
# run_morph_lm_d512_csynth.sh — CSYNTH-ONLY feasibility gate for the D=512 rebuild of the
# config-B MorphSSM LM training kernel (forge_train_step_morph_lm), BEFORE the multi-hour full AFI.
#
# WHY: the deployed kernel is width D=128 (sentinel AFI agfi-0433370d93cb58a78). The user is
# rebuilding at D=512 (a legal power of two -> ML_LOG2_D=9, >>9 shift-normalize; the D^2 out-proj
# region grows 16x). This script runs Vitis HLS C-synthesis at D=512 on a cheap build box (NO f2 /
# NO FPGA hardware) and reports the per-loop II table + timing estimate + resource (LUT/FF/BRAM/
# URAM/DSP) so we learn TIMING (250 MHz? the tcl sets target_period_ns=4.0) and AREA (URAM/BRAM
# blow-up from the 16x-larger mp_buf/g_mp / 4x embed/head) feasibility CHEAPLY. DSP MUST stay 0
# (multiplier-free invariant) — the script greps the report and flags any DSP>0.
#
# Same robust NO-SSH / S3 / IAM-profile / self-terminate mechanics as run_muzero_afi_f2_power.sh
# and run_kernel_csynth.sh, but a z1d BUILD box (FPGA Developer AMI has Vitis HLS), not an f2.
#
# HOW THE D=512 FLAG REACHES THE TCL: synth_morph_lm.tcl reads env ML_D —
#     if {[info exists ::env(ML_D)]} { add_files ... -cflags "-DML_D=$::env(ML_D)" }
# so we simply export ML_D=512 in the build-box user-data before invoking vitis_hls. ML_LOG2_D is
# DERIVED in forge_buffer_morph_lm.hpp (ml_ilog2(ML_D)) and static_asserted == log2(ML_D) — it is
# NOT a -D macro (passing -DML_LOG2_D collides with the constexpr and fails to compile), so ML_D
# alone is the only knob. The tcl already does csynth_design ONLY (no cosim/export), i.e. it IS the
# CSYNTH_ONLY path; CSYNTH_ONLY=1 is exported for convention/logging.
#
# UNCOMMITTED-TREE NOTE: the D=512 parameterization (forge_buffer_morph_lm.hpp #ifndef ML_D guard,
# derived ML_LOG2_D, the tcl's env-ML_D branch) currently lives in the WORKING TREE (house rules:
# do not commit). `git archive HEAD` would ship the OLD pre-parameterization files, so we capture
# the working tree with `git stash create` (produces a commit object WITHOUT touching the index or
# working tree) and archive THAT — falling back to HEAD only if the tree is clean.
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_morph_lm_d512_csynth.sh launch      # ML_D=512 by default; ML_D=256 bash ... to sweep
#   bash scripts/run_morph_lm_d512_csynth.sh status       # / fetch / terminate
#   DRY_RUN=1 bash scripts/run_morph_lm_d512_csynth.sh launch   # render user-data, no EC2
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"          # cheap build box (NOT an f2); Vitis HLS only
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                    # FPGA Developer AMI (ships Vitis HLS)
MAX_MINUTES="${MAX_MINUTES:-90}"                       # D=512 csynth is bigger than D=128; give headroom
ROLE="${ROLE:-tiki-sweep-runner}"
ML_D="${ML_D:-512}"                                    # THE width knob -> -DML_D=$ML_D via the tcl
KTOP="${KTOP:-forge_train_step_morph_lm}"              # top fn = proj prefix (report path)
KTCL="${KTCL:-synth_morph_lm.tcl}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_morphlm_d${ML_D}_csynth"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

render_user_data(){ local s3_src="$1" s3_results="$2"; cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki morph-lm d${ML_D} csynth deadline" || true
exec > /var/log/tiki-morphlm-csynth.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; ML_D="${ML_D}"; KTOP="${KTOP}"; KTCL="${KTCL}"
export HOME=/root
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-morphlm-csynth.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname; mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz; up
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
# THE D=512 FLAG: synth_morph_lm.tcl reads env ML_D and adds -DML_D=\$ML_D to the kernel cflags.
export ML_D="\$ML_D"; export CSYNTH_ONLY=1
echo "=== csynth \$KTOP ML_D=\$ML_D START \$(date -u) ==="
"\$VHLS" -f "\$KTCL" 2>&1 | tail -60
echo "=== csynth \$KTOP done \$(date -u) ==="
RPT="\${KTOP}_proj/solution1/syn/report/\${KTOP}_csynth.rpt"
# ---- on-box summary: timing estimate + resource (LUT/FF/BRAM/URAM/DSP) + per-loop II ----
{
  echo "###### D=\$ML_D FEASIBILITY SUMMARY (target 250 MHz / 4.0 ns) ######"
  echo "--- Timing (Estimated vs Target) ---"; grep -iE "Target|Estimated|CP |clock" "\$RPT" | head -20
  echo "--- Resource (Utilization Estimates: DSP MUST be 0) ---"
  grep -iE "DSP|BRAM|URAM|LUT|FF|Utilization|Total|Available|SLICE" "\$RPT" | head -60
  echo "--- Per-loop latency + Initiation Interval (II) table ---"
  awk '/Loop Name|Latency|Initiation Interval|Trip Count|Pipelined/{p=1} p{print} /Utilization Estimates/{p=0}' "\$RPT" | head -120
  echo "--- DSP=0 assertion ---"
  DLINE="\$(grep -iE "^\|[[:space:]]*DSP" "\$RPT" | head -1)"
  echo "DSP row: \$DLINE"
  echo "\$DLINE" | grep -qE "[1-9]" && echo "WARNING: DSP>0 — multiplier-free invariant VIOLATED at D=\$ML_D" || echo "OK: DSP appears 0"
} 2>&1 | tee /tmp/d\${ML_D}_summary.txt
aws s3 cp "\$RPT"                    "\$S3_RESULTS/\${KTOP}_d\${ML_D}_csynth.rpt" --only-show-errors || true
aws s3 cp /tmp/d\${ML_D}_summary.txt "\$S3_RESULTS/\${KTOP}_d\${ML_D}_summary.txt" --only-show-errors || true
aws s3 cp vitis_hls.log             "\$S3_RESULTS/\${KTOP}_d\${ML_D}_vitis_hls.log" --only-show-errors || true
up; echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID TREE
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="morphlm-d${ML_D}-csynth-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  # Capture the WORKING TREE (the D=512 parameterization is uncommitted; house rules: no commit).
  # git stash create builds a commit object from the working tree WITHOUT mutating index/tree; if the
  # tree is clean it prints nothing -> fall back to HEAD.
  TREE="$(git -C "$PROJECT_DIR" stash create 2>/dev/null || true)"; [ -z "$TREE" ] && TREE="HEAD"
  echo "archiving tree: $TREE"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" "$TREE"
  # Sanity: the needed files AND the parameterization must be present in the shipped tarball.
  for f in "tikitorch/hardware/hls/forge_kernels/${KTCL}" \
           "tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp" \
           "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp"; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not in shipped tree"; exit 1; }; done
  # Verify the tcl actually has the env-ML_D branch and the header has the #ifndef ML_D guard
  # (otherwise ML_D=512 would silently synth the D=128 default — the whole point would be lost).
  local TMPD; TMPD="$(mktemp -d)"; tar xzf "$STATE_DIR/src.tar.gz" -C "$TMPD" \
    "tikitorch/hardware/hls/forge_kernels/${KTCL}" \
    "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp"
  grep -q 'env(ML_D)' "$TMPD/tikitorch/hardware/hls/forge_kernels/${KTCL}" \
    || { echo "FATAL: $KTCL has no env(ML_D) branch — ML_D=$ML_D would NOT reach cflags"; rm -rf "$TMPD"; exit 1; }
  grep -qE '#ifndef[[:space:]]+ML_D' "$TMPD/tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp" \
    || { echo "FATAL: header has no '#ifndef ML_D' guard — -DML_D=$ML_D would not override"; rm -rf "$TMPD"; exit 1; }
  rm -rf "$TMPD"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN (ML_D=$ML_D); UD:"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-morphlm-d${ML_D}-csynth},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then
      echo "retry $tries ($(tr -d '\n' <"$STATE_DIR/launch.err" | tail -c 120))"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; echo "ML_D=$ML_D"; echo "KTOP=$KTOP"; } > "$STATE_FILE"
  echo "LAUNCHED morph-lm D=$ML_D csynth -> $INSTANCE_ID -> $S3_RESULTS"; echo "next: $0 status"
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "(running)"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -20; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local D="$PROJECT_DIR/results/fpga_training_rig"; mkdir -p "$D"
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_summary.txt" "$D/${KTOP}_d${ML_D}_summary.txt" --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_csynth.rpt"  "$D/${KTOP}_d${ML_D}_csynth.rpt"  --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_vitis_hls.log" "$D/${KTOP}_d${ML_D}_vitis_hls.log" --only-show-errors 2>/dev/null || true
  echo "--- summary ---"; cat "$D/${KTOP}_d${ML_D}_summary.txt" 2>/dev/null || echo "(no summary — check status/boot.log)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo "no state"; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;;
  *) echo "usage: [ML_D=512] AWS_PROFILE=... $0 {launch|status|fetch|terminate}";; esac
