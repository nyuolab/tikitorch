#!/bin/bash
# run_kernel_csynth.sh — generic, self-terminating, NO-SSH Vitis HLS C-synthesis runner for any forge
# kernel. Same robust pattern as run_wgrad_probe_csynth.sh but parameterized so multiple kernels can be
# synthesized concurrently (per-kernel state dir). csynth needs NO FPGA hardware (cheap z1d build box).
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   KTOP=forge_fwd_gemm KTCL=synth_fwd_gemm.tcl bash scripts/run_kernel_csynth.sh launch
#   KTOP=forge_fwd_gemm bash scripts/run_kernel_csynth.sh status   # / fetch / terminate
# Requires (committed): tikitorch/hardware/hls/forge_kernels/{$KTCL, <$KTOP>.cpp}; the tcl must set
# proj_dir to "${KTOP}_proj" and top to $KTOP (so the report is <KTOP>_proj/solution1/syn/report/<KTOP>_csynth.rpt).
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
KTOP="${KTOP:?set KTOP (top function = proj prefix)}"
KTCL="${KTCL:-synth_${KTOP#forge_}.tcl}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
MAX_MINUTES="${MAX_MINUTES:-75}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_csynth_${KTOP}"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

render_user_data(){ local s3_src="$1" s3_results="$2"; cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki deadline" || true
exec > /var/log/tiki-csynth.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-csynth.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname; mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz; up
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
echo "=== csynth ${KTOP} START \$(date -u) ==="; "\$VHLS" -f ${KTCL} 2>&1 | tail -40
echo "=== csynth ${KTOP} done \$(date -u) ==="
aws s3 cp ${KTOP}_proj/solution1/syn/report/${KTOP}_csynth.rpt "\$S3_RESULTS/${KTOP}_csynth.rpt" --only-show-errors || true
aws s3 cp vitis_hls.log "\$S3_RESULTS/${KTOP}_vitis_hls.log" --only-show-errors || true
up; echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_launch(){ [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="${KTOP}-csynth-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  for f in "tikitorch/hardware/hls/forge_kernels/${KTCL}" "tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp"; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not in committed tree"; exit 1; }; done
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN; UD head:"; sed -n '1,30p' "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-${KTOP}-csynth},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ] && { sleep 10; continue; }
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1; done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $KTOP -> $INSTANCE_ID -> $S3_RESULTS"; }

cmd_status(){ source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "(running)"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -15; }; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/results/fpga_training_rig"; mkdir -p "$D"
  aws_ s3 cp "$S3_RESULTS/${KTOP}_csynth.rpt" "$D/${KTOP}_csynth.rpt" --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/${KTOP}_vitis_hls.log" "$D/${KTOP}_vitis_hls.log" --only-show-errors 2>/dev/null || true
  ls -l "$D/${KTOP}_csynth.rpt" 2>/dev/null || echo "(no report — check status/boot.log)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || exit 0; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: KTOP=.. KTCL=.. $0 {launch|status|fetch|terminate}";; esac
