#!/bin/bash
# run_wgrad_probe_csynth.sh — Vitis HLS C-synthesis of the FPGA-training-rig GATE-2 probe
# (forge_wgrad_probe: the operand-quantized weight-grad GEMM at d768). Same robust, self-terminating,
# NO-SSH pattern as run_muzero_chess_csynth_fpga.sh: git-archive HEAD -> S3, launch a cheap z1d build
# box with the FPGA Developer AMI (Vitis pre-installed; csynth needs NO FPGA hardware), run csynth via
# user-data, upload the report to S3, self-terminate. No F2 (F2 is only for AFI loading on silicon).
#
#   export AWS_PROFILE=my-profile
#   bash scripts/run_wgrad_probe_csynth.sh launch     # (DRY_RUN=1 to preview, no launch)
#   bash scripts/run_wgrad_probe_csynth.sh status     # poll S3 (no SSH)
#   bash scripts/run_wgrad_probe_csynth.sh fetch       # download the csynth report
#   bash scripts/run_wgrad_probe_csynth.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"            # AWS-recommended Vitis build instance (~$0.74/hr)
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (Ubuntu), Vitis 2024.1
MAX_MINUTES="${MAX_MINUTES:-60}"                          # csynth ~10-25 min; self-terminate backstop
ROLE="${ROLE:-tiki-sweep-runner}"                         # reuse the instance-profile from prior runs
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_wgrad_csynth_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-wgrad-csynth deadline" || true
exec > /var/log/tiki-csynth.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-csynth.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
up
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; \$VHLS -version 2>&1 | head -2 || true
up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
echo "=== csynth wgrad_probe START \$(date -u) ==="; "\$VHLS" -f synth_wgrad_probe.tcl 2>&1 | tail -30
echo "=== csynth wgrad_probe done \$(date -u) ==="
aws s3 cp forge_wgrad_probe_proj/solution1/syn/report/forge_wgrad_probe_csynth.rpt "\$S3_RESULTS/forge_wgrad_probe_csynth.rpt" --only-show-errors || true
aws s3 cp vitis_hls.log "\$S3_RESULTS/vitis_hls_wgrad.log" --only-show-errors || true
up
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET (prior runs create it); aborting"; exit 1; }
  TAG="wgrad-csynth-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== build source tarball (git archive HEAD) =="
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "tikitorch/hardware/hls/forge_kernels/synth_wgrad_probe.tcl" \
    || { echo "FATAL: synth_wgrad_probe.tcl not in committed tree (commit it first)"; exit 1; }
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "tikitorch/hardware/hls/forge_kernels/forge_wgrad_probe.cpp" \
    || { echo "FATAL: forge_wgrad_probe.cpp not in committed tree (commit it first)"; exit 1; }
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching. UD head:"; sed -n '1,40p' "$UD"; return 0; fi
  echo "== upload source -> $S3_SRC =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-wgrad-csynth},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- boot.log tail --"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -25; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_training_rig"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/forge_wgrad_probe_csynth.rpt" "$DEST/forge_wgrad_probe_csynth.rpt" --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/vitis_hls_wgrad.log" "$DEST/wgrad_vitis_hls.log" --only-show-errors 2>/dev/null || true
  ls -1 "$DEST"/forge_wgrad_probe_csynth.rpt 2>/dev/null || echo "(none yet)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true
  rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate} (DRY_RUN=1 for launch)";; esac
