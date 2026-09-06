#!/bin/bash
# run_muzero_chess_csynth_fpga.sh — robust one-shot Vitis HLS C-synthesis of the chess-scale dyadic
# MuZero kernels (forward + train) for axis 4 of CHESS_MUZERO_COMPARISON.md. Same robust pattern as
# run_connect4_prior_sweep_ec2.sh: source -> S3, IAM instance-profile, shutdown backstop, self-terminate,
# NO SSH. csynth needs only the Vitis toolchain (no FPGA hardware), so it runs on a cheap z1d build
# instance with the FPGA Developer AMI (Ubuntu, Vitis pre-installed) — NOT an F2 (F2 is only needed for
# AFI loading on real silicon, which is gated behind a strength result).
#
#   export AWS_PROFILE=my-profile
#   bash scripts/run_muzero_chess_csynth_fpga.sh launch     # build src -> S3 -> launch FPGA-dev box
#   bash scripts/run_muzero_chess_csynth_fpga.sh status      # poll S3 (no SSH)
#   bash scripts/run_muzero_chess_csynth_fpga.sh fetch       # download the chess csynth reports
#   bash scripts/run_muzero_chess_csynth_fpga.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"            # AWS-recommended Vitis build instance (~$0.74/hr)
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (Ubuntu) 1.16.1, Vitis 2024.1 (proven by existing reports)
MAX_MINUTES="${MAX_MINUTES:-75}"                          # csynth ~10-25 min; backstop 75 (train kernel needs more)
CSYNTH_ONLY="${CSYNTH_ONLY:-both}"                        # forward | train | both (train is the slow one)
ROLE="${ROLE:-tiki-sweep-runner}"                         # reuse the instance-profile from the prior runs
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_csynth_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET; run the connect4 runner's 'setup' first"; exit 1; }
  TAG="csynth-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== build source tarball (git archive HEAD) =="
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "tikitorch/hardware/hls/forge_kernels/synth_muzero_chess.tcl" \
    || { echo "FATAL: synth_muzero_chess.tcl not in committed tree"; exit 1; }
  aws_ ec2 describe-images --image-ids "$AMI" --query 'Images[0].State' --output text | grep -qx available \
    || { echo "FATAL: AMI $AMI not available"; exit 1; }
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching. UD:"; sed -n '1,60p' "$UD"; return 0; fi
  echo "== upload source -> $S3_SRC =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-muzero-csynth},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-csynth deadline" || true
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
# source Vitis HLS (FPGA Dev AMI ships it under /opt/Xilinx; license auto-configured)
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; \$VHLS -version 2>&1 | head -2 || true
up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
# Per-kernel: run csynth, then upload its report + log IMMEDIATELY (so a slow/hung kernel never loses the
# one that finished). CSYNTH_ONLY=forward|train|both selects which to run (train is the slow one).
ONLY="${CSYNTH_ONLY}"
if [ "\$ONLY" != "train" ]; then
  echo "=== csynth FORWARD (chess) START \$(date -u) ==="; "\$VHLS" -f synth_muzero_chess.tcl 2>&1 | tail -20
  echo "=== csynth FORWARD done \$(date -u) ==="
  aws s3 cp forge_muzero_chess_proj/solution1/syn/report/forge_train_step_muzero_csynth.rpt "\$S3_RESULTS/forge_train_step_muzero_CHESS_csynth.rpt" --only-show-errors || true
  aws s3 cp vitis_hls.log "\$S3_RESULTS/vitis_hls_forward.log" --only-show-errors || true
  up
fi
if [ "\$ONLY" != "forward" ]; then
  echo "=== csynth TRAIN (chess) START \$(date -u) ==="; "\$VHLS" -f synth_muzero_train_chess.tcl 2>&1 | tail -20
  echo "=== csynth TRAIN done \$(date -u) ==="
  aws s3 cp forge_muzero_train_chess_proj/solution1/syn/report/forge_train_step_muzero_train_csynth.rpt "\$S3_RESULTS/forge_train_step_muzero_train_CHESS_csynth.rpt" --only-show-errors || true
  aws s3 cp vitis_hls.log "\$S3_RESULTS/vitis_hls_train.log" --only-show-errors || true
  up
fi
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- boot.log tail --"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -20; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --exclude "*" --include "*CHESS_csynth.rpt" --only-show-errors
  aws_ s3 cp "$S3_RESULTS/vitis_hls.log" "$DEST/muzero_chess_vitis_hls.log" --only-show-errors 2>/dev/null || true
  ls -1 "$DEST"/*CHESS_csynth.rpt 2>/dev/null || echo "(none yet)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate} (DRY_RUN=1 for launch)";; esac
