#!/bin/bash
# run_pc_multif2.sh — the multi-F2 scale-out demo: a 2-layer PC backward SHARDED ACROSS 2 FPGAs (slots 0+1
# of an f2.12xlarge), neighbour-message-only, verified bit-exact to cascade_grads. Loads the PC AGFI on BOTH
# slots, builds forge_pc_2layer_shard.c, runs it. Isolated state/tag. Robust one-shot (S3+IAM+self-terminate).
#   export AWS_PROFILE=...; AGFI=agfi-063e2dd0d3255dcaa bash scripts/run_pc_multif2.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.12xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"   # f2.12xlarge = 2 FPGAs
AGFI="${AGFI:?set AGFI}"; MAX_MINUTES="${MAX_MINUTES:-25}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pc_multif2_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="multif2-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pc-multif2},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity|Unsupported" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { echo "retry ($tries): $(tail -1 "$STATE_DIR/launch.err")"; sleep 20; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-multif2.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-multif2.log "\$S3_RESULTS/multif2.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname; nproc
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FD="\$(command -v fpga-describe-local-image)"; FL="\$(command -v fpga-load-local-image)"; FC="\$(command -v fpga-clear-local-image)"
# load the SAME PC AGFI on BOTH FPGAs (slot 0 = layer 0, slot 1 = layer 1)
for S in 0 1; do sudo \$FC -S \$S || true; sudo \$FL -S \$S -I "\$AGFI"; done; sleep 4
echo "== loaded images (both slots) =="; for S in 0 1; do echo "-- slot \$S --"; sudo \$FD -S \$S 2>&1; done | tee /tmp/img.txt; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
gcc -O2 -std=gnu11 -I"\$SDK/include" \
    /opt/tiki/tikitorch/fpga/silicon/forge_pc_2layer_shard.c -lfpga_mgmt -o /tmp/forge_pc_2layer_shard 2>&1 | tee /tmp/build.txt
up
( sudo /tmp/forge_pc_2layer_shard 2>&1 | tee /tmp/shard.txt; echo SHARD_EXIT=\${PIPESTATUS[0]} >> /tmp/shard.txt ) || true
for f in img build shard; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/manuscript/results/predictive_coding/multif2"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
