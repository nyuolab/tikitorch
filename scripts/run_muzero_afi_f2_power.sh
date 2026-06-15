#!/bin/bash
# run_muzero_afi_f2_power.sh — Stage 3 of the forward MuZero AFI: load the registered AGFI on a REAL F2
# instance and measure on-silicon power (the energy axis of CHESS_MUZERO_COMPARISON.md). Same robust pattern
# (S3 + IAM profile + self-terminate, no SSH). This is the FIRST step that needs actual FPGA hardware (an F2),
# not a build box. Launches our OWN f2 instance (distinct Name tag) — leaves any other instance's F2 alone.
#
# What it measures: fpga-describe-local-image -M reports the loaded image's power (Vccint/Vccaux/Vccbram rails
# + total CL power in W). Loaded-fabric power of the routed DSP=0 design is a real on-silicon datapoint; a
# DYNAMIC (kernel-executing) measurement needs the PCIM host driver (follow-on; see README/AFI record).
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   AGFI=agfi-0b2c228a17e516e0a bash scripts/run_muzero_afi_f2_power.sh launch
#   bash scripts/run_muzero_afi_f2_power.sh status
#   bash scripts/run_muzero_afi_f2_power.sh fetch
#   bash scripts/run_muzero_afi_f2_power.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"             # 1 FPGA; smallest F2
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (has the SDK runtime: fpga-* tools)
AGFI="${AGFI:?set AGFI (the agfi-... to load)}"
MAX_MINUTES="${MAX_MINUTES:-30}"                          # load + measure is quick
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_f2power_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available \
    || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="f2power-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> UD:"; cat "$UD"; return 0; fi
  echo "== launch $INSTANCE_TYPE ($AMI; load $AGFI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-muzero-f2power},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then echo "retry $tries ($(tr -d '\n' <"$STATE_DIR/launch.err" | tail -c 120))"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_results="$1"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-f2power deadline" || true
exec > /var/log/tiki-f2power.log 2>&1
set -x
S3_RESULTS="${s3_results}"; AGFI="${AGFI}"
export HOME=/root
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-f2power.log "\$S3_RESULTS/f2power.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
# SDK runtime (fpga-* tools). FPGA Dev AMI ships aws-fpga; sdk_setup.sh builds/install the mgmt tools.
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
echo "AWS_FPGA=\$AF"
cd "\$AF" && source sdk_setup.sh || true
FD="\$(command -v fpga-describe-local-image || echo /usr/bin/fpga-describe-local-image)"
FL="\$(command -v fpga-load-local-image   || echo /usr/bin/fpga-load-local-image)"
FC="\$(command -v fpga-clear-local-image  || echo /usr/bin/fpga-clear-local-image)"
echo "TOOLS: \$FD / \$FL"; up
{ echo "== initial slot state =="; sudo \$FD -S 0; echo "== clear =="; sudo \$FC -S 0; } 2>&1 | tee /tmp/pre.txt; up
echo "== LOAD AGFI \$AGFI =="; sudo \$FL -S 0 -I "\$AGFI" -F 2>&1 | tee /tmp/load.txt; up
sleep 5
echo "== loaded state =="; sudo \$FD -S 0 2>&1 | tee /tmp/loaded.txt
# power/metrics: -M (metrics) includes the power rails + total. Sample a few times.
for n in 1 2 3 4 5; do echo "=== metrics sample \$n \$(date -u) ==="; sudo \$FD -S 0 -M 2>&1; sleep 3; done | tee /tmp/metrics.txt
up
aws s3 cp /tmp/pre.txt     "\$S3_RESULTS/pre_state.txt"   --only-show-errors || true
aws s3 cp /tmp/load.txt    "\$S3_RESULTS/load.txt"        --only-show-errors || true
aws s3 cp /tmp/loaded.txt  "\$S3_RESULTS/loaded_state.txt" --only-show-errors || true
aws s3 cp /tmp/metrics.txt "\$S3_RESULTS/metrics.txt"     --only-show-errors || true
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- f2power.log tail --"; aws_ s3 cp "$S3_RESULTS/f2power.log" - 2>/dev/null | tail -30; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/f2power"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -1 "$DEST"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
