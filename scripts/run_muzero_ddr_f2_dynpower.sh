#!/bin/bash
# run_muzero_ddr_f2_dynpower.sh — DYNAMIC per-inference power of the DDR forward MuZero AFI on real F2.
# Loads the DDR AGFI, builds muzero_ddr_power.c (stages the ~12 MB buffer into on-card DDR via fpga_dma, then
# kick-loops the kernel reading DDR), and samples Vccint BEFORE/DURING/AFTER. No hugepage / no host IOMMU —
# the kernel reads on-card DDR, so this is the drivable path. Robust one-shot (S3+IAM+self-terminate, no SSH).
#   export AWS_PROFILE=...; AGFI=agfi-085dca07f374aa1cd bash scripts/run_muzero_ddr_f2_dynpower.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; NKICKS="${NKICKS:-4000}"; MAX_MINUTES="${MAX_MINUTES:-35}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_dynpower_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="dynpower-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-muzero-ddr-dynpower},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { sleep 15; continue; }
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
exec > /var/log/tiki-dynpower.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; NKICKS="${NKICKS}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-dynpower.log "\$S3_RESULTS/dynpower.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FD="\$(command -v fpga-describe-local-image)"; FL="\$(command -v fpga-load-local-image)"; FC="\$(command -v fpga-clear-local-image)"
sudo \$FC -S 0 || true; sudo \$FL -S 0 -I "\$AGFI"; sleep 4
echo "== IDLE metrics (pre) =="; sudo \$FD -S 0 -M 2>&1 | tee /tmp/m_idle.txt; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
# compile as C (gcc): fpga_dma.h C99 array-param syntax is invalid in C++; driver hardcodes the buffer size
gcc -O2 -std=gnu11 -I"\$SDK/include" \
    /opt/tiki/tikitorch/fpga/silicon/muzero_ddr_power.c -lfpga_mgmt -o /tmp/muzero_ddr_power 2>&1 | tee /tmp/build.txt
up
( sudo /tmp/muzero_ddr_power 0 "\$NKICKS" 2>&1 | tee /tmp/driver.txt; echo DRIVER_EXIT=\${PIPESTATUS[0]} >> /tmp/driver.txt ) &
DRV=\$!
for n in \$(seq 1 14); do echo "=== during \$n \$(date -u) ==="; sudo \$FD -S 0 -M 2>&1 | grep -iE "Vccint|Last measured|Average|Max measured|pcim-|ddr"; sleep 2; done | tee /tmp/m_during.txt
wait \$DRV || true
echo "== FINAL metrics (post) =="; sudo \$FD -S 0 -M 2>&1 | tee /tmp/m_post.txt
for f in m_idle m_during m_post driver build; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/manuscript/results/fpga_silicon/ddr_dynpower"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
