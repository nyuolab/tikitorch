#!/bin/bash
# run_pc_train_f2.sh — TRAIN a net ENTIRELY ON F2: load the train-step AGFI, download MNIST bins, build +
# run forge_pc_train_f2.c (host stages a batch + kicks forge_pc_train_step per step; weights persist in DDR;
# host reads weights back each epoch + evals -> accuracy climbs ON FABRIC). Isolated state/tag.
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_pc_train_f2.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; EPOCHS="${EPOCHS:-5}"; MAX_MINUTES="${MAX_MINUTES:-40}"; ROLE="${ROLE:-tiki-sweep-runner}"; VERIFY="${VERIFY:-}"
BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/pc_mnist_bins}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pc_train_f2_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="pctrain-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pc-train-f2},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { sleep 15; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-pctrain.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; EPOCHS="${EPOCHS}"; BINS_S3="${BINS_S3}"; VERIFY="${VERIFY}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-pctrain.log "\$S3_RESULTS/pctrain.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
sudo \$FC -S 0 || true; sudo \$FL -S 0 -I "\$AGFI"; sleep 4
sudo \$FD -S 0 2>&1 | tee /tmp/img.txt; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
mkdir -p /tmp/pc_mnist; for f in train_x.i8 train_y.u8 test_x.i8 test_y.u8; do aws s3 cp "\$BINS_S3/\$f" /tmp/pc_mnist/\$f --only-show-errors; done
gcc -O2 -std=gnu11 -I"\$SDK/include" /opt/tiki/tikitorch/fpga/silicon/forge_pc_train_f2.c -lfpga_mgmt -o /tmp/pctrain 2>&1 | tee /tmp/build.txt
gcc -O2 -std=gnu11 -I"\$SDK/include" /opt/tiki/tikitorch/fpga/silicon/forge_pc_train_verify.c -lfpga_mgmt -o /tmp/pcverify 2>&1 | tee -a /tmp/build.txt
up
if [ -n "\$VERIFY" ]; then ( sudo /tmp/pcverify 0 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true
else ( sudo /tmp/pctrain 0 /tmp/pc_mnist "\$EPOCHS" 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true; fi
for f in img build train; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
for w in trainedW0.i32 trainedW1.i32; do [ -f /tmp/\$w ] && aws s3 cp /tmp/\$w "\$S3_RESULTS/\$w" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/manuscript/results/predictive_coding/onfabric_train"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
