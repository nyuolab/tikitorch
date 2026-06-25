#!/bin/bash
# run_pc_xformer_shard_f2.sh — train a TRANSFORMER MODEL-PARALLEL across the FPGAs of an f2.48xlarge (8 slots):
# load the forge_pc_xformer_block AGFI on every slot, build + run forge_pc_xformer_shard.cpp (one block per chip;
# only a neighbour hidden-state + error cross; CE/bpb on the output chip). Cloned from run_pc_shard_train_f2.sh;
# distinct STATE_DIR + tag; compiles the driver with g++ (it includes the C++ config header). Isolated state.
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_pc_xformer_shard_f2.sh launch   # then status/fetch/terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.48xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; STEPS="${STEPS:-3000}"; NSLOTS="${NSLOTS:-8}"; MAX_MINUTES="${MAX_MINUTES:-90}"; ROLE="${ROLE:-tiki-sweep-runner}"
BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/pc_shakespeare_bins}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_xformer_shard_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="xfshard-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-xformer-shard},{Key=tiki-sweep,Value=$TAG}]" \
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
exec > /var/log/tiki-xfshard.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; STEPS="${STEPS}"; NSLOTS="${NSLOTS}"; BINS_S3="${BINS_S3}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-xfshard.log "\$S3_RESULTS/xfshard.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
SLOTS=""; for s in \$(seq 0 \$((NSLOTS-1))); do SLOTS="\$SLOTS \$s"; sudo \$FC -S \$s || true; sudo \$FL -S \$s -I "\$AGFI"; done
sleep 4; for s in \$SLOTS; do sudo \$FD -S \$s 2>&1 | tee -a /tmp/img.txt; done; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
mkdir -p /tmp/pc_shakespeare; for fn in train.u16 val.u16; do aws s3 cp "\$BINS_S3/\$fn" /tmp/pc_shakespeare/\$fn --only-show-errors; done
g++ -O2 -std=c++14 -I"\$SDK/include" -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    /opt/tiki/tikitorch/fpga/silicon/forge_pc_xformer_shard.cpp -lfpga_mgmt -o /tmp/xfshard 2>&1 | tee /tmp/build.txt; up
( sudo /tmp/xfshard "\$STEPS" /tmp/pc_shakespeare \$SLOTS 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true
for fdump in img build train; do aws s3 cp /tmp/\$fdump.txt "\$S3_RESULTS/\$fdump.txt" --only-show-errors || true; done
for w in /tmp/xfW*.i32; do [ -f "\$w" ] && aws s3 cp "\$w" "\$S3_RESULTS/\$(basename \$w)" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; sync; shutdown -h now; exit 0
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/results/predictive_coding/xformer_shard"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
