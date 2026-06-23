#!/bin/bash
# run_s1_train_f2.sh — TRAIN a multiplier-free TRANSFORMER BLOCK ENTIRELY ON F2: load the S1 trainer AGFI,
# build + run forge_train_step_s1_f2.c (host stages the ~6.15 MB working-set buffer into on-card DDR over
# BAR4 DMA-PCIS, points the kernel at DDR addr 0, then ONE mode-3 kick runs N_STEPS fwd->bwd->dyadic-Adam
# resident in DDR — no per-step host round-trip; weights/state persist across the fused run). Host reads back
# LOSS + trained W for the bit-exact / loss-curve check vs the CSIM golden. Cloned from run_pc_train_f2.sh:
# same structure (AGFI guard, f2.6xlarge, FPGA Developer AMI, IAM role, 40m backstop, gcc -std=gnu11 the
# driver, run it, upload results, self-terminate). Differences: arg is N_STEPS (not MNIST epochs); no MNIST
# bins fetch (S1 stages its own buffer init in the driver — exp2 LUTs + W_fp are host-staged, BRAM static
# init reads ZERO on AFI, see plan R4/R10); results -> results/fpga_training_rig/onfabric_train/.
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_s1_train_f2.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; N_STEPS="${N_STEPS:-200}"; MAX_MINUTES="${MAX_MINUTES:-40}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_s1_train_f2_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="s1train-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-s1-train-f2},{Key=tiki-sweep,Value=$TAG}]" \
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
exec > /var/log/tiki-s1train.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; N_STEPS="${N_STEPS}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-s1train.log "\$S3_RESULTS/s1train.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
sudo \$FC -S 0 || true; sudo \$FL -S 0 -I "\$AGFI"; sleep 4
sudo \$FD -S 0 2>&1 | tee /tmp/img.txt; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
# build the S1 host driver (compile as C, -std=gnu11; the shared forge_s1_config.hpp is C++ and does
# `#include <cstdint>`, so we add the cstdint_shim/ dir on the -I path (a one-line cstdint->stdint.h
# redirect) — see the driver header comment. -I the forge_kernels dir for forge_s1_config.hpp itself).
gcc -O2 -std=gnu11 -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/silicon/forge_train_step_s1_f2.c -lfpga_mgmt -lm -o /tmp/s1_f2 2>&1 | tee /tmp/build.txt
up
# ONE mode-3 kick = N_STEPS fwd->bwd->dyadic-Adam resident in DDR (stages buffer + exp2 LUTs + W_fp first)
( sudo /tmp/s1_f2 0 "\$N_STEPS" 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true
for f in img build train; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/results/fpga_training_rig/onfabric_train"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... N_STEPS=200 $0 {launch|status|fetch|terminate}";; esac
