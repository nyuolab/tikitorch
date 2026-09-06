#!/bin/bash
# run_pc_scale_gpu.sh — robust one-shot GPU runner for the tropical predictive-coding
# scale tests (Phase 0 MNIST/Fashion-MNIST deep MLP; Phase 1 CIFAR conv). Same
# S3 / IAM-instance-profile / shutdown-backstop / self-terminate / no-SSH pattern as
# run_mzgeneral_chess_gpu.sh. Ships only committed files (git archive HEAD), runs the
# selected experiment on one A10G, streams JSONL results to S3, self-terminates.
#
#   export AWS_PROFILE=my-profile
#   PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh launch | status | fetch | terminate
#   (PC_TASK=cifar for Phase 1;  DRY_RUN=1 to render user-data without launching)
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.4xlarge}"             # 1x A10G (24GB) + 16 vCPU
AMI="${AMI:-ami-01011b868ec560823}"                     # DL Base OSS Nvidia Driver, Ubuntu 22.04 (driver+CUDA)
MAX_MINUTES="${MAX_MINUTES:-200}"                        # hard cost backstop (self-terminates earlier on completion)
PC_TASK="${PC_TASK:-mnist}"                              # mnist (Phase 0) | cifar (Phase 1)
PC_ARGS="${PC_ARGS:-}"                                   # optional extra args to the experiment script
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pcscale_${PC_TASK}_state"; mkdir -p "$STATE_DIR"  # per-task so mnist/cifar don't collide
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate/fetch first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET; run the connect4 runner's setup first"; exit 1; }
  TAG="pcscale-${PC_TASK}-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== build source tarball =="; git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  local need="scripts/_pc_scale_onbox.sh tikitorch/nn/predictive_coding.py"
  case "$PC_TASK" in
    mnist) need="$need scripts/pc_scale_mnist.py" ;;
    cifar) need="$need scripts/pc_scale_cifar.py" ;;
    convlm) need="$need scripts/pc_charlm_conv.py" ;;
    *) echo "FATAL: unknown PC_TASK=$PC_TASK (want mnist|cifar|convlm)"; exit 1 ;;
  esac
  for f in $need; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not committed (git archive HEAD ships only committed files)"; exit 1; }
  done
  aws_ ec2 describe-images --image-ids "$AMI" --query 'Images[0].State' --output text | grep -qx available \
    || { echo "FATAL: AMI $AMI not available"; exit 1; }
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching ($(wc -l < "$UD") line user-data)"; return 0; fi
  echo "== upload source =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; PC_TASK=$PC_TASK; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pcscale-$PC_TASK},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "PC_TASK=$PC_TASK"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-pcscale deadline" || true
exec > /var/log/tiki-pcscale.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
export DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:\$PATH
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-pcscale.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
for i in 1 2 3; do apt-get update -y && apt-get install -y python3-venv python3-pip awscli git && break; sleep 15; done
python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
up
for i in 1 2 3; do /opt/venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 && break; sleep 15; done
/opt/venv/bin/python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available())" || true
up
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
up
export REPO=/opt/tiki S3_RESULTS="\$S3_RESULTS" PYBIN=/opt/venv/bin/python PC_TASK="${PC_TASK}" PC_ARGS="${PC_ARGS}"
bash scripts/_pc_scale_onbox.sh
rc=\$?
aws s3 cp /var/log/tiki-pcscale.log "\$S3_RESULTS/boot.log" --only-show-errors || true
echo "rc=\$rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "== DONE present ==" || { echo "-- driver.log tail --"; aws_ s3 cp "$S3_RESULTS/driver.log" - 2>/dev/null | tail -25; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  # land each run under its own TAG dir so concurrent/sequential runs never clobber
  # one another's scale_*.jsonl (a shared filename + `aws s3 cp` silently overwrites).
  local DEST="$PROJECT_DIR/results/predictive_coding/$TAG"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --exclude "*" --include "scale_*.jsonl" --include "driver.log" --only-show-errors
  ls -1 "$DEST"/scale_*.jsonl 2>/dev/null || echo "(none yet)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true
  rm -f "$STATE_FILE"; echo "(state cleared; ready to launch again)"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}  (PC_TASK=mnist|cifar  DRY_RUN=1)";; esac
