#!/bin/bash
# run_mzgeneral_chess_gpu.sh — robust one-shot dense MuZero-on-GPU chess baseline (axes 2-3 of
# CHESS_MUZERO_COMPARISON.md: throughput + energy). Same S3/IAM-instance-profile/shutdown-backstop/
# self-terminate/no-SSH pattern as the other runners. Runs Werner Duvaud's muzero-general (the user's
# chosen reference impl) with our chess env on one A10G, measuring self-play + train throughput and
# GPU energy (nvidia-smi). NOT run to strength — DeepMind chess is the literature ceiling, this is a
# measured systems datapoint only.
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_mzgeneral_chess_gpu.sh launch | status | fetch | terminate   (DRY_RUN=1 for launch)
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.4xlarge}"             # 1x A10G + 16 vCPU (so CPU self-play isn't starved)
AMI="${AMI:-ami-01011b868ec560823}"                      # DL Base OSS Nvidia Driver, Ubuntu 22.04 (driver+CUDA)
MAX_MINUTES="${MAX_MINUTES:-60}"                          # bootstrap (~8m) + 15m window + margin; backstop 60
MZ_MAX_SECONDS="${MZ_MAX_SECONDS:-900}"                   # measurement window
MZ_NUM_WORKERS="${MZ_NUM_WORKERS:-12}"                    # CPU self-play actors (g5.4xlarge has 16 vCPU)
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_mzgpu_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET; run the connect4 runner's setup first"; exit 1; }
  TAG="mzgpu-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== build source tarball =="; git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  for f in scripts/mzgeneral_chessgame.py scripts/mzgeneral_gpu_driver.py scripts/_mzgeneral_gpu_onbox.sh; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not committed"; exit 1; }
  done
  aws_ ec2 describe-images --image-ids "$AMI" --query 'Images[0].State' --output text | grep -qx available \
    || { echo "FATAL: AMI $AMI not available"; exit 1; }
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching ($(wc -l < "$UD") line user-data)"; return 0; fi
  echo "== upload source =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop; ${MZ_MAX_SECONDS}s window) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-mzgpu-chess},{Key=tiki-sweep,Value=$TAG}]" \
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
shutdown -h +${MAX_MINUTES} "tiki-mzgpu deadline" || true
exec > /var/log/tiki-mzgpu.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
export DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:\$PATH
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-mzgpu.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
for i in 1 2 3; do apt-get update -y && apt-get install -y python3-venv python3-pip awscli git && break; sleep 15; done
python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
up
# CUDA torch + muzero-general deps + python-chess
for i in 1 2 3; do /opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cu121 && break; sleep 15; done
for i in 1 2 3; do /opt/venv/bin/pip install -q ray tensorboard nevergrad numpy seaborn "gym[classic_control]" chess && break; sleep 15; done
/opt/venv/bin/python -c "import torch,ray,chess;print('torch',torch.__version__,'cuda',torch.cuda.is_available())" || true
up
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
up
export REPO=/opt/tiki S3_RESULTS="\$S3_RESULTS" PYBIN=/opt/venv/bin/python MZ_MAX_SECONDS="${MZ_MAX_SECONDS}" MZ_NUM_WORKERS="${MZ_NUM_WORKERS}"
bash scripts/_mzgeneral_gpu_onbox.sh
rc=\$?
aws s3 cp /var/log/tiki-mzgpu.log "\$S3_RESULTS/boot.log" --only-show-errors || true
echo "rc=\$rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && { echo "DONE present:"; aws_ s3 cp "$S3_RESULTS/mz_summary.json" - 2>/dev/null; } || { echo "-- boot.log tail --"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -20; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/chess_gpu_baseline"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --exclude "*" --include "mz_*.json*" --include "power.csv" --include "driver.log" --only-show-errors
  ls -1 "$DEST" 2>/dev/null || echo "(none yet)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate} (DRY_RUN=1)";; esac
