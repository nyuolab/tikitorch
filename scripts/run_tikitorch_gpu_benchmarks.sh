#!/bin/bash
# Historical image/sequence experiment launcher on an EC2 GPU.
# Explicit launch provisions a paid g5.4xlarge, syncs package + runner scripts,
# and runs sequential-MNIST/CIFAR via scripts/run_closed_benchmarks.py. These
# experiments use soft training and hard-tau evaluation; they do not certify
# multiplier-free training, physical zero DSPs, or current release performance.
# AMI/dependency choices are historical defaults to review before launch.
#
# Usage:
#   bash scripts/run_tikitorch_gpu_benchmarks.sh launch
#   bash scripts/run_tikitorch_gpu_benchmarks.sh monitor
#   bash scripts/run_tikitorch_gpu_benchmarks.sh download
#   bash scripts/run_tikitorch_gpu_benchmarks.sh terminate
set -e
usage(){
  printf 'Usage: %s {launch|monitor|download|terminate|ssh}\n' "$0"
  printf '%s\n' 'Only an explicit launch provisions paid AWS resources and changes SSH ingress.' \
    'Set AWS_PROFILE, KEY_NAME, and SG for operational commands; review AMI and REGION.' \
    'No arguments or --help prints this help without reading credentials or creating state.'
}
if [ "$#" -gt 1 ]; then usage >&2; exit 2; fi
case "${1:-help}" in
  help|-h|--help) usage; exit 0;;
  launch|monitor|download|terminate|ssh) COMMAND="$1";;
  *) usage >&2; exit 2;;
esac
# --- REQUIRED: your own AWS account/infra (this script never defaults to one) ---
PROFILE="${AWS_PROFILE:?set AWS_PROFILE to your AWS CLI profile}"
KEY_NAME="${KEY_NAME:?set KEY_NAME to your EC2 key pair name}"; SG="${SG:?set SG to your security-group id (must allow inbound SSH)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.4xlarge}"; REGION="${REGION:-us-east-1}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; PEM="${PEM:-$HOME/.ssh/$KEY_NAME.pem}"
EBS_SIZE=200
AMI="${AMI:-ami-05603a42e5254c4bb}"               # GPU AMI used for training
STATE_DIR="$PROJECT_DIR/.tiki_gpu_state"
STATE_FILE="$STATE_DIR/instance.state"

ssh_to(){ local IP=$1; shift; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$PEM" ubuntu@"$IP" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "exists: $(cat $STATE_FILE); terminate first"; exit 1; }
  mkdir -p "$STATE_DIR"
  SG_VPC=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" --group-ids "$SG" --query 'SecurityGroups[0].VpcId' --output text)
  if [ -n "${SUBNET_OVERRIDE:-}" ]; then SUBNET="$SUBNET_OVERRIDE"; else
    SUBNET=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" --filters "Name=vpc-id,Values=$SG_VPC" --query 'Subnets[0].SubnetId' --output text); fi
  MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
  aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" 2>&1 | head -1 || true
  echo "Subnet $SUBNET; launching $INSTANCE_TYPE"
  INSTANCE_ID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-gpu-bench}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "Instance: $INSTANCE_ID"
  aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID"
  IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$INSTANCE_ID|$IP" > "$STATE_FILE"; echo "IP: $IP"
  echo "Waiting for SSH..."; for i in $(seq 1 60); do ssh_to "$IP" "echo ok" 2>/dev/null && break; sleep 10; done
  echo "Syncing tikitorch package + scripts ..."
  # First match wins: exclusions must precede the subtree-wide includes.
  # Include only the two required roots; do not traverse the artifact archive.
  rsync -avz --exclude='__pycache__/' --exclude='*.py[cod]' \
    --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.cache/' \
    --exclude='.git/' --exclude='zoo/' --exclude='/manuscript/' --exclude='/results/' \
    --exclude='/tikitorch/results/' \
    --include='/tikitorch/***' --include='/scripts/***' --exclude='*' \
    -e "ssh -i $PEM -o StrictHostKeyChecking=no" "$PROJECT_DIR/" ubuntu@"$IP":~/tikitorch/ 2>&1 | tail -2
  echo "Python deps (torch + torchvision) ..."
  ssh_to "$IP" 'set -e
    if ! python3 -c "import torch" 2>/dev/null; then pip3 install -q torch --index-url https://download.pytorch.org/whl/cu121; fi
    pip3 install -q torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || pip3 install -q torchvision || true
    python3 -c "import torch,torchvision; print(\"torch\",torch.__version__,\"cuda\",torch.cuda.is_available())"' 2>&1 | tail -3
  echo "Run historical hard-eval benchmarks (smnist + scifar) in tmux ..."
  ssh_to "$IP" "cd ~/tikitorch; mkdir -p manuscript/results/closed_benchmarks logs
    tmux new-session -d -s bench 'bash -e -o pipefail -c \"cd ~/tikitorch
      rm -f bench_DONE
      export PYTHONPATH=~/tikitorch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
      python3 scripts/run_closed_benchmarks.py --task smnist --epochs 6 --batch-size 32 --accum-steps 16 --device cuda --out manuscript/results/closed_benchmarks 2>&1 | tee logs/bench_smnist.log
      python3 scripts/run_closed_benchmarks.py --task scifar --epochs 6 --batch-size 24 --accum-steps 16 --device cuda --out manuscript/results/closed_benchmarks 2>&1 | tee logs/bench_scifar.log
      touch bench_DONE\"'
    sleep 3; tmux list-sessions"
  echo "Launched on $IP ($INSTANCE_ID). monitor: $0 monitor"
}
cmd_monitor(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"
  ssh_to "$IP" 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    cd ~/tikitorch; echo "== smnist =="; tail -6 logs/bench_smnist.log 2>/dev/null
    echo "== scifar =="; tail -6 logs/bench_scifar.log 2>/dev/null
    echo "== done? =="; [ -f bench_DONE ] && echo YES || echo no'; }
cmd_download(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"
  mkdir -p "$PROJECT_DIR/manuscript/results/closed_benchmarks_remote"
  rsync -avz --include='*.json' --include='*.tt' --include='*.log' --include='*/' --exclude='*' \
    -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
    ubuntu@"$IP":~/tikitorch/manuscript/results/closed_benchmarks/ "$PROJECT_DIR/manuscript/results/closed_benchmarks_remote/" 2>/dev/null || true
  scp -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP:~/tikitorch/logs/bench_*.log" \
    "$PROJECT_DIR/manuscript/results/closed_benchmarks_remote/" 2>/dev/null || true; }
cmd_ssh(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"; exec ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; IFS='|' read -r ID IP < "$STATE_FILE"
  # Preserve the only local instance record if AWS rejects/fails the request.
  aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$ID"
  rm -f "$STATE_FILE"; }
case "$COMMAND" in launch) cmd_launch;; monitor) cmd_monitor;; download) cmd_download;; ssh) cmd_ssh;; terminate) cmd_terminate;; esac
