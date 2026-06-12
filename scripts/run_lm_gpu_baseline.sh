#!/bin/bash
# LM GPU baseline for the hallmark figure: nanoGPT (dense) + MorphSSM-float on
# CUDA with nvidia-smi power integration -> measured tok/s AND mJ/token.
# Modeled on scripts/run_tikitorch_gpu_benchmarks.sh (launch/monitor/download/
# terminate); payload swapped for scripts/nanogpt_shakespeare.py BPE arms.
#
# Usage:
#   AWS_PROFILE=... KEY_NAME=... SG=... bash scripts/run_lm_gpu_baseline.sh launch
#   ... monitor | download | terminate | ssh
set -e
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
KEY_NAME="${KEY_NAME:?set KEY_NAME}"; SG="${SG:?set SG}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"; REGION="${REGION:-us-east-1}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; PEM="${PEM:-$HOME/.ssh/$KEY_NAME.pem}"
AMI="${AMI:-ami-05603a42e5254c4bb}"               # DLAMI PyTorch 2.7 / Ubuntu 22.04
STATE_DIR="$PROJECT_DIR/.tiki_lm_gpu_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/instance.state"

ssh_to(){ local IP=$1; shift; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$PEM" ubuntu@"$IP" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "exists: $(cat $STATE_FILE); terminate first"; exit 1; }
  SG_VPC=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" --group-ids "$SG" --query 'SecurityGroups[0].VpcId' --output text)
  SUBNET=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" --filters "Name=vpc-id,Values=$SG_VPC" --query 'Subnets[0].SubnetId' --output text)
  MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
  aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" 2>&1 | head -1 || true
  echo "Subnet $SUBNET; launching $INSTANCE_TYPE"
  INSTANCE_ID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-lm-gpu-baseline}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "Instance: $INSTANCE_ID"
  aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID"
  IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$INSTANCE_ID|$IP" > "$STATE_FILE"; echo "IP: $IP"
  echo "Waiting for SSH..."; for i in $(seq 1 60); do ssh_to "$IP" "echo ok" 2>/dev/null && break; sleep 10; done
  echo "Syncing package + scripts + data ..."
  rsync -avz \
    --include='tikitorch/***' --include='scripts/***' --include='data/tinyshakespeare.txt' \
    --include='tests/***' --include='*/' --exclude='*' \
    --exclude='*/__pycache__/***' --exclude='tikitorch/zoo/***' \
    -e "ssh -i $PEM -o StrictHostKeyChecking=no" "$PROJECT_DIR/" ubuntu@"$IP":~/tikitorch/ 2>&1 | tail -2
  ssh_to "$IP" 'pip3 install -q tokenizers 2>&1 | tail -1; python3 -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
  echo "Launching LM baseline in tmux (power-logged) ..."
  ssh_to "$IP" "cd ~/tikitorch; mkdir -p results/lm_gpu_baseline logs
    tmux new-session -d -s lmgpu 'cd ~/tikitorch &&
      export PYTHONPATH=~/tikitorch PYTHONUNBUFFERED=1 &&
      nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv,noheader -lms 200 > results/lm_gpu_baseline/power_dense.csv &
      PWPID=\$!;
      python3 scripts/nanogpt_shakespeare.py --runs dense --tokenizer bpe --steps 1500 --device cuda --adam-lr 3e-3 --out results/lm_gpu_baseline --tag gpu_dense_bpe 2>&1 | tee logs/gpu_dense.log;
      kill \$PWPID;
      nvidia-smi --query-gpu=timestamp,power.draw,utilization.gpu --format=csv,noheader -lms 200 > results/lm_gpu_baseline/power_morph.csv &
      PWPID=\$!;
      python3 scripts/nanogpt_shakespeare.py --runs tropical-float --variant morphssm --tokenizer bpe --steps 1500 --device cuda --adam-lr 1e-2 --out results/lm_gpu_baseline --tag gpu_morphfloat_bpe 2>&1 | tee logs/gpu_morph.log;
      kill \$PWPID;
      touch LM_DONE'
    sleep 3; tmux list-sessions"
  echo "Launched on $IP ($INSTANCE_ID). monitor: $0 monitor"
}
cmd_monitor(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"
  ssh_to "$IP" 'nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader 2>/dev/null
    cd ~/tikitorch; echo == dense ==; tail -4 logs/gpu_dense.log 2>/dev/null
    echo == morph ==; tail -4 logs/gpu_morph.log 2>/dev/null
    echo "== done? =="; [ -f LM_DONE ] && echo YES || echo no'; }
cmd_download(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"
  mkdir -p "$PROJECT_DIR/results/lm_gpu_baseline"
  rsync -avz -e "ssh -i $PEM -o StrictHostKeyChecking=no" \
    ubuntu@"$IP":~/tikitorch/results/lm_gpu_baseline/ "$PROJECT_DIR/results/lm_gpu_baseline/" 2>&1 | tail -3
  scp -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP:~/tikitorch/logs/gpu_*.log" \
    "$PROJECT_DIR/results/lm_gpu_baseline/" 2>/dev/null || true; }
cmd_ssh(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP < "$STATE_FILE"; exec ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; IFS='|' read -r ID IP < "$STATE_FILE"
  aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$ID" 2>&1 | head -3 || true; rm -f "$STATE_FILE"; }
case "${1:-launch}" in launch) cmd_launch;; monitor) cmd_monitor;; download) cmd_download;; ssh) cmd_ssh;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|monitor|download|terminate|ssh}";; esac
