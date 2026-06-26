#!/bin/bash
# Phase A1 GPU run: train the PC-fabric LM recipe "#3" (warm dot-attention + multiplier-free PC bulk;
# scripts/pc_xformer_charlm.py) on a GPU and SAMPLE coherent TinyShakespeare. Also runs the full-float
# (MULTFREE=0) NanoGPT-style control for side-by-side. Cloned from run_lm_gpu_baseline.sh (same launch/sync/
# terminate). Scale via CFG (default = the ~10.6M NanoGPT-char config that produces coherent Shakespeare;
# set CFG=med for the ~350M GPT-2-Medium target).
#   AWS_PROFILE=... KEY_NAME=... SG=... bash scripts/run_charlm_gpu.sh launch    # then monitor|sample|terminate
set -e
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; KEY_NAME="${KEY_NAME:?set KEY_NAME}"; SG="${SG:?set SG}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"; REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-05603a42e5254c4bb}"               # DLAMI PyTorch 2.7 / Ubuntu 22.04 (same as run_lm_gpu_baseline.sh)
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; PEM="${PEM:-$HOME/.ssh/$KEY_NAME.pem}"
STATE_DIR="$PROJECT_DIR/.tiki_charlm_gpu_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/instance.state"
CFG="${CFG:-small}"
# scale presets: small ~= NanoGPT-char (coherent Shakespeare); med ~= GPT-2-Medium (~350M)
if [ "$CFG" = med ]; then ENVCFG="XF_D=1024 XF_LAYER=24 XF_HEAD=16 XF_BLOCK=256 XF_BATCH=24 XF_STEPS=20000 XF_WARMUP=1000 XF_LR=1e-3";
else                      ENVCFG="XF_D=384 XF_LAYER=6 XF_HEAD=6 XF_BLOCK=256 XF_BATCH=64 XF_STEPS=6000 XF_WARMUP=300 XF_LR=3e-3"; fi
ssh_to(){ local IP=$1; shift; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$PEM" ubuntu@"$IP" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "exists: $(cat $STATE_FILE); terminate first"; exit 1; }
  SG_VPC=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" --group-ids "$SG" --query 'SecurityGroups[0].VpcId' --output text)
  SUBNET=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" --filters "Name=vpc-id,Values=$SG_VPC" --query 'Subnets[0].SubnetId' --output text)
  MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
  aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" 2>&1 | head -1 || true
  echo "launch $INSTANCE_TYPE (CFG=$CFG: $ENVCFG)"
  INSTANCE_ID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" --image-id "$AMI" \
    --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --instance-initiated-shutdown-behavior terminate \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-charlm-gpu}]" \
    --query 'Instances[0].InstanceId' --output text)
  aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID"
  IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$INSTANCE_ID|$IP" > "$STATE_FILE"; echo "Instance $INSTANCE_ID  IP $IP"
  for i in $(seq 1 60); do ssh_to "$IP" "echo ok" 2>/dev/null && break; sleep 10; done
  rsync -avz --include='scripts/***' --include='data/tinyshakespeare.txt' --include='*/' --exclude='*' \
    -e "ssh -i $PEM -o StrictHostKeyChecking=no" "$PROJECT_DIR/" ubuntu@"$IP":~/tikitorch/ 2>&1 | tail -2
  # data: the script reads ~/.cache/tiur/tinyshakespeare.txt; stage it (from the repo or fetch)
  ssh_to "$IP" 'mkdir -p ~/.cache/tiur; [ -f ~/tikitorch/data/tinyshakespeare.txt ] && cp ~/tikitorch/data/tinyshakespeare.txt ~/.cache/tiur/ || curl -sL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o ~/.cache/tiur/tinyshakespeare.txt
    python3 -c "import torch;print(\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0))"'
  echo "training #3 (mult-free) + float control in tmux ..."
  ssh_to "$IP" "cd ~/tikitorch && export PYTHONPATH=~/tikitorch PYTHONUNBUFFERED=1 && mkdir -p results/charlm_gpu &&
    tmux new-session -d -s charlm 'cd ~/tikitorch && export PYTHONPATH=~/tikitorch PYTHONUNBUFFERED=1 &&
      $ENVCFG MULTFREE=1 python3 scripts/pc_xformer_charlm.py > results/charlm_gpu/mf.log 2>&1 ;
      $ENVCFG MULTFREE=0 python3 scripts/pc_xformer_charlm.py > results/charlm_gpu/float.log 2>&1 ;
      echo DONE > results/charlm_gpu/DONE'"
  echo "launched. monitor:  AWS_PROFILE=$PROFILE KEY_NAME=$KEY_NAME SG=$SG bash $0 sample"
}
cmd_state(){ cat "$STATE_FILE"; }
cmd_sample(){ IFS='|' read -r ID IP < "$STATE_FILE"; ssh_to "$IP" 'tail -25 ~/tikitorch/results/charlm_gpu/mf.log 2>/dev/null; echo "=== float control ==="; tail -8 ~/tikitorch/results/charlm_gpu/float.log 2>/dev/null'; }
cmd_download(){ IFS='|' read -r ID IP < "$STATE_FILE"; mkdir -p "$PROJECT_DIR/results/charlm_gpu"; rsync -avz -e "ssh -i $PEM -o StrictHostKeyChecking=no" ubuntu@"$IP":~/tikitorch/results/charlm_gpu/ "$PROJECT_DIR/results/charlm_gpu/"; }
cmd_terminate(){ IFS='|' read -r ID IP < "$STATE_FILE"; aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; state) cmd_state;; sample) cmd_sample;; download) cmd_download;; terminate) cmd_terminate;; *) echo "usage: AWS_PROFILE=.. KEY_NAME=.. SG=.. CFG=small|med $0 {launch|sample|download|terminate}";; esac
