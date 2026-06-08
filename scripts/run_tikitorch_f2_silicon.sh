#!/bin/bash
# Reference: TikiTorch model.fpga() char-LM trained on REAL AWS F2 silicon.
# Provisions an f2.6xlarge, syncs the tikitorch package, sets up the aws-fpga SDK,
# loads the validated monolithic forge_train_step AGFI, and runs the proven
# scripts/run_fpga_f2_verify.sh (binds model.fpga(), runs 50 on-board train_step()s,
# reports bound/dsp=0/params-evolved). The on-silicon zero-DSP TRAINING reference.
#
# Usage:
#   bash scripts/run_tikitorch_f2_silicon.sh launch
#   bash scripts/run_tikitorch_f2_silicon.sh monitor
#   bash scripts/run_tikitorch_f2_silicon.sh download
#   bash scripts/run_tikitorch_f2_silicon.sh terminate
set -e
PROFILE="NYUMCResearchAdminFullAccess-739966866637"
KEY_NAME="mnemo-dev"; SG="sg-0efe5c4b6eeb6fda6"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; REGION="us-east-1"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; PEM="$HOME/.ssh/mnemo-dev.pem"
EBS_SIZE=200
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # F2 FPGA dev AMI
AGFI="${AGFI:-agfi-0a845113b088cf62e}"                    # cl_forge_train_step_v7_monolithic
STATE_DIR="$PROJECT_DIR/.tiki_f2_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/instance.state"

ssh_to(){ local IP=$1; shift; ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i "$PEM" ubuntu@"$IP" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "exists: $(cat $STATE_FILE); terminate first"; exit 1; }
  SG_VPC=$(aws ec2 describe-security-groups --profile "$PROFILE" --region "$REGION" --group-ids "$SG" --query 'SecurityGroups[0].VpcId' --output text)
  if [ -n "${SUBNET_OVERRIDE:-}" ]; then SUBNET="$SUBNET_OVERRIDE"; else
    SUBNET=$(aws ec2 describe-subnets --profile "$PROFILE" --region "$REGION" --filters "Name=vpc-id,Values=$SG_VPC" --query 'Subnets[0].SubnetId' --output text); fi
  MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
  aws ec2 authorize-security-group-ingress --profile "$PROFILE" --region "$REGION" --group-id "$SG" --protocol tcp --port 22 --cidr "$MY_IP/32" 2>&1 | head -1 || true
  echo "Subnet $SUBNET; launching $INSTANCE_TYPE; AGFI=$AGFI"
  INSTANCE_ID=$(aws ec2 run-instances --profile "$PROFILE" --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG" --subnet-id "$SUBNET" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-f2-silicon}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "Instance: $INSTANCE_ID"
  aws ec2 wait instance-running --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID"
  IP=$(aws ec2 describe-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  echo "$INSTANCE_ID|$IP|$AGFI" > "$STATE_FILE"; echo "IP: $IP"
  echo "Waiting for SSH..."; for i in $(seq 1 60); do ssh_to "$IP" "echo ok" 2>/dev/null && break; sleep 10; done
  echo "Syncing tikitorch package + scripts to ~/tikitorch ..."
  rsync -avz --include='tikitorch/***' --include='scripts/***' --include='*/' --exclude='*' \
    --exclude='tikitorch/zoo/***' --exclude='tikitorch/results/***' --exclude='*/__pycache__/***' \
    -e "ssh -i $PEM -o StrictHostKeyChecking=no" "$PROJECT_DIR/" ubuntu@"$IP":~/tikitorch/ 2>&1 | tail -2
  echo "SDK setup + system-wide Python deps (torch CPU + numpy) ..."
  ssh_to "$IP" 'set -e; cd ~; mkdir -p ~/tikitorch/logs
    if [ ! -d aws-fpga ]; then git clone --depth 1 --branch f2 https://github.com/aws/aws-fpga.git; fi
    cd aws-fpga && source ./sdk_setup.sh > ~/sdk_setup.log 2>&1; tail -3 ~/sdk_setup.log
    sudo apt-get install -y -q python3-pip 2>&1 | tail -1 || true
    # deps must be SYSTEM-WIDE (the bridge runs under sudo for PCI register access).
    sudo python3 -m pip install -q "numpy==1.24.4" "typing-extensions==4.12.2" 2>&1 | tail -1 || true
    sudo python3 -m pip install -q "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -1 || true
    sudo python3 -c "import numpy,torch; print(\"sudo deps OK np\",numpy.__version__,\"torch\",torch.__version__)"' 2>&1 | tail -6
  echo "Run model.fpga() char-LM on silicon (tmux) ..."
  ssh_to "$IP" "cd ~/tikitorch
    tmux new-session -d -s tikif2 'cd ~/tikitorch && AGFI=$AGFI bash scripts/run_fpga_f2_verify.sh 2>&1 | tee logs/tikitorch_f2.log; touch tiki_f2_DONE'
    sleep 3; tmux list-sessions"
  echo "Launched on $IP ($INSTANCE_ID). monitor: $0 monitor"
}
cmd_monitor(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  ssh_to "$IP" 'cd ~/tikitorch; echo "== tikitorch_f2 log =="; tail -30 logs/tikitorch_f2.log 2>/dev/null; echo "== done? =="; [ -f tiki_f2_DONE ] && echo YES || echo no'; }
cmd_download(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  mkdir -p "$PROJECT_DIR/tikitorch/results/fpga_silicon"
  scp -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP:~/tikitorch/logs/tikitorch_f2.log" \
    "$PROJECT_DIR/tikitorch/results/fpga_silicon/tikitorch_f2.log" 2>/dev/null || echo "(no log yet)"; }
cmd_ssh(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"; exec ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$ID" 2>&1 | head -3 || true; rm -f "$STATE_FILE"; }
case "${1:-launch}" in launch) cmd_launch;; monitor) cmd_monitor;; download) cmd_download;; ssh) cmd_ssh;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|monitor|download|terminate|ssh}";; esac
