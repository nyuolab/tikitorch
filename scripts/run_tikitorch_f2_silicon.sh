#!/bin/bash
# Historical TikiTorch model.fpga() char-LM demonstration on AWS F2.
# Explicit launch provisions a paid f2.6xlarge, syncs package + scripts, sets up
# the SDK, loads the caller's AGFI, and invokes scripts/run_fpga_f2_verify.sh.
# A binding, a kernel's declared DSP metadata, and evolving parameters do not
# establish current numerical parity, whole-loop closure, or physical DSP use.
# Those claims require a matching build, ABI, and routed resource evidence.
# Review the verifier's historical layout limitations and AMI/dependency defaults.
#
# Usage:
#   bash scripts/run_tikitorch_f2_silicon.sh launch
#   bash scripts/run_tikitorch_f2_silicon.sh monitor
#   bash scripts/run_tikitorch_f2_silicon.sh download
#   bash scripts/run_tikitorch_f2_silicon.sh terminate
set -e
usage(){
  printf 'Usage: %s {launch|monitor|download|terminate|ssh}\n' "$0"
  printf '%s\n' 'Only an explicit launch provisions paid AWS resources and changes SSH ingress.' \
    'Set AWS_PROFILE, KEY_NAME, SG, and your own AGFI; review AMI, REGION, and image ABI.' \
    'This historical demonstration is not a current physical zero-DSP certificate.' \
    'No arguments or --help prints this help without reading credentials or creating state.'
}
if [ "$#" -gt 1 ]; then usage >&2; exit 2; fi
case "${1:-help}" in
  help|-h|--help) usage; exit 0;;
  launch|monitor|download|terminate|ssh) COMMAND="$1";;
  *) usage >&2; exit 2;;
esac
# --- REQUIRED: your own AWS account/infra. This script NEVER defaults to anyone's
# account; set these in your environment before running. ---
PROFILE="${AWS_PROFILE:?set AWS_PROFILE to your AWS CLI profile (e.g. export AWS_PROFILE=my-profile)}"
KEY_NAME="${KEY_NAME:?set KEY_NAME to your EC2 key pair name}"; SG="${SG:?set SG to your security-group id (must allow inbound SSH)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; REGION="${REGION:-us-east-1}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"; PEM="${PEM:-$HOME/.ssh/$KEY_NAME.pem}"
EBS_SIZE=200
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # overridable: the AWS FPGA Developer AMI for $REGION
AGFI="${AGFI:?set AGFI to your OWN AFI global id — this library ships NO AFI; build one from the bundled HLS (see tikitorch/fpga/BUILD_NATIVE.md)}"
STATE_DIR="$PROJECT_DIR/.tiki_f2_state"
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
  # First match wins: exclusions must precede the subtree-wide includes.
  # Include only the two required roots; do not traverse the artifact archive.
  rsync -avz --exclude='__pycache__/' --exclude='*.py[cod]' \
    --exclude='.pytest_cache/' --exclude='.ruff_cache/' --exclude='.cache/' \
    --exclude='.git/' --exclude='zoo/' --exclude='/manuscript/' --exclude='/results/' \
    --exclude='/tikitorch/results/' \
    --include='/tikitorch/***' --include='/scripts/***' --exclude='*' \
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
    tmux new-session -d -s tikif2 'bash -e -o pipefail -c \"cd ~/tikitorch
      rm -f tiki_f2_DONE
      AGFI=$AGFI bash scripts/run_fpga_f2_verify.sh 2>&1 | tee logs/tikitorch_f2.log
      touch tiki_f2_DONE\"'
    sleep 3; tmux list-sessions"
  echo "Launched on $IP ($INSTANCE_ID). monitor: $0 monitor"
}
cmd_monitor(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  ssh_to "$IP" 'cd ~/tikitorch; echo "== tikitorch_f2 log =="; tail -30 logs/tikitorch_f2.log 2>/dev/null; echo "== done? =="; [ -f tiki_f2_DONE ] && echo YES || echo no'; }
cmd_download(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  mkdir -p "$PROJECT_DIR/manuscript/results/fpga_silicon"
  scp -i "$PEM" -o StrictHostKeyChecking=no "ubuntu@$IP:~/tikitorch/logs/tikitorch_f2.log" \
    "$PROJECT_DIR/manuscript/results/fpga_silicon/tikitorch_f2.log" 2>/dev/null || echo "(no log yet)"; }
cmd_ssh(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; IFS='|' read -r ID IP AG < "$STATE_FILE"; exec ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$IP"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; IFS='|' read -r ID IP AG < "$STATE_FILE"
  # Preserve the only local instance record if AWS rejects/fails the request.
  aws ec2 terminate-instances --profile "$PROFILE" --region "$REGION" --instance-ids "$ID"
  rm -f "$STATE_FILE"; }
case "$COMMAND" in launch) cmd_launch;; monitor) cmd_monitor;; download) cmd_download;; ssh) cmd_ssh;; terminate) cmd_terminate;; esac
