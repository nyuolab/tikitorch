#!/bin/bash
# run_pc_shard_net_f2.sh — launch TWO f2 instances and train ONE net MODEL-PARALLEL across BOTH boxes
# (2*NSLOTS physical FPGAs) with forge_pc_train_shard_net.c: the tail listens, the head connects, and ONLY the
# boundary activation (fwd) + boundary error (bwd) cross the VPC network. Structural demo (TCP<PCIe<fabric):
# NOT a throughput win — it shows the no-global-coordination scheme runs across two physically separate boxes.
# Cheap proof: NSLOTS=2 INSTANCE_TYPE=f2.12xlarge -> 4 FPGAs across 2 boxes. Full: NSLOTS=8 f2.48xlarge -> 16.
#   export AWS_PROFILE=...; AGFI=agfi-... INITPERM_S3=s3://.../pc_initperm_net16 NSLOTS=8 INSTANCE_TYPE=f2.48xlarge \
#     bash scripts/run_pc_shard_net_f2.sh launch        # then: status / fetch / terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.12xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; EPOCHS="${EPOCHS:-4}"; NSLOTS="${NSLOTS:-2}"; PORT="${PORT:-5599}"; MAX_MINUTES="${MAX_MINUTES:-120}"; ROLE_PROF="${ROLE_PROF:-tiki-sweep-runner}"
BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/pc_fashion_bins}"
INITPERM_S3="${INITPERM_S3:-s3://tiki-sweep-739966866637/pc_initperm_net16}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pc_shardnet_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

# user-data for one box. $1=role(head|tail) $2=peer-private-ip(head only) $3=goff-tag $4=S3_RESULTS
render(){
  local role="$1" peer="$2" tag="$3" s3r="$4" listen=""
  [ "$role" = tail ] && listen="LISTEN=1"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-shardnet.log 2>&1
set -x; export HOME=/root
up(){ aws s3 cp /var/log/tiki-shardnet.log "${s3r}/${role}.log" --only-show-errors || true; }
trap up EXIT
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"
for s in \$(seq 0 \$((${NSLOTS}-1))); do sudo \$FC -S \$s || true; sudo \$FL -S \$s -I "${AGFI}"; done; sleep 4
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "${S3_SRC}" s.tgz && break; sleep 8; done && tar xzf s.tgz
mkdir -p /tmp/pc_mnist; for f in train_x.i8 train_y.u8 test_x.i8 test_y.u8; do aws s3 cp "${BINS_S3}/\$f" /tmp/pc_mnist/\$f --only-show-errors; done
mkdir -p /tmp/ip; aws s3 cp "${INITPERM_S3}/" /tmp/ip/ --recursive --only-show-errors
gcc -O2 -std=gnu11 -I"\$SDK/include" /opt/tiki/tikitorch/fpga/silicon/forge_pc_train_shard_net.c -lfpga_mgmt -o /tmp/shardnet 2>&1 | tee /tmp/build.txt; up
SLOTS=""; for s in \$(seq 0 \$((${NSLOTS}-1))); do SLOTS="\$SLOTS \$s"; done
sudo ROLE=${role} ${listen} PEER="${peer}" PORT=${PORT} PC_INITPERM=/tmp/ip /tmp/shardnet "${EPOCHS}" /tmp/pc_mnist \$SLOTS 2>&1 | tee /tmp/run.txt
echo EXIT=\${PIPESTATUS[0]} >> /tmp/run.txt
for w in /tmp/netW*.i32; do aws s3 cp "\$w" "${s3r}/\$(basename \$w)" --only-show-errors || true; done
aws s3 cp /tmp/run.txt "${s3r}/${role}_run.txt" --only-show-errors || true; aws s3 cp /tmp/build.txt "${s3r}/${role}_build.txt" --only-show-errors || true
echo done | aws s3 cp - "${s3r}/${role}_DONE" --only-show-errors || true
up; sync; shutdown -h now
UDEOF
}

ensure_sg(){   # a security group that allows all intra-group TCP (so head<->tail can talk on $PORT)
  local vpc sg
  vpc="$(aws_ ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)"
  sg="$(aws_ ec2 describe-security-groups --filters Name=group-name,Values=tiki-shardnet Name=vpc-id,Values=$vpc --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo None)"
  if [ "$sg" = None ] || [ -z "$sg" ]; then
    sg="$(aws_ ec2 create-security-group --group-name tiki-shardnet --description 'tiki cross-instance PC' --vpc-id "$vpc" --query GroupId --output text)"
    aws_ ec2 authorize-security-group-ingress --group-id "$sg" --protocol tcp --port 0-65535 --source-group "$sg" >/dev/null
  fi
  echo "$sg"
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG SG SUBNET TAIL_ID TAIL_IP HEAD_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="shardnet-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  export S3_SRC
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  SG="$(ensure_sg)"; echo "SG=$SG"
  SUBNET="$(aws_ ec2 describe-subnets --filters Name=default-for-az,Values=true --query 'Subnets[0].SubnetId' --output text)"
  # launch TAIL (listens), get its private IP, then launch HEAD pointed at it
  render tail "" tail "$S3_RESULTS" > "$STATE_DIR/ud_tail.sh"
  TAIL_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --iam-instance-profile "Name=$ROLE_PROF" \
     --instance-initiated-shutdown-behavior terminate --security-group-ids "$SG" --subnet-id "$SUBNET" \
     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-shardnet-tail},{Key=tiki-sweep,Value=$TAG}]" \
     --user-data "file://$STATE_DIR/ud_tail.sh" --query 'Instances[0].InstanceId' --output text)"
  echo "tail $TAIL_ID; waiting for private IP..."
  for i in $(seq 1 30); do TAIL_IP="$(aws_ ec2 describe-instances --instance-ids "$TAIL_ID" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)"; [ "$TAIL_IP" != None ] && [ -n "$TAIL_IP" ] && break; sleep 5; done
  echo "tail private IP=$TAIL_IP"
  render head "$TAIL_IP" head "$S3_RESULTS" > "$STATE_DIR/ud_head.sh"
  HEAD_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --iam-instance-profile "Name=$ROLE_PROF" \
     --instance-initiated-shutdown-behavior terminate --security-group-ids "$SG" --subnet-id "$SUBNET" \
     --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-shardnet-head},{Key=tiki-sweep,Value=$TAG}]" \
     --user-data "file://$STATE_DIR/ud_head.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "TAIL_ID=$TAIL_ID"; echo "HEAD_ID=$HEAD_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED head=$HEAD_ID tail=$TAIL_ID -> $S3_RESULTS"
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$HEAD_ID" "$TAIL_ID" --query 'Reservations[].Instances[].[InstanceId,State.Name]' --output text; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/results/predictive_coding/shard_net"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$HEAD_ID" "$TAIL_ID" --query 'TerminatingInstances[].CurrentState.Name' --output text 2>&1 | head; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
