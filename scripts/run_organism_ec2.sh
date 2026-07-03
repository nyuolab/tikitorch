#!/bin/bash
# run_organism_ec2.sh — launcher for Organism v0: a PERSISTENT always-learning service
# (ontogenic_learner/organism/). Adapted from run_connect4_prior_sweep_ec2.sh (the robust
# no-SSH pattern: S3-only interaction, IAM instance profile, SSM-resolved AMI) with the
# service-critical differences:
#   * NO self-terminate backstop — this box must LIVE; end-of-life is software (day 14),
#     termination only via the CLI. shutdown-behavior=stop (not terminate).
#   * systemd unit (Restart=always, enabled) owns the daemon; user-data runs once and exits.
#   * cron watchdog: heartbeat stale >10 min -> restart service; >30 min -> reboot box.
#   * DATA is gitignored -> a separate data.tar.gz ships the world (text8/shakespeare/MNIST).
#
# Usage:
#   export AWS_PROFILE=g1b
#   bash scripts/run_organism_ec2.sh setup      # idempotent bucket/role (+SSM emergency shell)
#   bash scripts/run_organism_ec2.sh launch     # commit first! (tarball preflight enforces)
#   bash scripts/run_organism_ec2.sh status
#   bash scripts/run_organism_ec2.sh terminate  # emergency kill (prefer: organism CLI terminate)
#   SMOKE=1 ... launch                          # minutes-scale life for the shakedown
set -euo pipefail

PROFILE="${AWS_PROFILE:?set AWS_PROFILE (e.g. g1b)}"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.2xlarge}"   # 8 vCPU, ~$0.357/hr
EBS_SIZE="${EBS_SIZE:-50}"
ROLE="${ROLE:-tiki-sweep-runner}"
SMOKE="${SMOKE:-0}"
LIFE_DAYS="${LIFE_DAYS:-14}"
PROBE_HOURS="${PROBE_HOURS:-6}"
HOURLY_RATE_DEFAULT="0.357"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_organism_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"

aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }
acct(){ aws_ sts get-caller-identity --query Account --output text; }

cmd_setup(){
  local ACCT BUCKET; ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws_ s3api create-bucket --bucket "$BUCKET" >/dev/null
  aws_ iam get-role --role-name "$ROLE" >/dev/null 2>&1 || aws_ iam create-role --role-name "$ROLE" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws_ iam put-role-policy --role-name "$ROLE" --policy-name s3-sweep \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:DeleteObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${BUCKET}\",\"arn:aws:s3:::${BUCKET}/*\"]}]}" >/dev/null
  aws_ iam attach-role-policy --role-name "$ROLE" \
    --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore 2>/dev/null || true
  aws_ iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1 || \
    aws_ iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws_ iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null || true
  echo "SETUP OK bucket=$BUCKET role=$ROLE (+SSM emergency shell)"
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists ($STATE_FILE) — a life is recorded; terminate + rm it first"; exit 1; }
  local ACCT BUCKET AMI TAG S3P UD INSTANCE_ID
  ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="org-$(date -u +%Y%m%dT%H%M%SZ)"
  S3P="s3://${BUCKET}/organism/${TAG}"

  echo "== source tarball (git archive HEAD) with organism preflight =="
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "ontogenic_learner/organism/daemon.py" \
    || { echo "FATAL: organism/ not in committed tree — commit it first (the trap!)"; exit 1; }

  echo "== data tarball (gitignored world) =="
  tar czf "$STATE_DIR/data.tar.gz" -C "$PROJECT_DIR" \
    data/text8_20m.txt data/tinyshakespeare.txt data/MNIST/raw
  ls -lh "$STATE_DIR"/src.tar.gz "$STATE_DIR"/data.tar.gz

  echo "== organism.json =="
  PYTHONPATH="$PROJECT_DIR" python3 - "$STATE_DIR/organism.json" <<PYEOF
import sys
from ontogenic_learner.organism.config import make_config
cfg = make_config(smoke=($SMOKE == 1))
cfg.life_tag = "$TAG"; cfg.s3_prefix = "$S3P"
cfg.max_life_days = float($LIFE_DAYS) if not cfg.smoke else cfg.max_life_days
cfg.probe_every_hours = float($PROBE_HOURS) if not cfg.smoke else cfg.probe_every_hours
cfg.save(sys.argv[1]); print("config hash", cfg.config_hash())
PYEOF

  AMI="$(aws_ ssm get-parameters \
    --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id \
    --query 'Parameters[0].Value' --output text)"
  [ -n "$AMI" ] && [ "$AMI" != "None" ] || { echo "FATAL: no AMI"; exit 1; }

  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3P" > "$UD"
  echo "== uploading src/data/config -> $S3P =="
  aws_ s3 cp "$STATE_DIR/src.tar.gz"  "$S3P/src.tar.gz"  --only-show-errors
  aws_ s3 cp "$STATE_DIR/data.tar.gz" "$S3P/data.tar.gz" --only-show-errors
  aws_ s3 cp "$STATE_DIR/organism.json" "$S3P/organism.json" --only-show-errors

  echo "== launching $INSTANCE_TYPE (PERSISTENT: stop-on-shutdown, no backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances \
        --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" \
        --instance-initiated-shutdown-behavior stop \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-organism},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" \
        --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then
      break
    fi
    tries=$((tries+1))
    grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ] \
      && { echo "  IAM propagating (try $tries)..."; sleep 10; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done

  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"
    echo "PROFILE=$PROFILE"; echo "S3_PREFIX=$S3P"; echo "BUCKET=$BUCKET"
    echo "INSTANCE_TYPE=$INSTANCE_TYPE"; echo "HOURLY_RATE=$HOURLY_RATE_DEFAULT"
  } > "$STATE_FILE"
  echo "BORN: $INSTANCE_ID -> $S3P"
  echo "next: python3 -m ontogenic_learner.organism.cli status"
}

render_user_data(){
  local s3p="$1"
  cat <<UDEOF
#!/bin/bash
# Organism v0 box bootstrap — runs ONCE; the systemd service owns the life afterwards.
# Deliberately NO shutdown backstop: this is a persistent service (terminate via CLI only).
exec > /var/log/organism-boot.log 2>&1
set -x
S3P="${s3p}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/organism-boot.log "\$S3P/boot.log" --only-show-errors || true; }
trap up EXIT
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update -y && apt-get install -y python3-venv python3-pip unzip && break; sleep 15; done
python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
for i in 1 2 3; do /opt/venv/bin/pip install -q awscli && break; sleep 10; done
up
for i in 1 2 3; do /opt/venv/bin/pip install -q numpy matplotlib && break; sleep 10; done
for i in 1 2 3; do /opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu && break; sleep 15; done
ln -sf /opt/venv/bin/aws /usr/local/bin/aws
up
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3P/src.tar.gz" . && break; sleep 10; done
tar xzf src.tar.gz
for i in 1 2 3; do aws s3 cp "\$S3P/data.tar.gz" . && break; sleep 10; done
tar xzf data.tar.gz
aws s3 cp "\$S3P/organism.json" /opt/tiki/organism.json
up

cat > /etc/systemd/system/organism.service <<'UNIT'
[Unit]
Description=Organism v0 — always-on continual learner
After=network-online.target
[Service]
ExecStart=/opt/venv/bin/python -u /opt/tiki/ontogenic_learner/organism/daemon.py --config /opt/tiki/organism.json --workdir /opt/tiki/life
WorkingDirectory=/opt/tiki
Environment=PYTHONPATH=/opt/tiki
Environment=PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
Restart=always
RestartSec=10
MemoryMax=12G
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now organism

# watchdog: heartbeat stale >10min -> restart; >30min -> reboot. disk guard: pause at <5%.
cat > /usr/local/bin/organism-watchdog <<'WD'
#!/bin/bash
exec 9>/var/lock/organism-wd; flock -n 9 || exit 0
ST=/opt/tiki/life/status.json
now=\$(date +%s)
if [ -f "\$ST" ]; then
  ts=\$(python3 -c "import json;print(int(json.load(open('\$ST'))['ts']))" 2>/dev/null || echo 0)
  age=\$((now - ts))
  [ \$age -gt 1800 ] && { echo "\$(date) reboot (age \$age)" >> /var/log/organism-wd.log; /sbin/reboot; }
  [ \$age -gt 600 ] && { echo "\$(date) restart (age \$age)" >> /var/log/organism-wd.log; systemctl restart organism; }
fi
free_pct=\$(df --output=pcent /opt/tiki | tail -1 | tr -dc '0-9')
[ "\$free_pct" -gt 95 ] && touch /opt/tiki/life/PAUSE && echo "\$(date) disk>95% PAUSE" >> /var/log/organism-wd.log
WD
chmod +x /usr/local/bin/organism-watchdog
echo "*/5 * * * * root /usr/local/bin/organism-watchdog" > /etc/cron.d/organism
echo "BOOTSTRAP DONE \$(date -u)"
UDEOF
}

cmd_status(){
  [ -f "$STATE_FILE" ] || { echo "no state; launch first"; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,LaunchTime]' --output text || true
  aws_ s3 cp "$S3_PREFIX/status.json" - 2>/dev/null || \
    { echo "(no status.json yet — booting?)"; aws_ s3 cp "$S3_PREFIX/boot.log" - 2>/dev/null | tail -8 || true; }
}

cmd_terminate(){
  [ -f "$STATE_FILE" ] || { echo "no state"; exit 0; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  echo "PREFER: python3 -m ontogenic_learner.organism.cli terminate  (final checkpoint + ack)"
  read -r -p "hard-terminate $INSTANCE_ID without final checkpoint? [y/N] " a
  [ "$a" = "y" ] && aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
}

case "${1:-}" in
  setup) cmd_setup;;
  launch) cmd_launch;;
  status) cmd_status;;
  terminate) cmd_terminate;;
  *) echo "usage: AWS_PROFILE=g1b $0 {setup|launch|status|terminate}  [SMOKE=1] [LIFE_DAYS=14]";;
esac
