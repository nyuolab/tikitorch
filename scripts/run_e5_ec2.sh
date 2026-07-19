#!/bin/bash
# run_e5_ec2.sh — launcher for E5: the second life (st7_life.py). Ported from
# run_organism_ec2.sh (persistent service: NO self-terminate backstop,
# stop-on-shutdown, systemd Restart=always with --resume, cron watchdog on
# status.json ts, cron S3 sync). Differences from the organism launcher:
# procedural worlds (no data tarball); pip needs torch+minatar+minigrid.
#   bash scripts/run_e5_ec2.sh launch          # LIFE=pilot (default) | full
#   bash scripts/run_e5_ec2.sh status|terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.2xlarge}"
EBS_SIZE="${EBS_SIZE:-40}"
ROLE="${ROLE:-tiki-sweep-runner}"
LIFE="${LIFE:-pilot}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_e5_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }
acct(){ aws_ sts get-caller-identity --query Account --output text; }

write_cfg(){
  python3 - "$1" <<PYEOF
import json, sys
pilot = dict(era_steps=100_000, biome_steps=25_000, cycles=4,
             probe_every_hours=1.0, ckpt_every_steps=25_000,
             ckpt_every_secs=900, budget_wall_hours=40.0,
             mech_lever="headobsk", fire_schedule="always")
full = dict(era_steps=500_000, biome_steps=125_000, cycles=10,
            probe_every_hours=6.0, ckpt_every_steps=50_000,
            ckpt_every_secs=900, budget_wall_hours=380.0,
            mech_lever="headobsk", fire_schedule="always")
cfg = pilot if "$LIFE" == "pilot" else full
json.dump(cfg, open(sys.argv[1], "w"), indent=1)
print("cfg:", "$LIFE", cfg)
PYEOF
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists — a life is recorded; terminate + rm first"; exit 1; }
  local ACCT BUCKET AMI TAG S3P UD IID
  ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="e5-${LIFE}-$(date -u +%Y%m%dT%H%M%SZ)"
  S3P="s3://${BUCKET}/e5/${TAG}"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "ontogenic_learner/experiments/st7_life.py" \
    || { echo "FATAL: st7_life.py not committed"; exit 1; }
  write_cfg "$STATE_DIR/e5.json"
  AMI="$(aws_ ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query 'Parameters[0].Value' --output text)"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3P" > "$UD"
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3P/src.tar.gz" --only-show-errors
  aws_ s3 cp "$STATE_DIR/e5.json" "$S3P/e5.json" --only-show-errors
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --iam-instance-profile "Name=$ROLE" \
      --instance-initiated-shutdown-behavior stop \
      --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-e5-life},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_PREFIX=$S3P"
    echo "INSTANCE_TYPE=$INSTANCE_TYPE"; echo "LIFE=$LIFE"; } > "$STATE_FILE"
  echo "E5 $LIFE BORN: $IID -> $S3P"
}

render_user_data(){
  local s3p="$1"
  cat <<UDEOF
#!/bin/bash
exec > /var/log/e5-boot.log 2>&1
set -x
S3P="${s3p}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/e5-boot.log "\$S3P/boot.log" --only-show-errors || true; }
trap up EXIT
export DEBIAN_FRONTEND=noninteractive
for i in 1 2 3; do apt-get update -y && apt-get install -y python3-venv python3-pip && break; sleep 15; done
python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
for i in 1 2 3; do /opt/venv/bin/pip install -q awscli numpy && break; sleep 10; done
for i in 1 2 3; do /opt/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu && break; sleep 15; done
for i in 1 2 3; do /opt/venv/bin/pip install -q minatar minigrid gymnasium tokenizers && break; sleep 10; done
ln -sf /opt/venv/bin/aws /usr/local/bin/aws
up
mkdir -p /opt/tiki && cd /opt/tiki
aws s3 cp "\$S3P/src.tar.gz" . && tar xzf src.tar.gz
aws s3 cp "\$S3P/e5.json" /opt/tiki/e5.json
up
cat > /etc/systemd/system/e5life.service <<'UNIT'
[Unit]
Description=E5 second life (st7_life)
After=network-online.target
[Service]
ExecStart=/opt/venv/bin/python -u /opt/tiki/ontogenic_learner/experiments/st7_life.py --workdir /opt/tiki/life --config /opt/tiki/e5.json --resume
WorkingDirectory=/opt/tiki
Environment=PYTHONPATH=/opt/tiki
Environment=OMP_NUM_THREADS=4
Environment=MKL_NUM_THREADS=4
Restart=always
RestartSec=10
MemoryMax=10G
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now e5life
cat > /usr/local/bin/e5-watchdog <<'WD'
#!/bin/bash
exec 9>/var/lock/e5-wd; flock -n 9 || exit 0
ST=/opt/tiki/life/status.json
now=\$(date +%s)
if [ -f "\$ST" ]; then
  ts=\$(python3 -c "import json;print(int(json.load(open('\$ST'))['ts']))" 2>/dev/null || echo 0)
  age=\$((now - ts))
  [ \$age -gt 1800 ] && { echo "\$(date) reboot (age \$age)" >> /var/log/e5-wd.log; /sbin/reboot; }
  [ \$age -gt 600 ] && { echo "\$(date) restart (age \$age)" >> /var/log/e5-wd.log; systemctl restart e5life; }
fi
WD
chmod +x /usr/local/bin/e5-watchdog
echo "*/5 * * * * root /usr/local/bin/e5-watchdog" > /etc/cron.d/e5wd
cat > /usr/local/bin/e5-sync <<SY
#!/bin/bash
exec 9>/var/lock/e5-sync; flock -n 9 || exit 0
cd /opt/tiki/life 2>/dev/null || exit 0
aws s3 cp status.json "\$S3P/status.json" --only-show-errors 2>/dev/null
aws s3 cp tele.jsonl "\$S3P/tele.jsonl" --only-show-errors 2>/dev/null
aws s3 cp probes.jsonl "\$S3P/probes.jsonl" --only-show-errors 2>/dev/null
aws s3 sync ledger "\$S3P/ledger" --only-show-errors 2>/dev/null
L=\$(cat ckpt/LATEST 2>/dev/null); [ -n "\$L" ] && aws s3 cp "ckpt/\$L" "\$S3P/latest.pt" --only-show-errors 2>/dev/null
aws s3 cp /var/log/e5-wd.log "\$S3P/wd.log" --only-show-errors 2>/dev/null
SY
chmod +x /usr/local/bin/e5-sync
echo "*/5 * * * * root S3P=\$S3P /usr/local/bin/e5-sync" > /etc/cron.d/e5sync
echo "BOOTSTRAP DONE \$(date -u)"
UDEOF
}

cmd_status(){
  [ -f "$STATE_FILE" ] || { echo "no state"; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,LaunchTime]' --output text || true
  aws_ s3 cp "$S3_PREFIX/status.json" - 2>/dev/null || aws_ s3 cp "$S3_PREFIX/boot.log" - 2>/dev/null | tail -8 || true
}

cmd_terminate(){
  [ -f "$STATE_FILE" ] || { echo "no state"; exit 0; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  read -r -p "terminate $INSTANCE_ID? [y/N] " a
  [ "$a" = "y" ] && aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
}

case "${1:-}" in
  launch) cmd_launch;;
  status) cmd_status;;
  terminate) cmd_terminate;;
  *) echo "usage: $0 {launch|status|terminate} [LIFE=pilot|full]";;
esac
