#!/bin/bash
# run_e6_ec2.sh — E6 fleet launcher (forked from run_e5_ec2.sh). One instance
# per (ARM, LIFE) running SEEDS as templated systemd units e6life@N with
# per-seed workdirs/configs/S3 subprefixes. Same persistent-service posture:
# NO self-terminate, stop-on-shutdown, Restart=always --resume, cron watchdog
# (per-life restart; reboot only if ALL lives stale), cron per-seed S3 sync
# incl. ckpt_era/ (all-era retention pipe).
#   ARM=base LIFE=screen SEEDS="0 1 2" bash scripts/run_e6_ec2.sh launch
#   ARM=base LIFE=full   SEEDS="1 2"   bash scripts/run_e6_ec2.sh launch  # Tier-1
#   ARM=... bash scripts/run_e6_ec2.sh status|terminate
# Arms: base anchor idbd-adam idbd-dyadic idbd-sgd ent-floor ent-pull
#       head-renorm lrfixed
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.2xlarge}"
EBS_SIZE="${EBS_SIZE:-40}"
ROLE="${ROLE:-tiki-sweep-runner}"
ARM="${ARM:-base}"
LIFE="${LIFE:-screen}"
SEEDS="${SEEDS:-0 1 2}"
OMP="${OMP:-2}"
BRANCH_FROM_S3="${BRANCH_FROM_S3:-}"
BRANCH_SCHEDULE="${BRANCH_SCHEDULE:-novel}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_e6_state"; mkdir -p "$STATE_DIR"
BATCH="${BATCH:-}"
STATE_FILE="$STATE_DIR/${ARM}-${LIFE}${BATCH:+-$BATCH}.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }
acct(){ aws_ sts get-caller-identity --query Account --output text; }

arm_overlay(){
  case "$ARM" in
    base)        echo '{}' ;;
    anchor)      echo '{"anchor":"adam","anchor_lam":0.01}' ;;
    idbd-adam)   echo '{"optimizer":"idbd-adam","lr_decay":"none"}' ;;
    idbd-dyadic) echo '{"optimizer":"idbd-dyadic","lr_decay":"none"}' ;;
    idbd-sgd)    echo '{"optimizer":"idbd-sgd","lr_decay":"none"}' ;;
    ent-floor)   echo '{"act_eps":0.05}' ;;
    ent-pull)    echo '{"ent_k":-3}' ;;
    ent-pull-weak) echo '{"ent_k":-4}' ;;
    ent-pull-full) echo '{"ent_k":-3}' ;;
    head-renorm) echo '{"head_renorm_M":4.0}' ;;
    lrfixed)     echo '{"lr_decay":"fixed","lr_fixed_div":8}' ;;
    shadow)      echo '{"freeze_first_steps":50000}' ;;
    nb-base)     echo '{}' ;;
    nb-pull)     echo '{"ent_k":-4}' ;;
    nb-lo)       echo '{"lr":0.000125}' ;;
    nb-lopull)   echo '{"lr":0.000125,"ent_k":-4}' ;;
    *) echo "FATAL: unknown ARM=$ARM" >&2; exit 1 ;;
  esac
}

write_cfgs(){
  local outdir="$1" overlay; overlay="$(arm_overlay)"
  python3 - "$outdir" "$overlay" <<PYEOF
import json, sys
outdir, overlay = sys.argv[1], json.loads(sys.argv[2])
screen = dict(era_steps=250_000, biome_steps=62_500, cycles=10,
              probe_every_hours=3.0, ckpt_every_steps=25_000,
              ckpt_every_secs=900, budget_wall_hours=260.0,
              mech_lever="headobsk", fire_schedule="always",
              babyai_enabled=True, babyai_from_cycle=6,
              retain_era_ckpts=True, tele_margin=True)
full = dict(era_steps=500_000, biome_steps=125_000, cycles=10,
            probe_every_hours=6.0, ckpt_every_steps=50_000,
            ckpt_every_secs=900, budget_wall_hours=380.0,
            mech_lever="headobsk", fire_schedule="always",
            babyai_enabled=True, babyai_from_cycle=6,
            retain_era_ckpts=True, tele_margin=True)
micro = dict(era_steps=100_000, biome_steps=25_000, cycles=1,
             env_order=["babyai"], probe_every_hours=0.0,
             ckpt_every_steps=50_000, ckpt_every_secs=3600,
             budget_wall_hours=24.0, mech_lever="headobsk",
             fire_schedule="always", retain_era_ckpts=True,
             tele_margin=True)
base = micro if "$LIFE" == "micro" else screen if "$LIFE" == "screen" else full
for s in "$SEEDS".split():
    cfg = dict(base); cfg.update(overlay)
    cfg.update(seed=int(s), world_seed=int(s))
    p = f"{outdir}/e6-s{s}.json"
    json.dump(cfg, open(p, "w"), indent=1)
    print("cfg:", "$ARM", "$LIFE", f"s{s}", {k: cfg[k] for k in
          sorted(set(cfg) - set(base) | {"seed"})})
PYEOF
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists for ${ARM}-${LIFE}; terminate + rm first"; exit 1; }
  local ACCT BUCKET AMI TAG S3P UD IID
  ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="e6-${ARM}-${LIFE}${BATCH:+-$BATCH}-$(date -u +%Y%m%dT%H%M%SZ)"
  S3P="s3://${BUCKET}/e6/${TAG}"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  for f in ontogenic_learner/experiments/st7_life.py \
           ontogenic_learner/experiments/st7_optim.py; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" \
      || { echo "FATAL: $f not committed"; exit 1; }
  done
  write_cfgs "$STATE_DIR"
  AMI="$(aws_ ssm get-parameters --names /aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id --query 'Parameters[0].Value' --output text)"
  UD="$STATE_DIR/user-data-${ARM}-${LIFE}.sh"; render_user_data "$S3P" > "$UD"
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3P/src.tar.gz" --only-show-errors
  for s in $SEEDS; do
    aws_ s3 cp "$STATE_DIR/e6-s$s.json" "$S3P/e6-s$s.json" --only-show-errors
  done
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --iam-instance-profile "Name=$ROLE" \
      --instance-initiated-shutdown-behavior stop \
      --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-e6-${ARM}-${LIFE}},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_PREFIX=$S3P"
    echo "INSTANCE_TYPE=$INSTANCE_TYPE"; echo "ARM=$ARM"; echo "LIFE=$LIFE"
    echo "SEEDS=\"$SEEDS\""; } > "$STATE_FILE"
  echo "E6 ${ARM}-${LIFE} BORN: $IID seeds=[$SEEDS] -> $S3P"
}

render_user_data(){
  local s3p="$1"
  cat <<UDEOF
#!/bin/bash
exec > /var/log/e6-boot.log 2>&1
set -x
S3P="${s3p}"
SEEDS="${SEEDS}"
BRANCH_FROM_S3="${BRANCH_FROM_S3}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/e6-boot.log "\$S3P/boot.log" --only-show-errors || true; }
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
for s in \$SEEDS; do aws s3 cp "\$S3P/e6-s\$s.json" /opt/tiki/e6-s\$s.json; done
[ -n "\$BRANCH_FROM_S3" ] && aws s3 cp "\$BRANCH_FROM_S3" /opt/tiki/branch_root.pt
up
cat > /etc/systemd/system/e6life@.service <<'UNIT'
[Unit]
Description=E6 life seed %i (st7_life)
After=network-online.target
[Service]
ExecStart=/opt/venv/bin/python -u /opt/tiki/ontogenic_learner/experiments/st7_life.py --workdir /opt/tiki/life-s%i --config /opt/tiki/e6-s%i.json --resume ${BRANCH_FROM_S3:+--branch-from /opt/tiki/branch_root.pt --branch-schedule ${BRANCH_SCHEDULE}}
WorkingDirectory=/opt/tiki
Environment=PYTHONPATH=/opt/tiki
Environment=OMP_NUM_THREADS=${OMP}
Environment=MKL_NUM_THREADS=${OMP}
Restart=always
RestartSec=10
MemoryMax=8G
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
for s in \$SEEDS; do systemctl enable --now e6life@\$s; done
cat > /usr/local/bin/e6-watchdog <<'WD'
#!/bin/bash
exec 9>/var/lock/e6-wd; flock -n 9 || exit 0
# boot grace: births + pip + first heartbeats take a while (review BLOCKER:
# the E5 [ -f ] guard was dropped -> fresh boots reboot-looped)
up=\$(cut -d. -f1 /proc/uptime); [ "\$up" -lt 2700 ] && exit 0
now=\$(date +%s)
all_stale=1
for d in /opt/tiki/life-s*; do
  [ -d "\$d" ] || continue
  s=\${d##*-s}
  ST="\$d/status.json"
  # missing status = pre-heartbeat life: not stale, no restart vote
  [ -f "\$ST" ] || { all_stale=0; continue; }
  ts=\$(python3 -c "import json;print(int(json.load(open('\$ST'))['ts']))" 2>/dev/null || stat -c %Y "\$ST" 2>/dev/null || echo 0)
  age=\$((now - ts))
  if [ \$age -gt 1800 ]; then
    echo "\$(date) restart s\$s (age \$age)" >> /var/log/e6-wd.log
    systemctl restart e6life@\$s
  fi
  [ \$age -le 5400 ] && all_stale=0
done
[ \$all_stale -eq 1 ] && { echo "\$(date) reboot (all stale)" >> /var/log/e6-wd.log; /sbin/reboot; }
exit 0
WD
chmod +x /usr/local/bin/e6-watchdog
echo "*/5 * * * * root /usr/local/bin/e6-watchdog" > /etc/cron.d/e6wd
cat > /usr/local/bin/e6-sync <<'SY'
#!/bin/bash
exec 9>/var/lock/e6-sync; flock -n 9 || exit 0
for d in /opt/tiki/life-s*; do
  [ -d "\$d" ] || continue
  s=\${d##*-s}
  cd "\$d" || continue
  aws s3 cp status.json "\$S3P/s\$s/status.json" --only-show-errors 2>/dev/null
  aws s3 cp tele.jsonl "\$S3P/s\$s/tele.jsonl" --only-show-errors 2>/dev/null
  aws s3 cp probes.jsonl "\$S3P/s\$s/probes.jsonl" --only-show-errors 2>/dev/null
  aws s3 sync ledger "\$S3P/s\$s/ledger" --only-show-errors 2>/dev/null
  [ -d ckpt_era ] && aws s3 sync ckpt_era "\$S3P/s\$s/ckpt_era" --only-show-errors 2>/dev/null
  [ -f probe_windows.pt ] && aws s3 cp probe_windows.pt "\$S3P/s\$s/probe_windows.pt" --only-show-errors 2>/dev/null
  L=\$(cat ckpt/LATEST 2>/dev/null); [ -n "\$L" ] && aws s3 cp "ckpt/\$L" "\$S3P/s\$s/latest.pt" --only-show-errors 2>/dev/null
done
aws s3 cp /var/log/e6-wd.log "\$S3P/wd.log" --only-show-errors 2>/dev/null
SY
chmod +x /usr/local/bin/e6-sync
echo "*/5 * * * * root S3P=\$S3P /usr/local/bin/e6-sync" > /etc/cron.d/e6sync
echo "BOOTSTRAP DONE \$(date -u)"
UDEOF
}

cmd_status(){
  [ -f "$STATE_FILE" ] || { echo "no state for ${ARM}-${LIFE}"; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,LaunchTime]' --output text || true
  for s in $SEEDS; do
    echo "-- s$s --"
    aws_ s3 cp "$S3_PREFIX/s$s/status.json" - 2>/dev/null || true
    echo
  done
}

cmd_terminate(){
  [ -f "$STATE_FILE" ] || { echo "no state for ${ARM}-${LIFE}"; exit 0; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  read -r -p "terminate ${ARM}-${LIFE} ($INSTANCE_ID)? [y/N] " a
  [ "$a" = "y" ] && aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].CurrentState.Name' --output text
}

case "${1:-}" in
  launch) cmd_launch;;
  status) cmd_status;;
  terminate) cmd_terminate;;
  *) echo "usage: ARM=<arm> LIFE=screen|full SEEDS=\"0 1 2\" $0 {launch|status|terminate}";;
esac
