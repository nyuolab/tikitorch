#!/bin/bash
# run_pc_shard_train_f2.sh — train ONE net MODEL-PARALLEL across the 2 FPGAs of an f2.12xlarge: load the
# forge_pc_block AGFI on BOTH slots, build + run forge_pc_train_shard.c (slot0 = layer0/hidden, slot1 =
# layer1/output; only a neighbour activation+error cross; weights persist per chip; accuracy climbs ON FABRIC
# and a Gate-C audit proves no weight/global-gradient ever crosses). Isolated state/tag. Cloned from
# run_pc_train_f2.sh; distinct STATE_DIR + tag + f2.12xlarge + 2-slot load.
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_pc_shard_train_f2.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.12xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; EPOCHS="${EPOCHS:-5}"; NSLOTS="${NSLOTS:-2}"; MAX_MINUTES="${MAX_MINUTES:-50}"; ROLE="${ROLE:-tiki-sweep-runner}"; VERIFY="${VERIFY:-}"
BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/pc_mnist_bins}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pc_shard_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="pcshard-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pc-shard-train},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { sleep 15; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-pcshard.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; EPOCHS="${EPOCHS}"; NSLOTS="${NSLOTS}"; BINS_S3="${BINS_S3}"; VERIFY="${VERIFY}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-pcshard.log "\$S3_RESULTS/pcshard.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
SLOTS=""; for s in \$(seq 0 \$((NSLOTS-1))); do SLOTS="\$SLOTS \$s"; sudo \$FC -S \$s || true; sudo \$FL -S \$s -I "\$AGFI"; done
sleep 4; for s in \$SLOTS; do sudo \$FD -S \$s 2>&1 | tee -a /tmp/img.txt; done; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
mkdir -p /tmp/pc_mnist; for f in train_x.i8 train_y.u8 test_x.i8 test_y.u8; do aws s3 cp "\$BINS_S3/\$f" /tmp/pc_mnist/\$f --only-show-errors; done
gcc -O2 -std=gnu11 -I"\$SDK/include" /opt/tiki/tikitorch/fpga/silicon/forge_pc_train_shard.c -lfpga_mgmt -o /tmp/pcshard 2>&1 | tee /tmp/build.txt
up
if [ -n "\$VERIFY" ]; then
  gcc -O2 -std=gnu11 -I"\$SDK/include" /opt/tiki/tikitorch/fpga/silicon/forge_pc_block_verify.c -lfpga_mgmt -o /tmp/pcbverify 2>&1 | tee -a /tmp/build.txt
  ( sudo /tmp/pcbverify 0 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true
  for f in img build train; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
  echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
  up; sync; shutdown -h now; exit 0
fi
# Vccint (core rail) energy: read 'Last measured: N watts' under each slot's Vccint. helper:
vcc(){ sudo \$FD -S \$1 -M 2>&1 | awk 'BEGIN{IGNORECASE=1} /vccint/{f=1} f&&/last measured/{for(i=1;i<=NF;i++)if(\$i~/^[0-9]+\$/){print \$i;exit}}'; }
echo "== IDLE Vccint (per slot, pre-run) ==" | tee /tmp/power.txt
for s in \$SLOTS; do echo "idle slot=\$s W=\$(vcc \$s)" | tee -a /tmp/power.txt; done
# run the instrumented driver in the BACKGROUND; sample per-slot Vccint concurrently every ~2s
rm -f /tmp/scdone
( sudo /tmp/pcshard "\$EPOCHS" /tmp/pc_mnist \$SLOTS 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt; touch /tmp/scdone ) &
while [ ! -f /tmp/scdone ]; do TS=\$(date +%s.%N); for s in \$SLOTS; do echo "during t=\$TS slot=\$s W=\$(vcc \$s)"; done >> /tmp/power.txt; sleep 2; done
wait
# compute energy (J, J/step) from the sampled Vccint + the driver's ENERGY_WINDOW line
python3 - <<'PY' | tee /tmp/energy.txt
import re
slots={}; idle={}
for ln in open('/tmp/power.txt'):
    m=re.search(r'(idle|during).*slot=(\d+).*W=(\d+)', ln)
    if not m: continue
    ph,s,w=m.group(1),int(m.group(2)),int(m.group(3))
    if ph=='idle': idle[s]=w
    else: slots.setdefault(s,[]).append(w)
wall=steps=samples=None
for ln in open('/tmp/train.txt'):
    m=re.search(r'ENERGY_WINDOW wall_s=([\d.]+) steps=(\d+) samples=(\d+)', ln)
    if m: wall,steps,samples=float(m.group(1)),int(m.group(2)),int(m.group(3))
print("=== ENERGY (Vccint core rail, per slot) ===")
tot_avg=0.0; tot_idle=0.0
for s in sorted(slots):
    a=sum(slots[s])/len(slots[s]); i=idle.get(s,0); tot_avg+=a; tot_idle+=i
    print(f"slot {s}: Vccint avg-during {a:.2f} W (idle {i} W, dynamic {a-i:+.2f} W; n={len(slots[s])})")
if wall:
    E=tot_avg*wall; Edyn=(tot_avg-tot_idle)*wall
    print(f"board Vccint: during {tot_avg:.2f} W, idle {tot_idle:.2f} W  | wall {wall:.2f}s, {steps} steps")
    print(f"ENERGY: {E:.1f} J total ({E/steps*1e3:.2f} mJ/step, {E/samples*1e3:.3f} mJ/sample)")
    print(f"ENERGY(dynamic, during-idle): {Edyn:.1f} J ({Edyn/steps*1e3:.2f} mJ/step)")
    print("NOTE: Vccint is the core rail (not whole-board); a DSP_MULT=0 core's dynamic draw is often at/below the sensor's 1W resolution -> dynamic may read ~0 (honest: structural demo, not a power record).")
PY
for f in img build train power energy; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
for w in s1_perm.i32 shardW0.i32 shardW1.i32 s1_iW0.i32 s1_iW1.i32 s1_x.i32 s1_y.i32 s1_W0.i32 s1_W1.i32; do [ -f /tmp/\$w ] && aws s3 cp /tmp/\$w "\$S3_RESULTS/\$w" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/results/predictive_coding/shard_train"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
