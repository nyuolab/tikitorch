#!/bin/bash
# run_s2_shard_f2.sh — run the SCALE-2 (30M, D=512) DDR-streaming PC TRANSFORMER on f2.48xlarge slots: load the
# forge_pc_xformer_s2_block AGFI on every slot, build + run forge_pc_xformer_s2_shard.cpp. Two modes:
#   (a) KICKMEAS=<n>  — single-chip roofline: NSLOTS=1, one OUTPUT-role chip, n FWD+BWD steps, per-phase kick ms.
#   (b) 8-chip 30M training run — NSLOTS=8, one block per chip, neighbour wide-state + wide-error cross, bpb on OUT.
# Thin clone of run_mp_xformer_shard_f2.sh; distinct STATE_DIR + s2shard- S3 tag; byte-level enwik8 bins.
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_s2_shard_f2.sh launch     # then status/fetch/terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.48xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; STEPS="${STEPS:-3000}"; NSLOTS="${NSLOTS:-8}"; MAX_MINUTES="${MAX_MINUTES:-180}"; ROLE="${ROLE:-tiki-sweep-runner}"
XF_LR="${XF_LR:-3e-3}"; XF_WARMUP="${XF_WARMUP:-200}"; SEED="${SEED:-7}"; SAMPLE="${SAMPLE:-400}"; ACTSTATS="${ACTSTATS:-0}"; FIXEDOFF="${FIXEDOFF:--1}"; ACCUM="${ACCUM:-8}"; INIT_S3="${INIT_S3:-}"; LR_DECAY="${LR_DECAY:-1}"; LR_MINFRAC="${LR_MINFRAC:-0.1}"; KICKMEAS="${KICKMEAS:-0}"; PIPE="${PIPE:-0}"
# byte-level enwik8 token stream (uint16 ids 0-255). Provide ENWIK_S3 (an s3://.../train.u16) OR a BINS_S3 dir that
# contains train.u16. ENWIK_S3 wins if set. The parent already staged /tmp/enwik8_20m.u16 -> put it at ENWIK_S3.
ENWIK_S3="${ENWIK_S3:-}"; BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/pc_enwik8_bins}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; STATE_DIR="${STATE_DIR:-$PROJECT_DIR/.tiki_s2_shard_state}"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"; aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="s2shard-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  cat > "$STATE_DIR/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-s2shard.log 2>&1
set -x; export HOME=/root
S3_SRC="${S3_SRC}"; S3_RESULTS="${S3_RESULTS}"; AGFI="${AGFI}"; STEPS="${STEPS}"; NSLOTS="${NSLOTS}"; BINS_S3="${BINS_S3}"; ENWIK_S3="${ENWIK_S3}"
XF_LR="${XF_LR}"; XF_WARMUP="${XF_WARMUP}"; SEED="${SEED}"; SAMPLE="${SAMPLE}"; ACTSTATS="${ACTSTATS}"; FIXEDOFF="${FIXEDOFF}"; ACCUM="${ACCUM}"; INIT_S3="${INIT_S3}"; LR_DECAY="${LR_DECAY}"; LR_MINFRAC="${LR_MINFRAC}"; KICKMEAS="${KICKMEAS}"; PIPE="${PIPE}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-s2shard.log "\$S3_RESULTS/s2shard.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 30; up; done ) &
echo "BOOT \$(date -u)"; hostname; free -g
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
SLOTS=""; for s in \$(seq 0 \$((NSLOTS-1))); do SLOTS="\$SLOTS \$s"; sudo \$FC -S \$s || true; sudo \$FL -S \$s -I "\$AGFI"; done
sleep 4; for s in \$SLOTS; do sudo \$FD -S \$s 2>&1 | tee -a /tmp/img.txt; done; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
# byte-level enwik8 bins -> /tmp/pc_enwik8/train.u16 (ids = byte values 0-255; no itos needed)
mkdir -p /tmp/pc_enwik8
if [ -n "\$ENWIK_S3" ]; then aws s3 cp "\$ENWIK_S3" /tmp/pc_enwik8/train.u16 --only-show-errors;
else aws s3 cp "\$BINS_S3/train.u16" /tmp/pc_enwik8/train.u16 --only-show-errors; fi
ls -la /tmp/pc_enwik8; up
if [ -n "\$INIT_S3" ]; then mkdir -p /tmp/init; for l in 0 1 2 3 4 5 6 7; do aws s3 cp "\$INIT_S3/master\$l.f32" /tmp/init/master\$l.f32 --only-show-errors 2>/dev/null || true; done; export INIT_DIR=/tmp/init; ls -la /tmp/init; fi
g++ -O2 -std=c++14 -I"\$SDK/include" -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    /opt/tiki/tikitorch/fpga/silicon/forge_pc_xformer_s2_shard.cpp -lfpga_mgmt -o /tmp/xfs2shard 2>&1 | tee /tmp/build.txt; up
export XF_LR XF_WARMUP SEED SAMPLE ACTSTATS FIXEDOFF ACCUM LR_DECAY LR_MINFRAC KICKMEAS PIPE DUMP_DIR=/tmp   # INIT_DIR exported above when INIT_S3 set
( sudo -E /tmp/xfs2shard "\$STEPS" /tmp/pc_enwik8 \$SLOTS 2>&1 | tee /tmp/train.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/train.txt ) || true
for fdump in img build train; do aws s3 cp /tmp/\$fdump.txt "\$S3_RESULTS/\$fdump.txt" --only-show-errors || true; done
for w in /tmp/xfs2_master*.f32; do [ -f "\$w" ] && aws s3 cp "\$w" "\$S3_RESULTS/\$(basename \$w)" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; sync; shutdown -h now; exit 0
UDEOF
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$STATE_DIR/user-data.sh"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-s2-xformer-shard},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE_DIR/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $IID -> $S3_RESULTS"
}
cmd_status(){ source "$STATE_FILE"; echo -n "instance: "; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; echo "--- tail train.txt ---"; aws_ s3 cp "$S3_RESULTS/train.txt" - 2>/dev/null | tail -25 || aws_ s3 cp "$S3_RESULTS/s2shard.log" - 2>/dev/null | tail -25 || echo "(nothing yet)"; }
cmd_fetch(){ source "$STATE_FILE"; D="$PROJECT_DIR/results/predictive_coding/multi_fpga/s2_xformer_run"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; echo "=== train tail ==="; tail -40 "$D/train.txt" 2>/dev/null; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
