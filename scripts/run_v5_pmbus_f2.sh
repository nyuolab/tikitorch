#!/bin/bash
# run_v5_pmbus_f2.sh — V5 SILICON ENERGY session, v2 (NORTHSTAR_VALIDATION.md
# gate V5; runbook + scoping amendments in ontogenic_learner/V5_PMBUS_PROTOCOL.md).
#
# TWO-IMAGE design on the S1 kernel family (pre-launch review killed the
# morph-LM single-image plan: its fused AGFI synthesizes ONLY mode 3, so a
# forward-only serve arm cannot exist on it):
#   SERVE image  agfi-082aa26aac7495798  (S1 kv2 FORWARD-ONLY build, D=128/L4,
#                bpc 2.4230 measured bit-exact) — arm S = sustained token
#                GENERATION (sequential mode-0 kicks = literal serving).
#   TRAIN image  agfi-0179e963c146d39f7  (S1 trainer v17: fwd+bwd on fabric,
#                FREEZE_CODES) — arm SL = looped mode-3 fused train kicks.
#   Per-image IDLE floors; SL-gated is DERIVED per the runbook's own identity
#   (mJ/tok_gated ~= S + fire_rate x (SL - S), fire_rate = V4's 1.65%).
# Both arms are driven by the PROVEN host binaries (forge_s1_infer_f2.c,
# forge_train_step_s1_f2.c) with their proven build lines — no new silicon
# driver code. 1 Hz power sampling via fpga-describe-local-image -S 0 -M with
# the proven field grep + an on-box empty-sampler gate.
# Own f2.6xlarge (isolation rule).
#   export AWS_PROFILE=...; bash scripts/run_v5_pmbus_f2.sh launch
#   bash scripts/run_v5_pmbus_f2.sh status|fetch|terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI_SERVE="${AGFI_SERVE:-agfi-082aa26aac7495798}"
AGFI_TRAIN="${AGFI_TRAIN:-agfi-0179e963c146d39f7}"
S3_DEPLOY="${S3_DEPLOY:-s3://tiki-sweep-739966866637/s1kv2-deploy}"
MINUTES="${MINUTES:-10}"; REPS="${REPS:-3}"; MAX_MINUTES="${MAX_MINUTES:-240}"
ROLE="${ROLE:-tiki-sweep-runner}"
KDEFS="-DS1_KV2=1 -DS1_D=128 -DS1_LOG2_D=7 -DS1_H=4 -DS1_DH=32 -DS1_FFN=512 -DS1_N_LAYERS=4 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_v5pmbus_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS IID A
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  for A in "$AGFI_SERVE" "$AGFI_TRAIN"; do
    aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$A" \
      --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available \
      || { echo "FATAL: $A not available"; exit 1; }
  done
  aws_ s3 ls "$S3_DEPLOY/s1_kv2_deploy.npy" >/dev/null || { echo "FATAL: deploy bundle missing at $S3_DEPLOY"; exit 1; }
  TAG="v5pmbus-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  for f in tikitorch/fpga/silicon/forge_s1_infer_f2.c tikitorch/fpga/silicon/forge_train_step_s1_f2.c; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not in committed tree"; exit 1; }
  done
  cat > "$STATE_DIR/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-v5 deadline" || true
exec > /var/log/tiki-v5.log 2>&1
set -x; export HOME=/root
S3_SRC="${S3_SRC}"; S3_RESULTS="${S3_RESULTS}"; S3_DEPLOY="${S3_DEPLOY}"
AGFI_SERVE="${AGFI_SERVE}"; AGFI_TRAIN="${AGFI_TRAIN}"; MINUTES="${MINUTES}"; REPS="${REPS}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-v5.log "\$S3_RESULTS/v5.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 30; up; done ) &
echo "BOOT \$(date -u)"; hostname
die(){ echo "V5-FAILED: \$1" | aws s3 cp - "\$S3_RESULTS/FAILED" --only-show-errors; up; shutdown -h now; exit 1; }
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done; tar xzf s.tgz || die "src"
aws s3 cp "\$S3_DEPLOY/s1_kv2_deploy.npy" /tmp/buf.npy || die "deploy-npy"
aws s3 cp "\$S3_DEPLOY/s1_vocab.txt" /tmp/vocab.txt || die "deploy-vocab"
# ---- build BOTH proven hosts with their proven lines ----
gcc -O2 -std=gnu11 -DHAVE_FPGA_CLKGEN=0 ${KDEFS} -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/silicon/forge_s1_infer_f2.c -lfpga_mgmt -lm -o /tmp/s1_infer \
    2>&1 | tee /tmp/build_infer.txt
[ -x /tmp/s1_infer ] || die "infer-build"
gcc -O2 -std=gnu11 -DHAVE_FPGA_CLKGEN=0 -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/silicon/forge_train_step_s1_f2.c -lfpga_mgmt -lm -o /tmp/s1_f2 \
    2>&1 | tee /tmp/build_train.txt
[ -x /tmp/s1_f2 ] || die "train-build"
up
sample(){ # sample <outfile>: 1 Hz until killed (proven field grep)
  while true; do
    echo "TS=\$(date +%s)"
    \$FD -S 0 -M 2>&1 | grep -iE "Vccint|Last measured|Average|Max measured|power"
    sleep 1
  done >> "\$1" 2>&1
}
run_timed(){ # run_timed <name> <minutes> <cmd...>: loop cmd until deadline, sampling power
  local NAME=\$1 MIN=\$2; shift 2
  local PWR=/tmp/power_\${NAME}.log LOG=/tmp/arm_\${NAME}.log
  echo "ARM=\$NAME T_START=\$(date +%s)" > "\$LOG"
  sample "\$PWR" & local SPID=\$!
  local DEADLINE=\$(( \$(date +%s) + MIN*60 ))
  local ITERS=0
  while [ "\$(date +%s)" -lt "\$DEADLINE" ]; do
    if [ "\$#" -eq 0 ]; then sleep 5; else "\$@" >> "\$LOG" 2>&1 || echo "ITER-FAILED" >> "\$LOG"; fi
    ITERS=\$((ITERS+1))
  done
  echo "ARM=\$NAME T_END=\$(date +%s) ITERS=\$ITERS" >> "\$LOG"
  kill \$SPID 2>/dev/null || true
  grep -c "Last measured" "\$PWR" | grep -qv "^0\$" || die "power-sampler-empty-\$NAME"
  aws s3 cp "\$PWR" "\$S3_RESULTS/\$(basename \$PWR)" --only-show-errors || true
  aws s3 cp "\$LOG" "\$S3_RESULTS/\$(basename \$LOG)" --only-show-errors || true
  up
}
# ================= PHASE A: SERVE image =================
sudo \$FC -S 0 || true
sudo \$FL -S 0 -I "\$AGFI_SERVE" || die "load-serve"
sleep 4; sudo \$FD -S 0 | tee /tmp/img_serve.txt
# KAT-lite: a short generation must exit 0 and emit tokens
( sudo /tmp/s1_infer 0 /tmp/buf.npy 200 - 0 128 1234 /tmp/vocab.txt 2>&1 | tee /tmp/kat_serve.txt )
grep -qiE "tok/s|generated|bpc" /tmp/kat_serve.txt || die "serve-kat"
aws s3 cp /tmp/kat_serve.txt "\$S3_RESULTS/kat_serve.txt" --only-show-errors; up
for R in \$(seq 1 \$REPS); do
  run_timed idleS_r\$R 2
  run_timed serve_r\$R "\$MINUTES" sudo /tmp/s1_infer 0 /tmp/buf.npy 5000 - 0 128 1234 /tmp/vocab.txt
done
# ================= PHASE B: TRAIN image =================
sudo \$FC -S 0 || true
sudo \$FL -S 0 -I "\$AGFI_TRAIN" || die "load-train"
sleep 4; sudo \$FD -S 0 | tee /tmp/img_train.txt
( sudo /tmp/s1_f2 0 32 freeze 4242 2>&1 | tee /tmp/kat_train.txt )
grep -qiE "step|tok|converge|acc" /tmp/kat_train.txt || die "train-kat"
aws s3 cp /tmp/kat_train.txt "\$S3_RESULTS/kat_train.txt" --only-show-errors; up
for R in \$(seq 1 \$REPS); do
  run_timed idleT_r\$R 2
  run_timed sl_r\$R "\$MINUTES" sudo /tmp/s1_f2 0 512 freeze 4242
done
for f in img_serve img_train build_infer build_train; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; sync; shutdown -h now; exit 0
UDEOF
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$STATE_DIR/user-data.sh"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-v5-pmbus},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE_DIR/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $IID -> $S3_RESULTS"
}
cmd_status(){ source "$STATE_FILE"; echo -n "instance: "; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; D="$PROJECT_DIR/ontogenic_learner/results/v5_pmbus"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
