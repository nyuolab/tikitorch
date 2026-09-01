#!/bin/bash
# run_morph_lm_a2_f2.sh — A2 ISO-MODEL serve-vs-update power run on a REAL F2.
#
# Loads the SINGLE deployed morph-LM AFI (cl_forge_morph_lm_v8_aligned,
# agfi-0433370d93cb58a78; DSP=0; fused mode 3) ONCE and runs ALL arms against it
# — NO serve/train AFI swap (that iso-model property is the whole point of A2).
# On this AFI only fused mode 3 is synthesized, so:
#     serve  = fused mode-3 kick with a HUGE SignSGD threshold  -> weights frozen
#     update = fused mode-3 kick with the real threshold        -> weights learn
# Serve and update are the IDENTICAL fabric computation; the A2 claim is that the
# optimizer step is below the mgmt-tool (Vccint core-rail) instrument floor.
#
# Pattern forked from run_muzero_afi_f2_power.sh (S3 + IAM profile + self-
# terminate, no SSH); the on-box user-data is forked from
# .tiki_v5pmbus_state/user-data.sh (its field-proven 1 Hz Vccint sample()/
# run_timed() are reused VERBATIM). Measures the Vccint CORE rail via
# fpga-describe-local-image -M — NOT true full-card PMBus (disclosed in the docs).
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_morph_lm_a2_f2.sh launch
#   bash scripts/run_morph_lm_a2_f2.sh status
#   bash scripts/run_morph_lm_a2_f2.sh fetch
#   bash scripts/run_morph_lm_a2_f2.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"             # 1 FPGA; smallest F2
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (fpga-* tools + SDK)
AGFI="${AGFI:-agfi-0433370d93cb58a78}"                    # cl_forge_morph_lm_v8_aligned (fused mode 3)
REPS="${REPS:-3}"
IDLE_MIN="${IDLE_MIN:-2}"                                 # idle arm minutes
SERVE_MIN="${SERVE_MIN:-10}"                              # serve arm minutes
UPDATE_MIN="${UPDATE_MIN:-10}"                            # update arm minutes
GATED_MIN="${GATED_MIN:-10}"                              # gated arm minutes
# Backstop deadline: REPS*(idle+serve+update+gated) + generous build/load slack.
MAX_MINUTES="${MAX_MINUTES:-$(( REPS*(IDLE_MIN+SERVE_MIN+UPDATE_MIN+GATED_MIN) + 45 ))}"
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_a2power_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available \
    || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="a2power-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"
  S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  # ---- package the source the on-box build needs (bridge + kernels + silicon drivers) ----
  echo "== packaging src.tar.gz =="
  tar -C "$PROJECT_DIR" -czf "$STATE_DIR/src.tar.gz" \
    tikitorch/fpga/bridge.cpp \
    tikitorch/fpga/silicon/morph_lm_a2_power.py \
    tikitorch/fpga/silicon/morph_lm_silicon.py \
    tikitorch/fpga/silicon/gen_buffer_init.py \
    tikitorch/fpga/silicon/cstdint_shim \
    tikitorch/hardware/hls/forge_kernels
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> UD:"; cat "$UD"; return 0; fi
  echo "== launch $INSTANCE_TYPE ($AMI; load $AGFI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-morph-a2power},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then echo "retry $tries ($(tr -d '\n' <"$STATE_DIR/launch.err" | tail -c 120))"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_SRC=$S3_SRC"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-a2power deadline" || true
exec > /var/log/tiki-a2power.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"
REPS="${REPS}"; IDLE_MIN="${IDLE_MIN}"; SERVE_MIN="${SERVE_MIN}"; UPDATE_MIN="${UPDATE_MIN}"; GATED_MIN="${GATED_MIN}"
export A2_STEPS_PER_KICK="${A2_STEPS_PER_KICK:-5}" A2_GATE_EVERY="${A2_GATE_EVERY:-20}" A2_SGD_THR="${A2_SGD_THR:-2048}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-a2power.log "\$S3_RESULTS/a2power.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 30; up; done ) &
echo "BOOT \$(date -u)"; hostname
die(){ echo "A2-FAILED: \$1" | aws s3 cp - "\$S3_RESULTS/FAILED" --only-show-errors; up; shutdown -h now; exit 1; }
# ---- SDK runtime (fpga-* tools) ----
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
echo "TOOLS: \$FD / \$FL / \$FC"; up
# ---- fetch + unpack src ----
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done; tar xzf s.tgz || die "src"
# ---- build the PCI host lib EXACTLY as morph_lm_silicon.py documents ----
g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_PCI \
    -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/bridge.cpp \
    -lfpga_mgmt -o /tmp/_libtikifpga_pci.so 2>&1 | tee /tmp/build_lib.txt
[ -f /tmp/_libtikifpga_pci.so ] || die "lib-build"
export TIKITORCH_FPGA_LIB=/tmp/_libtikifpga_pci.so
# ---- shared ground-truth init buffer (SAME bytes the CSIM golden consumes) ----
python3 /opt/tiki/tikitorch/fpga/silicon/gen_buffer_init.py /tmp/buffer_init.bin || die "gen-init"
A2=/opt/tiki/tikitorch/fpga/silicon/morph_lm_a2_power.py
# ---- load the SINGLE AFI ONCE (no swap for the whole run) ----
sudo \$FC -S 0 || true
sudo \$FL -S 0 -I "\$AGFI" -F || die "load-afi"
sleep 5; sudo \$FD -S 0 | tee /tmp/img_a2.txt; up
# =================== KAT GATE (must pass BEFORE any timed arm) ===================
# A short fused kick must reproduce the golden wcrc/preds16, else the host is
# mis-built and every power number would be junk -> die loudly.
( sudo -E python3 "\$A2" kat /tmp/buffer_init.bin 2>&1 | tee /tmp/kat.txt )
aws s3 cp /tmp/kat.txt "\$S3_RESULTS/kat.txt" --only-show-errors || true
grep -q "KAT_PASS" /tmp/kat.txt || die "kat-mismatch (host mis-built / AFI wrong)"
up
# ---- proven 1 Hz Vccint sampler (VERBATIM from .tiki_v5pmbus_state/user-data.sh) ----
sample(){ # sample <outfile>: 1 Hz until killed
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
# ============================== TIMED ARMS (REPS x) ==============================
# The python arm runs kicks for (arm_minutes*60 + 20 s), so it OUTLIVES run_timed's
# deadline -> run_timed's while-loop invokes it exactly ONCE per arm (a second call
# never starts because the deadline has already passed on return). The sampler runs
# for the whole invocation and is joined by unix timestamp in a2_reduce.py.
for R in \$(seq 1 \$REPS); do
  run_timed idle_r\$R   "\$IDLE_MIN"   sudo -E python3 "\$A2" idle   /tmp/buffer_init.bin \$(( IDLE_MIN*60   + 20 ))
  run_timed serve_r\$R  "\$SERVE_MIN"  sudo -E python3 "\$A2" serve  /tmp/buffer_init.bin \$(( SERVE_MIN*60  + 20 ))
  run_timed update_r\$R "\$UPDATE_MIN" sudo -E python3 "\$A2" update /tmp/buffer_init.bin \$(( UPDATE_MIN*60 + 20 ))
  run_timed gated_r\$R  "\$GATED_MIN"  sudo -E python3 "\$A2" gated  /tmp/buffer_init.bin \$(( GATED_MIN*60  + 20 ))
done
for f in img_a2 build_lib kat; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; sync; shutdown -h now; exit 0
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- a2power.log tail --"; aws_ s3 cp "$S3_RESULTS/a2power.log" - 2>/dev/null | tail -30; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/a2power"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -1 "$DEST"
  echo "-- reduce --"; python3 "$PROJECT_DIR/scripts/a2_reduce.py" "$DEST" || true; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
