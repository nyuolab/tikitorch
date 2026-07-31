#!/bin/bash
# run_morph_lm_d512_recsynth.sh — RE-CSYNTH the D=512 morph-LM kernel AFTER the
# #if ML_D>=512 timing pragmas (URAM BIND_STORAGE + 16-way ARRAY_PARTITION on
# mp_buf/wh_buf) to MEASURE the improved critical path against the REDUCED A3 clock.
#
# Forked VERBATIM from run_morph_lm_d512_csynth.sh (EC2 mechanics: z1d build box,
# source->S3, IAM profile, self-terminate, NO SSH). Two differences only:
#   (1) KTCL = synth_morph_lm_d512.tcl  — the D=512 driver that ALWAYS builds
#       -DML_D=512 and targets TT_CLK_NS (reduced clock), so we learn the clock
#       the pragmas actually close, not the original 4.0 ns target.
#   (2) TT_CLK_NS (default 6.0 ns = ~166 MHz; A3 accepts 6-8 ns) is exported into
#       the build so create_clock uses it.
# The committed baseline (pre-pragma, 4.0 ns target) is results/a2_power/
# d512_csynth.rpt (11.425 ns). This run's report lands beside it as
# results/a2_power/d512_recsynth.rpt so the improvement is a direct diff.
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_morph_lm_d512_recsynth.sh launch          # TT_CLK_NS=6.0 default
#   TT_CLK_NS=8.0 bash scripts/run_morph_lm_d512_recsynth.sh launch   # sweep the target
#   bash scripts/run_morph_lm_d512_recsynth.sh {status|fetch|terminate}
#   DRY_RUN=1 bash scripts/run_morph_lm_d512_recsynth.sh launch       # render user-data only
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"          # cheap build box (NOT an f2); Vitis HLS only
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                    # FPGA Developer AMI (ships Vitis HLS)
MAX_MINUTES="${MAX_MINUTES:-90}"
ROLE="${ROLE:-tiki-sweep-runner}"
ML_D="${ML_D:-512}"                                    # this driver is D=512
TT_CLK_NS="${TT_CLK_NS:-6.0}"                          # REDUCED target (ns): 6.0=~166MHz, 8.0=125MHz
KTOP="${KTOP:-forge_train_step_morph_lm}"
KTCL="${KTCL:-synth_morph_lm_d512.tcl}"                # the D=512 reduced-clock driver
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_morphlm_d${ML_D}_recsynth"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

render_user_data(){ local s3_src="$1" s3_results="$2"; cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki morph-lm d${ML_D} recsynth deadline" || true
exec > /var/log/tiki-morphlm-recsynth.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; ML_D="${ML_D}"; TT_CLK_NS="${TT_CLK_NS}"; KTOP="${KTOP}"; KTCL="${KTCL}"
export HOME=/root
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-morphlm-recsynth.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname; mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz; up
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
export ML_D="\$ML_D"; export TT_CLK_NS="\$TT_CLK_NS"; export CSYNTH_ONLY=1
echo "=== recsynth \$KTOP ML_D=\$ML_D TT_CLK_NS=\$TT_CLK_NS START \$(date -u) ==="
"\$VHLS" -f "\$KTCL" 2>&1 | tail -80
echo "=== recsynth \$KTOP done \$(date -u) ==="
RPT="\${KTOP}_proj/solution1/syn/report/\${KTOP}_csynth.rpt"
{
  echo "###### D=\$ML_D RECSYNTH (reduced target \$TT_CLK_NS ns) — post-URAM/partition pragmas ######"
  echo "--- Timing (Estimated vs Target) ---"; grep -iE "Target|Estimated|CP |clock" "\$RPT" | head -20
  echo "--- Resource (DSP MUST be 0; BRAM should DROP vs baseline; URAM should RISE) ---"
  grep -iE "DSP|BRAM|URAM|LUT|FF|Utilization|Total|Available|SLICE" "\$RPT" | head -60
  echo "--- Per-loop latency + II table ---"
  awk '/Loop Name|Latency|Initiation Interval|Trip Count|Pipelined/{p=1} p{print} /Utilization Estimates/{p=0}' "\$RPT" | head -140
  echo "--- DSP=0 assertion ---"
  DLINE="\$(grep -iE "^\|[[:space:]]*DSP" "\$RPT" | head -1)"; echo "DSP row: \$DLINE"
  echo "\$DLINE" | grep -qE "[1-9]" && echo "WARNING: DSP>0 — multiplier-free VIOLATED" || echo "OK: DSP appears 0"
} 2>&1 | tee /tmp/d\${ML_D}_recsynth_summary.txt
aws s3 cp "\$RPT"                          "\$S3_RESULTS/\${KTOP}_d\${ML_D}_recsynth.rpt" --only-show-errors || true
aws s3 cp /tmp/d\${ML_D}_recsynth_summary.txt "\$S3_RESULTS/\${KTOP}_d\${ML_D}_recsynth_summary.txt" --only-show-errors || true
aws s3 cp vitis_hls.log                   "\$S3_RESULTS/\${KTOP}_d\${ML_D}_recsynth_vitis_hls.log" --only-show-errors || true
up; echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
shutdown -h now
UDEOF
}

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID TREE
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="morphlm-d${ML_D}-recsynth-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  # Capture the WORKING TREE (the D=512 pragmas + new tcl are uncommitted; house rules: no commit).
  TREE="$(git -C "$PROJECT_DIR" stash create 2>/dev/null || true)"; [ -z "$TREE" ] && TREE="HEAD"
  echo "archiving tree: $TREE"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" "$TREE"
  for f in "tikitorch/hardware/hls/forge_kernels/${KTCL}" \
           "tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp" \
           "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp"; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not in shipped tree"; exit 1; }; done
  local TMPD; TMPD="$(mktemp -d)"; tar xzf "$STATE_DIR/src.tar.gz" -C "$TMPD" \
    "tikitorch/hardware/hls/forge_kernels/${KTCL}" \
    "tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp" \
    "tikitorch/hardware/hls/forge_kernels/forge_buffer_morph_lm.hpp"
  grep -qE '#if[[:space:]]+ML_D[[:space:]]*>=[[:space:]]*512' "$TMPD/tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp" \
    || { echo "FATAL: kernel has no '#if ML_D >= 512' timing block — recsynth would measure the un-pragma'd kernel"; rm -rf "$TMPD"; exit 1; }
  grep -q 'BIND_STORAGE   variable=mp_buf type=RAM_T2P impl=URAM' "$TMPD/tikitorch/hardware/hls/forge_kernels/${KTOP}.cpp" \
    || { echo "FATAL: mp_buf URAM bind pragma missing"; rm -rf "$TMPD"; exit 1; }
  rm -rf "$TMPD"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN (ML_D=$ML_D TT_CLK_NS=$TT_CLK_NS); UD:"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-morphlm-d${ML_D}-recsynth},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then
      echo "retry $tries ($(tr -d '\n' <"$STATE_DIR/launch.err" | tail -c 120))"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; echo "ML_D=$ML_D"; echo "TT_CLK_NS=$TT_CLK_NS"; echo "KTOP=$KTOP"; } > "$STATE_FILE"
  echo "LAUNCHED morph-lm D=$ML_D recsynth (target $TT_CLK_NS ns) -> $INSTANCE_ID -> $S3_RESULTS"; echo "next: $0 status"
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "(running)"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -20; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local D="$PROJECT_DIR/results/a2_power"; mkdir -p "$D"
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_recsynth_summary.txt" "$D/d${ML_D}_recsynth_summary.txt" --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_recsynth.rpt"         "$D/d${ML_D}_recsynth.rpt"         --only-show-errors || true
  aws_ s3 cp "$S3_RESULTS/${KTOP}_d${ML_D}_recsynth_vitis_hls.log" "$D/d${ML_D}_recsynth_vitis_hls.log" --only-show-errors 2>/dev/null || true
  echo "--- recsynth summary (compare Estimated vs baseline 11.425 ns) ---"; cat "$D/d${ML_D}_recsynth_summary.txt" 2>/dev/null || echo "(no summary — check status/boot.log)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo "no state"; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;;
  *) echo "usage: [ML_D=512] [TT_CLK_NS=6.0] AWS_PROFILE=... $0 {launch|status|fetch|terminate}";; esac
