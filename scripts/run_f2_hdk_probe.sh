#!/bin/bash
# run_f2_hdk_probe.sh — Phase A of the forward MuZero AFI build: extract the AWS F2 HDK shell interface
# (cl_ports.vh PCIM port names/widths + any cl_dram_dma PCIM-master example + the unused_*_template.inc
# names) so the PCIM section of cl_forge_muzero.sv can be wired against GROUND TRUTH, not a guess — before
# spending a 3-4h P&R. Same robust pattern as run_muzero_chess_csynth_fpga.sh: source->S3, IAM
# instance-profile, shutdown backstop, self-terminate, NO SSH. Cheap z1d + FPGA Dev AMI (no FPGA hardware
# needed to read the HDK).
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_f2_hdk_probe.sh launch    # set up HDK on a box -> dump interface -> S3 -> self-terminate
#   bash scripts/run_f2_hdk_probe.sh status     # poll S3 (no SSH)
#   bash scripts/run_f2_hdk_probe.sh fetch       # download the dumped HDK interface files
#   bash scripts/run_f2_hdk_probe.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (Ubuntu) 1.16.1 (proven by csynth runs)
MAX_MINUTES="${MAX_MINUTES:-40}"                          # probe is ~15-25 min (clone + lfs pull dominate)
ROLE="${ROLE:-tiki-sweep-runner}"
AWS_FPGA_REF="${AWS_FPGA_REF:-}"                          # optional git ref of aws-fpga to checkout (F2 branch/tag)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_hdkprobe_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="hdkprobe-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching. UD:"; cat "$UD"; return 0; fi
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-hdk-probe},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_results="$1"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-hdkprobe deadline" || true
exec > /var/log/tiki-hdkprobe.log 2>&1
set -x
S3_RESULTS="${s3_results}"; AWS_FPGA_REF="${AWS_FPGA_REF}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-hdkprobe.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname; whoami
# locate aws-fpga (FPGA Dev AMI ships it under common paths) or clone it
AF=""
for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/hdk" ] && AF="\$p" && break; done
if [ -z "\$AF" ]; then
  AF=/opt/aws-fpga
  git clone https://github.com/aws/aws-fpga.git "\$AF" || true
fi
echo "AWS_FPGA=\$AF"; ( cd "\$AF" && git log --oneline -1 && git branch -a | head -40 && git tag | tail -40 ) > /tmp/aws_fpga_gitinfo.txt 2>&1 || true
[ -n "\$AWS_FPGA_REF" ] && ( cd "\$AF" && git fetch --all && git checkout "\$AWS_FPGA_REF" ) || true
( cd "\$AF" && git lfs install && git lfs pull ) > /tmp/lfs_pull.txt 2>&1 || true
up
# source HDK setup (sets HDK_SHELL_DIR / HDK_COMMON_DIR). Do NOT pipe (subshell eats exports).
cd "\$AF" && source hdk_setup.sh > /tmp/hdk_setup.txt 2>&1 || true
echo "HDK_SHELL_DIR=\$HDK_SHELL_DIR" | tee /tmp/hdk_env.txt
echo "HDK_DIR=\$HDK_DIR HDK_COMMON_DIR=\$HDK_COMMON_DIR" | tee -a /tmp/hdk_env.txt
up
# DUMP the interface ground truth -> S3 (capture early, before anything can fail)
OUT=/tmp/hdkdump; mkdir -p \$OUT
cp /tmp/aws_fpga_gitinfo.txt /tmp/lfs_pull.txt /tmp/hdk_setup.txt /tmp/hdk_env.txt \$OUT/ 2>/dev/null || true
# cl_ports.vh wherever it lives + the whole interfaces dir listing
find "\$AF" -name cl_ports.vh 2>/dev/null | tee \$OUT/cl_ports_locations.txt
for f in \$(find "\$AF" -name cl_ports.vh 2>/dev/null); do cp "\$f" "\$OUT/cl_ports.\$(echo "\$f"|tr / _).vh"; done
[ -n "\$HDK_SHELL_DIR" ] && ls -R "\$HDK_SHELL_DIR/design/interfaces" > \$OUT/interfaces_listing.txt 2>&1 || true
[ -n "\$HDK_SHELL_DIR" ] && cp "\$HDK_SHELL_DIR"/design/interfaces/*.inc \$OUT/ 2>/dev/null || true
[ -n "\$HDK_SHELL_DIR" ] && cp "\$HDK_SHELL_DIR"/design/interfaces/*.vh  \$OUT/ 2>/dev/null || true
# grep the PCIM port family out of cl_ports.vh + templates (the names we need to wire)
{ echo "==== PCIM ports (cl_sh_pcim_* / sh_cl_pcim_*) ===="; grep -rniE "pcim" "\$HDK_SHELL_DIR/design/interfaces" 2>/dev/null;
  echo "==== unused_*_template.inc files ===="; find "\$AF" -name "unused_*_template.inc" 2>/dev/null;
  echo "==== ocl rename check (ocl_cl_/cl_ocl_) ===="; grep -rniE "ocl_cl_|cl_ocl_|ocl_" "\$HDK_SHELL_DIR/design/interfaces/cl_ports.vh" 2>/dev/null | head -40;
} > \$OUT/pcim_grep.txt 2>&1
# any CL example that MASTERS pcim (cl_dram_dma / cl_*); copy its top .sv for a wiring reference
find "\$AF/hdk/cl" -maxdepth 2 -name "*.sv" 2>/dev/null | tee \$OUT/cl_examples.txt
for d in cl_dram_dma cl_dram_hbm_dma cl_mem_perf cl_hello_world; do
  for f in \$(find "\$AF/hdk/cl/examples/\$d" -name "*.sv" 2>/dev/null | grep -iE "cl_\$d(\b|\.|_)" | head -3); do
    cp "\$f" "\$OUT/example_\$d_\$(basename "\$f")" 2>/dev/null || true
  done
done
aws s3 cp \$OUT/ "\$S3_RESULTS/" --recursive --only-show-errors || true
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- boot.log tail --"; aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -25; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/hdk_probe"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -1 "$DEST"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate} (DRY_RUN=1 for launch)";; esac
