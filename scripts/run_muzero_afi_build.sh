#!/bin/bash
# run_muzero_afi_build.sh — Phase B of the forward MuZero AFI: HLS the forward kernel, scaffold the CL from
# the HDK template, overlay our committed cl_forge_muzero design/+build/, gen IP, and run
# aws_build_dcp_from_cl.py (synth -> P&R -> DCP -> to_aws tarball). Same robust pattern as the csynth/probe
# runners: source->S3, IAM instance-profile, shutdown backstop, self-terminate, NO SSH. Streams the build log
# to S3 continuously so a synth failure (~20 min, FAIL-FAST before the ~3h P&R) is diagnosable without SSH.
# AFI registration (create-fpga-image) is done separately from the dev box (this box only produces the tarball).
#   export AWS_PROFILE=my-profile
#   bash scripts/run_muzero_afi_build.sh launch      # build src -> S3 -> launch z1d build box
#   bash scripts/run_muzero_afi_build.sh status        # poll S3 (no SSH)
#   bash scripts/run_muzero_afi_build.sh fetch          # download to_aws tarball + reports
#   bash scripts/run_muzero_afi_build.sh terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"            # AWS-recommended Vivado build instance
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (Vivado+Vitis+aws-fpga); proven by probe
MAX_MINUTES="${MAX_MINUTES:-300}"                         # synth ~20m + P&R ~2h43m + margin; backstop 5h
CL="${CL:-cl_forge_muzero}"
PKG_REL="${PKG_REL:-tikitorch/hardware/hdk/cl_forge_muzero}"   # CL package (design/ + build/); override for DDR variant
KERNEL_TCL="${KERNEL_TCL:-synth_muzero_chess.tcl}"        # HLS the FORWARD kernel (top forge_train_step_muzero)
KERNEL_PROJ="${KERNEL_PROJ:-forge_muzero_chess_proj}"
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_afibuild_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="afibuild-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== source tarball (git archive HEAD) =="
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$PKG_REL/design/$CL.sv" \
    || { echo "FATAL: $PKG_REL/design/$CL.sv not in committed tree"; exit 1; }
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> UD:"; cat "$UD"; return 0; fi
  echo "== upload source -> $S3_SRC =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-muzero-afibuild},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-afibuild deadline" || true
exec > /var/log/tiki-afibuild.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; CL="${CL}"; PKG_REL="${PKG_REL}"
KERNEL_TCL="${KERNEL_TCL}"; KERNEL_PROJ="${KERNEL_PROJ}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-afibuild.log "\$S3_RESULTS/build.log" --only-show-errors || true; }
trap up EXIT
# continuous log uploader (every 60s) so a synth failure is visible without SSH
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; hostname; nproc; free -g
# cloud-init runs as root with HOME unset -> Vivado's realTime.tcl chokes expanding ~/.oasys
# ("Failed to load feature 'vivado'"). Set HOME before any Vivado/Vitis invocation.
export HOME=/root; mkdir -p /root/.oasys /root/.Xilinx
# git-lfs is NOT preinstalled on the AMI; the shell DCP/manifest are LFS objects, so without it
# 'git lfs pull' is a no-op and P&R links against pointer stubs. Install it first.
( command -v git-lfs >/dev/null 2>&1 || apt-get update -y && apt-get install -y git-lfs || yum install -y git-lfs ) || true
# --- locate aws-fpga + tools ---
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/hdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
echo "AWS_FPGA=\$AF"
for s in /opt/Xilinx/Vivado/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
( cd "\$AF" && git lfs install && git lfs pull ) > /tmp/lfs.txt 2>&1 || true
cd "\$AF" && source hdk_setup.sh || true
echo "HDK_DIR=\$HDK_DIR HDK_SHELL_DIR=\$HDK_SHELL_DIR"
# --- EARLY DIAGNOSTICS to S3 (so even a failure teaches the box layout) ---
{ echo "== aws_build_dcp_from_cl.py locations =="; find "\$AF" -name aws_build_dcp_from_cl.py 2>/dev/null;
  echo "== CL example/template dirs =="; ls -d "\$AF"/hdk/cl/examples/* "\$AF"/hdk/cl/developer_designs/* 2>/dev/null;
  echo "== HDK_SHELL_DIR interfaces =="; ls "\$HDK_SHELL_DIR/design/interfaces" 2>/dev/null;
  echo "== vivado =="; command -v vivado; vivado -version 2>&1 | head -2;
} > /tmp/diag.txt 2>&1; aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors || true
# --- fetch our repo source ---
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
# --- 1. HLS the forward kernel -> RTL ---
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "=== HLS \$KERNEL_TCL START \$(date -u) ==="; "\$VHLS" -f "\$KERNEL_TCL" 2>&1 | tail -25
KVDIR="/opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ/solution1/syn/verilog"
ls -la "\$KVDIR" | head; ls "\$KVDIR" | wc -l
up
# --- 2. scaffold the CL from the HDK template (prefer the minimal cl_template; fall back to an example) ---
TPL=""
for cand in CL_TEMPLATE cl_template cl_dram_hbm_dma cl_dram_dma cl_mem_perf; do
  hit="\$(find "\$AF" -path "*/\$cand/build/scripts/aws_build_dcp_from_cl.py" 2>/dev/null | head -1)"
  [ -n "\$hit" ] && { TPL="\$(cd "\$(dirname "\$hit")/../.." && pwd)"; break; }
done
[ -z "\$TPL" ] && { echo "FATAL: no CL template with aws_build_dcp_from_cl.py found"; aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors; shutdown -h now; }
echo "TEMPLATE=\$TPL"
CLROOT="/opt/tiki/CL/\$CL"; mkdir -p "\$CLROOT"
cp -r "\$TPL"/. "\$CLROOT"/                      # template structure (build/scripts harness, verif, etc.)
# CL_TEMPLATE's build/ harness uses RELATIVE symlinks into the HDK that dangle after relocation
# (=> "./aws_build_dcp_from_cl.py: No such file or directory"). Re-copy build/ dereferencing symlinks.
rm -rf "\$CLROOT"/build && cp -rL "\$TPL/build" "\$CLROOT"/build
rm -rf "\$CLROOT"/design                          # drop template RTL; use ours
cp -r "/opt/tiki/\$PKG_REL/design" "\$CLROOT"/design
mkdir -p "\$CLROOT/design/ip"                      # create_ip -dir requires the dir to exist
cp -f "/opt/tiki/\$PKG_REL/build/scripts/"*.tcl "\$CLROOT"/build/scripts/ 2>/dev/null || true
mkdir -p "\$CLROOT/build/constraints"; cp -f "/opt/tiki/\$PKG_REL/build/constraints/"*.xdc "\$CLROOT"/build/constraints/ 2>/dev/null || true
cp "\$KVDIR"/*.v "\$CLROOT"/design/ 2>/dev/null || true     # flatten HLS verilog into design/
echo "== aws_build entry (shebang/type) =="; head -3 "\$CLROOT/build/scripts/aws_build_dcp_from_cl.py" 2>/dev/null; file "\$CLROOT/build/scripts/aws_build_dcp_from_cl.py" 2>/dev/null
echo "== CL design dir =="; ls "\$CLROOT/design" | head -40; ls "\$CLROOT/design"/*.v 2>/dev/null | wc -l
aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors || true
{ echo "== CLROOT design listing =="; ls -la "\$CLROOT/design"; echo "== build/scripts =="; ls -la "\$CLROOT/build/scripts"; } | aws s3 cp - "\$S3_RESULTS/cl_layout.txt" --only-show-errors || true
up
# --- 3. gen IP ---
echo "=== gen_ip START \$(date -u) ==="
vivado -mode batch -source "\$CLROOT/build/scripts/gen_ip.tcl" -tclargs "\$CLROOT/design/ip" 2>&1 | tail -30
echo "== gen_ip produced =="; ls -la "\$CLROOT/design/ip" 2>&1 | head -20; up
# --- 4. synth + P&R via the HDK harness (FAIL-FAST at synth ~20m before the ~3h P&R) ---
export CL_DIR="\$CLROOT"
BUILD_PY="\$(find "\$CLROOT/build" -name aws_build_dcp_from_cl.py 2>/dev/null | head -1)"
echo "BUILD_PY=\$BUILD_PY"; [ -z "\$BUILD_PY" ] && { echo "FATAL: aws_build_dcp_from_cl.py absent in CL"; ls -R "\$CLROOT/build" | head -60; }
chmod +x "\$BUILD_PY" 2>/dev/null
cd "\$(dirname "\$BUILD_PY")"
echo "=== aws_build_dcp_from_cl START \$(date -u) ==="
# capture the FULL build output (tail -80 previously hid generate_dcp_tarball); upload it whole
./aws_build_dcp_from_cl.py --cl "\$CL" --no-encrypt > /tmp/aws_build.full.log 2>&1; echo "BUILD_RC=\$?"
tail -100 /tmp/aws_build.full.log
aws s3 cp /tmp/aws_build.full.log "\$S3_RESULTS/aws_build.full.log" --only-show-errors || true
echo "=== build done \$(date -u) ==="
# --- 5. collect outputs ---
# The tarball create-fpga-image needs is checkpoints/<tag>.Developer_CL.tar (the script tars ./to_aws then
# DELETES to_aws/, so upload the .tar itself + the post_route DCP as backup), plus a checkpoints listing.
ls -la "\$CLROOT/build/checkpoints/" "\$CLROOT/build/checkpoints/to_aws/" 2>&1 | aws s3 cp - "\$S3_RESULTS/checkpoints_listing.txt" --only-show-errors || true
aws s3 cp "\$CLROOT/build/checkpoints/" "\$S3_RESULTS/to_aws/" --recursive --exclude "*" --include "*Developer_CL.tar" --include "*.tar" --include "*post_route*.dcp" --only-show-errors 2>/dev/null || true
for r in \$(find "\$CLROOT/build" -name "*.rpt" -o -name "*timing*summary*" 2>/dev/null); do aws s3 cp "\$r" "\$S3_RESULTS/reports/\$(basename "\$r")" --only-show-errors 2>/dev/null || true; done
aws s3 cp "\$CLROOT/build/scripts/" "\$S3_RESULTS/scriptlogs/" --recursive --exclude "*" --include "*.log" --only-show-errors 2>/dev/null || true
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up
shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- build.log tail --"; aws_ s3 cp "$S3_RESULTS/build.log" - 2>/dev/null | tail -40; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/afibuild"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -R "$DEST" | head -60; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate} (DRY_RUN=1 for launch)";; esac
