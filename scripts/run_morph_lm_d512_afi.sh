#!/bin/bash
# run_morph_lm_d512_afi.sh — FULL D=512 AFI build (A3 reduced-clock) for the config-B
# MorphSSM-LM training kernel: HLS(-DML_D=512) -> scaffold cl_forge_morph_lm_d512 CL
# -> gen IP (two 4 MB URAM banks = 8 MB staging, NK=1) -> synth -> P&R -> DCP -> create-fpga-image.
#
# THE USER-RUN STEP. Forked from run_pc_afi_build.sh (EC2 mechanics: z1d build box,
# source->S3, IAM profile, continuous log upload, shutdown backstop, self-terminate,
# NO SSH). Differences from the PC AFI build:
#   * CL / PKG_REL / KERNEL_TCL point at the D=512 package + D=512 HLS driver.
#   * The D=512 sources (kernel #if ML_D>=512 pragmas, forge_buffer_morph_lm.hpp
#     width parameterization, and the WHOLE cl_forge_morph_lm_d512 package) are
#     UNCOMMITTED (house rules: no commit) — so we archive the WORKING TREE via
#     `git stash create` (a commit object, no index/tree mutation), NOT HEAD.
#   * CLOCK (v3 CLKGEN, after the v2 250 MHz P&R MEASURED WNS -2.726 ns): the
#     failing paths were INSIDE the 4 MB staging banks (BRAM_CTL write control
#     -> 16-deep URAM cascade, ~6.7 ns physical) — unfixable at the shell-fixed
#     4 ns clk_main_a0. The CL now instantiates AWS_CLK_GEN (the S1 pattern,
#     cut at the SHELL edge): the ENTIRE user design (crossbars, banks,
#     controllers, kernel) runs on clk_extra_a1 = 125 MHz (8 ns; worst measured
#     path closes with >1.2 ns slack), with async AXI clock converters on the
#     DMA-PCIS + OCL boundaries. The build MUST pass --aws_clk_gen AND
#     --clock_recipe_a A1 (CLOCK_ARGS below; A1 -> clk_extra_a1 = 125 MHz).
#     The kernel stays HLS-scheduled at TT_CLK_NS=4.0 (verified est 3.376 ns,
#     II parity) — over-pipelined at 8 ns is harmless, and the HLS gate keeps
#     guarding regressions.
#   * HLS GATE: after the on-box csynth, the estimated CP and the DSP total are
#     parsed from the report; DSP!=0 or estimate > target ABORTS before the ~3 h
#     P&R (report is uploaded either way).
#   * After the DCP tarball, register the AFI with `aws ec2 create-fpga-image`
#     (DO_REGISTER=1 default; needs ec2:CreateFpgaImage on the runner role — if it
#     is denied the tarball is left in S3 for manual `create-fpga-image`).
#
# PREREQ: a PASSING 4.0 ns recsynth (run_morph_lm_d512_recsynth.sh with
# TT_CLK_NS=4.0 + the champion flags). The DEPLOYED D=128 AFI
# (agfi-0433370d93cb58a78) is NOT touched by this build (separate CL package +
# separate AFI).
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   CLOCK_RECIPE_A=A1 bash scripts/run_morph_lm_d512_afi.sh launch     # pick recipe per recsynth
#   bash scripts/run_morph_lm_d512_afi.sh {status|fetch|terminate}
#   DRY_RUN=1 bash scripts/run_morph_lm_d512_afi.sh launch             # render user-data only
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"            # AWS-recommended Vivado build instance
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (Vivado+Vitis+aws-fpga)
MAX_MINUTES="${MAX_MINUTES:-360}"                         # D=512 all-SLR P&R is longer; 6h backstop
CL="${CL:-cl_forge_morph_lm_d512}"
PKG_REL="${PKG_REL:-tikitorch/hardware/hdk/cl_forge_morph_lm_d512}"
KERNEL_TCL="${KERNEL_TCL:-synth_morph_lm_d512.tcl}"      # HLS the D=512 kernel (forces -DML_D=512)
KERNEL_PROJ="${KERNEL_PROJ:-forge_train_step_morph_lm_proj}"
ML_D="${ML_D:-512}"
# the DEPLOYED champion config: weights resident in URAM. REQUIRED for d512 timing
# closure -- recsynth measured plain -DML_D=512 at 12.9 ns (gmem carried deps in the
# backward loops) vs 4.38 ns with these flags (gmem II violations 34->6, DSP=0,
# URAM 189%/SLR = the all-SLR floorplan). Do NOT build plain.
TT_BUILD_FLAGS="${TT_BUILD_FLAGS:--DTT_RESIDENT_W -DTT_ONCHIP_GRAD -DTT_URAM_W}"
TT_CLK_NS="${TT_CLK_NS:-4.0}"                            # HLS target = the FIXED 250 MHz clk_main_a0 (see CLOCK note above)
CLOCK_RECIPE_A="${CLOCK_RECIPE_A:-A1}"                   # A1 -> clk_extra_a1 = 125 MHz (the user-design clock; AWS_CLK_GEN in the CL). A0 -> 62.5 MHz relief knob.
DO_REGISTER="${DO_REGISTER:-1}"                          # 1 -> create-fpga-image on the box after the DCP
AFI_NAME="${AFI_NAME:-cl_forge_morph_lm_d512}"
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_morphlm_d512_afibuild"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID TREE
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  TAG="morphlm-d512-afibuild-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  # WORKING-TREE archive (the D=512 kernel pragmas + the whole d512 CL package are uncommitted).
  echo "== source tarball (git stash create -> working tree) =="
  TREE="$(git -C "$PROJECT_DIR" stash create 2>/dev/null || true)"; [ -z "$TREE" ] && TREE="HEAD"
  echo "archiving tree: $TREE"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" "$TREE"
  # The whole d512 package + the D=512 HLS driver + the pragma'd kernel MUST be present.
  for f in "$PKG_REL/design/$CL.sv" \
           "$PKG_REL/build/scripts/synth_$CL.tcl" \
           "$PKG_REL/build/scripts/gen_ip.tcl" \
           "$PKG_REL/build/constraints/cl_pnr_user_floorplan.xdc" \
           "$PKG_REL/build/constraints/small_shell_cl_pnr_user.xdc" \
           "tikitorch/hardware/hls/forge_kernels/$KERNEL_TCL" \
           "tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp"; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "$f" || { echo "FATAL: $f not in shipped (working-tree) tarball"; exit 1; }; done
  # sanity: the kernel actually carries the D>=512 timing block
  local TMPD; TMPD="$(mktemp -d)"; tar xzf "$STATE_DIR/src.tar.gz" -C "$TMPD" "tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp"
  grep -qE '#if[[:space:]]+ML_D[[:space:]]*>=[[:space:]]*512' "$TMPD/tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp" \
    || { echo "FATAL: kernel missing '#if ML_D >= 512' timing block"; rm -rf "$TMPD"; exit 1; }
  rm -rf "$TMPD"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" "$BUCKET" "$TAG" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN (CL=$CL CLOCK_RECIPE_A='${CLOCK_RECIPE_A:-<shell default>}' TT_CLK_NS=$TT_CLK_NS) -> UD:"; cat "$UD"; return 0; fi
  echo "== upload source -> $S3_SRC =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-morphlm-d512-afibuild},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then echo "retry $tries"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "CL=$CL"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2" bucket="$3" tag="$4"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-morphlm-d512-afibuild deadline" || true
exec > /var/log/tiki-afibuild.log 2>&1
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; BUCKET="${bucket}"; TAG="${tag}"
CL="${CL}"; PKG_REL="${PKG_REL}"; KERNEL_TCL="${KERNEL_TCL}"; KERNEL_PROJ="${KERNEL_PROJ}"; TT_BUILD_FLAGS="${TT_BUILD_FLAGS}"
ML_D="${ML_D}"; TT_CLK_NS="${TT_CLK_NS}"; CLOCK_RECIPE_A="${CLOCK_RECIPE_A}"
DO_REGISTER="${DO_REGISTER}"; AFI_NAME="${AFI_NAME}"; REGION="${REGION}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-afibuild.log "\$S3_RESULTS/build.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; hostname; nproc; free -g
export HOME=/root; mkdir -p /root/.oasys /root/.Xilinx
( command -v git-lfs >/dev/null 2>&1 || apt-get update -y && apt-get install -y git-lfs || yum install -y git-lfs ) || true
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/hdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
echo "AWS_FPGA=\$AF"
for s in /opt/Xilinx/Vivado/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
( cd "\$AF" && git lfs install && git lfs pull ) > /tmp/lfs.txt 2>&1 || true
cd "\$AF" && source hdk_setup.sh || true
echo "HDK_DIR=\$HDK_DIR HDK_SHELL_DIR=\$HDK_SHELL_DIR"
# print the clock-recipe table so a valid CLOCK_RECIPE_A can be chosen for the reduced clock
{ echo "== clock recipes (pick CLOCK_RECIPE_A per the recsynth-measured clock) =="; \
  find "\$AF/hdk" -iname "*clock_recipe*" 2>/dev/null | head; \
  cat "\$AF"/hdk/common/shell_stable/build/scripts/*clock_recipe* 2>/dev/null | head -80; \
  cat "\$AF"/hdk/cl/examples/cl_template/build/scripts/aws_build_dcp_from_cl.py 2>/dev/null | grep -iE "recipe|clock" | head -40; \
} > /tmp/clock_recipes.txt 2>&1; aws s3 cp /tmp/clock_recipes.txt "\$S3_RESULTS/clock_recipes.txt" --only-show-errors || true
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
# --- 1. HLS the D=512 kernel -> RTL (forces -DML_D=512, reduced TT_CLK_NS) ---
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
export ML_D="\$ML_D"; export TT_CLK_NS="\$TT_CLK_NS"; export TT_BUILD_FLAGS="\$TT_BUILD_FLAGS"
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "=== HLS \$KERNEL_TCL (ML_D=\$ML_D TT_CLK_NS=\$TT_CLK_NS) START \$(date -u) ==="; "\$VHLS" -f "\$KERNEL_TCL" 2>&1 | tail -30
KVDIR="/opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ/solution1/syn/verilog"
ls -la "\$KVDIR" | head; ls "\$KVDIR" | wc -l; up
# --- 1b. HLS GATE: DSP must be 0 (mult-free) and estimated CP <= target, else ABORT before the ~3h P&R.
#     Parse failures do NOT abort (a P&R attempt still yields the utilization feedback the pblock TODO needs).
KRPT="/opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ/solution1/syn/report/forge_train_step_morph_lm_csynth.rpt"
aws s3 cp "\$KRPT" "\$S3_RESULTS/csynth.rpt" --only-show-errors || true
EST="\$(awk -F'|' '/ap_clk/{gsub(/[ ns]/,"",\$4); print \$4; exit}' "\$KRPT")"
DSPTOT="\$(awk -F'|' '/^\|Total/{gsub(/ /,"",\$4); print \$4; exit}' "\$KRPT")"
echo "HLS GATE: est=\${EST:-unparsed} target=\$TT_CLK_NS dsp=\${DSPTOT:-unparsed}"
GATEFAIL=0
[ -n "\$DSPTOT" ] && [ "\$DSPTOT" != "0" ] && { echo "GATE FAIL: DSP=\$DSPTOT != 0 (mult-free invariant)"; GATEFAIL=1; }
[ -n "\$EST" ] && awk -v e="\$EST" -v t="\$TT_CLK_NS" 'BEGIN{exit !(e>t)}' && { echo "GATE FAIL: estimated \$EST ns > target \$TT_CLK_NS ns"; GATEFAIL=1; }
if [ "\$GATEFAIL" = "1" ]; then echo "est=\$EST dsp=\$DSPTOT" | aws s3 cp - "\$S3_RESULTS/HLS_GATE_FAIL" --only-show-errors || true; up; shutdown -h now; exit 1; fi
# --- 2. scaffold the CL from the HDK template, overlay OUR d512 design/+build/ ---
TPL=""
for cand in CL_TEMPLATE cl_template cl_dram_hbm_dma cl_dram_dma cl_mem_perf; do
  hit="\$(find "\$AF" -path "*/\$cand/build/scripts/aws_build_dcp_from_cl.py" 2>/dev/null | head -1)"
  [ -n "\$hit" ] && { TPL="\$(cd "\$(dirname "\$hit")/../.." && pwd)"; break; }
done
[ -z "\$TPL" ] && { echo "FATAL: no CL template found"; shutdown -h now; }
echo "TEMPLATE=\$TPL"
CLROOT="/opt/tiki/CL/\$CL"; mkdir -p "\$CLROOT"
cp -r "\$TPL"/. "\$CLROOT"/
rm -rf "\$CLROOT"/build && cp -rL "\$TPL/build" "\$CLROOT"/build
rm -rf "\$CLROOT"/design && cp -r "/opt/tiki/\$PKG_REL/design" "\$CLROOT"/design
mkdir -p "\$CLROOT/design/ip"
cp -f "/opt/tiki/\$PKG_REL/build/scripts/"*.tcl "\$CLROOT"/build/scripts/ 2>/dev/null || true
mkdir -p "\$CLROOT/build/constraints"; cp -f "/opt/tiki/\$PKG_REL/build/constraints/"*.xdc "\$CLROOT"/build/constraints/ 2>/dev/null || true
cp "\$KVDIR"/*.v "\$CLROOT"/design/ 2>/dev/null || true
{ echo "== CL design =="; ls -la "\$CLROOT/design"; echo "== build/scripts =="; ls -la "\$CLROOT/build/scripts"; } | aws s3 cp - "\$S3_RESULTS/cl_layout.txt" --only-show-errors || true
up
# --- 3. gen IP (two 4 MB URAM banks, 1->2 crossbars both sides) ---
echo "=== gen_ip START \$(date -u) ==="
vivado -mode batch -source "\$CLROOT/build/scripts/gen_ip.tcl" -tclargs "\$CLROOT/design/ip" 2>&1 | tail -30
ls -la "\$CLROOT/design/ip" 2>&1 | head -20; up
# --- 4. synth + P&R (FAIL-FAST at synth before the long all-SLR P&R) ---
export CL_DIR="\$CLROOT"
BUILD_PY="\$(find "\$CLROOT/build" -name aws_build_dcp_from_cl.py 2>/dev/null | head -1)"
echo "BUILD_PY=\$BUILD_PY"; chmod +x "\$BUILD_PY" 2>/dev/null; cd "\$(dirname "\$BUILD_PY")"
# v3 CLKGEN: AWS_CLK_GEN is instantiated in the CL, so --aws_clk_gen is MANDATORY
# on EVERY build (pulls in aws_clk_gen.sv + its IP + clkgen clock constraints; the
# S1 lesson: recipe WITHOUT clkgen silently leaves the kernel with no 125MHz clock).
CLKARG="--aws_clk_gen --clock_recipe_a \${CLOCK_RECIPE_A:-A1}"
echo "=== aws_build_dcp_from_cl (CL=\$CL \$CLKARG) START \$(date -u) ==="
./aws_build_dcp_from_cl.py --cl "\$CL" --no-encrypt \$CLKARG > /tmp/aws_build.full.log 2>&1; echo "BUILD_RC=\$?"
tail -120 /tmp/aws_build.full.log
aws s3 cp /tmp/aws_build.full.log "\$S3_RESULTS/aws_build.full.log" --only-show-errors || true
# --- 5. collect DCP tarball + reports ---
ls -la "\$CLROOT/build/checkpoints/" "\$CLROOT/build/checkpoints/to_aws/" 2>&1 | aws s3 cp - "\$S3_RESULTS/checkpoints_listing.txt" --only-show-errors || true
aws s3 cp "\$CLROOT/build/checkpoints/" "\$S3_RESULTS/to_aws/" --recursive --exclude "*" --include "*Developer_CL.tar" --include "*.tar" --include "*post_route*.dcp" --only-show-errors 2>/dev/null || true
for r in \$(find "\$CLROOT/build" -name "*.rpt" -o -name "*timing*summary*" 2>/dev/null); do aws s3 cp "\$r" "\$S3_RESULTS/reports/\$(basename "\$r")" --only-show-errors 2>/dev/null || true; done
up
# --- 6. register the AFI (create-fpga-image). Needs the DCP .tar in an owned S3 bucket + ec2:CreateFpgaImage. ---
TARB="\$(find "\$CLROOT/build/checkpoints" -name "*Developer_CL.tar" 2>/dev/null | head -1)"
if [ "\$DO_REGISTER" = "1" ] && [ -n "\$TARB" ]; then
  DCP_KEY="\${TAG}/dcp/\$(basename "\$TARB")"; LOG_KEY="\${TAG}/afi-logs"
  aws s3 cp "\$TARB" "s3://\$BUCKET/\$DCP_KEY" --only-show-errors || true
  echo "=== create-fpga-image (name=\$AFI_NAME) ==="
  aws ec2 create-fpga-image --region "\$REGION" \
    --name "\$AFI_NAME" \
    --description "MorphSSM-LM D=512 reduced-clock (\${TAG}); DSP=0 mult-free; parallel to agfi-0433 D=128 sentinel" \
    --input-storage-location "Bucket=\$BUCKET,Key=\$DCP_KEY" \
    --logs-storage-location  "Bucket=\$BUCKET,Key=\$LOG_KEY" \
    > /tmp/afi_ids.json 2>/tmp/afi_err.txt \
    && cat /tmp/afi_ids.json | aws s3 cp - "\$S3_RESULTS/afi_ids.json" --only-show-errors \
    || { echo "create-fpga-image FAILED (likely ec2:CreateFpgaImage not on the role) — tarball is at s3://\$BUCKET/\$DCP_KEY for manual registration"; cat /tmp/afi_err.txt; aws s3 cp /tmp/afi_err.txt "\$S3_RESULTS/afi_err.txt" --only-show-errors || true; }
else
  echo "DO_REGISTER=\$DO_REGISTER or no tarball (\$TARB) — skipping create-fpga-image"
fi
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- build.log tail --"; aws_ s3 cp "$S3_RESULTS/build.log" - 2>/dev/null | tail -40; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/afibuild_d512"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -R "$DEST" | head -80
  echo "-- AFI ids (if registered) --"; cat "$DEST/afi_ids.json" 2>/dev/null || echo "(none — check afi_err.txt / register the to_aws tarball manually)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;;
  *) echo "usage: [CLOCK_RECIPE_A=A1] [TT_CLK_NS=6.0] [DO_REGISTER=1] AWS_PROFILE=... $0 {launch|status|fetch|terminate}";; esac
