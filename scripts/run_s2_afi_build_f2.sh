#!/bin/bash
# run_s2_afi_build_f2.sh — build the AFI for forge_pc_xformer_s2_block (SCALE-2 D=512, 30M DDR-streaming K-wide-MAC
# PC transformer kernel). Thin clone of run_mp_afi_build_f2.sh pointing at the s2 CL + s2 kernel: HLS the s2 kernel
# -> RTL, scaffold the cl_forge_pc_xformer_s2 CL (instantiating forge_pc_xformer_s2_block), gen_ip,
# aws_build_dcp_from_cl.py -> DCP tarball in S3, then create-fpga-image -> AGFI. Distinct STATE_DIR + s2afi- S3 tag.
#   AWS_PROFILE=... bash scripts/run_s2_afi_build_f2.sh build      # launch the DCP build (~3-5h, self-terminates)
#   AWS_PROFILE=... bash scripts/run_s2_afi_build_f2.sh status     # poll the build
#   AWS_PROFILE=... bash scripts/run_s2_afi_build_f2.sh afi        # create-fpga-image from the tarball (after build DONE)
#   AWS_PROFILE=... bash scripts/run_s2_afi_build_f2.sh afistatus  # poll AFI -> prints AGFI when available
# Optional csynth-only gate (read the per-loop II table before the ~3-5h P&R): CSYNTH_ONLY=1 S2_K=32 ... build
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"; ITYPE="${ITYPE:-z1d.6xlarge}"; ROLE="${ROLE:-tiki-sweep-runner}"
# CL main clock recipe. On F2, clk_main_a0 is HARD-LOCKED at 250MHz for EVERY recipe; the kernel runs on a derived
# 150MHz clk_kernel (CDC in the CL top) so timing closes. Leave CLOCK_A empty = build default (A1/250MHz).
CLOCK_A="${CLOCK_A:-}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; STATE="${STATE_DIR:-$PROJECT_DIR/.tiki_s2_afi_state}"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_build(){
  [ -f "$STATE/build.env" ] && { echo "build state exists; terminate/clear first"; cat "$STATE/build.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="s2afi-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE/src.tar.gz" HEAD
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +330 "tiki-s2afi deadline" || true
exec > /var/log/tiki-s2afi.log 2>&1
set -x
S3_SRC="${S3SRC}"; S3_RESULTS="${S3RES}"; CLOCK_A="${CLOCK_A}"; CSYNTH_ONLY="${CSYNTH_ONLY:-}"
CL="cl_forge_pc_xformer_s2"; PKG_REL="tikitorch/hardware/hdk/cl_forge_pc_xformer_s2"
KERNEL_TCL="${KERNEL_TCL:-synth_pc_xformer_s2_block.tcl}"; KERNEL_PROJ="${KERNEL_PROJ:-forge_pc_xformer_s2_block_proj}"; export S2_K="${S2_K:-}"
# sweep knobs -> the synth tcl (empty = kernel-header defaults): vocab, the (innocent) aggressive-sched engine
# gate, the de-pipeline gate, and a csynth clock override (S2_CLK_NS, e.g. 6.667) for sub-250MHz CDC points.
export S2_VOCAB="${S2_VOCAB:-}"; export S2_AGGRESSIVE_SCHED="${S2_AGGRESSIVE_SCHED:-}"; export S2_DEPIPELINE_REDUCE="${S2_DEPIPELINE_REDUCE:-}"; export S2_CLK_NS="${S2_CLK_NS:-}"
export S2_SEQ_TILE="${S2_SEQ_TILE:-}"; export S2_TILE_O="${S2_TILE_O:-}"; export S2_TILE_I="${S2_TILE_I:-}"   # tiling knobs (the synth tcl already forwards these)
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-s2afi.log "\$S3_RESULTS/build.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; hostname; nproc; free -g
export HOME=/root; mkdir -p /root/.oasys /root/.Xilinx
# git-lfs is REQUIRED: hdk_setup.sh uses it to pull the F2 shell DCP, then sets HDK_IP_SRC_DIR. This AMI is
# Ubuntu (hdk_setup's auto-install tries yum -> fails). Install via apt, else fall back to the static github binary.
if ! command -v git-lfs >/dev/null 2>&1; then apt-get update -y >/dev/null 2>&1 || true; apt-get install -y git-lfs >/dev/null 2>&1 || true; fi
if ! command -v git-lfs >/dev/null 2>&1; then
  ( cd /tmp && curl -fsSL https://github.com/git-lfs/git-lfs/releases/download/v3.5.1/git-lfs-linux-amd64-v3.5.1.tar.gz -o glfs.tgz \
      && tar xzf glfs.tgz && cp git-lfs-3.5.1/git-lfs /usr/local/bin/git-lfs && chmod +x /usr/local/bin/git-lfs ) || true
fi
echo "git-lfs: $(command -v git-lfs || echo MISSING) $(git-lfs version 2>/dev/null | head -1)"
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/hdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
echo "AWS_FPGA=\$AF"
for s in /opt/Xilinx/Vivado/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
( cd "\$AF" && git lfs install && git lfs pull ) > /tmp/lfs.txt 2>&1 || true
cd "\$AF" && source hdk_setup.sh || true
echo "HDK_DIR=\$HDK_DIR HDK_SHELL_DIR=\$HDK_SHELL_DIR HDK_IP_SRC_DIR=\$HDK_IP_SRC_DIR"
# FAST-FAIL if hdk_setup didn't fully set up (don't waste ~20min reaching aws_build's HDK_IP_SRC_DIR error).
if [ -z "\$HDK_IP_SRC_DIR" ]; then
  echo "FATAL: hdk_setup did not set HDK_IP_SRC_DIR (git-lfs / shell-DCP pull failed)"
  { echo "=== HDK SETUP FAILED ==="; echo "git-lfs: \$(command -v git-lfs)"; git-lfs version 2>&1; echo "--- lfs.txt ---"; tail -30 /tmp/lfs.txt 2>/dev/null; } | aws s3 cp - "\$S3_RESULTS/HDK_SETUP_FAILED.txt" --only-show-errors || true
  echo "hdk-setup-fail \$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
  shutdown -h now; exit 1
fi
{ echo "== aws_build_dcp_from_cl.py =="; find "\$AF" -name aws_build_dcp_from_cl.py 2>/dev/null;
  echo "== CL templates =="; ls -d "\$AF"/hdk/cl/examples/* "\$AF"/hdk/cl/developer_designs/* 2>/dev/null;
  echo "== vivado =="; command -v vivado; vivado -version 2>&1 | head -2; } > /tmp/diag.txt 2>&1
aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors || true
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
# 1. HLS the s2 kernel -> RTL
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "=== HLS \$KERNEL_TCL START \$(date -u) ==="; "\$VHLS" -f "\$KERNEL_TCL" 2>&1 | tail -30
KVDIR="/opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ/solution1/syn/verilog"
ls -la "\$KVDIR" | head; ls "\$KVDIR" | wc -l; up
# ---- csynth per-loop latency report(s) — concatenate EVERY module report (INLINE-off sub-functions s2_gemm/
#      s2_attn_*/s2_adam_stream each with its own report) so the parent can read the II table + DSP_MULT ----
: > /tmp/csynth.rpt
for r in \$(find /opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ -name '*_csynth.rpt' 2>/dev/null | sort); do
  echo "===================== \$r =====================" >> /tmp/csynth.rpt; cat "\$r" >> /tmp/csynth.rpt; echo >> /tmp/csynth.rpt; done
echo "=== CSYNTH reports concatenated: \$(grep -c '=====================' /tmp/csynth.rpt) modules, \$(wc -l </tmp/csynth.rpt) lines ==="
aws s3 cp /tmp/csynth.rpt "\$S3_RESULTS/csynth.rpt" --only-show-errors || true; up
if [ -n "\$CSYNTH_ONLY" ]; then echo "=== CSYNTH_ONLY: skipping CL/P&R ==="; echo "csynth-only \$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true; sync; shutdown -h now; exit 0; fi
# 2. scaffold CL from HDK template
TPL=""
for cand in CL_TEMPLATE cl_template cl_dram_hbm_dma cl_dram_dma cl_mem_perf; do
  hit="\$(find "\$AF" -path "*/\$cand/build/scripts/aws_build_dcp_from_cl.py" 2>/dev/null | head -1)"
  [ -n "\$hit" ] && { TPL="\$(cd "\$(dirname "\$hit")/../.." && pwd)"; break; }
done
[ -z "\$TPL" ] && { echo "FATAL: no CL template"; aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors; shutdown -h now; }
echo "TEMPLATE=\$TPL"
CLROOT="/opt/tiki/CL/\$CL"; mkdir -p "\$CLROOT"; cp -r "\$TPL"/. "\$CLROOT"/
rm -rf "\$CLROOT"/build && cp -rL "\$TPL/build" "\$CLROOT"/build
rm -rf "\$CLROOT"/design; cp -r "/opt/tiki/\$PKG_REL/design" "\$CLROOT"/design; mkdir -p "\$CLROOT/design/ip"
cp -f "/opt/tiki/\$PKG_REL/build/scripts/"*.tcl "\$CLROOT"/build/scripts/ 2>/dev/null || true
mkdir -p "\$CLROOT/build/constraints"; cp -f "/opt/tiki/\$PKG_REL/build/constraints/"*.xdc "\$CLROOT"/build/constraints/ 2>/dev/null || true
cp "\$KVDIR"/*.v "\$CLROOT"/design/ 2>/dev/null || true
# stage the HLS floating-point operator IP GENERATORS (the *_ip.tcl that csynth emits next to the wrappers);
# synth_cl_forge_pc_xformer_s2.tcl sources them to create+inline-synth the floating_point cores the BF16 backward
# + attention P@V blend reference.
mkdir -p "\$CLROOT/design/fpip"; cp "\$KVDIR"/*_ip.tcl "\$CLROOT/design/fpip/" 2>/dev/null || true
echo "FPIP generators staged: \$(ls "\$CLROOT/design/fpip"/*_ip.tcl 2>/dev/null | wc -l)"
{ echo "== CL design =="; ls -la "\$CLROOT/design"; echo "== design/fpip =="; ls -la "\$CLROOT/design/fpip"; echo "== build/scripts =="; ls -la "\$CLROOT/build/scripts"; } | aws s3 cp - "\$S3_RESULTS/cl_layout.txt" --only-show-errors || true
up
# 3. gen IP
echo "=== gen_ip START \$(date -u) ==="; vivado -mode batch -source "\$CLROOT/build/scripts/gen_ip.tcl" -tclargs "\$CLROOT/design/ip" 2>&1 | tail -30
ls -la "\$CLROOT/design/ip" 2>&1 | head -20; up
# 4. synth + P&R
export CL_DIR="\$CLROOT"; BUILD_PY="\$(find "\$CLROOT/build" -name aws_build_dcp_from_cl.py 2>/dev/null | head -1)"
echo "BUILD_PY=\$BUILD_PY"; chmod +x "\$BUILD_PY" 2>/dev/null; cd "\$(dirname "\$BUILD_PY")"
echo "=== aws_build_dcp_from_cl START \$(date -u) ==="
CLKARG=""; [ -n "\$CLOCK_A" ] && CLKARG="--aws_clk_gen --clock_recipe_a \$CLOCK_A"
echo "=== aws_build clock args: \${CLKARG:-(default A1/250MHz)} ==="
./aws_build_dcp_from_cl.py --cl "\$CL" \$CLKARG --no-encrypt > /tmp/aws_build.full.log 2>&1; echo "BUILD_RC=\$?"
tail -120 /tmp/aws_build.full.log; aws s3 cp /tmp/aws_build.full.log "\$S3_RESULTS/aws_build.full.log" --only-show-errors || true
# 5. collect the Developer_CL.tar (the create-fpga-image input). Only upload if timing met (no *.VIOLATED.dcp).
ls -la "\$CLROOT/build/checkpoints/" "\$CLROOT/build/checkpoints/to_aws/" 2>&1 | aws s3 cp - "\$S3_RESULTS/checkpoints_listing.txt" --only-show-errors || true
TAR="\$(find "\$CLROOT/build/checkpoints" -name '*Developer_CL.tar' 2>/dev/null | head -1)"
VIOL="\$(find "\$CLROOT/build/checkpoints" -name '*post_route*VIOLATED*.dcp' 2>/dev/null | head -1)"
if [ -n "\$TAR" ] && [ -z "\$VIOL" ]; then
  echo "TIMING MET -> uploading \$TAR"
  aws s3 cp "\$TAR" "\$S3_RESULTS/to_aws/\$(basename "\$TAR")" --only-show-errors || true
else
  echo "NOT uploading tarball (TAR=\$TAR VIOLATED=\$VIOL) — timing not met or no tar"
fi
for r in \$(find "\$CLROOT/build/reports" -name "*.rpt" 2>/dev/null); do aws s3 cp "\$r" "\$S3_RESULTS/reports/\$(basename "\$r")" --only-show-errors 2>/dev/null || true; done
# TIMING DIAGNOSTIC: open the post-route DCP -> report_clocks + worst paths (decisive for the CDC).
DCP="\$(find "\$CLROOT/build/checkpoints" -name '*post_route*.dcp' 2>/dev/null | head -1)"
if [ -n "\$DCP" ]; then
  cat > /tmp/tdiag.tcl <<'TCL'
open_checkpoint DCPPATH
puts "===CLOCKS==="; report_clocks
puts "===SUMMARY==="; report_timing_summary -delay_type max -max_paths 1 -no_detailed_paths
puts "===WORST6==="; report_timing -max_paths 6 -nworst 1 -delay_type max -setup -input_pins
TCL
  sed -i "s#DCPPATH#\$DCP#" /tmp/tdiag.tcl
  vivado -mode batch -source /tmp/tdiag.tcl 2>&1 | grep -iE "===|clk_kernel|clk_main_a0|clk_out|WNS|TNS|Slack|Source:|Destination:|Source Clock|Destination Clock|Data Path Delay|Logic Levels|Path Group|Requirement|Period|^ *[0-9.]+ +ns|Waveform" | head -120 | aws s3 cp - "\$S3_RESULTS/TIMING_DIAG.txt" --only-show-errors || true
fi
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-s2afi},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3RES=$S3RES"; echo "BUCKET=$BUCKET"; } > "$STATE/build.env"
  echo "LAUNCHED DCP build $IID ($ITYPE) -> $S3RES"
}
cmd_status(){ source "$STATE/build.env"; echo -n "instance: "; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3RES/" 2>/dev/null; echo "--- to_aws ---"; aws_ s3 ls "$S3RES/to_aws/" 2>/dev/null || echo " (no tarball yet)"; }
cmd_afi(){
  source "$STATE/build.env"
  TAR="$(aws_ s3 ls "$S3RES/to_aws/" | grep -oE '[^ ]+Developer_CL.tar' | head -1)"
  [ -z "$TAR" ] && { echo "no Developer_CL.tar in $S3RES/to_aws/ yet"; exit 1; }
  echo "AFI from $S3RES/to_aws/$TAR"
  OUT="$(aws_ ec2 create-fpga-image --name "tiki-s2-xformer-$TAG" --description "SCALE-2 D=512 PC-xformer s2 kernel" \
      --input-storage-location "Bucket=$BUCKET,Key=$TAG/results/to_aws/$TAR" \
      --logs-storage-location "Bucket=$BUCKET,Key=$TAG/afi-logs" --output json)"
  echo "$OUT" | tee "$STATE/afi.json"
  FID="$(echo "$OUT" | grep -oE 'afi-[a-z0-9]+' | head -1)"; AGFI="$(echo "$OUT" | grep -oE 'agfi-[a-z0-9]+' | head -1)"
  { echo "FID=$FID"; echo "AGFI=$AGFI"; } >> "$STATE/build.env"; echo "FID=$FID AGFI=$AGFI"
}
cmd_afistatus(){ source "$STATE/build.env"; aws_ ec2 describe-fpga-images --fpga-image-ids "${FID:?run afi first}" --query 'FpgaImages[0].State.Code' --output text; echo "AGFI=${AGFI:-?}"; }
cmd_fetch(){ source "$STATE/build.env"; D="$PROJECT_DIR/results/predictive_coding/multi_fpga/s2_afi"; mkdir -p "$D"; aws_ s3 cp "$S3RES/" "$D/" --recursive --exclude "to_aws/*" --only-show-errors; ls -la "$D"; }
cmd_terminate(){ [ -f "$STATE/build.env" ] && { source "$STATE/build.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; }; }
case "${1:-}" in build) cmd_build;; status) cmd_status;; afi) cmd_afi;; afistatus) cmd_afistatus;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {build|status|afi|afistatus|fetch|terminate}";; esac
