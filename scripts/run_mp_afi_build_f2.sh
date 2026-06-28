#!/bin/bash
# run_mp_afi_build_f2.sh — build the AFI for forge_pc_xformer_mp_block (BF16 mixed-precision PC transformer kernel).
# Adapted from the proven int-kernel AFI build (.tiki_xformer_afibuild_state): HLS the mp kernel -> RTL, scaffold the
# cl_forge_pc_xformer CL (now instantiating forge_pc_xformer_mp_block), gen_ip, aws_build_dcp_from_cl.py -> DCP tarball
# in S3, then create-fpga-image -> AGFI.
#   AWS_PROFILE=... bash scripts/run_mp_afi_build_f2.sh build      # launch the DCP build (~3-5h, self-terminates)
#   AWS_PROFILE=... bash scripts/run_mp_afi_build_f2.sh status     # poll the build
#   AWS_PROFILE=... bash scripts/run_mp_afi_build_f2.sh afi        # create-fpga-image from the tarball (after build DONE)
#   AWS_PROFILE=... bash scripts/run_mp_afi_build_f2.sh afistatus  # poll AFI -> prints AGFI when available
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"; ITYPE="${ITYPE:-z1d.6xlarge}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; STATE="$PROJECT_DIR/.tiki_mp_afi_state"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_build(){
  [ -f "$STATE/build.env" ] && { echo "build state exists; terminate/clear first"; cat "$STATE/build.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="mpafi-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE/src.tar.gz" HEAD
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +330 "tiki-mpafi deadline" || true
exec > /var/log/tiki-mpafi.log 2>&1
set -x
S3_SRC="${S3SRC}"; S3_RESULTS="${S3RES}"
CL="cl_forge_pc_xformer"; PKG_REL="tikitorch/hardware/hdk/cl_forge_pc_xformer"
KERNEL_TCL="synth_pc_xformer_mp_block.tcl"; KERNEL_PROJ="forge_pc_xformer_mp_block_proj"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-mpafi.log "\$S3_RESULTS/build.log" --only-show-errors || true; }
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
{ echo "== aws_build_dcp_from_cl.py =="; find "\$AF" -name aws_build_dcp_from_cl.py 2>/dev/null;
  echo "== CL templates =="; ls -d "\$AF"/hdk/cl/examples/* "\$AF"/hdk/cl/developer_designs/* 2>/dev/null;
  echo "== vivado =="; command -v vivado; vivado -version 2>&1 | head -2; } > /tmp/diag.txt 2>&1
aws s3 cp /tmp/diag.txt "\$S3_RESULTS/diag.txt" --only-show-errors || true
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz
# 1. HLS the mp kernel -> RTL
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "=== HLS \$KERNEL_TCL START \$(date -u) ==="; "\$VHLS" -f "\$KERNEL_TCL" 2>&1 | tail -30
KVDIR="/opt/tiki/tikitorch/hardware/hls/forge_kernels/\$KERNEL_PROJ/solution1/syn/verilog"
ls -la "\$KVDIR" | head; ls "\$KVDIR" | wc -l; up
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
{ echo "== CL design =="; ls -la "\$CLROOT/design"; echo "== build/scripts =="; ls -la "\$CLROOT/build/scripts"; } | aws s3 cp - "\$S3_RESULTS/cl_layout.txt" --only-show-errors || true
up
# 3. gen IP
echo "=== gen_ip START \$(date -u) ==="; vivado -mode batch -source "\$CLROOT/build/scripts/gen_ip.tcl" -tclargs "\$CLROOT/design/ip" 2>&1 | tail -30
ls -la "\$CLROOT/design/ip" 2>&1 | head -20; up
# 4. synth + P&R
export CL_DIR="\$CLROOT"; BUILD_PY="\$(find "\$CLROOT/build" -name aws_build_dcp_from_cl.py 2>/dev/null | head -1)"
echo "BUILD_PY=\$BUILD_PY"; chmod +x "\$BUILD_PY" 2>/dev/null; cd "\$(dirname "\$BUILD_PY")"
echo "=== aws_build_dcp_from_cl START \$(date -u) ==="
./aws_build_dcp_from_cl.py --cl "\$CL" --no-encrypt > /tmp/aws_build.full.log 2>&1; echo "BUILD_RC=\$?"
tail -120 /tmp/aws_build.full.log; aws s3 cp /tmp/aws_build.full.log "\$S3_RESULTS/aws_build.full.log" --only-show-errors || true
# 5. collect the Developer_CL.tar (the create-fpga-image input)
ls -la "\$CLROOT/build/checkpoints/" "\$CLROOT/build/checkpoints/to_aws/" 2>&1 | aws s3 cp - "\$S3_RESULTS/checkpoints_listing.txt" --only-show-errors || true
aws s3 cp "\$CLROOT/build/checkpoints/to_aws/" "\$S3_RESULTS/to_aws/" --recursive --exclude "*" --include "*Developer_CL.tar" --only-show-errors 2>/dev/null || true
for r in \$(find "\$CLROOT/build" -name "*timing_summary*" 2>/dev/null); do aws s3 cp "\$r" "\$S3_RESULTS/reports/\$(basename "\$r")" --only-show-errors 2>/dev/null || true; done
echo "rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=300,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-mpafi},{Key=tiki-sweep,Value=$TAG}]" \
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
  OUT="$(aws_ ec2 create-fpga-image --name "tiki-mp-xformer-$TAG" --description "BF16 PC-xformer mp kernel" \
      --input-storage-location "Bucket=$BUCKET,Key=$TAG/results/to_aws/$TAR" \
      --logs-storage-location "Bucket=$BUCKET,Key=$TAG/afi-logs" --output json)"
  echo "$OUT" | tee "$STATE/afi.json"
  FID="$(echo "$OUT" | grep -oE 'afi-[a-z0-9]+' | head -1)"; AGFI="$(echo "$OUT" | grep -oE 'agfi-[a-z0-9]+' | head -1)"
  { echo "FID=$FID"; echo "AGFI=$AGFI"; } >> "$STATE/build.env"; echo "FID=$FID AGFI=$AGFI"
}
cmd_afistatus(){ source "$STATE/build.env"; aws_ ec2 describe-fpga-images --fpga-image-ids "${FID:?run afi first}" --query 'FpgaImages[0].State.Code' --output text; echo "AGFI=${AGFI:-?}"; }
cmd_fetch(){ source "$STATE/build.env"; D="$PROJECT_DIR/results/predictive_coding/multi_fpga/mp_afi"; mkdir -p "$D"; aws_ s3 cp "$S3RES/" "$D/" --recursive --exclude "to_aws/*" --only-show-errors; ls -la "$D"; }
cmd_terminate(){ [ -f "$STATE/build.env" ] && { source "$STATE/build.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; }; }
case "${1:-}" in build) cmd_build;; status) cmd_status;; afi) cmd_afi;; afistatus) cmd_afistatus;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {build|status|afi|afistatus|fetch|terminate}";; esac
