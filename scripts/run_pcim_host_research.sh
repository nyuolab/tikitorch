#!/bin/bash
# run_pcim_host_research.sh — Stage-3b RESEARCH probe (cheap z1d, no F2): resolve the ONE unknown for driving
# our PCIM-to-host-DRAM AFI from the host — how the host hands the CL a PCIM-reachable host buffer. Dumps to
# S3: the AWS HDK examples that master host memory over PCIM (cl_mem_perf / cl_dram_hbm_dma SW test sources),
# the fpga_dma / fpga_pci API headers, any DMA helper, AND regenerates forge_train_step_muzero's HLS control
# RTL so we can confirm the s_axilite register offsets (vs bridge.cpp's hardcoded morph-LM offsets). No F2
# needed — this is reading SDK source + an HLS csynth. Same robust pattern (S3+IAM+self-terminate, no SSH).
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_pcim_host_research.sh launch ; ... status ; ... fetch ; ... terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
MAX_MINUTES="${MAX_MINUTES:-40}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_pcimresearch_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="pcimresearch-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pcim-research},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ] && { sleep 10; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-pcimresearch.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-pcimresearch.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
( command -v git-lfs >/dev/null 2>&1 || apt-get update -y && apt-get install -y git-lfs ) || true
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
echo "AWS_FPGA=\$AF"; OUT=/tmp/pcim; mkdir -p \$OUT
# (1) AWS HDK examples that MASTER host memory over PCIM + the SW test host code
{ echo "== examples =="; ls "\$AF/hdk/cl/examples";
  echo "== find PCIM host/test sources =="; find "\$AF/hdk/cl/examples" -path "*software*" \( -name "*.c" -o -name "*.h" \) 2>/dev/null;
} > \$OUT/example_index.txt 2>&1
for d in cl_mem_perf cl_dram_hbm_dma cl_sde; do
  for f in \$(find "\$AF/hdk/cl/examples/\$d/software" \( -name "*.c" -o -name "*.h" \) 2>/dev/null | head -8); do
    cp "\$f" "\$OUT/ex_\${d}_\$(basename "\$f")" 2>/dev/null || true; done
done
# (2) fpga_dma / fpga_pci API + any DMA helper that gives a host buffer the CL can master via PCIM
for h in fpga_dma.h fpga_pci.h fpga_mgmt.h; do find "\$AF" -name "\$h" -exec cp {} "\$OUT/api_\$h" \; 2>/dev/null; done
grep -rniE "pcim|cl_sh_pcim|host.*addr|XDMA|/dev/xdma|fpga_dma_burst|hugepage|pagemap|map_buffer|pci_peer|buffer.*phys" "\$AF/sdk" 2>/dev/null | grep -iE "pcim|xdma|hugepage|pagemap|dma_burst|phys" | head -80 > \$OUT/pcim_mechanism_grep.txt
# the sv DMA / fpga_dma example app (the canonical CL<->host DMA driver)
find "\$AF/sdk" -path "*fpga_dma*" \( -name "*.c" -o -name "*.h" \) -exec sh -c 'cp "\$1" "/tmp/pcim/dma_\$(basename \$1)"' _ {} \; 2>/dev/null || true
# (3) regenerate the forward kernel HLS -> control_s_axi.v (the real s_axilite register offsets)
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null|head -1)"
"\$VHLS" -f synth_muzero_chess.tcl 2>&1 | tail -5 || true
CS="\$(find forge_muzero_chess_proj -name '*control_s_axi.v' 2>/dev/null | head -1)"
echo "CONTROL_S_AXI=\$CS"; cp "\$CS" \$OUT/forge_train_step_muzero_control_s_axi.v 2>/dev/null || true
# the register offsets are in the control_s_axi.v as localparams / address comments
grep -nE "ADDR_|_DATA|_CTRL|localparam|0x[0-9a-fA-F]+|offset" "\$CS" 2>/dev/null | head -80 > \$OUT/control_offsets.txt || true
aws s3 cp \$OUT/ "\$S3_RESULTS/" --recursive --only-show-errors || true
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/results/fpga_silicon/pcim_research"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
