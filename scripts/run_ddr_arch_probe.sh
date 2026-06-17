#!/bin/bash
# run_ddr_arch_probe.sh — DDR-rebuild RESEARCH probe (cheap z1d, no F2): dump the F2 on-card-memory CL
# architecture so we can rewire cl_forge_muzero to read on-card DDR/HBM (drivable; no host IOMMU) instead of
# host-PCIM. Dumps to S3: the shell DDR/HBM port family in cl_ports.vh, the HDK sh_ddr/HBM controller module,
# the cl_dram_hbm_dma + cl_mem_perf reference CL design (.sv/.vh), their gen_ip + constraints, and the DDR/HBM
# unused-template includes. From these we adapt morph-LM's DMA-PCIS-crossbar .sv (DMA-PCIS + kernel m_axi ->
# on-card memory, replacing URAM). Same robust pattern (S3+IAM+self-terminate, no SSH).
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-z1d.2xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
MAX_MINUTES="${MAX_MINUTES:-30}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_ddrprobe_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="ddrprobe-$(date -u +%Y%m%dT%H%M%SZ)"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-ddr-probe},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ] && { sleep 10; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_results="$1"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-ddrprobe.log 2>&1
set -x; export HOME=/root
S3_RESULTS="${s3_results}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-ddrprobe.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
( command -v git-lfs >/dev/null 2>&1 || apt-get update -y && apt-get install -y git-lfs ) || true
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/hdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && git lfs install && git lfs pull >/dev/null 2>&1 || true
source hdk_setup.sh || true
echo "HDK_SHELL_DIR=\$HDK_SHELL_DIR"; OUT=/tmp/ddr; mkdir -p \$OUT
# (1) on-card memory ports in cl_ports.vh (DDR: *_ddr_*, *DDR*; HBM: *hbm*) + the sh_ddr / hbm includes
grep -niE "ddr|hbm|cl_sh_ddr|sh_cl_ddr|sh_ddr|stat_" "\$HDK_SHELL_DIR/design/interfaces/cl_ports.vh" 2>/dev/null > \$OUT/cl_ports_mem.txt || true
ls "\$HDK_SHELL_DIR/design/interfaces" > \$OUT/interfaces_ls.txt 2>&1
cp "\$HDK_SHELL_DIR"/design/interfaces/*ddr*.inc "\$HDK_SHELL_DIR"/design/interfaces/*hbm*.inc \$OUT/ 2>/dev/null || true
# (2) the HDK sh_ddr / DDR-HBM controller modules the CL instantiates
for n in sh_ddr cl_sh_ddr sh_hbm; do find "\$AF/hdk" -name "\$n.sv" -o -name "\$n.v" 2>/dev/null | head -2 | while read f; do cp "\$f" "\$OUT/mod_\$(basename \$f)"; done; done
find "\$AF/hdk" -iname "*sh_ddr*" -o -iname "*ddr4*sv" 2>/dev/null | head -20 > \$OUT/ddr_module_index.txt
# (3) the reference CL designs + FULL build setup (the proven DDR flow: synth_cl tcl, encrypt/IP, sh_ddr inclusion, filelists)
for d in cl_dram_hbm_dma cl_mem_perf; do
  D="\$AF/hdk/cl/examples/\$d"
  echo "== \$d design ==" >> \$OUT/ref_index.txt; ls -R "\$D/design" >> \$OUT/ref_index.txt 2>&1
  echo "== \$d build ==" >> \$OUT/ref_index.txt; ls -R "\$D/build" >> \$OUT/ref_index.txt 2>&1
  for f in \$(find "\$D/design" -name "*.sv" 2>/dev/null | head -16); do cp "\$f" "\$OUT/\${d}_\$(basename \$f)" 2>/dev/null; done
  # the WHOLE build/scripts (synth_cl_*.tcl, encrypt.tcl, *.f filelists, gen IP) + constraints + any IP cfg
  for f in \$(find "\$D/build" \( -name "*.tcl" -o -name "*.f" -o -name "*.xdc" -o -name "cl_*.vh" -o -name "encrypt*" \) 2>/dev/null | head -40); do cp "\$f" "\$OUT/\${d}_build_\$(basename \$f)" 2>/dev/null; done
  # how sh_ddr is referenced (read_verilog? read_ip? a filelist? the encrypt list?)
  grep -rniE "sh_ddr|ddr4|encrypt|read_verilog|read_ip|add_files|aws_v1|\.f\b" "\$D/build/scripts" 2>/dev/null | head -40 >> \$OUT/\${d}_sh_ddr_refs.txt 2>&1
done
# the canonical sh_ddr.f / filelist the HDK uses to compile sh_ddr (+ its DDR4 IP)
find "\$AF/hdk" -name "*.f" 2>/dev/null | grep -iE "ddr|sh_ddr|cl" | head > \$OUT/hdk_filelists.txt
find "\$AF/hdk/common/shell_stable" -iname "*ddr*" \( -name "*.f" -o -name "*.tcl" -o -name "*.xci" \) 2>/dev/null | head -20 >> \$OUT/hdk_filelists.txt
# (4) how the host stages into on-card DDR/HBM (the fpga_dma DDR write path the test uses)
grep -rniE "fpga_dma|0x[0-9a-f]+|XDMA|burst_write|ddr|hbm|init_ddr|sv_map" "\$AF/hdk/cl/examples/cl_mem_perf/software" 2>/dev/null | grep -iE "fpga_dma|burst|ddr|hbm|xdma" | head -40 > \$OUT/host_ddr_stage.txt || true
aws s3 cp \$OUT/ "\$S3_RESULTS/" --recursive --only-show-errors || true
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/results/fpga_silicon/ddr_arch"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
