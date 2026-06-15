#!/bin/bash
# run_muzero_afi_f2_dynpower.sh — Stage 3b: DYNAMIC (per-inference) power of the forward MuZero chess AFI on
# real F2. Reserves a 1 GB hugepage, loads the AGFI, builds muzero_pcim_power.c (drives the kernel over
# PCIM-to-host-DRAM), runs a kick loop, and samples Vccint + the PCIM transaction counters BEFORE / DURING /
# AFTER so we get (a) dynamic vs idle power and (b) proof the CL actually streamed host DRAM (pcim-read-count
# delta). Robust one-shot (S3+IAM+self-terminate, no SSH); own f2 instance.
#   export AWS_PROFILE=...; AGFI=agfi-0b2c228a17e516e0a bash scripts/run_muzero_afi_f2_dynpower.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; NKICKS="${NKICKS:-4000}"; MAX_MINUTES="${MAX_MINUTES:-35}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_dynpower_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="dynpower-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-muzero-dynpower},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { sleep 15; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-dynpower.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; NKICKS="${NKICKS}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-dynpower.log "\$S3_RESULTS/dynpower.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
# (1) reserve 1 GB hugepages (fresh boot = low fragmentation) for the contiguous host buffer the CL masters
echo 2 > /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || true
HP1G=\$(cat /sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages 2>/dev/null || echo 0)
echo "1GB hugepages reserved: \$HP1G"
# (2) SDK runtime + load AGFI
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FD="\$(command -v fpga-describe-local-image)"; FL="\$(command -v fpga-load-local-image)"; FC="\$(command -v fpga-clear-local-image)"
sudo \$FC -S 0 || true; sudo \$FL -S 0 -I "\$AGFI"; sleep 4
echo "== IDLE metrics (pre) =="; sudo \$FD -S 0 -M 2>&1 | tee /tmp/m_idle.txt; up
# (3) build the PCIM power driver
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
HLS=/opt/tiki/tikitorch/hardware/hls
g++ -O2 -std=c++17 -DMZ_OBS_OVERRIDE=769 -DMZ_L_OVERRIDE=64 -DMZ_A_OVERRIDE=4096 \
    -I"\$SDK/include" -I"\$HLS/forge_kernels" -I"\$HLS" \
    /opt/tiki/tikitorch/fpga/silicon/muzero_pcim_power.c -lfpga_mgmt -o /tmp/muzero_pcim_power 2>&1 | tee /tmp/build.txt
up
# (4) run the driver in the background; sample power + PCIM counters DURING the kick loop
( sudo /tmp/muzero_pcim_power 0 "\$NKICKS" 2>&1 | tee /tmp/driver.txt; echo DRIVER_EXIT=\${PIPESTATUS[0]} >> /tmp/driver.txt ) &
DRV=\$!
for n in \$(seq 1 14); do echo "=== during sample \$n \$(date -u) ==="; sudo \$FD -S 0 -M 2>&1 | grep -iE "Vccint|Last measured|Average|Max measured|pcim-read-count|pcim-write-count|pcim-range-error=|pcim-axi-protocol-error="; sleep 2; done | tee /tmp/m_during.txt
wait \$DRV || true
echo "== FINAL metrics (post) =="; sudo \$FD -S 0 -M 2>&1 | tee /tmp/m_post.txt
# (5) collect
for f in m_idle m_during m_post driver build; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local D="$PROJECT_DIR/results/fpga_silicon/dynpower"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... $0 {launch|status|fetch|terminate}";; esac
