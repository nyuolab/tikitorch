#!/bin/bash
# run_mp_csynth_f2.sh — Vitis HLS csynth gate for forge_pc_xformer_mp_block on EC2 (z1d, FPGA-dev AMI, no F2).
# Cloned from the int-kernel csynth that produced forge_pc_xformer_block_csynth.rpt.
#   AWS_PROFILE=... bash scripts/run_mp_csynth_f2.sh launch | status | fetch | terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"; ITYPE="${ITYPE:-z1d.2xlarge}"; ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; STATE="$PROJECT_DIR/.tiki_mp_csynth_state"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }
cmd_launch(){
  [ -f "$STATE/run.env" ] && { echo "state exists; terminate first"; cat "$STATE/run.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="mpcsynth-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE/src.tar.gz" HEAD
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +180 "tiki deadline" || true
exec > /var/log/tiki-mpcsynth.log 2>&1
set -x; echo "BOOT \$(date -u)"; hostname
S3RES="${S3RES}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-mpcsynth.log "\$S3RES/boot.log" --only-show-errors || true; }
trap up EXIT
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "${S3SRC}" src.tgz && break; sleep 10; done
tar xzf src.tgz; up
for s in /opt/Xilinx/Vitis_HLS/*/settings64.sh /opt/Xilinx/Vitis/*/settings64.sh; do [ -f "\$s" ] && source "\$s"; done
VHLS="\$(command -v vitis_hls || ls /opt/Xilinx/Vitis_HLS/*/bin/vitis_hls 2>/dev/null | head -1)"
echo "VITIS_HLS=\$VHLS"; up
cd /opt/tiki/tikitorch/hardware/hls/forge_kernels
echo "=== csynth mp START \$(date -u) ==="
"\$VHLS" -f synth_pc_xformer_mp_block.tcl 2>&1 | tail -80
echo "=== csynth mp DONE \$(date -u) ==="
R="\$(find . -name 'forge_pc_xformer_mp_block_csynth.rpt' 2>/dev/null | head -1)"
[ -n "\$R" ] && aws s3 cp "\$R" "\$S3RES/mp_csynth.rpt" --only-show-errors || echo "NO REPORT"
aws s3 cp vitis_hls.log "\$S3RES/mp_vitis_hls.log" --only-show-errors || true
up; echo "done \$(date -u)" | aws s3 cp - "\$S3RES/DONE" --only-show-errors || true
shutdown -h now
UDEOF
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=120,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-mpcsynth},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3RES=$S3RES"; } > "$STATE/run.env"
  echo "LAUNCHED $IID ($ITYPE) -> $S3RES"
}
cmd_status(){ source "$STATE/run.env"; echo -n "instance: "; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; echo "S3:"; aws_ s3 ls "$S3RES/" 2>/dev/null || echo " (nothing yet)"; }
cmd_fetch(){ source "$STATE/run.env"; D="$PROJECT_DIR/results/predictive_coding/multi_fpga/mp_csynth"; mkdir -p "$D"; aws_ s3 cp "$S3RES/" "$D/" --recursive --only-show-errors; ls -la "$D"; }
cmd_terminate(){ [ -f "$STATE/run.env" ] && { source "$STATE/run.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE/run.env"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
