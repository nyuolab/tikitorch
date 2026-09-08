#!/bin/bash
# run_s1_infer_f2.sh — GENERATE text on real F2 with the COHERENT strictly-multiplier-free (DSP=0)
# tropical-transformer LM (kernel-v2, D=128/L4, bit-exact 2.42 bpc). Loads the kernel-v2 AGFI, builds
# forge_s1_infer_f2.c (KV2/D=128/L4 defines so S1_BUFLEN/offsets match the AFI + the export buffer),
# downloads the exported deploy buffer + vocab + val ids from S3, then autoregressively generates
# (write tokens -> mode-0 kick -> read logits -> sample -> slide) and prints the on-silicon bpc.
# Cloned from run_s1_train_f2.sh: same AGFI guard / f2.6xlarge / CLKGEN -DHAVE_FPGA_CLKGEN=0 / self-term.
# DEDICATED BOX (see memory f2-instance-isolation): tags tiki-s1-infer-f2; NEVER touches tiki-f2-silicon
# or tiki-mp-xformer-shard (the PC instance's boxes).
#   export AWS_PROFILE=...; AGFI=agfi-... bash scripts/run_s1_infer_f2.sh launch
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"; AMI="${AMI:-ami-092fc5deb8f3c0f7d}"
AGFI="${AGFI:?set AGFI}"; GEN="${GEN:-400}"; TEMP="${TEMP:-128}"; NWIN="${NWIN:-60}"; SEED="${SEED:-1234}"
MAX_MINUTES="${MAX_MINUTES:-40}"; ROLE="${ROLE:-tiki-sweep-runner}"
S3_DEPLOY="${S3_DEPLOY:-s3://tiki-sweep-739966866637/s1kv2-deploy}"
KDEFS="-DS1_KV2=1 -DS1_D=128 -DS1_LOG2_D=7 -DS1_H=4 -DS1_DH=32 -DS1_FFN=512 -DS1_N_LAYERS=4 -DS1_FWD_ONLY=1 -DS1_ATTN_HARD=1 -DS1_HEAD_MODE=1 -DS1_POS_EMBED=1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_s1_infer_f2_state"; mkdir -p "$STATE_DIR"; STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="s1infer-$(date -u +%Y%m%dT%H%M%SZ)"; S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$UD"; return 0; }
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-s1-infer-f2},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1)); grep -qiE "instance profile|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ] && { sleep 15; continue; }
    echo "FATAL:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> $S3_RESULTS"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-s1infer.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"; S3_DEPLOY="${S3_DEPLOY}"
GEN="${GEN}"; TEMP="${TEMP}"; NWIN="${NWIN}"; SEED="${SEED}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-s1infer.log "\$S3_RESULTS/s1infer.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"; hostname
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
sudo \$FC -S 0 || true; sudo \$FL -S 0 -I "\$AGFI"; sleep 4
sudo \$FD -S 0 2>&1 | tee /tmp/img.txt; up
mkdir -p /opt/tiki && cd /opt/tiki && for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done && tar xzf s.tgz
# fetch the exported deploy buffer + vocab + val ids
aws s3 cp "\$S3_DEPLOY/s1_kv2_deploy.npy" /tmp/buf.npy
aws s3 cp "\$S3_DEPLOY/s1_vocab.txt" /tmp/vocab.txt
aws s3 cp "\$S3_DEPLOY/val_ids.bin" /tmp/val.bin || true
# build the KV2/D=128/L4 inference driver (CLKGEN fallback -DHAVE_FPGA_CLKGEN=0; cstdint shim for the C++ header)
gcc -O2 -std=gnu11 -DHAVE_FPGA_CLKGEN=0 ${KDEFS} -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/silicon/forge_s1_infer_f2.c -lfpga_mgmt -lm -o /tmp/s1_infer 2>&1 | tee /tmp/build.txt
up
# generate on silicon + on-fabric bpc:  s1_infer <slot> <buf.npy> <gen> <valbin> <nwin> <temp> <seed> <vocab.txt>
( sudo /tmp/s1_infer 0 /tmp/buf.npy "\$GEN" /tmp/val.bin "\$NWIN" "\$TEMP" "\$SEED" /tmp/vocab.txt 2>&1 | tee /tmp/gen.txt; echo EXIT=\${PIPESTATUS[0]} >> /tmp/gen.txt ) || true
for f in img build gen; do aws s3 cp /tmp/\$f.txt "\$S3_RESULTS/\$f.txt" --only-show-errors || true; done
echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
up; shutdown -h now
UDEOF
}
cmd_status(){ source "$STATE_FILE"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null; aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null; }
cmd_fetch(){ source "$STATE_FILE"; local Dd="$PROJECT_DIR/manuscript/results/lm_shakespeare/onfabric_gen"; mkdir -p "$Dd"; aws_ s3 cp "$S3_RESULTS/" "$Dd/" --recursive --only-show-errors; ls -1 "$Dd"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] && { source "$STATE_FILE"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1; rm -f "$STATE_FILE"; } || true; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: AGFI=agfi-... GEN=400 TEMP=128 $0 {launch|status|fetch|terminate}";; esac
