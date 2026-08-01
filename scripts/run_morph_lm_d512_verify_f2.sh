#!/bin/bash
# run_morph_lm_d512_verify_f2.sh — A3 D=512 SILICON VERIFY on a fresh f2 box.
#
# Forked from run_morph_lm_a2_f2.sh (proven F2 mechanics: SDK setup, PCI lib
# build, fpga-load, S3 logging, self-terminate, NO SSH). Runs against the v3
# CLKGEN AFI (agfi-003a8a2f2b1ba7df7: D=512, two-bank 8MB staging, whole user
# design on clk_extra_a1=125MHz, WNS +0.050):
#   (1) build the CSIM golden lib ON-BOX (ML_D=512 + champion flags — the
#       #error in forge_buffer_morph_lm.hpp enforces the trio) and emit the
#       GOLDEN fused line for TT_FUSED_STEPS=30 from buffer_init.bin;
#   (2) build the PCI lib, fpga-load the AFI, TT_CLKGEN_BRINGUP=1 (MANDATORY:
#       clkgen AFIs boot with the whole 125MHz domain in reset — the S1
#       lesson; bridge.cpp polls MMCM_LOCK + clears SYS_RST), run the SAME
#       driver -> SILICON fused line; diff => BITEXACT / MISMATCH marker;
#   (3) on-fabric training curve: fresh-init fused kicks at 30 / 200 / 2000
#       steps (acc + wcrc + tok/s per kick, all fwd+bwd+signsgd on fabric).
# NEVER touches tiki-f2-silicon (launches its OWN f2.6xlarge).
#
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637
#   bash scripts/run_morph_lm_d512_verify_f2.sh {launch|status|fetch|terminate}
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-f2.6xlarge}"             # 1 FPGA; smallest F2
AMI="${AMI:-ami-092fc5deb8f3c0f7d}"                       # FPGA Developer AMI (fpga-* tools + SDK)
AGFI="${AGFI:-agfi-003a8a2f2b1ba7df7}"                    # cl_forge_morph_lm_d512 v3 clkgen (fused mode 3)
MAX_MINUTES="${MAX_MINUTES:-90}"                          # builds+load ~10m, kicks ~15m, big slack
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_d512verify_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first:"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET"; exit 1; }
  aws_ ec2 describe-fpga-images --filters "Name=fpga-image-global-id,Values=$AGFI" --query 'FpgaImages[0].State.Code' --output text 2>/dev/null | grep -qx available \
    || { echo "FATAL: $AGFI not available"; exit 1; }
  TAG="d512verify-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"
  S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== packaging src.tar.gz =="
  tar -C "$PROJECT_DIR" -czf "$STATE_DIR/src.tar.gz" \
    tikitorch/fpga/bridge.cpp \
    tikitorch/fpga/silicon/morph_lm_silicon.py \
    tikitorch/fpga/silicon/gen_buffer_init.py \
    tikitorch/fpga/silicon/cstdint_shim \
    tikitorch/hardware/hls/forge_kernels
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  UD="$STATE_DIR/user-data.sh"; render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> UD:"; cat "$UD"; return 0; fi
  echo "== launch $INSTANCE_TYPE ($AMI; load $AGFI; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-morph-d512verify},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qiE "instance profile|InsufficientInstanceCapacity|capacity" "$STATE_DIR/launch.err" && [ "$tries" -lt 8 ]; then echo "retry $tries"; sleep 15; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"; echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "AGFI=$AGFI"; } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} "tiki-d512verify deadline" || true
exec > /var/log/tiki-d512verify.log 2>&1
set -x; export HOME=/root
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"; AGFI="${AGFI}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-d512verify.log "\$S3_RESULTS/verify.log" --only-show-errors || true; }
trap up EXIT
( while true; do sleep 30; up; done ) &
echo "BOOT \$(date -u)"; hostname
die(){ echo "D512VERIFY-FAILED: \$1" | aws s3 cp - "\$S3_RESULTS/FAILED" --only-show-errors; up; shutdown -h now; exit 1; }
# ---- SDK runtime (fpga-* tools) ----
AF=""; for p in ~/aws-fpga /home/*/src/project_data/aws-fpga /home/*/aws-fpga /opt/aws-fpga; do [ -d "\$p/sdk" ] && AF="\$p" && break; done
[ -z "\$AF" ] && { AF=/opt/aws-fpga; git clone https://github.com/aws/aws-fpga.git "\$AF"; }
cd "\$AF" && source sdk_setup.sh || true; SDK="\$AF/sdk/userspace"
FC="\$(command -v fpga-clear-local-image)"; FL="\$(command -v fpga-load-local-image)"; FD="\$(command -v fpga-describe-local-image)"
echo "TOOLS: \$FD / \$FL / \$FC"; up
# ---- fetch + unpack src ----
mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" s.tgz && break; sleep 8; done; tar xzf s.tgz || die "src"
export ML_D=512
# ---- (1) CSIM GOLDEN: host build of the SAME kernel at champion flags ----
g++ -O2 -std=c++17 -fPIC -shared -DCSIM_ONLY -DTT_KERNEL_MORPH_LM \
    -DML_D=512 -DTT_RESIDENT_W -DTT_ONCHIP_GRAD -DTT_URAM_W \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/bridge.cpp \
    /opt/tiki/tikitorch/hardware/hls/forge_kernels/forge_train_step_morph_lm.cpp \
    -o /tmp/_libcsim_d512.so 2>&1 | tail -5
[ -f /tmp/_libcsim_d512.so ] || die "csim-build"
python3 /opt/tiki/tikitorch/fpga/silicon/gen_buffer_init.py /tmp/buffer_init.bin || die "gen-init"
DRV=/opt/tiki/tikitorch/fpga/silicon/morph_lm_silicon.py
TT_FUSED_STEPS=30 TIKITORCH_FPGA_LIB=/tmp/_libcsim_d512.so python3 "\$DRV" /tmp/buffer_init.bin > /tmp/golden_fused.jsonl || die "golden-run"
cat /tmp/golden_fused.jsonl; aws s3 cp /tmp/golden_fused.jsonl "\$S3_RESULTS/golden_fused.jsonl" --only-show-errors
# ---- (2) SILICON: PCI lib + AFI load + clkgen bring-up + fused KAT ----
g++ -O2 -std=c++17 -fPIC -shared -DTT_FPGA_PCI \
    -I"\$SDK/include" \
    -I/opt/tiki/tikitorch/hardware/hls/forge_kernels \
    -I/opt/tiki/tikitorch/fpga/silicon/cstdint_shim \
    /opt/tiki/tikitorch/fpga/bridge.cpp \
    -lfpga_mgmt -o /tmp/_libtikifpga_pci.so 2>&1 | tee /tmp/build_lib.txt
[ -f /tmp/_libtikifpga_pci.so ] || die "lib-build"
sudo \$FC -S 0 || true
sudo \$FL -S 0 -I "\$AGFI" -F || die "load-afi"
sleep 5; sudo \$FD -S 0 | tee /tmp/img.txt; up
export TIKITORCH_FPGA_LIB=/tmp/_libtikifpga_pci.so TT_CLKGEN_BRINGUP=1 TT_FUSED_STEPS=30
sudo -E python3 "\$DRV" /tmp/buffer_init.bin > /tmp/silicon_fused.jsonl 2>/tmp/silicon_fused.err || die "silicon-kat-run"
cat /tmp/silicon_fused.jsonl /tmp/silicon_fused.err
aws s3 cp /tmp/silicon_fused.jsonl "\$S3_RESULTS/silicon_fused.jsonl" --only-show-errors
aws s3 cp /tmp/silicon_fused.err   "\$S3_RESULTS/silicon_fused.err"   --only-show-errors
if diff -q /tmp/golden_fused.jsonl /tmp/silicon_fused.jsonl; then
  echo "BITEXACT: d512 silicon fused(30) == CSIM golden" | aws s3 cp - "\$S3_RESULTS/BITEXACT" --only-show-errors
else
  { echo "MISMATCH golden vs silicon:"; diff /tmp/golden_fused.jsonl /tmp/silicon_fused.jsonl; } | aws s3 cp - "\$S3_RESULTS/MISMATCH" --only-show-errors
fi
up
# ---- (3) on-fabric training curve: fresh init per kick; 30/200/2000 steps ----
for N in 30 200 2000; do
  TT_FUSED_STEPS=\$N sudo -E python3 "\$DRV" /tmp/buffer_init.bin > /tmp/train_\$N.jsonl 2>/tmp/train_\$N.err || die "train-\$N"
  cat /tmp/train_\$N.jsonl /tmp/train_\$N.err
  aws s3 cp /tmp/train_\$N.jsonl "\$S3_RESULTS/train_\$N.jsonl" --only-show-errors
  aws s3 cp /tmp/train_\$N.err   "\$S3_RESULTS/train_\$N.err"   --only-show-errors
done
echo "done \$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors
up; shutdown -h now
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" 2>/dev/null || echo "(nothing yet)"
  aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1 && echo "DONE present" || { echo "-- verify.log tail --"; aws_ s3 cp "$S3_RESULTS/verify.log" - 2>/dev/null | tail -30; }; }
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/fpga_silicon/d512_verify"; mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --only-show-errors; ls -la "$DEST"
  echo "-- verdict --"; cat "$DEST/BITEXACT" 2>/dev/null || cat "$DEST/MISMATCH" 2>/dev/null || echo "(no verdict marker)"; }
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text 2>&1 | head -1 || true; rm -f "$STATE_FILE"; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;;
  *) echo "usage: AWS_PROFILE=... $0 {launch|status|fetch|terminate}";; esac
