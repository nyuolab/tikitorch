#!/bin/bash
# run_nanogpt_baseline_ec2.sh -- launch a self-terminating GPU box (g5.xlarge / A10G, DLAMI) that trains the
# MATCHED float-BP nanoGPT baseline (nanogpt_bins_baseline.py) on the s2_bpe8192 bins and uploads the jsonl +
# log to S3. The design-doc gate for the 30M silicon run: silicon converged val-bpb within ~0.3-0.5 nat of this.
#   AWS_PROFILE=... bash scripts/pc_science/run_nanogpt_baseline_ec2.sh launch   # then status / fetch / terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-g5.xlarge}"; AMI="${AMI:-ami-05603a42e5254c4bb}"; ROLE="${ROLE:-tiki-sweep-runner}"   # DLAMI PyTorch 2.7 / Ubuntu 22.04
BINS_S3="${BINS_S3:-s3://tiki-sweep-739966866637/s2_bpe8192_bins}"
STEPS="${STEPS:-25000}"; BATCH="${BATCH:-8}"; SEQ="${SEQ:-256}"; EVAL_EVERY="${EVAL_EVERY:-500}"
D="${D:-512}"; HH="${HH:-8}"; LL="${LL:-8}"; FFN="${FFN:-2048}"; VOCAB="${VOCAB:-8192}"
LR="${LR:-6e-4}"; WARMUP="${WARMUP:-500}"; DROPOUT="${DROPOUT:-0.1}"; BYTES_PER_TOKEN="${BYTES_PER_TOKEN:-3.114}"
MAX_MINUTES="${MAX_MINUTES:-300}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"; STATE="${STATE_DIR:-$PROJECT_DIR/.tiki_nanogpt_base}"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE/run.env" ] && { echo "state exists; terminate first"; cat "$STATE/run.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="nanogpt-base-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  # stage = git archive HEAD + working-tree pc_science overlay (same recipe as run_exp3_ec2.sh)
  STG="$STATE/stage"; rm -rf "$STG"; mkdir -p "$STG"
  git -C "$PROJECT_DIR" archive HEAD | tar -x -C "$STG"
  mkdir -p "$STG/scripts/pc_science"; cp "$PROJECT_DIR"/scripts/pc_science/*.py "$STG/scripts/pc_science/" 2>/dev/null || true
  tar -C "$STG" -czf "$STATE/src.tar.gz" .
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-nanogpt.log 2>&1
set -x; export HOME=/root
S3_RESULTS="${S3RES}"
up(){ aws s3 cp /var/log/tiki-nanogpt.log "\$S3_RESULTS/run.log" --only-show-errors || true; aws s3 cp /tmp/out "\$S3_RESULTS/" --recursive --exclude '*' --include '*.jsonl' --only-show-errors || true; }
trap up EXIT
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; nvidia-smi -L || true
# DLAMI torch interpreter (venv or conda), fallback system python3
PY=""; for c in /opt/pytorch/bin/python /opt/conda/bin/python /usr/bin/python3; do [ -x "\$c" ] && "\$c" -c "import torch" 2>/dev/null && PY="\$c" && break; done
[ -z "\$PY" ] && { echo "FATAL: no torch python"; echo fail | aws s3 cp - "\$S3_RESULTS/DONE" || true; shutdown -h now; }
"\$PY" -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"
mkdir -p /opt/tiki /tmp/bins /tmp/out && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "${S3SRC}" s.tgz && break; sleep 8; done && tar xzf s.tgz
aws s3 cp "${BINS_S3}/train.u16" /tmp/bins/train.u16 --only-show-errors
aws s3 cp "${BINS_S3}/val.u16"   /tmp/bins/val.u16   --only-show-errors
up
TOKBIN=/tmp/bins/train.u16 VALBIN=/tmp/bins/val.u16 OUT=/tmp/out/baseline.jsonl \
D=${D} H=${HH} L=${LL} FFN=${FFN} SEQ=${SEQ} VOCAB=${VOCAB} BATCH=${BATCH} STEPS=${STEPS} \
LR=${LR} WARMUP=${WARMUP} DROPOUT=${DROPOUT} EVAL_EVERY=${EVAL_EVERY} BYTES_PER_TOKEN=${BYTES_PER_TOKEN} \
  "\$PY" scripts/pc_science/nanogpt_bins_baseline.py
up; echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
sync; shutdown -h now
UDEOF
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$STATE/user-data.sh"; return 0; }
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-nanogpt-base},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_RESULTS=$S3RES"; } > "$STATE/run.env"
  echo "LAUNCHED $IID ($ITYPE) -> $S3RES"
}
cmd_status(){ source "$STATE/run.env"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; aws_ s3 cp "$S3_RESULTS/baseline.jsonl" - 2>/dev/null | tail -3 || true; }
cmd_fetch(){ source "$STATE/run.env"; DD="$PROJECT_DIR/results/predictive_coding/pc_science_runs"; aws_ s3 cp "$S3_RESULTS/baseline.jsonl" "$DD/nanogpt_bpe8192_baseline.jsonl" --only-show-errors; echo "-> $DD/nanogpt_bpe8192_baseline.jsonl"; }
cmd_terminate(){ [ -f "$STATE/run.env" ] && { source "$STATE/run.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text | head -1; rm -f "$STATE/run.env"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
