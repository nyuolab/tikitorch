#!/bin/bash
# run_pc_scale_gpu.sh — robust one-shot GPU runner for the tropical predictive-coding
# scale tests (Phase 0 MNIST/Fashion-MNIST deep MLP; Phase 1 CIFAR conv). Same
# S3 / IAM-instance-profile / shutdown-backstop / self-terminate / no-SSH pattern as
# run_mzgeneral_chess_gpu.sh. Ships only committed files (git archive HEAD), runs the
# selected experiment on one A10G, streams JSONL results to S3, self-terminates.
#
#   export AWS_PROFILE=my-profile
#   PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh launch
#   PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh status
#   PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh fetch
#   PC_TASK=mnist bash scripts/run_pc_scale_gpu.sh terminate
#   (PC_TASK=cifar or convlm for other tasks; DRY_RUN=1 renders without launching)
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${REGION:-}"
if [ -z "$REGION" ]; then
  REGION="$(aws configure get region --profile "$PROFILE" 2>/dev/null || true)"
fi
REGION="${REGION:-us-east-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.4xlarge}"             # 1x A10G (24GB) + 16 vCPU
AMI="${AMI:-ami-01011b868ec560823}"                     # DL Base OSS Nvidia Driver, Ubuntu 22.04 (driver+CUDA)
MAX_MINUTES="${MAX_MINUTES:-200}"                        # hard cost backstop (self-terminates earlier on completion)
PC_TASK="${PC_TASK:-mnist}"                              # mnist (Phase 0) | cifar (Phase 1)
PC_ARGS="${PC_ARGS:-}"                                   # optional extra args to the experiment script
ROLE="${ROLE:-tiki-sweep-runner}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
case "$PC_TASK" in
  mnist|cifar|convlm) ;;
  *) echo "FATAL: PC_TASK must be mnist, cifar, or convlm"; exit 2 ;;
esac
case "$MAX_MINUTES" in
  ""|*[!0-9]*) echo "FATAL: MAX_MINUTES must be a positive integer"; exit 2 ;;
esac
if [ "$MAX_MINUTES" -le 0 ]; then
  echo "FATAL: MAX_MINUTES must be a positive integer"
  exit 2
fi
STATE_DIR="$PROJECT_DIR/.tiki_pcscale_${PC_TASK}_state"; mkdir -p "$STATE_DIR"  # per-task so mnist/cifar don't collide
STATE_FILE="$STATE_DIR/run.env"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists; terminate first (fetch anytime):"; cat "$STATE_FILE"; exit 1; }
  local ACCT BUCKET TAG S3_SRC S3_RESULTS UD INSTANCE_ID SOURCE_COMMIT SOURCE_ARCHIVE_SHA256
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET; run the connect4 runner's setup first"; exit 1; }
  TAG="pcscale-${PC_TASK}-$(date -u +%Y%m%dT%H%M%SZ)"
  S3_SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3_RESULTS="s3://${BUCKET}/${TAG}/results"
  echo "== build source tarball =="; git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  SOURCE_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD)"
  SOURCE_ARCHIVE_SHA256="$(shasum -a 256 "$STATE_DIR/src.tar.gz" | cut -d' ' -f1)"
  local need="scripts/_pc_scale_onbox.sh tikitorch/nn/predictive_coding.py"
  case "$PC_TASK" in
    mnist) need="$need scripts/pc_scale_mnist.py" ;;
    cifar) need="$need scripts/pc_scale_cifar.py" ;;
    convlm) need="$need scripts/pc_charlm_conv.py" ;;
    *) echo "FATAL: unknown PC_TASK=$PC_TASK (want mnist|cifar|convlm)"; exit 1 ;;
  esac
  for f in $need; do
    tar tzf "$STATE_DIR/src.tar.gz" | grep -Fx -- "$f" >/dev/null \
      || { echo "FATAL: $f not committed (git archive HEAD ships only committed files)"; exit 1; }
  done
  aws_ ec2 describe-images --image-ids "$AMI" --query 'Images[0].State' --output text | grep -qx available \
    || { echo "FATAL: AMI $AMI not available"; exit 1; }
  UD="$STATE_DIR/user-data.sh"
  render_user_data "$S3_SRC" "$S3_RESULTS" "$SOURCE_COMMIT" "$SOURCE_ARCHIVE_SHA256" > "$UD"
  if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY_RUN -> not launching ($(wc -l < "$UD") line user-data)"; return 0; fi
  echo "== upload source =="; aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors
  echo "== launch $INSTANCE_TYPE ($AMI; PC_TASK=$PC_TASK; ${MAX_MINUTES}m backstop) =="
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":120,"VolumeType":"gp3"}}]' \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pcscale-$PC_TASK},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then break; fi
    tries=$((tries+1))
    if grep -qi "instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then echo "IAM not propagated (try $tries)"; sleep 10; continue; fi
    echo "FATAL run-instances:"; cat "$STATE_DIR/launch.err"; exit 1
  done
  {
    printf 'TAG=%q\n' "$TAG"
    printf 'INSTANCE_ID=%q\n' "$INSTANCE_ID"
    printf 'REGION=%q\n' "$REGION"
    printf 'BUCKET=%q\n' "$BUCKET"
    printf 'S3_RESULTS=%q\n' "$S3_RESULTS"
    printf 'PC_TASK=%q\n' "$PC_TASK"
    printf 'SOURCE_COMMIT=%q\n' "$SOURCE_COMMIT"
    printf 'SOURCE_ARCHIVE_SHA256=%q\n' "$SOURCE_ARCHIVE_SHA256"
  } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID -> results under $S3_RESULTS"; echo "next: $0 status"
}

render_user_data(){
  local s3_src="$1" s3_results="$2" source_commit="$3" expected_sha="$4"
  local pc_args_q pc_task_q ami_q instance_type_q region_q
  printf -v pc_args_q '%q' "$PC_ARGS"
  printf -v pc_task_q '%q' "$PC_TASK"
  printf -v ami_q '%q' "$AMI"
  printf -v instance_type_q '%q' "$INSTANCE_TYPE"
  printf -v region_q '%q' "$REGION"
  cat <<UDEOF
#!/bin/bash
exec > /var/log/tiki-pcscale.log 2>&1
set -u
set -x
S3_SRC="${s3_src}"; S3_RESULTS="${s3_results}"
SOURCE_COMMIT="${source_commit}"
EXPECTED_SOURCE_ARCHIVE_SHA256="${expected_sha}"
RUN_AMI=${ami_q}; RUN_INSTANCE_TYPE=${instance_type_q}; RUN_REGION=${region_q}
RUN_RC=""
export DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:\$PATH
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-pcscale.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
retry3(){
  local delay="\$1" attempt
  shift
  for attempt in 1 2 3; do
    "\$@" && return 0
    [ "\$attempt" -eq 3 ] || sleep "\$delay"
  done
  return 1
}
finish(){
  local trap_rc="\$?" final_rc marker
  final_rc="\${RUN_RC:-\$trap_rc}"
  marker=FAILED
  [ "\$final_rc" -eq 0 ] && marker=SUCCESS
  up
  echo "rc=\$final_rc marker=\$marker commit=\$SOURCE_COMMIT archive_sha256=\$EXPECTED_SOURCE_ARCHIVE_SHA256 done=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
  echo "rc=\$final_rc done=\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | aws s3 cp - "\$S3_RESULTS/\$marker" --only-show-errors || true
  shutdown -h now || true
}
trap finish EXIT
shutdown -h +${MAX_MINUTES} "tiki-pcscale deadline" \
  || { echo "FATAL: could not schedule the cost-backstop shutdown"; exit 1; }
echo "BOOT \$(date -u)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
retry3 15 apt-get update -y || { echo "FATAL: apt-get update failed"; exit 1; }
retry3 15 apt-get install -y python3-venv python3-pip awscli git \
  || { echo "FATAL: package installation failed"; exit 1; }
python3 -m venv /opt/venv || { echo "FATAL: venv creation failed"; exit 1; }
retry3 15 /opt/venv/bin/pip install -q --upgrade pip \
  || { echo "FATAL: pip upgrade failed"; exit 1; }
up
retry3 15 /opt/venv/bin/pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
  || { echo "FATAL: torch installation failed"; exit 1; }
/opt/venv/bin/python -c "import torch,torchvision;print('torch',torch.__version__,'tv',torchvision.__version__,'cuda',torch.cuda.is_available());assert torch.cuda.is_available(),'CUDA unavailable'" \
  || { echo "FATAL: CUDA validation failed"; exit 1; }
up
mkdir -p /opt/tiki && cd /opt/tiki
retry3 10 aws s3 cp "\$S3_SRC" src.tar.gz \
  || { echo "FATAL: source download failed"; exit 1; }
SOURCE_ARCHIVE_SHA256="\$(sha256sum src.tar.gz | cut -d' ' -f1)"
[ "\$SOURCE_ARCHIVE_SHA256" = "\$EXPECTED_SOURCE_ARCHIVE_SHA256" ] \
  || { echo "FATAL: source archive SHA-256 mismatch"; exit 1; }
tar tzf src.tar.gz >/dev/null || { echo "FATAL: invalid source archive"; exit 1; }
tar xzf src.tar.gz || { echo "FATAL: source extraction failed"; exit 1; }
up
export REPO=/opt/tiki S3_RESULTS="\$S3_RESULTS" PYBIN=/opt/venv/bin/python
export PC_TASK=${pc_task_q} PC_ARGS=${pc_args_q} SOURCE_ARCHIVE_SHA256 SOURCE_COMMIT
export RUN_AMI RUN_INSTANCE_TYPE RUN_REGION
bash scripts/_pc_scale_onbox.sh
RUN_RC=\$?
exit "\$RUN_RC"
UDEOF
}

cmd_status(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "(gone)"
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null || echo "(nothing yet)"
  local done_record marker_record
  if done_record="$(aws_ s3 cp "$S3_RESULTS/DONE" - 2>/dev/null)"; then
    echo "== run complete: $done_record =="
    case "$done_record" in
      "rc=0 "*) return 0 ;;
      *) echo "FATAL: remote run failed"; return 1 ;;
    esac
  fi
  if marker_record="$(aws_ s3 cp "$S3_RESULTS/FAILED" - 2>/dev/null)"; then
    echo "== run failed: $marker_record =="
    return 1
  fi
  if marker_record="$(aws_ s3 cp "$S3_RESULTS/SUCCESS" - 2>/dev/null)"; then
    echo "== run succeeded: $marker_record =="
    return 0
  fi
  echo "-- driver.log tail --"
  aws_ s3 cp "$S3_RESULTS/driver.log" - 2>/dev/null | tail -25
}
cmd_fetch(){ [ -f "$STATE_FILE" ] || { echo no state; exit 1; }; source "$STATE_FILE"
  # land each run under its own TAG dir so concurrent/sequential runs never clobber
  # one another's scale_*.jsonl (a shared filename + `aws s3 cp` silently overwrites).
  local DEST="$PROJECT_DIR/results/predictive_coding/$TAG" name found=0
  local -a result_names include_args
  case "$PC_TASK" in
    mnist) result_names=(scale_mnist_gamma_damped.jsonl scale_mnist.jsonl) ;;
    cifar) result_names=(scale_cifar.jsonl) ;;
    convlm) result_names=(charlm_conv_localpc.jsonl) ;;
  esac
  include_args=()
  for name in "${result_names[@]}"; do
    include_args+=(--include "$name")
  done
  mkdir -p "$DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --exclude "*" \
    "${include_args[@]}" --include "driver.log" --include "boot.log" \
    --include "DONE" --include "SUCCESS" --include "FAILED" --only-show-errors
  for name in "${result_names[@]}"; do
    if [ -f "$DEST/$name" ]; then
      printf '%s\n' "$DEST/$name"
      found=1
    fi
  done
  [ "$found" -eq 1 ] || echo "(none yet)"
}
cmd_terminate(){ [ -f "$STATE_FILE" ] || { echo no state; exit 0; }; source "$STATE_FILE"
  local termination_state
  if ! termination_state="$(aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text)"; then
    echo "FATAL: termination request failed; state retained at $STATE_FILE" >&2
    return 1
  fi
  case "$termination_state" in
    shutting-down|terminated) ;;
    *) echo "FATAL: unexpected termination state '$termination_state'; state retained at $STATE_FILE" >&2; return 1 ;;
  esac
  printf '%s\n' "$termination_state"
  rm -f "$STATE_FILE"
  echo "(state cleared; ready to launch again)"
}
case "${1:-}" in
  launch) cmd_launch ;;
  status) cmd_status ;;
  fetch) cmd_fetch ;;
  terminate) cmd_terminate ;;
  *) echo "usage: $0 {launch|status|fetch|terminate}  (PC_TASK=mnist|cifar|convlm  DRY_RUN=1)"; exit 2 ;;
esac
