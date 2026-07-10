#!/bin/bash
# run_connect4_prior_sweep_ec2.sh — a ROBUST, one-shot EC2 runner for the fully
# multiply-free Connect4 anti-collapse prior sweep (results/board_scaling/CONNECT4_PRIOR_SWEEP.md).
#
# WHY A NEW RUNNER. The ad-hoc SSH-polling runners (run_tikitorch_gpu_benchmarks.sh etc.)
# failed this job three times: (1) unpinned torch threads -> 13 h / $20 oversubscription;
# (2) the home IP rotated, the SSH completion-waiter went blind, the cost backstop nuked
# the box AND its EBS -> results lost; (3) the multi-step SSH setup hung BEFORE the cost
# backstop was armed -> the box ran unbacked for hours with the repo never synced. Total
# ~$30 with the apples-to-apples table never retrieved.
#
# This runner removes every one of those failure modes by removing SSH from the critical
# path entirely:
#   * THREADS PINNED  — the on-box driver runs each recipe with --threads 1 (fixes #1).
#   * RESULTS TO S3   — the box uploads each recipe JSON to S3 the instant it finishes,
#                       and the laptop polls S3 (not SSH), so an IP rotation can't blind
#                       us and a dead box never loses completed work (fixes #2).
#   * BACKSTOP FIRST  — `shutdown -h +MAX_MINUTES` is the FIRST line of user-data, before
#                       any setup can hang, paired with InstanceInitiatedShutdownBehavior=
#                       terminate, so the box self-destructs no matter what (fixes #3).
#   * IAM INSTANCE-PROFILE — the box gets fresh credentials from IMDS that never expire
#                       mid-run, instead of laptop SSO creds that die hourly.
#   * AMI VIA SSM     — the Ubuntu AMI id is resolved live from Canonical's SSM parameter,
#                       so it never goes stale.
# No key pair, no security-group edits, no inbound SSH: nothing IP-dependent to misconfigure.
#
# Usage (all state is in .tiki_sweep_state/, region/account come from your AWS profile):
#   export AWS_PROFILE=NYUMCResearchAdminFullAccess-739966866637   # admin in us-east-1
#   bash scripts/run_connect4_prior_sweep_ec2.sh setup       # idempotent: bucket + IAM role/profile
#   bash scripts/run_connect4_prior_sweep_ec2.sh launch      # build src tarball -> S3 -> launch box
#   bash scripts/run_connect4_prior_sweep_ec2.sh status      # poll S3 (+ instance state); no SSH
#   bash scripts/run_connect4_prior_sweep_ec2.sh fetch       # download result JSONs + logs from S3
#   bash scripts/run_connect4_prior_sweep_ec2.sh terminate   # belt-and-suspenders manual kill
#   bash scripts/run_connect4_prior_sweep_ec2.sh clean       # delete the S3 sweep prefix + local state
#   DRY_RUN=1 bash scripts/run_connect4_prior_sweep_ec2.sh launch   # build+validate, do NOT launch
set -euo pipefail

PROFILE="${AWS_PROFILE:?set AWS_PROFILE (e.g. NYUMCResearchAdminFullAccess-739966866637)}"
REGION="${REGION:-$(aws configure get region --profile "$PROFILE" 2>/dev/null || echo us-east-1)}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.9xlarge}"   # 36 vCPU, ~$1.53/hr on-demand in us-east-1
MAX_MINUTES="${MAX_MINUTES:-120}"              # hard self-terminate deadline (cost backstop)
WORKERS="${WORKERS:-4}"                         # self-play workers per recipe (7*4=28 < 36)
EBS_SIZE="${EBS_SIZE:-30}"                      # GB; torch-cpu + repo fit easily
ROLE="${ROLE:-tiki-sweep-runner}"              # IAM role == instance-profile name
# Generalized sweep template: DRIVER = the on-box sweep script (defaults to the Connect4 prior sweep);
# RESULT_GLOB = which result JSONs fetch pulls back. Override both to drive other sweeps on the same
# robust machinery, e.g.:  DRIVER=_chess_strength_onbox.sh RESULT_GLOB='chess_strength_*.json' ... launch
DRIVER="${DRIVER:-_connect4_prior_sweep_onbox.sh}"
RESULT_GLOB="${RESULT_GLOB:-connect4_fully_tropical_*.json}"
PIP_EXTRA="${PIP_EXTRA:-}"                      # extra pip pkgs the on-box bootstrap installs (e.g. "chess" for the chess sweep)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STATE_DIR="$PROJECT_DIR/.tiki_sweep_state"; mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/run.env"

aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }
acct(){ aws_ sts get-caller-identity --query Account --output text; }

# ---------------------------------------------------------------------------- setup
cmd_setup(){
  local ACCT BUCKET
  ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  echo "account=$ACCT region=$REGION bucket=$BUCKET role=$ROLE"

  # S3 bucket (idempotent). us-east-1 must NOT pass a LocationConstraint.
  if aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "bucket exists: $BUCKET"
  else
    if [ "$REGION" = "us-east-1" ]; then aws_ s3api create-bucket --bucket "$BUCKET" >/dev/null
    else aws_ s3api create-bucket --bucket "$BUCKET" \
           --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
    echo "bucket created: $BUCKET"
  fi

  # IAM role + instance profile (idempotent). Trust EC2; allow ONLY S3 on this bucket.
  if aws_ iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    echo "role exists: $ROLE"
  else
    aws_ iam create-role --role-name "$ROLE" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    echo "role created: $ROLE"
  fi
  aws_ iam put-role-policy --role-name "$ROLE" --policy-name s3-sweep \
    --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::${BUCKET}\",\"arn:aws:s3:::${BUCKET}/*\"]}]}" >/dev/null
  echo "inline policy s3-sweep attached (GetObject/PutObject/ListBucket on $BUCKET only)"

  if aws_ iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
    echo "instance-profile exists: $ROLE"
  else
    aws_ iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
    echo "instance-profile created: $ROLE"
  fi
  # add-role-to-instance-profile errors if already attached; tolerate it.
  aws_ iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" 2>/dev/null \
    && echo "role attached to instance-profile" || echo "role already in instance-profile"
  echo "SETUP OK. bucket=$BUCKET  instance-profile=$ROLE"
}

# ---------------------------------------------------------------------------- launch
cmd_launch(){
  [ -f "$STATE_FILE" ] && { echo "state exists ($STATE_FILE):"; cat "$STATE_FILE"; \
    echo "-> a run is already recorded; 'terminate' + 'clean' first."; exit 1; }
  local ACCT BUCKET AMI SRC_KEY S3_SRC S3_RESULTS UD INSTANCE_ID
  ACCT="$(acct)"; BUCKET="tiki-sweep-${ACCT}"
  aws_ s3api head-bucket --bucket "$BUCKET" 2>/dev/null || { echo "no bucket $BUCKET; run 'setup' first"; exit 1; }
  aws_ iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1 || { echo "no instance-profile $ROLE; run 'setup' first"; exit 1; }

  # Unique-per-launch prefix from a value the laptop ALREADY knows (instance id is not
  # known until after run-instances). Use the local epoch as the prefix tag.
  local TAG="sweep-$(date -u +%Y%m%dT%H%M%SZ)"
  SRC_KEY="s3://${BUCKET}/${TAG}/src.tar.gz"
  S3_SRC="$SRC_KEY"
  S3_RESULTS="s3://${BUCKET}/${TAG}/results"

  echo "== building source tarball (git archive HEAD, deterministic committed tree) =="
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE_DIR/src.tar.gz" HEAD
  ls -lh "$STATE_DIR/src.tar.gz"
  # sanity: the two scripts we depend on must be in the archive
  tar tzf "$STATE_DIR/src.tar.gz" | grep -qx "scripts/$DRIVER" \
    || { echo "FATAL: scripts/$DRIVER not in committed tree (commit it first)"; exit 1; }

  echo "== resolving AMI from SSM (${AMI_PARAM:-canonical ubuntu 22.04}) =="
  AMI="$(aws_ ssm get-parameters \
    --names "${AMI_PARAM:-/aws/service/canonical/ubuntu/server/22.04/stable/current/amd64/hvm/ebs-gp2/ami-id}" \
    --query 'Parameters[0].Value' --output text)"
  echo "AMI=$AMI"
  [ -n "$AMI" ] && [ "$AMI" != "None" ] || { echo "FATAL: could not resolve AMI"; exit 1; }

  UD="$STATE_DIR/user-data.sh"
  render_user_data "$S3_SRC" "$S3_RESULTS" > "$UD"
  echo "== user-data rendered ($(wc -l < "$UD") lines) -> $UD =="

  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1 -> NOT launching. Validated: tarball, AMI, user-data."
    echo "  would: aws s3 cp src.tar.gz $S3_SRC ; run-instances $INSTANCE_TYPE $AMI (terminate-on-shutdown), backstop ${MAX_MINUTES}m"
    echo "  results prefix would be: $S3_RESULTS"
    return 0
  fi

  echo "== uploading source tarball -> $S3_SRC =="
  aws_ s3 cp "$STATE_DIR/src.tar.gz" "$S3_SRC" --only-show-errors

  echo "== launching $INSTANCE_TYPE (terminate-on-shutdown; ${MAX_MINUTES}m hard backstop) =="
  # Retry: IAM instance-profile is eventually consistent right after setup.
  local tries=0
  while :; do
    if INSTANCE_ID="$(aws_ ec2 run-instances \
        --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
        --iam-instance-profile "Name=$ROLE" \
        --instance-initiated-shutdown-behavior terminate \
        --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${EBS_SIZE},\"VolumeType\":\"gp3\",\"DeleteOnTermination\":true}}]" \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-prior-sweep},{Key=tiki-sweep,Value=$TAG}]" \
        --user-data "file://$UD" \
        --query 'Instances[0].InstanceId' --output text 2>"$STATE_DIR/launch.err")"; then
      break
    fi
    tries=$((tries+1))
    if grep -qi "Invalid IAM Instance Profile\|instance profile" "$STATE_DIR/launch.err" && [ "$tries" -lt 6 ]; then
      echo "  IAM profile not yet propagated (try $tries); waiting 10s..."; sleep 10; continue
    fi
    echo "FATAL: run-instances failed:"; cat "$STATE_DIR/launch.err"; exit 1
  done

  {
    echo "TAG=$TAG"; echo "INSTANCE_ID=$INSTANCE_ID"; echo "REGION=$REGION"
    echo "DRIVER=$DRIVER"; echo "RESULT_GLOB=$RESULT_GLOB"
    echo "BUCKET=$BUCKET"; echo "S3_RESULTS=$S3_RESULTS"; echo "AMI=$AMI"
    echo "INSTANCE_TYPE=$INSTANCE_TYPE"; echo "MAX_MINUTES=$MAX_MINUTES"
  } > "$STATE_FILE"
  echo "LAUNCHED $INSTANCE_ID"; echo "state -> $STATE_FILE"
  echo "results will appear under $S3_RESULTS"
  echo "next: bash $0 status   (polls S3; safe across IP rotation)"
}

# render_user_data <s3_src> <s3_results>  — emitted to the box as cloud-init user-data.
render_user_data(){
  local s3_src="$1" s3_results="$2"
  cat <<UDEOF
#!/bin/bash
# ARM THE HARD BACKSTOP FIRST — before ANY setup can hang. Paired with the launch-time
# InstanceInitiatedShutdownBehavior=terminate, this guarantees the box self-destructs
# even if everything below wedges. (Fixes the "ran unbacked for hours" failure.)
shutdown -h +${MAX_MINUTES} "tiki-sweep hard deadline" || true

exec > /var/log/tiki-sweep.log 2>&1
set -x
S3_SRC="${s3_src}"
S3_RESULTS="${s3_results}"
export WORKERS="${WORKERS}"
up(){ command -v aws >/dev/null 2>&1 && aws s3 cp /var/log/tiki-sweep.log "\$S3_RESULTS/boot.log" --only-show-errors || true; }
trap up EXIT
echo "BOOT \$(date -u)"
export DEBIAN_FRONTEND=noninteractive PATH=/opt/venv/bin:\$PATH

for i in 1 2 3; do apt-get update -y && apt-get install -y python3-venv python3-pip && break; sleep 15; done
python3 -m venv /opt/venv
/opt/venv/bin/pip install -q --upgrade pip
# awscli first so heartbeat + source-fetch work; then numpy; then CPU-only torch.
for i in 1 2 3; do /opt/venv/bin/pip install -q awscli && break; sleep 10; done
up
for i in 1 2 3; do /opt/venv/bin/pip install -q numpy && break; sleep 10; done
TIDX="${TORCH_INDEX-https://download.pytorch.org/whl/cpu}"
if [ -n "\$TIDX" ]; then
  for i in 1 2 3; do /opt/venv/bin/pip install -q torch --index-url "\$TIDX" && break; sleep 15; done
else
  for i in 1 2 3; do /opt/venv/bin/pip install -q torch && break; sleep 15; done
fi
EXTRA="${PIP_EXTRA}"
[ -n "\$EXTRA" ] && for i in 1 2 3; do /opt/venv/bin/pip install -q \$EXTRA && break; sleep 10; done
up
python3 -c "import torch,numpy;print('torch',torch.__version__,'numpy',numpy.__version__)" || true

mkdir -p /opt/tiki && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "\$S3_SRC" /opt/tiki/src.tar.gz && break; sleep 10; done
tar xzf src.tar.gz   # git archive => flat tree (scripts/, tikitorch/ at /opt/tiki root)
up

export REPO=/opt/tiki S3_RESULTS="\$S3_RESULTS"
bash scripts/${DRIVER}
rc=\$?
echo "SWEEP_RC=\$rc \$(date -u)"
aws s3 cp /var/log/tiki-sweep.log "\$S3_RESULTS/boot.log" --only-show-errors || true
echo "rc=\$rc done=\$(date -u)" | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
# Self-terminate immediately on completion (don't wait for the ${MAX_MINUTES}m backstop).
shutdown -h now
UDEOF
}

# ---------------------------------------------------------------------------- status
cmd_status(){
  [ -f "$STATE_FILE" ] || { echo "no state ($STATE_FILE); launch first"; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  echo "== instance =="
  aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,PublicIpAddress,LaunchTime]' \
    --output text 2>/dev/null || echo "(instance not found — likely already self-terminated)"
  echo "== S3 ($S3_RESULTS) =="
  aws_ s3 ls "$S3_RESULTS/" --recursive 2>/dev/null | awk '{print $3, $4}' || echo "(nothing yet)"
  if aws_ s3 ls "$S3_RESULTS/DONE" >/dev/null 2>&1; then
    echo "== DONE marker present =="; aws_ s3 cp "$S3_RESULTS/DONE" - 2>/dev/null
    local n; n=$(aws_ s3 ls "$S3_RESULTS/" 2>/dev/null | grep -c "connect4_fully_tropical_.*\.json" || true)
    echo "result JSONs in S3: $n / 7  ->  fetch with: bash $0 fetch"
  else
    echo "== not done yet (no DONE marker). Re-run status to poll; tail boot.log below =="
    aws_ s3 cp "$S3_RESULTS/boot.log" - 2>/dev/null | tail -15 || echo "(no boot.log yet)"
  fi
}

# ---------------------------------------------------------------------------- fetch
cmd_fetch(){
  [ -f "$STATE_FILE" ] || { echo "no state; launch first"; exit 1; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  local DEST="$PROJECT_DIR/results/board_scaling"
  mkdir -p "$DEST/_sweep_logs"
  echo "downloading result JSONs -> $DEST"
  aws_ s3 cp "$S3_RESULTS/" "$DEST/" --recursive --exclude "*" --include "${RESULT_GLOB:-connect4_fully_tropical_*.json}" --only-show-errors
  aws_ s3 cp "$S3_RESULTS/logs/" "$DEST/_sweep_logs/" --recursive --only-show-errors 2>/dev/null || true
  aws_ s3 cp "$S3_RESULTS/boot.log" "$DEST/_sweep_logs/boot.log" --only-show-errors 2>/dev/null || true
  echo "== fetched =="; ls -1 "$DEST"/${RESULT_GLOB:-connect4_fully_tropical_*.json} 2>/dev/null || echo "(none yet)"
}

# ---------------------------------------------------------------------------- terminate / clean
cmd_terminate(){
  [ -f "$STATE_FILE" ] || { echo "no state"; exit 0; }
  # shellcheck disable=SC1090
  source "$STATE_FILE"
  aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --query 'TerminatingInstances[0].[InstanceId,CurrentState.Name]' --output text 2>&1 | head -3 || true
}
cmd_clean(){
  if [ -f "$STATE_FILE" ]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    read -r -p "delete S3 prefix s3://$BUCKET/$TAG and local state? [y/N] " a
    [ "$a" = "y" ] && { aws_ s3 rm "s3://$BUCKET/$TAG" --recursive --only-show-errors || true; rm -f "$STATE_FILE"; echo cleaned; }
  else echo "no state to clean"; fi
}

case "${1:-}" in
  setup) cmd_setup;;
  launch) cmd_launch;;
  status) cmd_status;;
  fetch) cmd_fetch;;
  terminate) cmd_terminate;;
  clean) cmd_clean;;
  *) echo "usage: AWS_PROFILE=... $0 {setup|launch|status|fetch|terminate|clean}  (DRY_RUN=1 for launch)";;
esac
