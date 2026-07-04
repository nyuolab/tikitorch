#!/bin/bash
# run_pc_science_ec2.sh — launch a compute-optimized EC2 box that runs Experiment 1 (cascade vs relaxation
# divergence vs depth) + Experiment 2 (PC-vs-backprop bpb gap vs depth, seeds) IN PARALLEL across cores,
# uploads jsonl results to S3, self-terminates. Pure CPU (numpy) — NO FPGA.
#   AWS_PROFILE=... bash scripts/pc_science/run_pc_science_ec2.sh launch     # then status / fetch / terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-c7i.48xlarge}"; AMI="${AMI:-ami-051bfa33df3949860}"; ROLE="${ROLE:-tiki-sweep-runner}"   # AL2023: aws CLI + python3 preinstalled
BIN_S3="${BIN_S3:-s3://tiki-sweep-739966866637/s2_bins/train.u16}"   # byte-level enwik8 (already uploaded)
DEPTHS="${DEPTHS:-2,4,8,16}"; SEEDS="${SEEDS:-0,1}"
E1_STEPS="${E1_STEPS:-2000}"; E2_STEPS="${E2_STEPS:-2500}"
D="${D:-256}"; HH="${HH:-4}"; DH="${DH:-64}"; FFN="${FFN:-1024}"; SEQ="${SEQ:-128}"; VOCAB="${VOCAB:-256}"
THREADS="${THREADS:-2}"; MAX_MINUTES="${MAX_MINUTES:-660}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"; STATE="${STATE_DIR:-$PROJECT_DIR/.tiki_pcsci_state}"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE/run.env" ] && { echo "state exists; terminate first"; cat "$STATE/run.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="pc-science-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  git -C "$PROJECT_DIR" archive --format=tar.gz -o "$STATE/src.tar.gz" HEAD
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-pcsci.log 2>&1
set -x; export HOME=/root
S3_RESULTS="${S3RES}"; BIN_S3="${BIN_S3}"
up(){ aws s3 cp /var/log/tiki-pcsci.log "\$S3_RESULTS/run.log" --only-show-errors || true; aws s3 cp /tmp/pcout "\$S3_RESULTS/" --recursive --exclude '*' --include '*.jsonl' --only-show-errors || true; }
trap up EXIT
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; nproc
# AL2023 ships aws CLI + python3; ensure aws + numpy are present (belt-and-suspenders for any AMI)
command -v aws >/dev/null 2>&1 || { dnf install -y awscli 2>/dev/null || pip3 install awscli 2>/dev/null || true; }
echo "aws: \$(command -v aws || echo MISSING)  python3: \$(command -v python3 || echo MISSING)"
dnf install -y python3-pip gcc >/dev/null 2>&1 || true
pip3 install --quiet "numpy>=1.24" 2>/dev/null || python3 -c "import numpy" 2>/dev/null || dnf install -y python3-numpy 2>/dev/null || true
python3 -c "import numpy; print('numpy',numpy.__version__); numpy.show_config()" 2>&1 | grep -iE "numpy 1|openblas|blas" | head -3 || { echo "FATAL: numpy missing"; echo fail | aws s3 cp - "\$S3_RESULTS/DONE" || true; shutdown -h now; }
# EARLY BLAS-SPEED CHECK: a 512x512x512 matmul must be <50ms (fast BLAS); if slow, we still run but flag it
python3 -c "import numpy as np,time; a=np.random.rand(512,512); t=time.time(); [a@a for _ in range(20)]; print('BLAS 512cube:', round((time.time()-t)/20*1000,1),'ms/matmul')"
up   # upload the boot log NOW so a launch can be verified within minutes
mkdir -p /opt/tiki /tmp/pcout && cd /opt/tiki
for i in 1 2 3; do aws s3 cp "${S3SRC}" s.tgz && break; sleep 8; done && tar xzf s.tgz
aws s3 cp "\$BIN_S3" /tmp/enwik8.u16 --only-show-errors
cd /opt/tiki/scripts/pc_science
export TOKBIN=/tmp/enwik8.u16 OMP_NUM_THREADS=${THREADS} OPENBLAS_NUM_THREADS=${THREADS} MKL_NUM_THREADS=${THREADS}
COMMON="D=${D} H=${HH} DH=${DH} FFN=${FFN} SEQ=${SEQ} VOCAB=${VOCAB} ACCUM=8 XF_LR=3e-3"
# ---- build the job list (each a background process; the OS scheduler packs them across cores) ----
: > /tmp/jobs.sh
IFS=',' read -ra DA <<< "${DEPTHS}"; IFS=',' read -ra SA <<< "${SEEDS}"
for L in "\${DA[@]}"; do
  echo "env \$COMMON L=\$L STEPS=${E1_STEPS} MEASURE_EVERY=500 SEED=0 OUT=/tmp/pcout/exp1_L\${L}.jsonl python3 exp1_formulation.py > /tmp/pcout/exp1_L\${L}.out 2>&1" >> /tmp/jobs.sh
  for S in "\${SA[@]}"; do
    echo "env \$COMMON L=\$L SEED=\$S STEPS=${E2_STEPS} WARMUP=100 LR_DECAY=1 EVAL_EVERY=250 EVAL_BATCHES=16 OUT=/tmp/pcout/exp2_L\${L}_s\${S}.jsonl python3 exp2_depthsweep.py > /tmp/pcout/exp2_L\${L}_s\${S}.out 2>&1" >> /tmp/jobs.sh
  done
done
echo "=== CPU: nproc=\$(nproc) lscpu-cpus=\$(lscpu 2>/dev/null | awk -F: '/^CPU\\(s\\)/{print \$2}' | head -1) ==="
echo "=== \$(wc -l < /tmp/jobs.sh) jobs, ${THREADS} threads each ==="; cat /tmp/jobs.sh
# run at most (nproc/THREADS) concurrently
PAR=\$(wc -l < /tmp/jobs.sh)   # run ALL jobs concurrently; kernel schedules across cores
cat /tmp/jobs.sh | xargs -P \$PAR -I{} bash -c '{}'
echo "=== ALL JOBS DONE \$(date -u) ==="
# summary: collect exp2 gaps + exp1 final divergence per L
python3 - <<'PYEOF' > /tmp/pcout/SUMMARY.txt 2>&1 || true
import json, glob
print("=== EXP2: PC-vs-BP bpb gap vs depth ===")
rows={}
for f in sorted(glob.glob('/tmp/pcout/exp2_*.jsonl')):
    for l in open(f):
        r=json.loads(l)
        if r.get("exp")=="exp2_summary":
            rows.setdefault(r["L"],[]).append(r)
for L in sorted(rows):
    import statistics as st
    g=[x["gap_PC_BP"] for x in rows[L]]; gf=[x["gap_PC_BPf"] for x in rows[L]]
    print(f"L={L:2d}  gap(PC-BP)={st.mean(g):+.4f}+-{(st.pstdev(g) if len(g)>1 else 0):.4f}  gap(PC-BPf)={st.mean(gf):+.4f}  n={len(g)}")
print("\n=== EXP1: cascade-vs-relax_sync divergence at final step vs depth ===")
for f in sorted(glob.glob('/tmp/pcout/exp1_*.jsonl')):
    last=None
    for l in open(f): last=json.loads(l)
    if last: print(f"L={last['L']:2d}  grad_cos(sync)={last.get('grad_cos_sync'):.6f}  eps_cos(sync)={last.get('eps_cos_sync')}  perblock_sync={last.get('eps_div_perblock_sync')}  grad_cos(async)={last.get('grad_cos_async'):.4f}")
PYEOF
up; echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
sync; shutdown -h now
UDEOF
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$STATE/user-data.sh"; return 0; }
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pc-science},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_RESULTS=$S3RES"; } > "$STATE/run.env"
  echo "LAUNCHED $IID ($ITYPE) -> $S3RES"
}
cmd_status(){ source "$STATE/run.env"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; echo "--- SUMMARY ---"; aws_ s3 cp "$S3_RESULTS/SUMMARY.txt" - 2>/dev/null || echo "(not done)"; }
cmd_fetch(){ source "$STATE/run.env"; D="$PROJECT_DIR/results/predictive_coding/pc_science_runs"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE/run.env" ] && { source "$STATE/run.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text | head -1; rm -f "$STATE/run.env"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
