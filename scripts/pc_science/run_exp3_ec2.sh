#!/bin/bash
# run_exp3_ec2.sh -- launch a compute-optimized EC2 box that runs Experiment 2 (CORRECTED):
# the deep-PC-traction CONTRAST -- {tropical net, smooth net} x {backprop ceiling, PC method} --
# across (L in DEPTHS) x (seed in SEEDS), ONE ARM PER JOB fanned across the cores, uploads the
# per-arm jsonl + a SUMMARY (the two gap-vs-depth curves) to S3, self-terminates. Pure CPU (numpy).
#
# Cloned from run_pc_science_ec2.sh -- reuses its AL2023 + CPU-online + pip-OpenBLAS +
# THREADS=16 + all-jobs-concurrent hardening. The ONLY changes: it dispatches exp3_contrast.py
# per-arm (4L x 2seed x 5arm = 40 jobs), and the SUMMARY recombines per-arm bpb into the
# gap_trop (should be ~0, flat vs depth) and gap_smooth_T* (should GROW with depth) curves.
#   AWS_PROFILE=... bash scripts/pc_science/run_exp3_ec2.sh launch     # then status / fetch / terminate
set -euo pipefail
PROFILE="${AWS_PROFILE:?set AWS_PROFILE}"; REGION="${REGION:-us-east-1}"
ITYPE="${ITYPE:-c7i.48xlarge}"; AMI="${AMI:-ami-051bfa33df3949860}"; ROLE="${ROLE:-tiki-sweep-runner}"   # AL2023: aws CLI + python3 preinstalled
BIN_S3="${BIN_S3:-s3://tiki-sweep-739966866637/s2_bins/train.u16}"   # byte-level enwik8 (already uploaded)
DEPTHS="${DEPTHS:-2,4,8,16}"; SEEDS="${SEEDS:-0,1}"
STEPS="${STEPS:-2500}"; RELAX_T="${RELAX_T:-5,20}"
D="${D:-192}"; HH="${HH:-4}"; DH="${DH:-48}"; FFN="${FFN:-768}"; SEQ="${SEQ:-128}"; VOCAB="${VOCAB:-256}"
THREADS="${THREADS:-16}"; MAX_MINUTES="${MAX_MINUTES:-660}"
PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"; STATE="${STATE_DIR:-$PROJECT_DIR/.tiki_exp3_state}"; mkdir -p "$STATE"
aws_(){ aws --profile "$PROFILE" --region "$REGION" "$@"; }

cmd_launch(){
  [ -f "$STATE/run.env" ] && { echo "state exists; terminate first"; cat "$STATE/run.env"; exit 1; }
  ACCT="$(aws_ sts get-caller-identity --query Account --output text)"; BUCKET="tiki-sweep-${ACCT}"
  TAG="pc-exp3-$(date -u +%Y%m%dT%H%M%SZ)"; S3SRC="s3://${BUCKET}/${TAG}/src.tar.gz"; S3RES="s3://${BUCKET}/${TAG}/results"
  # bundle HEAD, then OVERLAY the current working-tree scripts/pc_science (exp3 files are
  # uncommitted -- NO git in this workflow -- so git archive alone would miss them). We
  # stage a tree = git-archive extract + fresh copy of scripts/pc_science, then re-tar.
  STG="$STATE/stage"; rm -rf "$STG"; mkdir -p "$STG"
  git -C "$PROJECT_DIR" archive HEAD | tar -x -C "$STG"
  mkdir -p "$STG/scripts/pc_science"
  cp "$PROJECT_DIR"/scripts/pc_science/*.py "$STG/scripts/pc_science/" 2>/dev/null || true
  tar -C "$STG" -czf "$STATE/src.tar.gz" .
  aws_ s3 cp "$STATE/src.tar.gz" "$S3SRC" --only-show-errors
  cat > "$STATE/user-data.sh" <<UDEOF
#!/bin/bash
shutdown -h +${MAX_MINUTES} dl || true
exec > /var/log/tiki-exp3.log 2>&1
set -x; export HOME=/root
S3_RESULTS="${S3RES}"; BIN_S3="${BIN_S3}"
up(){ aws s3 cp /var/log/tiki-exp3.log "\$S3_RESULTS/run.log" --only-show-errors || true; aws s3 cp /tmp/pcout "\$S3_RESULTS/" --recursive --exclude '*' --include '*.jsonl' --include '*.txt' --only-show-errors || true; }
trap up EXIT
( while true; do sleep 60; up; done ) &
echo "BOOT \$(date -u)"; echo "nproc-before=\$(nproc)"
for f in /sys/devices/system/cpu/cpu[0-9]*/online; do echo 1 > "\$f" 2>/dev/null || true; done
echo "nproc-after-online=\$(nproc)  lscpu=\$(lscpu 2>/dev/null | awk -F: '/^CPU\\(s\\)/{gsub(/ /,\"\",\$2);print \$2}' | head -1)"
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
COMMON="D=${D} H=${HH} DH=${DH} FFN=${FFN} SEQ=${SEQ} VOCAB=${VOCAB} ACCUM=8 XF_LR=3e-3 STEPS=${STEPS} WARMUP=100 LR_DECAY=1 LR_MINFRAC=0.1 EVAL_EVERY=250 EVAL_BATCHES=16 RELAX_T=${RELAX_T}"
# ---- build the per-arm job list (each a background process; the OS scheduler packs them) ----
# 4 arms of the contrast; PC_smooth_T is dispatched ONCE (it expands internally to one run per
# RELAX_T budget) -> 5 training runs per (L,seed). ONE ARM PER JOB so the grid fans across cores
# and no single process serialises all 5 arms (the deep large-T arm is the long pole).
: > /tmp/jobs.sh
IFS=',' read -ra DA <<< "${DEPTHS}"; IFS=',' read -ra SA <<< "${SEEDS}"
for L in "\${DA[@]}"; do
  for S in "\${SA[@]}"; do
    for ARM in BP_trop PC_trop BP_smooth PC_smooth_T; do
      echo "env \$COMMON L=\$L SEED=\$S ARMS=\$ARM OUT=/tmp/pcout/exp3_L\${L}_s\${S}_\${ARM}.jsonl python3 exp3_contrast.py > /tmp/pcout/exp3_L\${L}_s\${S}_\${ARM}.out 2>&1" >> /tmp/jobs.sh
    done
  done
done
echo "=== CPU: nproc=\$(nproc) lscpu-cpus=\$(lscpu 2>/dev/null | awk -F: '/^CPU\\(s\\)/{print \$2}' | head -1) ==="
echo "=== \$(wc -l < /tmp/jobs.sh) jobs, ${THREADS} threads each ==="; cat /tmp/jobs.sh
# run ALL jobs concurrently; the kernel schedules them across cores (16 threads each)
PAR=\$(wc -l < /tmp/jobs.sh)
cat /tmp/jobs.sh | xargs -P \$PAR -I{} bash -c '{}'
echo "=== ALL JOBS DONE \$(date -u) ==="
# ---- SUMMARY: recombine per-arm finals into the two gap-vs-depth curves ----
python3 - <<'PYEOF' > /tmp/pcout/SUMMARY.txt 2>&1 || true
import json, glob, statistics as st
# collect finals per (L, seed, arm) from the per-arm summaries
finals = {}   # (L,seed) -> {armkey: bpb}
for f in sorted(glob.glob('/tmp/pcout/exp3_*.jsonl')):
    for l in open(f):
        try: r = json.loads(l)
        except Exception: continue
        if r.get("exp") == "exp3_summary":
            key = (r["L"], r["seed"])
            finals.setdefault(key, {}).update(r.get("finals", {}))
# discover the finite-T budgets present
Ts = sorted({int(k.split("PC_smooth_T")[1]) for kv in finals.values() for k in kv if k.startswith("PC_smooth_T")})
Ls = sorted({L for (L, s) in finals})
print("=== EXPERIMENT 2 (CORRECTED): deep-PC-traction contrast ===")
print("arms: BP_trop (dyadic-net BP ceiling, STE-thru-dyad) | PC_trop (cascade, ours) |")
print("      BP_smooth (float-net BP ceiling) | PC_smooth_T{T} (float-net finite-T PC)")
print()
print("--- final bpb per arm (mean +- pstdev over seeds) ---")
hdr = ["L", "BP_trop", "PC_trop", "BP_smooth"] + [f"PC_smoothT{T}" for T in Ts]
print("  " + "  ".join(f"{h:>11s}" for h in hdr))
def agg(L, arm):
    xs = [finals[(L, s)][arm] for s in {ss for (ll, ss) in finals if ll == L} if arm in finals.get((L, s), {})]
    if not xs: return None
    return (st.mean(xs), st.pstdev(xs) if len(xs) > 1 else 0.0, len(xs))
for L in Ls:
    cells = [f"{L:>11d}"]
    for arm in ["BP_trop", "PC_trop", "BP_smooth"] + [f"PC_smooth_T{T}" for T in Ts]:
        a = agg(L, arm)
        cells.append(f"{a[0]:>11.4f}" if a else f"{'--':>11s}")
    print("  " + "  ".join(cells))
print()
print("--- GAP CURVE 1: tropical  gap_trop = PC_trop - BP_trop  (CLAIM: ~0, FLAT vs depth) ---")
for L in Ls:
    pc, bp = agg(L, "PC_trop"), agg(L, "BP_trop")
    if pc and bp: print(f"  L={L:2d}  gap_trop = {pc[0]-bp[0]:+.4f}   (PC={pc[0]:.4f} BP={bp[0]:.4f}, n={pc[2]})")
print()
for T in Ts:
    print(f"--- GAP CURVE 2: smooth  gap_smooth_T{T} = PC_smooth_T{T} - BP_smooth  (CLAIM: GROWS w/ depth) ---")
    for L in Ls:
        pc, bp = agg(L, f"PC_smooth_T{T}"), agg(L, "BP_smooth")
        if pc and bp: print(f"  L={L:2d}  gap_smooth_T{T} = {pc[0]-bp[0]:+.4f}   (PC={pc[0]:.4f} BP={bp[0]:.4f}, n={pc[2]})")
    print()
print("VERDICT: gap_trop should stay ~flat near 0 across L; gap_smooth_T* should widen with L.")
PYEOF
up; echo done | aws s3 cp - "\$S3_RESULTS/DONE" --only-show-errors || true
sync; shutdown -h now
UDEOF
  [ "${DRY_RUN:-0}" = "1" ] && { echo "DRY_RUN"; cat "$STATE/user-data.sh"; return 0; }
  IID="$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$ITYPE" \
      --iam-instance-profile "Name=$ROLE" --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=40}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=tiki-pc-exp3},{Key=tiki-sweep,Value=$TAG}]" \
      --user-data "file://$STATE/user-data.sh" --query 'Instances[0].InstanceId' --output text)"
  { echo "TAG=$TAG"; echo "INSTANCE_ID=$IID"; echo "S3_RESULTS=$S3RES"; } > "$STATE/run.env"
  echo "LAUNCHED $IID ($ITYPE) -> $S3RES"
}
cmd_status(){ source "$STATE/run.env"; aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" --query 'Reservations[0].Instances[0].State.Name' --output text; aws_ s3 ls "$S3_RESULTS/" 2>/dev/null; echo "--- SUMMARY ---"; aws_ s3 cp "$S3_RESULTS/SUMMARY.txt" - 2>/dev/null || echo "(not done)"; }
cmd_fetch(){ source "$STATE/run.env"; D="$PROJECT_DIR/results/predictive_coding/exp3_runs"; mkdir -p "$D"; aws_ s3 cp "$S3_RESULTS/" "$D/" --recursive --only-show-errors; ls -1 "$D"; }
cmd_terminate(){ [ -f "$STATE/run.env" ] && { source "$STATE/run.env"; aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" --query 'TerminatingInstances[0].CurrentState.Name' --output text | head -1; rm -f "$STATE/run.env"; }; }
case "${1:-}" in launch) cmd_launch;; status) cmd_status;; fetch) cmd_fetch;; terminate) cmd_terminate;; *) echo "usage: $0 {launch|status|fetch|terminate}";; esac
