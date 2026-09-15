# Research and hardware scripts

These are experiment runners, diagnostics, and hardware orchestration helpers—not a
compatibility-stable command-line API. The installed public command is `tt-closure`;
the scripts in this directory are tied to their documented data, environment, and
source revision.

## Finding an entry point

| Prefix or area | Purpose |
| --- | --- |
| [`route_certificates/`](route_certificates/README.md) | Matched, replay, stability, and retained-state experiments |
| `tropical_*`, `induction_*`, `charlm_*` | Tropical model and language/induction experiments |
| `pc_*`, `bench_cascade_vs_bp.py` | Predictive-coding, cascade, scale, and local-learning studies |
| `muzero_*`, `run_*board*`, `run_*chess*`, `run_*connect4*` | Planning and board-game experiments |
| `run_*f2*`, `run_*afi*`, `run_*csynth*`, `build_fpga_local.sh` | Hardware builds, C-simulation, and board execution |
| `fpga_lm_pretrain.py`, `fpga_lm_layout.cpp`, `fpga_lm_aws.py` | New fresh-window integer LM pilot, native ABI probe and fail-closed cloud preparation |
| `_*.py`, `_*.sh` | Internal helpers invoked by another runner; not primary entry points |

For library onboarding, use [`examples/`](../examples/README.md), not these research
runners. For evidence interpretation, start at [`manuscript/results/`](../manuscript/results/README.md) and
the [claim ledger](../docs/research/CLAIM_LEDGER.md).

New output defaults use `manuscript/results/`; see the
[artifact preservation policy](../manuscript/research/EVIDENCE.md#artifact-preservation-and-layout). Research runner
defaults resolve the checkout root, or the caller's directory when installed
without a checkout. Pass explicit data/output paths for reproducible runs.
`nanogpt --strict` checks current hard training separately from a hardened
candidate. `pc_convrnn --strict-audit` is a legacy batch-2 hardened cascade
probe only; it does not certify the configured training trajectory.

## Running safely

1. Read the script source/docstring first: historical runners do not all have
   safe help or dry-run modes. The `run_tikitorch_gpu_benchmarks.sh`,
   `run_tikitorch_f2_silicon.sh` and `run_lm_gpu_baseline.sh` launchers require an explicit `launch`; their
   `--help` exits without AWS calls or local state creation. Do not assume a
   similarly named mode in another script has the same safety contract.
2. Record the exact Git revision, command, seed, dependencies, data digest, and hardware
   configuration. Do not infer those values from a filename.
3. Supply AWS profiles, keys, security groups, AGFIs, vendor-tool locations, and data
   paths explicitly. The repository provides no credentials or implicit AWS profile;
   some revision-bound runners retain documented S3 artifact locations as provenance
   defaults, so inspect and override them before launching in another account.
4. Write new outputs to a fresh run path. Promote only curated summaries and manifests
   to Git; keep large generated products in external artifact storage.

These remain historical orchestration helpers, not unattended job managers.
An interrupted launch before local state recording can require AWS-side
reconciliation. A completion marker means the workload commands succeeded;
it does not establish matched task quality or valid energy measurements.

A safe, download-free first check is:

```bash
python -m examples.closure_workflow
python -m scripts.route_certificates.actual_model_benchmark --dry-run
```

## Fresh-window FPGA language-model pilot

The active [research plan](../manuscript/research/PLAN.md) prioritizes FPGA
pretraining. `fpga_lm_pretrain.py` exercises the existing fixed-point MorphLM
mode-3 kernel: one update per new 128-character window, persistent weights,
frozen validation, full checkpoints and an independent integer forward/loss
reference. It rejects non-ASCII bytes, incompatible native layouts and PCI
libraries that cannot report transfer errors. It never silently substitutes
host execution for FPGA execution.

Run an offline plan with an existing, appropriately licensed ASCII corpus:

```bash
python scripts/fpga_lm_pretrain.py --plan --width 16 \
  --corpus data/tinyshakespeare.txt --steps 8 --eval-every 4 --eval-windows 2
```

For a local C++17/NumPy smoke test, build a **new** current-source library. The
builder prints its path (`libmorphlm.dylib` on macOS, `libmorphlm.so` on Linux):

```bash
python scripts/fpga_lm_pretrain.py --width 16 --build-csim build/fpga-lm-csim-d16
python scripts/fpga_lm_pretrain.py --backend csim --width 16 \
  --library /path/printed/by/the/build/command \
  --corpus data/tinyshakespeare.txt --output build/fpga-lm-pilot-d16 \
  --steps 8 --eval-every 4 --eval-windows 2
```

Use fresh build/output paths on reruns. The output includes an immutable run
manifest, append-only metrics, binary checkpoints with digests and a terminal
summary or failure. Large checkpoint payloads belong in ignored build storage
or private artifact storage; retain compact records under `manuscript/`.
`csim` is CPU execution, **not silicon throughput**. The D512 configuration is
a small character model, not a scaled LLM. Reported LUT-normalized NLL is not
softmax perplexity; neither a few updates nor matching predictions establishes
useful learning. Validation still invokes backward with learning rate zero.

For AWS, `python -S scripts/fpga_lm_aws.py` prints deliberately invalid
configuration placeholders. Its `plan`, legacy review-only `render` and `bundle`
are offline; `preflight` explicitly performs read-only AWS queries. Its direct
`launch` remains disabled. The separate pilot lifecycle consists of:

- `fpga_lm_control.py`: exclusive durable reservation, immutable launch intent,
  exact-instance reconciliation and termination confirmation.
- `fpga_lm_watchdog.py`: run-scoped Standard Step Functions definition and IAM
  policy, armed and health-checked before EC2 launch.
- `fpga_lm_guest.py`: current guest renderer, instance-bound S3 start permit,
  pinned offline NumPy runtime and SDK build, identical 64-update CSIM/PCI runs
  and full checkpoints. The Linux x86_64 runtime selects hash-verified NumPy
  1.24.4 for the historical AMI's CPython 3.8 or NumPy 2.2.6 for
  CPython 3.10/3.11/3.12, retains its license files, and imports
  from an isolated temporary directory before building. No guest package manager
  or public package index is used; unsupported environments fail closed.
  This legacy research-runner exception does not expand the library's supported
  Python versions or recommend the deprecated AMI for general deployment.
  Runner invocations use absolute paths: Python 3.8 can otherwise supply a
  relative `__file__`, breaking the frozen provenance check before compilation.
- `fpga_lm_cloud.py`: explicit AWS CLI adapter, independent policy/history
  readback, conditional start authorization and verified artifact retrieval.

`python scripts/fpga_lm_cloud.py status` and `--help` are offline. The adapter
requires explicit `run --config ... --state ... --output NEW_DIRECTORY --execute`;
`recover --config ... --state ... --execute` only reconciles and terminates an
owned run, never relaunches it. Configuration is deliberately not auto-discovered.
For an interrupted preparation that has **never** reserved or launched, explicit
`fpga_lm_aws.py preflight-staged --config ... --ledger ...` checks the same resources
and requires exactly the three approved source/SDK/NumPy objects, with bounded
expected-owner byte readback and SHA256 verification. It does not fabricate an
empty prefix or establish instance/watchdog absence. Prove those separately,
preserve the original configuration, and generate fresh timing into new files
before the normal live controller gates. Never refresh an attempted run's deadline.
`cancel-unlaunched --config ... --state ... --execute` is only for an intent
whose durable history proves that launch was never attempted. It authenticates
the account and exhaustively checks bounded exact-token discovery before closing
the local record; ambiguity or any launch attempt requires recovery instead.
Cancellation retains the full reservation and does not claim instance termination.
Preserve the same state ledger across runs, and do not reconcile its reservation
until termination and ancillary resources/costs have been reviewed. A $10
reservation and a $900 accounting ceiling plus $100 contingency are operational
controls, **not an AWS billing cap** or a distributed account-wide lock.
The historical sentinel is not adopted by this controller. See the research
plan for live integration status and the explicit temporary-public-IP exception.

Do not substitute the old `run_morph_lm_d512_*` launchers: their dry-run,
watchdog and failure-handling contracts do not meet this plan. The old
`_validate_kernel_morph_lm.py` / `_csim_train_morph_lm.py` helpers are disabled
before loading a native library: their obsolete buffer omits the aligned LUT
region and could cause an out-of-bounds access. Their historical reference
code is preserved for inspection; use the new runner for execution.
