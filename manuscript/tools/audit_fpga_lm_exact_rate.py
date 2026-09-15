"""Pinned-parent audit of one immutable exact-rate producer, never training.

The completed exact-fit audit is an explicit hash-pinned trust anchor, not a
newly repeated ancestor numerical audit. Every ancestor payload/source identity
is still checked. Only the exact-rate queries are independently rederived;
its saved gradients, not approximately equal recomputations, replay Adam.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from manuscript.tools import diagnose_fpga_lm_exact_rate as rate  # noqa: E402

SCHEMA = "tikitorch.fpga_lm_exact_rate.pinned_ancestor_audit.v1"
DEADLINE_S = 3600
PLAN = REPO / "manuscript/research/PLAN.md"
TRUST = {
    "rate_public_sha256": "e393750a7191c9a64c1a4f3cb89150b6355ba61838f551ba7361e42c7ff1859b",
    "rate_manifest_sha256": "15e0e7dbf963ea62354101a77737cf040859ef80e9670f690918ddf486b8a10a",
    "rate_seal_sha256": "c00578a494d38a856539ece1dc7e9b917a830729a954d18fb9f82658672d277f",
    "parent_audit_sha256": "8506cf77b68e2cb0a6e02aa43c441119f86aa226ebb46d7b1648444745bb92a0",
    "parent_public_sha256": "6c36b6145626b610b3cc46645dfb69e27b16b3f6162531c8fab5e3a852c1c74e",
    "parent_manifest_sha256": "eefa32acef07640554c23f3ab0b8ca6de91d61bafd3e37313658df3f529de12b",
    "parent_seal_sha256": "35c18b1016ab485ef19756866e628580a95e47f57f8931bdeb03123bacf4debf",
    "rate_source_count": 61, "rate_artifact_count": 403, "rate_artifact_bytes": 304332071,
    "parent_source_count": 59, "parent_artifact_count": 1203, "parent_artifact_bytes": 935065134,
    "ancestor_tree_count": 10, "ancestor_external_file_count": 163,
}


def source_identity():
    result = dict(rate.source_identity())
    for name in ("manuscript/tools/audit_fpga_lm_exact_rate.py", "tests/test_fpga_lm_exact_rate_audit.py"):
        result[name] = rate.old.digest((REPO / name).read_bytes())
    return result


def _pin(path, key):
    path = rate.head._regular(path)
    rate.equal(rate.old.digest(path.read_bytes()), TRUST[key], "immutable trust pin: " + key)
    return path


def _inventory(root, count_key, bytes_key):
    actual = rate.prior.artifact_manifest(root)
    rate.equal(rate.read(root / "artifact_manifest.json"), actual, "complete actual sealed payload")
    rate.equal(len(actual), TRUST[count_key], "pinned payload count")
    rate.equal(sum(row["bytes"] for row in actual), TRUST[bytes_key], "pinned payload bytes")
    return actual


def _predecessor(binding):
    root = Path(binding["prior_run"]).resolve(strict=True)
    record = _pin(binding["prior_record"], "parent_public_sha256")
    audit = _pin(binding["prior_audit"], "parent_audit_sha256")
    _pin(root / "manifest.json", "parent_manifest_sha256")
    _pin(root / "artifact_manifest.json", "parent_seal_sha256")
    manifest = rate.read(root / "manifest.json")
    rate.equal(manifest["schema"], rate.parent.SCHEMA, "known predecessor schema")
    rate.equal(manifest["run_directory"], str(root), "pinned predecessor directory")
    rate.equal(manifest["recipe"], rate.parent.recipe(), "unchanged predecessor recipe")
    rate.equal(manifest["source_hashes"], rate.parent.source_identity(), "all predecessor source bytes")
    rate.equal(len(manifest["source_hashes"]), TRUST["parent_source_count"], "predecessor source count")
    artifacts = _inventory(root, "parent_artifact_count", "parent_artifact_bytes")
    rate.equal({row["file"] for row in artifacts}, rate.parent.expected_artifacts(), "complete predecessor membership")
    rate.equal(rate.read(record), rate.parent.public_record(root, rate.read(root / "summary.json"), artifacts),
               "pinned predecessor public/raw agreement")
    # This verifies receipt identities, literal completion/count/scope fields
    # and timestamps only. It never invokes a predecessor model or optimizer.
    rate._audit_receipt(root, record, audit)
    receipt = rate.read(audit)
    prior_closure = manifest["binding"]["closure"]
    rate.equal(receipt["ancestor_raw_trees_verified"], len(prior_closure["roots"]), "completed audit ancestor count")
    rate.equal(receipt["external_ancestor_file_identities_verified"], len(prior_closure["files"]), "completed audit external count")
    rate.equal(receipt["independent_tropical_window_forwards"], 1048, "completed tropical forward coverage")
    rate.equal(receipt["independent_gru_batch_forwards"], 131, "completed explicit GRU coverage")
    rate.equal(receipt["independent_forward_window_equivalents"], 1572, "completed total window coverage")
    rate.equal(receipt["recomputed_gradient_pair_observations"], 196608, "completed gradient observation count")
    return root, record, audit, manifest


def validate_inputs(run, record):
    """Fresh full-byte validation with an explicitly pinned semantic ancestor.

    No CLI/config can supply alternate pins. A private token is created only
    after reconstructing the entire rate binding from the trusted parent and
    actual current source/input/artifact identities.
    """
    sources = source_identity()
    root = Path(run).resolve(strict=True)
    record = _pin(record, "rate_public_sha256")
    _pin(root / "manifest.json", "rate_manifest_sha256")
    _pin(root / "artifact_manifest.json", "rate_seal_sha256")
    manifest = rate.read(root / "manifest.json")
    rate.equal(manifest["schema"], rate.SCHEMA, "exact immutable producer schema")
    rate.equal(manifest["run_directory"], str(root), "exact immutable producer path")
    rate.equal(manifest["source_hashes"], rate.source_identity(), "all original rate source bytes")
    rate.equal(len(manifest["source_hashes"]), TRUST["rate_source_count"], "rate source count")
    rate.equal(manifest["recipe"], rate.recipe(), "unchanged producer recipe")
    rate.equal(rate.read(root / "protocol.json"), rate.recipe(), "unchanged saved protocol")
    rate.equal(rate.old.digest((root / "declaration.md").read_bytes()), manifest["plan_sha256"], "original declaration")
    artifacts = _inventory(root, "rate_artifact_count", "rate_artifact_bytes")
    rate.equal({row["file"] for row in artifacts}, rate.expected_artifacts(), "complete exact rate membership")
    rate.equal(rate.read(record), rate.public_record(root, rate.read(root / "summary.json"), artifacts),
               "pinned rate public/raw agreement")
    parent_root, parent_record, parent_audit, parent_manifest = _predecessor(manifest["binding"])
    print(json.dumps({"stage": "pinned_parent_receipt_verified", "ancestor_numeric_replay": False}), flush=True)
    closure = rate.head.ancestry_snapshot(parent_root, parent_record, parent_audit)
    rate.equal(len(closure["roots"]), TRUST["ancestor_tree_count"], "complete ancestor tree count")
    rate.equal(len(closure["files"]), TRUST["ancestor_external_file_count"], "complete external identity count")
    raw, dataset = rate.pilot.corpus_plan(Path(manifest["binding"]["dataset"]["path"]), 1024, 88)
    rate.equal(dataset, parent_manifest["binding"]["dataset"], "canonical same corpus and split")
    initial = parent_root / "mixed/state_000000.npz"
    canonical = {"prior_run": str(parent_root), "prior_record": str(parent_record), "prior_audit": str(parent_audit),
                 "prior_binding": parent_manifest["binding"], "dataset": dataset, "initial_state": str(initial),
                 "initial_state_identity": rate.old.file_identity(initial),
                 "initial_scores": [rate.old.file_identity(parent_root / "mixed" / f"score_initial_{index:03d}_1.npz")
                                    for index in rate.WINDOWS],
                 "controls": rate._controls(parent_record), "closure": closure}
    rate.equal(manifest["binding"], canonical, "fully reconstructed pinned-trust rate binding")
    _pin(record, "rate_public_sha256")
    _pin(root / "manifest.json", "rate_manifest_sha256")
    _pin(root / "artifact_manifest.json", "rate_seal_sha256")
    rate.equal(source_identity(), sources, "all auditor/source bytes unchanged during validation")
    validation = rate.head._Validation(canonical, manifest["source_hashes"])
    # Frozen guards now check the independently reconstructed private token and
    # actual closure again; no predecessor numeric check is substituted here.
    rate.frozen(manifest, validation)
    facts = {"rate_source_files_verified": len(manifest["source_hashes"]), "auditor_source_files_verified": len(sources),
             "rate_artifact_files_verified": len(artifacts), "rate_artifact_payload_bytes": sum(row["bytes"] for row in artifacts),
             "ancestor_raw_trees_verified": len(closure["roots"]), "external_ancestor_file_identities_verified": len(closure["files"]),
             "completed_predecessor_audit_reused": True, "predecessor_numerical_replay_executed": False,
             "trust": dict(TRUST), "source_hashes": sources}
    return manifest, raw, validation, facts


class Deadline:
    """Owned wall-clock and monotonic deadline, including input validation."""

    def __init__(self):
        self.started_utc = time.time()
        self.started_monotonic = time.monotonic()
        self.previous_handler = None
        self.installed = False
        self.stop = threading.Event()
        self.watcher = None

    def elapsed(self):
        return time.time() - self.started_utc, time.monotonic() - self.started_monotonic

    def check(self):
        utc, monotonic = self.elapsed()
        if not all(math.isfinite(value) for value in (utc, monotonic)) or min(utc, monotonic) < 0:
            raise RuntimeError("audit clock moved backwards")
        if max(utc, monotonic) >= DEADLINE_S:
            raise TimeoutError("fixed UTC/monotonic pinned-audit deadline exceeded")
        return utc, monotonic

    def _expired(self, signum, frame):
        raise TimeoutError("fixed UTC/monotonic pinned-audit deadline exceeded")

    def _watch(self):
        while not self.stop.wait(0.25):
            try:
                self.check()
            except BaseException:
                if not self.stop.is_set():
                    os.kill(os.getpid(), signal.SIGALRM)
                return

    def arm(self):
        self.check()
        self.previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._expired)
        self.installed = True
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_S - max(self.elapsed()))
        self.watcher = threading.Thread(target=self._watch, name="exact-rate-audit-deadline", daemon=True)
        self.watcher.start()

    def close(self):
        self.stop.set()
        try:
            if self.watcher is not None and self.watcher.ident is not None:
                self.watcher.join(timeout=1)
                if self.watcher.is_alive():
                    raise RuntimeError("owned audit deadline watcher did not stop")
        finally:
            if self.installed:
                try:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                finally:
                    signal.signal(signal.SIGALRM, self.previous_handler)
                self.installed = False


def _utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _unchanged(root, record, manifest, validation, facts):
    rate.equal(TRUST, facts["trust"], "fixed invocation trust anchors")
    rate.equal(source_identity(), facts["source_hashes"], "auditor and original sources unchanged")
    _pin(record, "rate_public_sha256")
    _pin(root / "manifest.json", "rate_manifest_sha256")
    _pin(root / "artifact_manifest.json", "rate_seal_sha256")
    binding = manifest["binding"]
    for path, key in ((binding["prior_record"], "parent_public_sha256"), (binding["prior_audit"], "parent_audit_sha256"),
                      (Path(binding["prior_run"]) / "manifest.json", "parent_manifest_sha256"),
                      (Path(binding["prior_run"]) / "artifact_manifest.json", "parent_seal_sha256")):
        _pin(path, key)
    _inventory(root, "rate_artifact_count", "rate_artifact_bytes")
    rate.frozen(manifest, validation)


def _receipt(root, record, output, audit_manifest, facts, result, clock):
    utc_elapsed, monotonic_elapsed = clock.check()
    return {"schema": SCHEMA, "complete": True, "independent_recompute_completed": True,
            "rate_run_directory": str(root), "rate_public_record": str(record), "audit_directory": str(output),
            "audit_manifest_sha256": rate.old.digest(audit_manifest),
            "started_at_utc": _utc(clock.started_utc),
            "verification_completed_at_utc": _utc(clock.started_utc + utc_elapsed),
            "verification_elapsed_utc_seconds": utc_elapsed,
            "verification_elapsed_monotonic_seconds": monotonic_elapsed, "deadline_seconds": DEADLINE_S,
            "publication_postchecks_within_deadline": True,
            "timing_scope": "verification completion precedes atomic publication; all publication/cleanup postchecks must also fit the same deadline",
            "verification": facts, "independent_tropical_window_vjps": 512,
            "independent_full_batch_gradients": 128, "independent_score_readouts": 12,
            "independent_tropical_window_forwards": 524, "complete_saved_states_verified": 129,
            "optimizer_commits_replayed": 128,
            "recomputed_gradient_pair_observations": result["new_gradient_pair_exposures"],
            "rate_numerical_replay_complete": True, "historical_numerical_gates": result["gates"],
            "source_input_raw_seal_public_bytes_unchanged": True,
            "predecessor_audit_trust": "hard-pinned completed exact-fit audit; full current ancestor bytes checked, predecessor numerical trajectory not repeated",
            "new_training_updates": 0, "new_gradient_pair_exposures": 0, "new_native_training_calls": 0,
            "new_optimizer_trajectory_executed": False, "heldout_corpus_evaluated": False,
            "physical_fpga_execution": False, "paid_execution_authorized": False,
            "recipe_promoted": False, "broader_study_authorized": False,
            "scope": "new independent rate-query reconstruction plus exact saved-gradient state replay; immutable completed parent audit reused explicitly, not independent-data learning"}


def run(config):
    clock = Deadline()  # Bound the whole invocation, not just its model loop.
    output = target = staged = staging_directory = None
    published = owned_inode = None
    created = False
    closed = False
    stage = "setup"
    try:
        clock.arm()
        output, target = (rate.prior.capacity._new_external_path(path) for path in (config.output, config.receipt))
        if target == output or output in target.parents or target in output.parents:
            raise ValueError("new public receipt must be outside the private audit output")
        if not target.parent.is_dir():
            raise FileNotFoundError("new receipt parent must already exist")
        root = Path(config.run).resolve(strict=True)
        record = rate.head._regular(config.record)
        sources, trust = source_identity(), dict(TRUST)
        output.mkdir(parents=True, exist_ok=False)
        created = True
        declaration = PLAN.read_bytes()
        audit_manifest = {"schema": SCHEMA, "complete": False, "operation": "independent rate audit with pinned completed ancestor",
                          "run_directory": str(root), "record": str(record), "audit_directory": str(output),
                          "receipt": str(target), "source_hashes": sources, "trust": trust,
                          "started_at_utc": _utc(clock.started_utc), "deadline_seconds": DEADLINE_S,
                          "declaration_sha256": rate.old.digest(declaration),
                          "command": ["python", "manuscript/tools/audit_fpga_lm_exact_rate.py", "--run", str(root),
                                      "--record", str(record), "--output", str(output), "--receipt", str(target)]}
        rate.pilot.write_json(output / "audit_manifest.json", audit_manifest)
        audit_manifest_bytes = (output / "audit_manifest.json").read_bytes()
        with (output / "declaration.md").open("xb") as stream:
            stream.write(declaration)
        rate.torch.set_num_threads(1)
        stage = "input_validation"
        print(json.dumps({"stage": stage, "deadline_seconds": DEADLINE_S, "new_training_updates": 0}), flush=True)
        manifest, raw, validation, facts = validate_inputs(root, record)
        rate.equal(facts["source_hashes"], sources, "no source relabeling during validation")
        rate.equal(facts["trust"], trust, "no trust relabeling during validation")
        clock.check()
        rate.pilot.write_json(output / "verification.json", facts)
        stage = "independent_rate_replay"
        print(json.dumps({"stage": stage, "window_vjps": 512, "score_readouts": 12, "ancestor_numerical_replay": False}), flush=True)
        # The original implementation rederives every independent VJP but uses
        # SAVED G for each eager-Adam transition, preserving exact state replay.
        result = rate._check(root, record, True, validation, raw)
        clock.check()
        stage = "final_integrity"
        _unchanged(root, record, manifest, validation, facts)
        rate.equal((output / "audit_manifest.json").read_bytes(), audit_manifest_bytes, "private declaration identity")
        rate.equal((output / "declaration.md").read_bytes(), declaration, "private protocol bytes")
        receipt = _receipt(root, record, output, audit_manifest_bytes, facts, result, clock)
        staging_directory = Path(tempfile.mkdtemp(prefix=".tt-pinned-rate-audit-", dir=target.parent))
        staged = staging_directory / "receipt.json"
        rate.pilot.write_json(staged, receipt)
        published = rate.old.digest(staged.read_bytes())
        identity = staged.stat()
        owned_inode = identity.st_dev, identity.st_ino
        clock.check()
        stage = "publication"
        os.link(staged, target)
        _unchanged(root, record, manifest, validation, facts)
        rate.equal(rate.old.digest(target.read_bytes()), published, "exact newly published receipt bytes")
        rate.equal((output / "audit_manifest.json").read_bytes(), audit_manifest_bytes, "private manifest unchanged through publication")
        rate.equal((output / "declaration.md").read_bytes(), declaration, "private declaration unchanged through publication")
        clock.check()
        staged.unlink()
        staging_directory.rmdir()
        clock.close()
        closed = True
        completed_utc_elapsed, completed_monotonic_elapsed = clock.check()
        print(json.dumps({"stage": "complete", "independent_rate_replay": True, "ancestor_numerical_replay": False,
                          "completed_at_utc": _utc(clock.started_utc + completed_utc_elapsed),
                          "elapsed_utc_seconds": completed_utc_elapsed,
                          "elapsed_monotonic_seconds": completed_monotonic_elapsed,
                          "receipt_sha256": published}), flush=True)
        return receipt
    except BaseException as error:
        if target is not None and target.is_file() and not target.is_symlink() and owned_inode is not None:
            identity = target.stat()
            if (identity.st_dev, identity.st_ino) == owned_inode and rate.old.digest(target.read_bytes()) == published:
                rejected = output / "rejected_public.json"
                if rejected.exists():
                    raise RuntimeError("cannot overwrite retained rejected audit receipt") from error
                target.rename(rejected)
        if created and not (output / "failure.json").exists():
            utc_elapsed, monotonic_elapsed = clock.elapsed()
            rate.pilot.write_json(output / "failure.json", {"schema": SCHEMA, "complete": False,
                "independent_recompute_completed": False, "stage": stage, "type": type(error).__name__, "error": str(error),
                "started_at_utc": _utc(clock.started_utc), "failure_observed_at_utc": _utc(time.time()),
                "elapsed_utc_seconds": utc_elapsed, "elapsed_monotonic_seconds": monotonic_elapsed,
                "deadline_seconds": DEADLINE_S, "new_training_updates": 0, "recipe_promoted": False})
        raise
    finally:
        try:
            if staged is not None and staged.exists():
                staged.unlink()
            if staging_directory is not None and staging_directory.exists():
                staging_directory.rmdir()
        finally:
            if not closed:
                clock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "record", "output", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args)
    print(json.dumps({"complete": receipt["complete"], "receipt": str(args.receipt),
                      "pinned_predecessor_audit_reused": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
