"""The paper's numerical tables must be derivable from identified raw records."""
import copy
import json
import subprocess
import sys

import pytest

from manuscript.tools import generate_tables as tables


def fixtures():
    sources = tables.read_sources()
    rows = [json.loads(line) for line in sources[f"{tables.SAVED}/saved_state.jsonl"].splitlines()]
    manifest = json.loads(sources[f"{tables.SAVED}/manifest.json"])
    return rows, manifest


def test_generated_tables_are_current():
    subprocess.run([sys.executable, "manuscript/tools/generate_tables.py", "--check"],
                   cwd=tables.REPO, check=True, capture_output=True, text=True)


def test_retained_ratios_use_boundary_matched_portable_bytes():
    rows, manifest = fixtures()
    result = tables.retained_records(rows, manifest)
    assert [round(r["supplied_cotangent_ratio"], 2) for r in result] == [1.97, 2.74, 2.96]
    assert [round(r["loss_inclusive_ratio"], 2) for r in result] == [1.69, 2.48, 2.79]
    assert result[1]["portable_bytes"] == 89799
    assert result[1]["boundary_target_bytes"] == 768


def test_legacy_mixed_boundary_ratios_are_explicitly_ignored():
    rows, manifest = fixtures()
    expected = tables.retained_records(rows, manifest)
    rows[0]["ratio_checkpoint_over_packed"] = 123456
    assert tables.retained_records(rows, manifest) == expected


@pytest.mark.parametrize("mutation,match", [
    (lambda rows: rows.append(copy.deepcopy(rows[0])), "duplicate"),
    (lambda rows: rows.pop(), "missing"),
    (lambda rows: rows[0].update(d_model=65), "unexpected"),
    (lambda rows: rows[0].update(boundary_logits=1), "boundary"),
    (lambda rows: rows[0].update(checkpoint_retained=1), "checkpoint-set"),
])
def test_ambiguous_or_inconsistent_selected_records_fail(mutation, match):
    rows, manifest = fixtures()
    mutation(rows)
    with pytest.raises(ValueError, match=match):
        tables.retained_records(rows, manifest)


def test_raw_source_digest_drift_is_not_silently_regenerated(tmp_path):
    first = next(iter(tables.SOURCE_SHA256))
    changed = tmp_path / first
    changed.parent.mkdir(parents=True)
    changed.write_text("altered raw evidence\n")
    with pytest.raises(ValueError, match="digest mismatch"):
        tables.read_sources(tmp_path)


def test_hardware_observables_are_not_full_state_or_independent_tokens():
    result = tables.hardware_records(tables.read_sources())
    assert len(result["d512_compared_record"]["preds16"]) == 16
    assert [r["observable_reference_match"] for r in result["d512"]] == [True, False, False]
    assert result["d512"][-1]["processed_positions"] == 256000
    assert result["mnist"][-1]["driver_test_accuracy"] == 0.7475
    assert result["mnist_independent_test_accuracy"] == 0.7413


def test_hardware_mismatch_and_ambiguous_log_fail():
    sources = tables.read_sources()
    sources[f"{tables.D512}/golden_fused.jsonl"] = "{}"
    with pytest.raises(ValueError, match="comparison failed"):
        tables.hardware_records(sources)
    sources = tables.read_sources()
    sources[f"{tables.D512}/train_30.err"] *= 2
    with pytest.raises(ValueError, match="ambiguous D512"):
        tables.hardware_records(sources)
