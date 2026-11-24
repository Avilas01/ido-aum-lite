"""Hand-calculated Step 2 tests: NumPy only, no downloads, model or torch."""
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from src.statistics import SampleStatistics


@pytest.fixture
def epochs():
    first = np.full((4, 10), -1000, dtype=np.float32)
    first[0, [0, 1]] = [3, 1]  # label 0: right, margin 2
    first[1, [0, 1]] = [4, 1]  # label 1: wrong, margin -3
    first[2, [0, 2]] = [1, 2]  # label 2: right, margin 1
    first[3, [0, 3]] = [2, 2]  # label 3: tied, argmax 0 is wrong, margin 0
    second = np.full((4, 10), -1000, dtype=np.float32)
    second[0, [0, 1]] = [-1, 2]  # margin -3
    second[1, [0, 1]] = [1, 3]   # margin 2
    second[2, [0, 2]] = [4, 1]   # margin -3
    second[3, [0, 3]] = [1, 3]   # margin 2
    return first, second, np.arange(4), np.arange(4)


def test_hand_calculated_counts_margins_ce_and_shuffled_ids(epochs):
    first, second, labels, ids = epochs
    tracker = SampleStatistics(4)
    order = np.array([2, 0, 3, 1])
    summary = tracker.update_epoch(first[order], labels[order], ids[order], epoch=1)
    np.testing.assert_array_equal(tracker.we, [0, 1, 0, 1])
    np.testing.assert_array_equal(tracker.margin_sum, [2, -3, 1, 0])
    np.testing.assert_array_equal(tracker.aum, [2, -3, 1, 0])
    expected_ce = (math.log1p(math.exp(-2)) + math.log1p(math.exp(3)) + math.log1p(math.exp(-1)) + math.log(2)) / 4
    assert summary["noisy_ce"] == pytest.approx(expected_ce)
    tracker.update_epoch(second, labels, ids, epoch=2)
    np.testing.assert_array_equal(tracker.we, [1, 1, 1, 1])
    np.testing.assert_array_equal(tracker.margin_sum, [-1, -1, -2, 2])
    np.testing.assert_array_equal(tracker.aum, [-.5, -.5, -1, 1])
    np.testing.assert_array_equal(tracker.observation_counts, [2, 2, 2, 2])
    assert tracker.we.dtype == tracker.observation_counts.dtype == np.int32
    assert tracker.margin_sum.dtype == tracker.aum.dtype == np.float32


def test_raw_logits_are_used_and_logit_shifts_do_not_change_margins(epochs):
    logits, _, labels, ids = epochs
    left, right = SampleStatistics(4), SampleStatistics(4)
    left.update_epoch(logits, labels, ids, epoch=1)
    right.update_epoch(logits + 500, labels, ids, epoch=1)
    np.testing.assert_array_equal(left.aum, right.aum)
    assert left.aum[1] == -3  # Would be impossible for a probability margin.


@pytest.mark.parametrize("failure", ["missing", "duplicate", "range", "float_ids", "bad_label", "float_labels", "nan", "inf", "shape", "overflow"])
def test_invalid_full_pass_never_partially_updates(epochs, failure):
    logits, _, labels, ids = epochs
    if failure == "missing": logits, labels, ids = logits[:-1], labels[:-1], ids[:-1]
    elif failure == "duplicate": ids = np.array([0, 1, 2, 2])
    elif failure == "range": ids = np.array([0, 1, 2, 4])
    elif failure == "float_ids": ids = ids.astype(float)
    elif failure == "bad_label": labels[0] = 10
    elif failure == "float_labels": labels = labels.astype(float)
    elif failure == "nan": logits[0, 0] = np.nan
    elif failure == "inf": logits[0, 0] = np.inf
    elif failure == "shape": logits = logits[:, :9]
    elif failure == "overflow": logits = logits.astype(np.float64); logits[0, 0] = 1e40
    tracker = SampleStatistics(4)
    with pytest.raises(ValueError):
        tracker.update_epoch(logits, labels, ids, epoch=1)
    assert tracker.tracked_epochs == 0
    np.testing.assert_array_equal(tracker.we, np.zeros(4))
    np.testing.assert_array_equal(tracker.margin_sum, np.zeros(4))
    np.testing.assert_array_equal(tracker.observation_counts, np.zeros(4))


def test_duplicate_or_skipped_epoch_is_rejected(epochs):
    first, _, labels, ids = epochs
    tracker = SampleStatistics(4)
    with pytest.raises(ValueError, match="next completed epoch"):
        tracker.update_epoch(first, labels, ids, epoch=2)
    tracker.update_epoch(first, labels, ids, epoch=1)
    with pytest.raises(ValueError, match="duplicate/skipped"):
        tracker.update_epoch(first, labels, ids, epoch=1)
    np.testing.assert_array_equal(tracker.observation_counts, np.ones(4))


def test_no_aum_before_first_observation(tmp_path):
    tracker = SampleStatistics(4)
    with pytest.raises(ValueError, match="undefined"):
        _ = tracker.aum
    with pytest.raises(ValueError):
        tracker.save_snapshot(tmp_path / "epoch_000.npz")
    cold = SampleStatistics.from_state_dict(tracker.state_dict())
    assert cold.tracked_epochs == 0


def test_history_roundtrip_branch_copies_and_conflicting_snapshot_protection(tmp_path, epochs):
    first, second, labels, ids = epochs
    tracker = SampleStatistics(4)
    tracker.update_epoch(first, labels, ids, epoch=1)
    first_path = tracker.save_snapshot(tmp_path / "history with spaces" / "epoch_001.npz")
    assert tracker.save_snapshot(first_path) == first_path  # Idempotent identical save.
    branch = tracker.copy()
    state = tracker.state_dict()
    state["we"][0] = 99
    assert tracker.we[0] == 0
    branch.update_epoch(second, labels, ids, epoch=2)
    np.testing.assert_array_equal(tracker.aum, [2, -3, 1, 0])
    second_path = branch.save_snapshot(first_path.parent / "epoch_002.npz")
    restored = SampleStatistics.load_snapshot(second_path)
    np.testing.assert_array_equal(restored.aum, [-.5, -.5, -1, 1])
    np.testing.assert_array_equal(SampleStatistics.load_snapshot(first_path).we, [0, 1, 0, 1])
    with pytest.raises(FileExistsError):
        branch.save_snapshot(first_path)


@pytest.mark.parametrize("bad", ["counts", "we", "margins", "dtype"])
def test_corrupted_state_rejected(epochs, bad):
    first, _, labels, ids = epochs
    tracker = SampleStatistics(4)
    tracker.update_epoch(first, labels, ids, epoch=1)
    state = tracker.state_dict()
    if bad == "counts": state["observation_counts"][0] = 0
    elif bad == "we": state["we"][0] = 2
    elif bad == "margins": state["margin_sum"][0] = np.nan
    elif bad == "dtype": state["we"] = state["we"].astype(float)
    with pytest.raises(ValueError):
        SampleStatistics.from_state_dict(state)


def test_imports_do_not_require_local_torch():
    root = Path(__file__).resolve().parents[1]
    script = "import sys; import src.model,src.statistics,src.smoke; assert 'torch' not in sys.modules and 'torchvision' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], cwd=root, check=True)


@pytest.mark.parametrize("field", ["aum", "tracked_epochs"])
def test_snapshot_rejects_inconsistent_aum_or_noninteger_epoch(tmp_path, epochs, field):
    first, _, labels, ids = epochs
    tracker = SampleStatistics(4)
    tracker.update_epoch(first, labels, ids, epoch=1)
    path = tracker.save_snapshot(tmp_path / "epoch_001.npz")
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if field == "aum":
        values["aum"][0] += 1
    else:
        values["tracked_epochs"] = np.array(1.5)
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError):
        SampleStatistics.load_snapshot(path)



def test_persistently_wrong_examples_count_every_epoch_not_only_transitions(epochs):
    first, _, labels, ids = epochs
    tracker = SampleStatistics(4)
    tracker.update_epoch(first, labels, ids, epoch=1)
    tracker.update_epoch(first, labels, ids, epoch=2)
    np.testing.assert_array_equal(tracker.we, [0, 2, 0, 2])
    np.testing.assert_array_equal(tracker.aum, [2, -3, 1, 0])
