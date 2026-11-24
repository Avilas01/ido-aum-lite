"""Step 1 construction/API tests: synthetic inputs only, no torch or downloads."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from config import load_config
from src import data


@pytest.fixture
def original_labels():
    return np.repeat(np.arange(10, dtype=np.int64), 5000)


@pytest.mark.parametrize("size,ratio,expected", [(20000, .2, 4000), (20000, .4, 8000), (10000, .2, 2000), (10000, .4, 4000)])
def test_exact_reproducible_noise_and_stratification(original_labels, size, ratio, expected):
    before = original_labels.copy()
    training, analysis = data.prepare_labels(original_labels, size, ratio, 42)
    again, again_analysis = data.prepare_labels(original_labels, size, ratio, 42)
    np.testing.assert_array_equal(original_labels, before)
    assert len(training.noisy_labels) == size
    assert len(np.unique(training.original_indices)) == size
    np.testing.assert_array_equal(np.bincount(analysis.clean_labels, minlength=10), np.full(10, size // 10))
    np.testing.assert_array_equal(analysis.clean_labels, original_labels[training.original_indices])
    mask = training.noisy_labels != analysis.clean_labels
    np.testing.assert_array_equal(mask, analysis.is_corrupted)
    assert mask.sum() == expected
    assert np.all(training.noisy_labels[mask] != analysis.clean_labels[mask])
    np.testing.assert_array_equal(training.noisy_labels[~mask], analysis.clean_labels[~mask])
    assert set(training.noisy_labels) == set(range(10))
    assert training.summary() == again.summary()
    np.testing.assert_array_equal(training.original_indices, again.original_indices)
    np.testing.assert_array_equal(analysis.is_corrupted, again_analysis.is_corrupted)
    assert data.noise_seed(42, .2) != data.noise_seed(42, .4)


def test_noise_conditions_use_the_same_image_ids(original_labels):
    low, _ = data.prepare_labels(original_labels, ratio=.2)
    high, _ = data.prepare_labels(original_labels, ratio=.4)
    np.testing.assert_array_equal(low.original_indices, high.original_indices)
    assert low.summary()["subset_checksum"] == high.summary()["subset_checksum"]
    assert low.summary()["noisy_label_checksum"] != high.summary()["noisy_label_checksum"]
    changed_seed, _ = data.prepare_labels(original_labels, seed=43)
    assert changed_seed.summary()["subset_checksum"] != low.summary()["subset_checksum"]


@pytest.mark.parametrize("ratio", [-.1, 0, .3, 1, float("nan"), True])
def test_invalid_noise_conditions_are_rejected(ratio):
    with pytest.raises(ValueError):
        data.symmetric_noise(np.arange(10), ratio, seed=42)


@pytest.mark.parametrize("labels", [[-1, 0], [0, 10], [0.5, 1.0], [], [[0, 1]], [True, False]])
def test_invalid_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        data.stratified_indices(labels, train_size=10)


@pytest.mark.parametrize("size", [0, 11, True, 50010])
def test_invalid_or_unavailable_subset_sizes_are_rejected(original_labels, size):
    with pytest.raises(ValueError):
        data.stratified_indices(original_labels, size)


@pytest.fixture
def tiny_split():
    return data.prepare_labels(np.repeat(np.arange(10), 5), train_size=20)


def test_metadata_is_separate_immutable_and_roundtrips_without_analysis_reads(tmp_path, tiny_split):
    training, analysis = tiny_split
    root = tmp_path / "run with spaces"
    data.save_metadata(training, analysis, root)
    data.save_metadata(training, analysis, root)  # Identical preparation is idempotent.
    loaded = data.load_training_metadata(root)
    assert loaded.summary() == training.summary()
    with np.load(root / "training.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"original_indices", "noisy_labels"}
    assert not hasattr(loaded, "clean_labels") and not hasattr(loaded, "is_corrupted")
    with pytest.raises(ValueError):
        loaded.noisy_labels[0] = 0
    # Training metadata can be loaded when analysis artifacts are unavailable.
    (root / "analysis.npz").unlink()
    assert data.load_training_metadata(root).summary() == training.summary()
    with pytest.raises(ValueError, match="Incomplete"):
        data.save_metadata(training, analysis, root)


def test_metadata_rejects_tampering_and_conflicting_configuration(tmp_path, tiny_split):
    training, analysis = tiny_split
    data.save_metadata(training, analysis, tmp_path)
    other_training, other_analysis = data.prepare_labels(np.repeat(np.arange(10), 5), train_size=20, ratio=.4)
    with pytest.raises(ValueError, match="refusing overwrite"):
        data.save_metadata(other_training, other_analysis, tmp_path)
    labels = training.noisy_labels.copy()
    labels[0] = (labels[0] + 1) % 10
    np.savez_compressed(tmp_path / "training.npz", original_indices=training.original_indices, noisy_labels=labels)
    with pytest.raises(ValueError, match="checksum"):
        data.load_training_metadata(tmp_path)


def test_training_views_keep_ids_and_do_not_leak_clean_labels_or_call_unused_strong(tiny_split):
    training, analysis = tiny_split
    images = np.zeros((20, 32, 32, 3), dtype=np.uint8)
    images[:, :, :, 0] = np.arange(20)[:, None, None]
    bundle = data.TrainingData(images, training)
    calls = []
    def weak(image):
        value = np.asarray(image).copy()
        image.putpixel((0, 0), (255, 255, 255))
        return value
    def strong(image):
        calls.append(1)
        return np.asarray(image).copy()
    datasets = data.make_training_datasets(bundle, {"weak": weak, "strong": strong, "eval": lambda im: np.asarray(im).copy()})
    for index in [19, 0, 12, 4]:
        image, label, position = datasets["standard"][index]
        assert position == index and label == training.noisy_labels[index]
        assert image[0, 0, 0] == index
        assert training.original_indices[position] // 5 == analysis.clean_labels[position]
    assert not calls
    weak_image, strong_image, label, position = datasets["adaptive"][3]
    np.testing.assert_array_equal(weak_image, strong_image)  # Strong starts from an independent copy.
    assert len(calls) == 1 and (label, position) == (int(training.noisy_labels[3]), 3)
    for dataset in datasets.values():
        assert not hasattr(dataset, "clean_labels") and not hasattr(dataset, "is_corrupted")
        assert not hasattr(dataset, "targets") and not hasattr(dataset, "source")
        assert not hasattr(dataset.metadata, "clean_labels")
    np.testing.assert_array_equal(datasets["statistics"][3][0], datasets["statistics"][3][0])
    with pytest.raises(IndexError):
        datasets["standard"][-1]


def test_cifar_factories_keep_train_and_test_separate_without_downloading(tmp_path, monkeypatch):
    train_labels = np.repeat(np.arange(10), 5000)
    test_labels = np.tile(np.arange(10), 1000)
    test_before = test_labels.copy()
    pixel = np.zeros((1, 32, 32, 3), dtype=np.uint8)
    calls = []
    def fake_cifar(root, *, train, download):
        calls.append((Path(root), train, download))
        labels = train_labels if train else test_labels
        return SimpleNamespace(targets=labels, data=np.broadcast_to(pixel, (len(labels), 32, 32, 3)))
    monkeypatch.setattr(data, "_cifar10", fake_cifar)
    settings = {**load_config(), "train_size": 20}
    bundle = data.prepare_cifar10_training(settings, .2, tmp_path / "cache with spaces", tmp_path / "metadata")
    assert len(bundle.images) == 20 and len(calls) == 1 and calls[0][1:] == (True, True)
    assert not hasattr(bundle, "clean_labels") and not hasattr(bundle, "is_corrupted")
    assert not hasattr(bundle, "source")
    test = data.load_cifar10_test(tmp_path / "cache with spaces", download=False, transform=lambda im: np.asarray(im))
    assert len(test) == 10000 and calls[-1][1:] == (False, False)
    np.testing.assert_array_equal(test.labels, test_before)
    np.testing.assert_array_equal(test_labels, test_before)
    assert test[501][1] == test_before[501] and test[0][0].shape == (32, 32, 3)


def test_synthetic_cli_imports_no_torch_and_downloads_nothing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = "import sys; from config import load_config; from src.data import synthetic_check; import json; print(json.dumps(synthetic_check(load_config()))); assert 'torch' not in sys.modules and 'torchvision' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", script], cwd=root, check=True, text=True, capture_output=True)
    reports = json.loads(result.stdout)
    assert [report["actual_corruptions"] for report in reports] == [4000, 8000]
    assert all(report["sample_shape"] == [3, 32, 32] and report["synthetic_only"] for report in reports)


@pytest.mark.parametrize("ids", [[-1, 2], [1, 1], [0, 50000], [0.5, 1.0]])
def test_original_ids_must_be_valid_and_distinct(ids):
    with pytest.raises(ValueError):
        data.TrainingMetadata(np.asarray(ids), np.array([0, 1]), 42, .2)


def test_analysis_loader_rejects_mismatched_mask(tmp_path, tiny_split):
    training, analysis = tiny_split
    data.save_metadata(training, analysis, tmp_path)
    np.testing.assert_array_equal(data.load_analysis_metadata(tmp_path).is_corrupted, analysis.is_corrupted)
    np.savez_compressed(tmp_path / "analysis.npz", clean_labels=analysis.clean_labels,
                        is_corrupted=np.zeros(len(analysis.clean_labels), dtype=bool))
    with pytest.raises(ValueError, match="Analysis metadata"):
        data.load_analysis_metadata(tmp_path)
