"""Instrumented preparation with synthetic data and fake torchvision I/O only."""
import json
from pathlib import Path
import sys
import threading
from types import ModuleType

import numpy as np
import pytest
from config import load_config
from src import data
from src.preparation_progress import PreparationProgress


@pytest.fixture
def fake_vision(monkeypatch):
    state = {"ready": False, "downloads": 0, "extractions": 0, "loads": 0, "checks": 0, "roots": []}
    class FakeCIFAR:
        url = "https://example.invalid/cifar.tar.gz"
        filename = "cifar-10-python.tar.gz"
        tgz_md5 = "fixture-md5"
        def __init__(self, root, train, download):
            self.root = root
            self.train = train
            if download:
                self.download()
            if not self._check_integrity():
                raise RuntimeError("Dataset not found or corrupted")
            state["loads"] += 1
            state["roots"].append(root)
            self.targets = np.repeat(np.arange(10), 5000) if train else np.tile(np.arange(10), 1000)
            self.data = np.broadcast_to(np.zeros((1, 32, 32, 3), dtype=np.uint8), (len(self.targets), 32, 32, 3))
        def _check_integrity(self):
            state["checks"] += 1
            return state["ready"]
    def download(url, root, filename, md5):
        assert url == FakeCIFAR.url and md5 == FakeCIFAR.tgz_md5
        state["downloads"] += 1
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / filename).write_bytes(b"fake archive, not CIFAR")
    def extract(archive, root):
        assert Path(archive).parent == Path(root)
        state["extractions"] += 1
        state["ready"] = True
    vision = ModuleType("torchvision")
    datasets = ModuleType("torchvision.datasets")
    utils = ModuleType("torchvision.datasets.utils")
    datasets.CIFAR10 = FakeCIFAR
    utils.download_url, utils.extract_archive = download, extract
    vision.datasets = datasets
    monkeypatch.setitem(sys.modules, "torchvision", vision)
    monkeypatch.setitem(sys.modules, "torchvision.datasets", datasets)
    monkeypatch.setitem(sys.modules, "torchvision.datasets.utils", utils)
    return state


def test_both_conditions_load_once_share_images_and_preserve_noise(tmp_path, fake_vision, capsys):
    settings = {**load_config(), "train_size": 20}
    prepared = data.prepare_cifar10_conditions(settings, [.2, .4], tmp_path / "cache", tmp_path / "metadata")
    assert fake_vision["downloads"] == fake_vision["extractions"] == fake_vision["loads"] == 1
    assert fake_vision["checks"] == 2  # Incomplete before extraction; verified once afterwards.
    assert np.shares_memory(prepared[.2].images, prepared[.4].images)
    original = np.repeat(np.arange(10), 5000)
    for ratio in (.2, .4):
        expected, _ = data.prepare_labels(original, 20, ratio, 42)
        assert prepared[ratio].metadata.summary() == expected.summary()
        assert (tmp_path / "metadata" / f"noise_{round(100*ratio)}" / "metadata.json").is_file()
    assert "Download/verify" in capsys.readouterr().out


def test_cached_rerun_never_redownloads_or_reextracts(tmp_path, fake_vision, capsys):
    settings = {**load_config(), "train_size": 20}
    first = data.prepare_cifar10_conditions(settings, [.2, .4], tmp_path / "cache", tmp_path / "metadata")
    second = data.prepare_cifar10_conditions(settings, [.2, .4], tmp_path / "cache", tmp_path / "metadata")
    assert fake_vision["downloads"] == fake_vision["extractions"] == 1
    assert fake_vision["loads"] == 2  # One source load per invocation, not per condition.
    assert fake_vision["checks"] == 3  # Cached object's constructor does not hash files twice.
    assert second[.4].metadata.summary() == first[.4].metadata.summary()
    assert "skipping download and extraction" in capsys.readouterr().out


def test_no_download_flag_fails_on_missing_cache_without_network(tmp_path, fake_vision):
    with pytest.raises(RuntimeError, match="Dataset not found"):
        data._cifar10(tmp_path, train=True, download=False)
    assert fake_vision["downloads"] == fake_vision["extractions"] == 0


@pytest.mark.parametrize("missing", ["training.npz", "analysis.npz", "metadata.json"])
def test_compatible_partial_metadata_recovers_without_rewriting_valid_files(tmp_path, missing):
    training, analysis = data.prepare_labels(np.repeat(np.arange(10), 5), 20)
    data.save_metadata(training, analysis, tmp_path)
    (tmp_path / missing).unlink()
    existing = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in tmp_path.iterdir()}
    (tmp_path / (missing + ".tmp")).write_bytes(b"interrupted temporary write")
    data.save_metadata(training, analysis, tmp_path)
    assert data.load_training_metadata(tmp_path).summary() == training.summary()
    np.testing.assert_array_equal(data.load_analysis_metadata(tmp_path).is_corrupted, analysis.is_corrupted)
    for name, (content, modified) in existing.items():
        assert (tmp_path / name).read_bytes() == content
        assert (tmp_path / name).stat().st_mtime_ns == modified


def test_conflicting_partial_metadata_is_not_overwritten(tmp_path):
    low, analysis = data.prepare_labels(np.repeat(np.arange(10), 5), 20, .2)
    data.save_metadata(low, analysis, tmp_path)
    (tmp_path / "metadata.json").unlink()
    saved = (tmp_path / "training.npz").read_bytes()
    high, other = data.prepare_labels(np.repeat(np.arange(10), 5), 20, .4)
    with pytest.raises(ValueError, match="refusing overwrite"):
        data.save_metadata(high, other, tmp_path)
    assert (tmp_path / "training.npz").read_bytes() == saved
    assert not (tmp_path / "metadata.json").exists()


def test_progress_reports_active_stage_and_stops_on_failure(capsys):
    heartbeat = threading.Event()
    class ObservedProgress(PreparationProgress):
        def log(self, text):
            super().log(text)
            if text.startswith("Still working"):
                heartbeat.set()
    with ObservedProgress(heartbeat_seconds=.01) as progress:
        with pytest.raises(ValueError, match="fixture failure"):
            with progress.stage("download fixture"):
                assert heartbeat.wait(timeout=2)
                raise ValueError("fixture failure")
    assert not progress._thread.is_alive()
    assert progress.events[-1]["status"] == "failed" and progress.events[-1]["seconds"] > 0
    output = capsys.readouterr().out
    assert "Still working: download fixture" in output and "FAILED download fixture" in output


def test_original_default_noise_fingerprints_have_not_changed():
    original = np.repeat(np.arange(10), 5000)
    expected = {.2: "c24980d3a4caa3e4143022faf271045c6aba9d459b721dd896f7c5de0e6c2341",
                .4: "bd099959b15b212d6cd1625d5b1b4996e78d8d29e82cf360f5b1443bf876fc5a"}
    for ratio, checksum in expected.items():
        training, _ = data.prepare_labels(original, 20000, ratio, 42)
        assert training.summary()["subset_checksum"] == "26070562aa182e081f764b4504be71d29f8f5785f630add73b4363d2dd972566"
        assert training.summary()["noisy_label_checksum"] == checksum



def test_two_noise_cli_loads_once_and_writes_the_existing_condition_layout(tmp_path, fake_vision, monkeypatch, capsys):
    settings = {**load_config(), "train_size": 20}
    config_path = tmp_path / "config with spaces.json"
    config_path.write_text(json.dumps(settings))
    metadata_root = tmp_path / "metadata with spaces"
    transform = lambda image: np.asarray(image).transpose(2, 0, 1)
    monkeypatch.setattr(data, "make_transforms", lambda: {"weak": transform, "strong": transform, "eval": transform})
    monkeypatch.setattr(sys, "argv", ["src.data", "--prepare", "--config", str(config_path), "--noise", "0.2", "0.4",
                                     "--data-root", str(tmp_path / "cache"), "--metadata-root", str(metadata_root)])
    data.main()
    assert fake_vision["loads"] == fake_vision["downloads"] == fake_vision["extractions"] == 1
    for ratio in (.2, .4):
        root = metadata_root / f"noise_{round(100 * ratio)}"
        assert data.load_training_metadata(root).noise_ratio == ratio
        assert data.load_analysis_metadata(root).is_corrupted.sum() == round(20 * ratio)
    output = capsys.readouterr().out
    assert '"preparations"' in output and '"stage_timings"' in output and "All conditions ready" in output
