"""CIFAR-10 preparation with exact noise and stable IDs.

NumPy-only label preparation and PIL dataset adapters are safe on Windows.
Torchvision is imported lazily for real CIFAR loading/transforms in Colab.
Clean labels/masks belong only to initial preparation and analysis artifacts;
model-facing datasets never retain the original torchvision dataset or targets.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from contextlib import nullcontext
from time import monotonic
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from .preparation_progress import PreparationProgress

NUM_CLASSES = 10
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
STRONG_JITTER = {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.05}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _labels(values):
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("Labels must be a nonempty one-dimensional integer array")
    if np.any((array < 0) | (array >= NUM_CLASSES)):
        raise ValueError("CIFAR-10 labels must be in [0, 9]")
    return np.array(array, dtype=np.int64, copy=True)


def _readonly(array):
    array.setflags(write=False)
    return array


def _noise_ratio(value):
    if isinstance(value, bool) or value not in (0.2, 0.4):
        raise ValueError("The experiment supports noise ratios 0.2 and 0.4 only")
    return float(value)


def array_checksum(values):
    """Portable checksum of an ID/label vector, independent of host byte order."""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def noise_seed(seed, ratio):
    """Distinct deterministic streams derived from the single experiment seed."""
    _integer(seed, "seed")
    ratio = _noise_ratio(ratio)
    return int(np.random.SeedSequence([seed, int(round(100 * ratio)), 1]).generate_state(1)[0])


def stratified_indices(clean_labels, train_size=20000, seed=42):
    labels = _labels(clean_labels)
    _integer(train_size, "train_size", 10)
    _integer(seed, "seed")
    if train_size % NUM_CLASSES:
        raise ValueError("train_size must be divisible by 10")
    rng = np.random.default_rng(seed)
    per_class = train_size // NUM_CLASSES
    parts = []
    for label in range(NUM_CLASSES):
        available = np.flatnonzero(labels == label)
        if len(available) < per_class:
            raise ValueError(f"Class {label} has fewer than {per_class} examples")
        parts.append(rng.choice(available, size=per_class, replace=False))
    indices = np.concatenate(parts).astype(np.int64)
    rng.shuffle(indices)
    return _readonly(indices)


def symmetric_noise(clean_labels, ratio, seed):
    """Change exactly round(ratio*N) labels, uniformly excluding their true class.

    This construction function may see originals. Its outputs must be separated
    into training labels and analysis-only masks before model training begins.
    """
    labels = _labels(clean_labels)
    ratio = _noise_ratio(ratio)
    _integer(seed, "seed")
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(labels), size=round(ratio * len(labels)), replace=False)
    draws = rng.integers(0, NUM_CLASSES - 1, size=len(positions))
    noisy = labels.copy()
    noisy[positions] = draws + (draws >= labels[positions])
    corrupted = noisy != labels
    assert int(corrupted.sum()) == len(positions)
    return _readonly(noisy), _readonly(corrupted)


@dataclass(frozen=True)
class TrainingMetadata:
    """Provided labels and IDs only: safe to pass to training/statistics code."""
    original_indices: np.ndarray
    noisy_labels: np.ndarray
    seed: int
    noise_ratio: float

    def __post_init__(self):
        ids = np.asarray(self.original_indices)
        labels = _labels(self.noisy_labels)
        if (ids.ndim != 1 or len(ids) != len(labels)
                or not np.issubdtype(ids.dtype, np.integer) or np.any((ids < 0) | (ids >= 50000))
                or len(np.unique(ids)) != len(ids)):
            raise ValueError("Original CIFAR IDs must be distinct integers in [0,49999] aligned with labels")
        _integer(self.seed, "seed")
        _noise_ratio(self.noise_ratio)
        object.__setattr__(self, "original_indices", _readonly(np.array(ids, dtype=np.int64, copy=True)))
        object.__setattr__(self, "noisy_labels", _readonly(labels))

    def summary(self):
        return {"train_size": len(self.noisy_labels), "seed": self.seed,
                "noise_ratio": self.noise_ratio, "noise_seed": noise_seed(self.seed, self.noise_ratio),
                "expected_corruptions": round(self.noise_ratio * len(self.noisy_labels)),
                "subset_checksum": array_checksum(self.original_indices),
                "noisy_label_checksum": array_checksum(self.noisy_labels)}


@dataclass(frozen=True)
class AnalysisMetadata:
    """Evaluation-only; never attach this object to a training dataset."""
    clean_labels: np.ndarray
    is_corrupted: np.ndarray

    def __post_init__(self):
        labels = _labels(self.clean_labels)
        mask = np.asarray(self.is_corrupted)
        if mask.shape != labels.shape or mask.dtype != np.bool_:
            raise ValueError("Analysis mask must be boolean and aligned with clean labels")
        object.__setattr__(self, "clean_labels", _readonly(labels))
        object.__setattr__(self, "is_corrupted", _readonly(mask.copy()))


def _stage(progress, name):
    return progress.stage(name) if progress is not None else nullcontext()


def _labels_for_indices(labels, ids, ratio, seed, progress=None):
    with _stage(progress, f"{ratio:.0%}: generate exact corrupted labels"):
        clean = labels[ids]
        noisy, mask = symmetric_noise(clean, ratio, noise_seed(seed, ratio))
        return TrainingMetadata(ids, noisy, seed, ratio), AnalysisMetadata(clean, mask)


def prepare_labels(original_train_labels, train_size=20000, ratio=0.2, seed=42, *, progress=None):
    """Initial construction only; return separate training and analysis objects."""
    with _stage(progress, "Select deterministic class-stratified sample IDs"):
        labels = _labels(original_train_labels)
        ids = stratified_indices(labels, train_size, seed)
    return _labels_for_indices(labels, ids, ratio, seed, progress)


def save_metadata(training, analysis, directory, *, progress=None):
    """Verify/reuse finals and complete compatible partial writes; never mix runs."""
    directory = Path(directory)
    expected_mask = training.noisy_labels != analysis.clean_labels
    if not np.array_equal(expected_mask, analysis.is_corrupted):
        raise ValueError("Corruption mask is inconsistent with clean/provided labels")
    if int(expected_mask.sum()) != training.summary()["expected_corruptions"]:
        raise ValueError("Incorrect corruption count")
    arrays = {
        "training.npz": {"original_indices": training.original_indices, "noisy_labels": training.noisy_labels},
        "analysis.npz": {"clean_labels": analysis.clean_labels, "is_corrupted": analysis.is_corrupted},
    }
    details = {**training.summary(), "normalization_mean": list(CIFAR_MEAN), "normalization_std": list(CIFAR_STD),
               "strong_jitter": STRONG_JITTER, "schema_version": 1}
    with _stage(progress, f"{training.noise_ratio:.0%}: verify existing metadata at {directory}"):
        for name, expected in arrays.items():
            path = directory / name
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        valid = set(saved.files) == set(expected) and all(np.array_equal(saved[key], value) for key, value in expected.items())
                except (OSError, ValueError, EOFError) as error:
                    raise ValueError(f"Unreadable metadata {path}; preserved for inspection. Use a fresh data_checks directory; do not delete experiment checkpoints.") from error
                if not valid:
                    raise ValueError(f"Existing metadata does not match this experiment: {path}; refusing overwrite")
        manifest = directory / "metadata.json"
        if manifest.exists():
            try:
                previous = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"Unreadable metadata {manifest}; preserved for inspection") from error
            if previous != details:
                raise ValueError(f"Existing metadata does not match this experiment: {manifest}; refusing overwrite")
    with _stage(progress, f"{training.noise_ratio:.0%}: ensure metadata directory"):
        directory.mkdir(parents=True, exist_ok=True)
    for name, expected in arrays.items():
        path = directory / name
        if path.exists():
            if progress is not None:
                progress.log(f"Reusing verified metadata: {path}")
            continue
        with _stage(progress, f"{training.noise_ratio:.0%}: save {path}"):
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as output:
                np.savez_compressed(output, **expected)
            with np.load(temporary, allow_pickle=False) as saved:
                if not all(np.array_equal(saved[key], value) for key, value in expected.items()):
                    raise ValueError(f"Metadata write verification failed: {temporary}")
            if path.exists():
                raise FileExistsError(f"Another preparation is writing {path}; run only one writer")
            temporary.replace(path)
    if not manifest.exists():
        with _stage(progress, f"{training.noise_ratio:.0%}: save {manifest}"):
            temporary = manifest.with_name(manifest.name + ".tmp")
            temporary.write_text(json.dumps(details, indent=2) + chr(10), encoding="utf-8")
            if json.loads(temporary.read_text(encoding="utf-8")) != details:
                raise ValueError(f"Metadata write verification failed: {temporary}")
            if manifest.exists():
                raise FileExistsError(f"Another preparation is writing {manifest}; run only one writer")
            temporary.replace(manifest)


def load_training_metadata(directory):
    """Does not open the clean-label/mask file."""
    directory = Path(directory)
    details = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with np.load(directory / "training.npz", allow_pickle=False) as data:
        training = TrainingMetadata(data["original_indices"], data["noisy_labels"], details["seed"], details["noise_ratio"])
    if any(details.get(key) != value for key, value in training.summary().items()):
        raise ValueError("Training metadata checksum/configuration mismatch")
    return training


def load_analysis_metadata(directory):
    """Analysis-only loader; verify masks against the matching provided labels."""
    with np.load(Path(directory) / "analysis.npz", allow_pickle=False) as data:
        analysis = AnalysisMetadata(data["clean_labels"], data["is_corrupted"])
    training = load_training_metadata(directory)
    if (len(analysis.clean_labels) != len(training.noisy_labels)
            or not np.array_equal(analysis.is_corrupted, training.noisy_labels != analysis.clean_labels)
            or int(analysis.is_corrupted.sum()) != training.summary()["expected_corruptions"]):
        raise ValueError("Analysis metadata does not match the saved training labels/noise count")
    return analysis


def _images(values, count):
    images = np.asarray(values)
    if images.dtype != np.uint8 or images.shape != (count, 32, 32, 3):
        raise ValueError("Images must be uint8 with shape [N,32,32,3]")
    return _readonly(images.view())


@dataclass(frozen=True)
class TrainingData:
    images: np.ndarray
    metadata: TrainingMetadata

    def __post_init__(self):
        object.__setattr__(self, "images", _images(self.images, len(self.metadata.noisy_labels)))


class NoisyLabelDataset:
    """Torch-compatible map-style dataset; no torch import or clean-label storage.

    One-view/statistics samples: (image, noisy_label, subset_position).
    Adaptive samples: (weak_image, strong_image, noisy_label, subset_position).
    Original CIFAR IDs are available via metadata.original_indices[position].
    """
    def __init__(self, data: TrainingData, weak_transform: Callable, strong_transform=None):
        self.images = data.images
        self.metadata = data.metadata
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.metadata.noisy_labels)

    def __getitem__(self, index):
        from PIL import Image
        position = int(index)
        if not 0 <= position < len(self):
            raise IndexError(position)
        image = Image.fromarray(self.images[position])
        weak = self.weak_transform(image.copy())
        label = int(self.metadata.noisy_labels[position])
        if self.strong_transform is None:
            return weak, label, position
        strong = self.strong_transform(image.copy())
        return weak, strong, label, position


class CleanTestDataset:
    """Official test images/labels, preserved in original order; evaluation only."""
    def __init__(self, images, labels, transform):
        self.labels = _readonly(_labels(labels))
        self.images = _images(images, len(self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        from PIL import Image
        return self.transform(Image.fromarray(self.images[index]).copy()), int(self.labels[index])


def make_transforms():
    """Call only in Colab with a verified torch/torchvision installation."""
    from torchvision import transforms as T
    def tail():
        return [T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return {
        "weak": T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), *tail()]),
        "strong": T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ColorJitter(**STRONG_JITTER), *tail()]),
        "eval": T.Compose(tail()),
    }


def make_training_datasets(data, transforms=None):
    transforms = make_transforms() if transforms is None else transforms
    return {"standard": NoisyLabelDataset(data, transforms["weak"]),
            "adaptive": NoisyLabelDataset(data, transforms["weak"], transforms["strong"]),
            "statistics": NoisyLabelDataset(data, transforms["eval"])}


def _cifar10(root, *, train, download, progress=None, archive_root=None):
    """Observable torchvision loading; no duplicate checksum pass per instance."""
    manager = nullcontext(progress) if progress is not None else PreparationProgress()
    with manager as reporter:
        root = Path(root).expanduser().resolve()
        reporter.log(f"Shared CIFAR cache: {root}; split={'train' if train else 'test'}; download={download}")
        if "/content/drive/" in root.as_posix():
            reporter.log("Data cache is on mounted Drive; /content/data avoids slow image/cache I/O. Metadata may remain on Drive.")
        with reporter.stage("Import torchvision CIFAR loader"):
            from torchvision.datasets import CIFAR10
            from torchvision.datasets.utils import download_url, extract_archive
            from src.archive_cache import ensure_archive

        class ObservableCIFAR10(CIFAR10):
            def _check_integrity(self):
                if getattr(self, "_verified_files", False):
                    return True
                with reporter.stage("Verify extracted CIFAR batch checksums"):
                    valid = super()._check_integrity()
                self._verified_files = valid
                return valid

            def download(self):
                if self._check_integrity():
                    reporter.log("Extracted CIFAR cache is valid; skipping download and extraction.")
                    if archive_root is not None:
                        archive = Path(self.root) / self.filename
                        if archive.is_file():
                            ensure_archive(self.root, archive_root, download=False, progress=reporter,
                                           url=self.url, filename=self.filename, checksum=self.tgz_md5)
                        else:
                            reporter.log("No local archive to back up. Run Block 6 once to create a persistent archive.")
                    return
                archive = ensure_archive(self.root, archive_root, progress=reporter,
                                         url=self.url, filename=self.filename, checksum=self.tgz_md5,
                                         downloader=download_url)
                with reporter.stage(f"Extract CIFAR archive into {self.root}"):
                    extract_archive(str(archive), self.root)

        # This total includes nested integrity/download/extraction times, not an
        # additional independent duration to add to them.
        with reporter.stage("Initialize CIFAR split (cache checks and batch loading)"):
            source = ObservableCIFAR10(root=str(root), train=train, download=download)
        reporter.log(f"Loaded {len(source.targets)} {'training' if train else 'test'} images into memory.")
        return source


def _prepare_conditions(settings, ratios, data_root, metadata_paths, download, progress, archive_root=None):
    ratios = [_noise_ratio(ratio) for ratio in ratios]
    if not ratios or len(set(ratios)) != len(ratios) or any(ratio not in settings["noise_ratios"] for ratio in ratios):
        raise ValueError("Use distinct configured noise ratios")
    kwargs = {"train": True, "download": download}
    if progress is not None:
        kwargs["progress"] = progress
    if archive_root is not None:
        kwargs["archive_root"] = archive_root
    source = _cifar10(data_root, **kwargs)
    if len(source.targets) != 50000:
        raise ValueError("Expected the official 50,000-image CIFAR-10 training split")
    with _stage(progress, "Select deterministic class-stratified sample IDs (shared by noise conditions)"):
        labels = _labels(source.targets)
        ids = stratified_indices(labels, settings["train_size"], settings["seed"])
    with _stage(progress, "Copy selected training images once (shared by noise conditions)"):
        # NumPy advanced indexing already owns a copy; a second np.array(copy=True)
        # duplicated the 20k image copy in the old one-condition implementation.
        images = source.data[ids]
    result = {}
    for ratio in ratios:
        training, analysis = _labels_for_indices(labels, ids, ratio, settings["seed"], progress)
        save_metadata(training, analysis, metadata_paths[ratio], progress=progress)
        result[ratio] = TrainingData(images, training)
        if progress is not None:
            progress.log(f"{ratio:.0%}: ready; {len(training.noisy_labels)} examples, {int(analysis.is_corrupted.sum())} changed labels; subset checksum={training.summary()['subset_checksum']}")
    return result


def prepare_cifar10_training(settings, ratio, data_root, metadata_root, *, download=True, progress=None, archive_root=None):
    """Single-condition facade retained for callers; analysis is saved separately."""
    ratio = _noise_ratio(ratio)
    return _prepare_conditions(settings, [ratio], data_root, {ratio: Path(metadata_root)}, download, progress, archive_root)[ratio]


def prepare_cifar10_conditions(settings, ratios, data_root, metadata_root, *, download=True, progress=None, archive_root=None):
    """One source load and image selection; separate reproducible noise artifacts."""
    ratios = [_noise_ratio(ratio) for ratio in ratios]
    paths = {ratio: Path(metadata_root) / f"noise_{round(100 * ratio)}" for ratio in ratios}
    return _prepare_conditions(settings, ratios, data_root, paths, download, progress, archive_root)


def load_cifar10_test(data_root, *, download=True, transform=None, progress=None, archive_root=None):
    """Separate entry point for evaluation; never corrupt or subsample test labels."""
    kwargs = {"train": False, "download": download}
    if progress is not None:
        kwargs["progress"] = progress
    if archive_root is not None:
        kwargs["archive_root"] = archive_root
    source = _cifar10(data_root, **kwargs)
    if len(source.targets) != 10000:
        raise ValueError("Expected the untouched official 10,000-image test split")
    if transform is None:
        transform = make_transforms()["eval"]
    return CleanTestDataset(source.data, source.targets, transform)


def synthetic_check(settings):
    """No network, torch or dataset download; synthetic labels and one RGB fixture."""
    original = np.repeat(np.arange(10), 5000)
    reports = []
    for ratio in settings["noise_ratios"]:
        training, analysis = prepare_labels(original, settings["train_size"], ratio, settings["seed"])
        pixel = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        data = TrainingData(np.broadcast_to(pixel, (settings["train_size"], 32, 32, 3)), training)
        transform = lambda image: np.asarray(image).transpose(2, 0, 1)
        datasets = make_training_datasets(data, {"weak": transform, "strong": transform, "eval": transform})
        reports.append({**training.summary(), "actual_corruptions": int(analysis.is_corrupted.sum()),
                        "selected_original_class_counts": np.bincount(analysis.clean_labels, minlength=10).tolist(),
                        "standard_length": len(datasets["standard"]), "adaptive_length": len(datasets["adaptive"]),
                        "statistics_length": len(datasets["statistics"]),
                        "sample_shape": list(datasets["adaptive"][0][0].shape), "synthetic_only": True})
    return reports


def main():
    from config import DEFAULT_CONFIG, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic-check", action="store_true")
    mode.add_argument("--prepare", action="store_true", help="Colab: download/prepare real training data")
    parser.add_argument("--noise", type=float, choices=(0.2, 0.4), nargs="+", help="One or both noise ratios; multiple ratios share one CIFAR load")
    parser.add_argument("--no-download", action="store_true", help="Use an existing verified CIFAR cache only")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--metadata-root", type=Path)
    parser.add_argument("--archive-root", type=Path, help="Persistent verified archive cache (e.g. Drive); extracted images stay local")
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.synthetic_check:
        print(json.dumps(synthetic_check(settings), indent=2))
    else:
        if args.noise is None or args.data_root is None or args.metadata_root is None:
            parser.error("--prepare requires --noise, --data-root and --metadata-root")
        with PreparationProgress() as progress:
            progress.log(f"Preparation only, no model training. Data root={args.data_root}; metadata root={args.metadata_root}")
            if len(args.noise) == 1:
                ratio = args.noise[0]
                prepared = {ratio: prepare_cifar10_training(settings, ratio, args.data_root, args.metadata_root,
                                                          download=not args.no_download, progress=progress, archive_root=args.archive_root)}
            else:
                prepared = prepare_cifar10_conditions(settings, args.noise, args.data_root, args.metadata_root,
                                                      download=not args.no_download, progress=progress, archive_root=args.archive_root)
            with progress.stage("Create transforms and check one weak/strong image per condition"):
                transforms = make_transforms()
                reports = []
                for ratio, data in prepared.items():
                    datasets = make_training_datasets(data, transforms)
                    weak, strong, _, position = datasets["adaptive"][0]
                    directory = args.metadata_root if len(args.noise) == 1 else args.metadata_root / f"noise_{round(100 * ratio)}"
                    reports.append({**data.metadata.summary(),
                                    "dataset_lengths": {key: len(value) for key, value in datasets.items()},
                                    "weak_shape": list(weak.shape), "strong_shape": list(strong.shape),
                                    "first_subset_position": position, "metadata_root": str(directory)})
            progress.log(f"All conditions ready in {monotonic()-progress.started:.2f}s; nested stage durations are not additive.")
            print(json.dumps({"preparations": reports, "stage_timings": progress.events,
                              "preparation_seconds": monotonic() - progress.started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
