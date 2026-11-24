"""Stable-ID epoch-end WE/AUM and common CE diagnostic.

Core calculations/snapshots use only NumPy, so hand-logit tests run on Windows.
The collection/loader helpers import torch lazily and are exercised in Colab.
No original labels, corruption masks, pseudo-labels or ROC metrics are accepted.
"""
from __future__ import annotations

from pathlib import Path
import random

import numpy as np


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _epoch_inputs(logits, noisy_labels, subset_positions, size):
    logits = np.asarray(logits)
    labels = np.asarray(noisy_labels)
    ids = np.asarray(subset_positions)
    if logits.shape != (size, 10) or not np.issubdtype(logits.dtype, np.number) or np.iscomplexobj(logits):
        raise ValueError("Expected full-subset raw logits with shape [N,10]")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite")
    if (labels.shape != (size,) or not np.issubdtype(labels.dtype, np.integer)
            or np.any((labels < 0) | (labels >= 10))):
        raise ValueError("Provided noisy labels must be N integers in [0,9]")
    if (ids.shape != (size,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any((ids < 0) | (ids >= size)) or len(np.unique(ids)) != size):
        raise ValueError("Every subset position 0..N-1 must occur exactly once")
    return logits.astype(np.float64), labels.astype(np.int64), ids.astype(np.int64)


def _ce(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    losses = np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]
    result = float(losses.mean())
    if not np.isfinite(result):
        raise ValueError("Nonfinite noisy-label cross-entropy")
    return result


class SampleStatistics:
    """Cumulative counts/margins indexed by subset_position, not batch order.

    update_epoch receives the complete deterministic pass and commits only after
    every input validates. Epoch numbers are global tracked epochs (1,2,...),
    including warm-up, preventing duplicate updates when a caller retries.
    """
    def __init__(self, size):
        self.size = _integer(size, "size", 1)
        self.tracked_epochs = 0
        self._we = np.zeros(size, dtype=np.int32)
        self._margin_sum = np.zeros(size, dtype=np.float32)
        self._observation_counts = np.zeros(size, dtype=np.int32)

    @property
    def we(self):
        return self._we.copy()

    @property
    def margin_sum(self):
        return self._margin_sum.copy()

    @property
    def observation_counts(self):
        return self._observation_counts.copy()

    @property
    def aum(self):
        if not self.tracked_epochs:
            raise ValueError("AUM is undefined before the first completed epoch")
        return self._margin_sum / np.float32(self.tracked_epochs)

    def _next_epoch(self, epoch):
        _integer(epoch, "epoch", 1)
        if epoch != self.tracked_epochs + 1:
            raise ValueError("Expected the next completed epoch; duplicate/skipped tracking is forbidden")
        if epoch > np.iinfo(np.int32).max:
            raise ValueError("Epoch count exceeds int32 capacity")

    def update_epoch(self, logits, noisy_labels, subset_positions, *, epoch):
        self._next_epoch(epoch)
        logits, labels, ids = _epoch_inputs(logits, noisy_labels, subset_positions, self.size)
        rows = np.arange(self.size)
        assigned = logits[rows, labels]
        alternatives = logits.copy()
        alternatives[rows, labels] = -np.inf
        margins = assigned - alternatives.max(axis=1)
        by_id = np.empty(self.size, dtype=np.float64)
        by_id[ids] = margins
        wrong = np.empty(self.size, dtype=np.int32)
        wrong[ids] = (logits.argmax(axis=1) != labels).astype(np.int32)
        candidate = self._margin_sum.astype(np.float64) + by_id
        if not np.isfinite(candidate).all() or np.any(np.abs(candidate) > np.finfo(np.float32).max):
            raise ValueError("Margin accumulation would overflow float32")
        ce = _ce(logits, labels)
        # All failure-prone calculations precede the commit below.
        self._we = self._we + wrong
        self._margin_sum = candidate.astype(np.float32)
        self._observation_counts = self._observation_counts + np.int32(1)
        self.tracked_epochs = epoch
        return {"sample_count": self.size, "noisy_ce": ce, "tracked_epochs": epoch}

    def state_dict(self):
        return {"schema_version": 1, "size": self.size, "tracked_epochs": self.tracked_epochs,
                "we": self.we, "margin_sum": self.margin_sum, "observation_counts": self.observation_counts}

    @classmethod
    def from_state_dict(cls, state):
        if type(state.get("schema_version")) is not int or state.get("schema_version") != 1:
            raise ValueError("Unsupported statistics schema")
        size = _integer(state["size"], "size", 1)
        epochs = _integer(state["tracked_epochs"], "tracked_epochs")
        arrays = {name: np.asarray(state[name]) for name in ("we", "margin_sum", "observation_counts")}
        dtypes = {"we": np.dtype("int32"), "margin_sum": np.dtype("float32"), "observation_counts": np.dtype("int32")}
        if any(value.shape != (size,) or value.dtype != dtypes[name] for name, value in arrays.items()):
            raise ValueError("Statistics arrays have invalid shapes/dtypes")
        if (not np.isfinite(arrays["margin_sum"]).all()
                or not np.all(arrays["observation_counts"] == epochs)
                or np.any(arrays["we"] < 0) or np.any(arrays["we"] > epochs)
                or (epochs == 0 and np.any(arrays["margin_sum"] != 0))):
            raise ValueError("Inconsistent cumulative statistics")
        result = cls(size)
        result.tracked_epochs = epochs
        result._we = arrays["we"].copy()
        result._margin_sum = arrays["margin_sum"].copy()
        result._observation_counts = arrays["observation_counts"].copy()
        return result

    def copy(self):
        """Independent branch state with the entire warm-up history intact."""
        return self.from_state_dict(self.state_dict())

    def save_snapshot(self, path):
        """Save cumulative WE/margins/counts/AUM; never overwrite conflicting history."""
        path = Path(path)
        current_aum = self.aum  # Reject unobserved snapshots.
        if path.exists():
            previous = self.load_snapshot(path)
            if (previous.size != self.size or previous.tracked_epochs != self.tracked_epochs
                    or any(not np.array_equal(previous.state_dict()[key], self.state_dict()[key])
                           for key in ("we", "margin_sum", "observation_counts"))):
                raise FileExistsError("A conflicting statistics snapshot already exists")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **self.state_dict(), aum=current_aum)
        self.load_snapshot(temporary)  # Read/validate before publishing.
        temporary.replace(path)
        return path

    @classmethod
    def load_snapshot(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            state = {}
            for key in ("schema_version", "size", "tracked_epochs"):
                scalar = archive[key]
                if scalar.shape != () or not np.issubdtype(scalar.dtype, np.integer):
                    raise ValueError("Snapshot scalar metadata must contain integers")
                state[key] = int(scalar.item())
            state.update({key: archive[key] for key in ("we", "margin_sum", "observation_counts")})
            result = cls.from_state_dict(state)
            saved_aum = archive["aum"]
            if saved_aum.dtype != np.float32 or not np.array_equal(saved_aum, result.aum):
                raise ValueError("Saved AUM does not match cumulative margins/counts")
        return result


def seed_worker(worker_id):
    """Top-level, picklable worker initialiser; torch is imported only in Colab."""
    import torch
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_statistics_loader(training_data, batch_size=128, num_workers=0, generator=None):
    import torch
    from .data import NoisyLabelDataset, make_transforms
    dataset = NoisyLabelDataset(training_data, make_transforms()["eval"])
    if generator is None:
        generator = torch.Generator().manual_seed(training_data.metadata.seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                      drop_last=False, num_workers=num_workers, generator=generator,
                                      worker_init_fn=seed_worker, persistent_workers=False)


def collect_epoch_statistics(model, loader, *, device, tracker=None, epoch=None):
    """Single eval/no-grad pass over deterministic (image,noisy_label,id) batches.

    With tracker=None, return the common CE diagnostic without updating WE/AUM
    (standard continuation). Deterministic transforms must be used; the loader
    helper above supplies them. Scrambled *IDs* remain safe because updates scatter
    by ID, but random sampling and partial/drop-last passes are rejected.
    """
    import torch
    from torch.utils.data import SequentialSampler
    if not isinstance(loader.sampler, SequentialSampler) or loader.drop_last:
        raise ValueError("Statistics require shuffle=False and drop_last=False")
    size = len(loader.dataset)
    if tracker is not None:
        tracker._next_epoch(epoch)
        if tracker.size != size:
            raise ValueError("Tracker size must match the complete training subset")
    elif epoch is not None:
        raise ValueError("An epoch number requires a tracker")
    modes = [(module, module.training) for module in model.modules()]
    output_batches, label_batches, id_batches = [], [], []
    try:
        model.eval()
        with torch.no_grad():
            for batch in loader:
                if len(batch) != 3:
                    raise ValueError("Use deterministic one-view (image,noisy_label,id) statistics data")
                images, noisy_labels, ids = batch
                logits = model(images.to(device))
                output_batches.append(logits.detach().float().cpu().numpy())
                label_batches.append(noisy_labels.detach().cpu().numpy())
                id_batches.append(ids.detach().cpu().numpy())
        if not output_batches:
            raise ValueError("Statistics loader is empty")
        logits, labels, ids = _epoch_inputs(np.concatenate(output_batches), np.concatenate(label_batches),
                                           np.concatenate(id_batches), size)
        metadata = getattr(loader.dataset, "metadata", None)
        if metadata is not None and not np.array_equal(labels, metadata.noisy_labels[ids]):
            raise ValueError("Loader labels do not match the provided noisy labels at those IDs")
        if tracker is None:
            return {"sample_count": size, "noisy_ce": _ce(logits, labels)}
        return tracker.update_epoch(logits, labels, ids, epoch=epoch)
    finally:
        # Preserve mixed modes too (for example, a deliberately frozen BN layer).
        for module, training in modes:
            module.training = training
