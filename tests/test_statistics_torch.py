"""Colab-only collection/loader tests with tiny synthetic fixtures."""
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.data import TrainingData, prepare_labels
from src.statistics import SampleStatistics, collect_epoch_statistics, make_statistics_loader


class Echo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(()))
        self.flags = []

    def forward(self, inputs):
        self.flags.append((self.training, torch.is_grad_enabled()))
        return inputs * self.scale


def fixture_dataset(ids=None):
    logits = torch.full((3, 10), -1000.)
    logits[0, 0:2] = torch.tensor([3., 1.])
    logits[1, 0:2] = torch.tensor([4., 1.])
    logits[2, [0, 2]] = torch.tensor([1., 2.])
    labels = torch.tensor([0, 1, 2])
    order = torch.tensor([2, 0, 1])
    actual_ids = order if ids is None else torch.tensor(ids)
    return torch.utils.data.TensorDataset(logits[order], labels[order], actual_ids)


def test_collection_scattered_ids_eval_no_grad_and_common_ce():
    dataset = fixture_dataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    model = Echo().train()
    tracker = SampleStatistics(3)
    summary = collect_epoch_statistics(model, loader, device="cpu", tracker=tracker, epoch=1)
    np.testing.assert_array_equal(tracker.we, [0, 1, 0])
    np.testing.assert_array_equal(tracker.aum, [2, -3, 1])
    assert model.training and model.scale.grad is None
    assert model.flags == [(False, False), (False, False)]
    expected = torch.nn.functional.cross_entropy(dataset.tensors[0], dataset.tensors[1]).item()
    assert summary["noisy_ce"] == pytest.approx(expected, rel=1e-6)
    diagnostic = collect_epoch_statistics(model, loader, device="cpu")
    assert diagnostic["noisy_ce"] == summary["noisy_ce"]
    assert tracker.tracked_epochs == 1


def test_duplicate_ids_cannot_change_tracker_and_modes_are_restored():
    loader = torch.utils.data.DataLoader(fixture_dataset([2, 0, 0]), batch_size=2)
    model = Echo().train()
    tracker = SampleStatistics(3)
    with pytest.raises(ValueError, match="exactly once"):
        collect_epoch_statistics(model, loader, device="cpu", tracker=tracker, epoch=1)
    assert model.training and tracker.tracked_epochs == 0
    np.testing.assert_array_equal(tracker.we, [0, 0, 0])


def test_collect_does_not_change_batchnorm_buffers_or_mixed_modes():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(10), torch.nn.Dropout(.5))
    model.train()
    model[1].eval()
    modes = [module.training for module in model.modules()]
    running_mean = model[0].running_mean.clone()
    loader = torch.utils.data.DataLoader(fixture_dataset(), batch_size=2)
    collect_epoch_statistics(model, loader, device="cpu")
    assert [module.training for module in model.modules()] == modes
    assert torch.equal(model[0].running_mean, running_mean)
    assert model[0].num_batches_tracked.item() == 0


def test_forward_failure_restores_train_mode():
    class Broken(Echo):
        def forward(self, inputs):
            raise RuntimeError("fixture forward failure")
    model = Broken().train()
    loader = torch.utils.data.DataLoader(fixture_dataset(), batch_size=2)
    tracker = SampleStatistics(3)
    with pytest.raises(RuntimeError, match="fixture forward failure"):
        collect_epoch_statistics(model, loader, device="cpu", tracker=tracker, epoch=1)
    assert model.training and tracker.tracked_epochs == 0


def test_random_or_drop_last_loaders_are_rejected():
    for options in ({"shuffle": True}, {"drop_last": True}):
        loader = torch.utils.data.DataLoader(fixture_dataset(), batch_size=2, **options)
        with pytest.raises(ValueError, match="shuffle=False"):
            collect_epoch_statistics(Echo(), loader, device="cpu")


def test_provided_label_mismatch_is_rejected():
    dataset = fixture_dataset()
    dataset.metadata = SimpleNamespace(noisy_labels=np.array([9, 9, 9]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    tracker = SampleStatistics(3)
    with pytest.raises(ValueError, match="provided noisy labels"):
        collect_epoch_statistics(Echo(), loader, device="cpu", tracker=tracker, epoch=1)
    assert tracker.tracked_epochs == 0


def test_statistics_loader_is_deterministic_single_view_and_covers_every_id():
    training, _ = prepare_labels(np.repeat(np.arange(10), 5), train_size=20)
    images = np.random.default_rng(42).integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
    loader = make_statistics_loader(TrainingData(images, training), batch_size=6)
    assert not loader.drop_last and loader.dataset.strong_transform is None
    assert isinstance(loader.sampler, torch.utils.data.SequentialSampler)
    assert torch.equal(loader.dataset[0][0], loader.dataset[0][0])
    seen = []
    for image_batch, labels, ids in loader:
        assert image_batch.shape[1:] == (3, 32, 32)
        np.testing.assert_array_equal(labels.numpy(), training.noisy_labels[ids.numpy()])
        seen.extend(ids.tolist())
    assert seen == list(range(20))
