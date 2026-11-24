"""Colab-only adapters/transforms/DataLoader tests; synthetic images, no downloads."""
import numpy as np
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src import data


def test_real_transforms_and_shuffled_loader_preserve_ids():
    training, _ = data.prepare_labels(np.repeat(np.arange(10), 5), train_size=20)
    # Nonuniform synthetic pixels exercise spatial/color transforms without CIFAR.
    images = np.random.default_rng(2).integers(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
    datasets = data.make_training_datasets(data.TrainingData(images, training))
    weak, strong, label, position = datasets["adaptive"][0]
    assert weak.shape == strong.shape == (3, 32, 32)
    assert weak.dtype == strong.dtype == torch.float32
    assert torch.isfinite(weak).all() and torch.isfinite(strong).all()
    assert 0 <= label < 10 and position == 0
    assert torch.equal(datasets["statistics"][0][0], datasets["statistics"][0][0])
    loader = torch.utils.data.DataLoader(datasets["standard"], batch_size=6, shuffle=True,
                                        generator=torch.Generator().manual_seed(42), num_workers=0)
    seen = []
    for batch_images, labels, ids in loader:
        assert batch_images.shape[1:] == (3, 32, 32)
        np.testing.assert_array_equal(labels.numpy(), training.noisy_labels[ids.numpy()])
        seen.extend(ids.tolist())
    assert sorted(seen) == list(range(20))
    assert seen != list(range(20))


def test_statistics_transform_uses_fixed_channel_normalization():
    image = np.full((1, 32, 32, 3), 255, dtype=np.uint8)
    training = data.TrainingMetadata(np.array([9]), np.array([2]), 42, .2)
    dataset = data.make_training_datasets(data.TrainingData(image, training))["statistics"]
    sample, label, position = dataset[0]
    expected = (torch.ones(3) - torch.tensor(data.CIFAR_MEAN)) / torch.tensor(data.CIFAR_STD)
    assert torch.allclose(sample[:, 0, 0], expected)
    assert (label, position) == (2, 0)
