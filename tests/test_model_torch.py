"""Colab-only model checks. No pretrained weights, data downloads or full training."""
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip("torch")
vision = pytest.importorskip("torchvision")

from config import load_config
from src.model import build_model
from src.smoke import run_smoke


def test_cifar_stem_output_and_no_pretrained_weights(monkeypatch):
    original = vision.models.resnet18
    calls = []
    def record(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(vision.models, "resnet18", record)
    model = build_model(load_config())
    assert calls == [{"weights": None, "num_classes": 10}]
    assert model.conv1.kernel_size == (3, 3) and model.conv1.stride == (1, 1)
    assert model.conv1.padding == (1, 1) and model.conv1.bias is None
    assert isinstance(model.maxpool, torch.nn.Identity)
    assert model.fc.out_features == 10
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 32, 32))
    assert output.shape == (2, 10) and torch.isfinite(output).all()


def test_smoke_refuses_missing_gpu_before_creating_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "no smoke outputs"
    with pytest.raises(RuntimeError, match="Colab GPU"):
        run_smoke(load_config(), output)
    assert not output.exists()
