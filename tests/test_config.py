"""CPU-only tests for configuration safety and portable loading."""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from config import DEFAULT_CONFIG, configuration_fingerprint, load_config


def write_config(tmp_path, changes):
    config = json.loads(DEFAULT_CONFIG.read_text())
    config.update(changes)
    path = tmp_path / "configuration with spaces.json"
    path.write_text(json.dumps(config))
    return path


@pytest.mark.parametrize("size,warmup,continuation", [(20000, 6, 6), (10000, 6, 6), (10000, 6, 4), (10000, 4, 4)])
def test_load_default_and_uniform_fallbacks(tmp_path, size, warmup, continuation):
    settings = load_config(write_config(tmp_path, {"train_size": size, "warmup_epochs": warmup, "continuation_epochs": continuation}))
    assert settings["test_size"] == 10000
    assert settings["cifar_stem"] is True
    assert settings["train_size"] == size
    assert settings["warmup_epochs"] + settings["continuation_epochs"] == warmup + continuation


@pytest.mark.parametrize("changes", [
    {"train_size": 19999}, {"train_size": 50010}, {"batch_size": True},
    {"warmup_epochs": 0}, {"num_workers": -1}, {"learning_rate": 0},
    {"learning_rate": float("nan")}, {"weight_decay": float("inf")},
    {"pseudo_threshold": 1.1}, {"clean_weight_floor": 0},
    {"noise_ratios": [0.2]}, {"model": "resnet50"}, {"pretrained": True},
    {"test_size": 5000}, {"cifar_stem": False},
])
def test_reject_invalid_or_out_of_scope_configuration(tmp_path, changes):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, changes))


def test_fingerprint_is_order_independent_but_detects_method_change():
    settings = load_config()
    assert configuration_fingerprint(settings) == configuration_fingerprint(dict(reversed(list(settings.items()))))
    assert configuration_fingerprint(settings) != configuration_fingerprint({**settings, "warmup_epochs": 4})


def test_config_cli_works_outside_repository_without_torch(tmp_path):
    # -S disables site-packages, including torch and Colab, for this subprocess.
    script = Path(__file__).resolve().parents[1] / "config.py"
    result = subprocess.run([sys.executable, "-S", str(script)], cwd=tmp_path, check=True, capture_output=True, text=True)
    assert '"test_size": 10000' in result.stdout
    assert "Scientific configuration SHA256:" in result.stdout
