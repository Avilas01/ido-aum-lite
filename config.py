"""Portable experiment configuration. No torch, CUDA or Colab dependency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load scientific settings; runtime storage paths are passed separately."""
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a JSON object")
    required = {"seed", "train_size", "test_size", "cifar_stem", "noise_ratios", "warmup_epochs",
                "continuation_epochs", "batch_size", "num_workers", "amp",
                "dataset", "model", "pretrained", "optimizer", "learning_rate",
                "weight_decay", "pseudo_threshold", "clean_weight_floor",
                "pseudo_coefficient", "consistency_coefficient", "near_zero_epsilon"}
    missing = required - settings.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")
    for key in ("seed", "train_size", "warmup_epochs", "continuation_epochs", "batch_size", "num_workers"):
        value = settings[key]
        minimum = 0 if key in ("seed", "num_workers") else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if settings["train_size"] % 10 or settings["train_size"] > 50000:
        raise ValueError("Full experiment train_size must be class-balanced and <= 50000")
    for key in ("amp", "pretrained", "cifar_stem"):
        if type(settings[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    for key in ("learning_rate", "weight_decay", "pseudo_threshold", "clean_weight_floor",
                "pseudo_coefficient", "consistency_coefficient", "near_zero_epsilon"):
        value = settings[key]
        if type(value) not in (int, float) or not (0 <= value < float("inf")):
            raise ValueError(f"{key} must be a finite nonnegative number")
    if settings["learning_rate"] == 0:
        raise ValueError("learning_rate must be positive")
    for key in ("pseudo_threshold", "clean_weight_floor"):
        if not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if (settings["dataset"] != "CIFAR10" or settings["model"] != "resnet18"
            or settings["pretrained"] or not settings["cifar_stem"]
            or type(settings["test_size"]) is not int or settings["test_size"] != 10000
            or settings["optimizer"] != "AdamW"
            or settings["noise_ratios"] != [0.2, 0.4]):
        raise ValueError("Configuration must preserve the planned dataset/model/noise/optimizer scope")
    return settings


def configuration_fingerprint(settings: dict[str, Any]) -> str:
    """Hash scientific settings independently of JSON formatting or host paths."""
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    selected = load_config(parser.parse_args().config)
    print(json.dumps(selected, indent=2))
    print("Scientific configuration SHA256:", configuration_fingerprint(selected))
