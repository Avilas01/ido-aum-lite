"""Shared experimental IDO-inspired objective for WE-only and hybrid scores.

PyTorch is imported only on invocation. The classification floor, ranks and
objective are deliberate simplifications, not original IDO/BMM. No method
switch or clean-label input is accepted.
"""
from __future__ import annotations

import math

COMPONENTS = ("supervised", "pseudo", "consistency", "total")


def adaptive_loss(logits_weak, logits_strong, noisy_labels, suspicion, *,
                  pseudo_threshold=0.7, clean_weight_floor=0.1,
                  pseudo_coefficient=0.5, consistency_coefficient=0.5,
                  near_zero_epsilon=1e-8):
    """Return (differentiable full-batch mean, detached scalar diagnostics).

    Pass configuration fields explicitly when settings differ from defaults.
    Gather frozen prior-epoch suspicion by stable subset IDs before calling.
    Pseudo targets/confidence are detached; consistency differentiates both
    views. Float32 softmax/CE support AMP. Every component uses the full batch
    denominator, including masked samples, never confident-only normalisation.
    """
    import torch
    import torch.nn.functional as F

    for name, value in (("pseudo_threshold", pseudo_threshold),
                        ("clean_weight_floor", clean_weight_floor)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{name} must lie in (0,1]")
    for name, value in (("pseudo_coefficient", pseudo_coefficient),
                        ("consistency_coefficient", consistency_coefficient),
                        ("near_zero_epsilon", near_zero_epsilon)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if (not isinstance(logits_weak, torch.Tensor) or not isinstance(logits_strong, torch.Tensor)
            or logits_weak.ndim != 2 or logits_weak.shape[1] != 10 or not len(logits_weak)
            or logits_strong.shape != logits_weak.shape
            or logits_strong.device != logits_weak.device
            or not logits_weak.is_floating_point() or not logits_strong.is_floating_point()
            or not torch.isfinite(logits_weak).all() or not torch.isfinite(logits_strong).all()):
        raise ValueError("Expected finite floating weak/strong logits with matching [B,10] shape/device")
    size = len(logits_weak)
    if (not isinstance(noisy_labels, torch.Tensor) or noisy_labels.shape != (size,)
            or noisy_labels.dtype != torch.long or noisy_labels.device != logits_weak.device
            or torch.any((noisy_labels < 0) | (noisy_labels >= 10))):
        raise ValueError("Provided noisy labels must be aligned int64 tensors in [0,9]")
    raw_scores = torch.as_tensor(suspicion, device=logits_weak.device)
    if raw_scores.is_complex():
        raise ValueError("Suspicion must be real")
    scores = raw_scores.detach().float()
    if (scores.shape != (size,) or not torch.isfinite(scores).all()
            or torch.any((scores < 0) | (scores > 1))):
        raise ValueError("Frozen suspicion must be an aligned finite vector in [0,1]")

    with torch.autocast(device_type=logits_weak.device.type, enabled=False):
        weak, strong = logits_weak.float(), logits_strong.float()
        p_weak, p_strong = weak.softmax(dim=1), strong.softmax(dim=1)
        confidence, targets = p_weak.detach().max(dim=1)
        mask = (confidence >= pseudo_threshold).float()
        c = clean_weight_floor + (1 - clean_weight_floor) * (1 - scores)
        d = 4 * scores * (1 - scores)
        supervised = c * F.cross_entropy(weak, noisy_labels, reduction="none")
        pseudo = pseudo_coefficient * scores * mask * F.cross_entropy(strong, targets, reduction="none")
        consistency = consistency_coefficient * d * (p_weak - p_strong).square().mean(dim=1)
        total = supervised + pseudo + consistency
        if not torch.isfinite(total).all():
            raise ValueError("Nonfinite adaptive objective")
        loss = total.mean()

    with torch.no_grad():
        logs = {"sample_count": size, "mean_c": c.mean().item(), "mean_d": d.mean().item(),
                "confidence_fraction": mask.mean().item()}
        for name, values in zip(COMPONENTS, (supervised, pseudo, consistency, total)):
            logs[f"{name}_loss"] = values.mean().item()
            logs[f"{name}_zero_fraction"] = (values == 0).float().mean().item()
            logs[f"{name}_near_zero_fraction"] = (values <= near_zero_epsilon).float().mean().item()
        _validate_logs(logs)
    return loss, logs


def _validate_logs(logs):
    expected = {"sample_count", "mean_c", "mean_d", "confidence_fraction"}
    expected.update(f"{name}_{suffix}" for name in COMPONENTS
                    for suffix in ("loss", "zero_fraction", "near_zero_fraction"))
    if set(logs) != expected or type(logs["sample_count"]) is not int or logs["sample_count"] <= 0:
        raise ValueError("Expected complete diagnostics and a positive integer sample_count")
    for key, value in logs.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid diagnostic: {key}")
        if (key.endswith("fraction") or key in ("mean_c", "mean_d")) and value > 1:
            raise ValueError(f"Diagnostic outside [0,1]: {key}")
    for name in COMPONENTS:
        if logs[f"{name}_zero_fraction"] > logs[f"{name}_near_zero_fraction"]:
            raise ValueError("Near-zero fractions must include exact zeros")
    if not math.isclose(sum(logs[f"{name}_loss"] for name in COMPONENTS[:-1]),
                        logs["total_loss"], rel_tol=2e-6, abs_tol=1e-7):
        raise ValueError("Weighted component means must sum to total loss")


class AdaptiveEpochSummary:
    """Sample-weighted reductions, including partial final minibatches.

    Stores detached numbers only. These are loss-value diagnostics, not gradient
    norms or the common deterministic unweighted noisy-label CE diagnostic.
    """
    def __init__(self):
        self.sample_count = 0
        self._sums = {}

    def update(self, logs):
        _validate_logs(logs)
        size = logs["sample_count"]
        for key, value in logs.items():
            if key != "sample_count":
                self._sums[key] = self._sums.get(key, 0.0) + size * value
        self.sample_count += size

    def compute(self):
        if not self.sample_count:
            raise ValueError("Cannot summarise an empty epoch")
        result = {key: value / self.sample_count for key, value in self._sums.items()}
        result["sample_count"] = self.sample_count
        _validate_logs(result)
        return result
