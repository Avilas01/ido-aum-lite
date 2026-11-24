"""Step 2 GPU-only model/tracker smoke: 512 generated images, one CE epoch.

No CIFAR download, official test-set evaluation, adaptive loss, full experiment
runner or runtime projection. Reports live in an isolated timestamped directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter


def run_smoke(settings, output_root, device="cuda"):
    import torch
    from .model import build_model
    from .statistics import SampleStatistics, collect_epoch_statistics

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Step 2 smoke requires a Colab GPU. Do not install torch/CUDA locally.")
    torch.manual_seed(settings["seed"])
    torch.cuda.manual_seed_all(settings["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    size = 512
    generator = torch.Generator().manual_seed(settings["seed"])
    images = torch.randn(size, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (size,), generator=generator)
    ids = torch.arange(size)
    dataset = torch.utils.data.TensorDataset(images, labels, ids)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=settings["batch_size"], shuffle=True,
                                              generator=torch.Generator().manual_seed(settings["seed"]), num_workers=0)
    stats_loader = torch.utils.data.DataLoader(dataset, batch_size=settings["batch_size"], shuffle=False,
                                              drop_last=False, num_workers=0)
    model = build_model(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    model.train()
    torch.cuda.synchronize(device)
    start = perf_counter()
    weighted_loss = 0.0
    for batch_images, batch_labels, _ in train_loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_images.to(device))
        if tuple(logits.shape) != (len(batch_labels), 10):
            raise RuntimeError("Unexpected CIFAR ResNet output shape")
        loss = torch.nn.functional.cross_entropy(logits, batch_labels.to(device))
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite smoke loss")
        loss.backward()
        if not any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("No gradients reached model parameters")
        if any(not torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
            raise RuntimeError("Nonfinite smoke gradients")
        optimizer.step()
        weighted_loss += float(loss.detach()) * len(batch_labels)
    torch.cuda.synchronize(device)
    training_seconds = perf_counter() - start
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite model parameters after smoke epoch")
    tracker = SampleStatistics(size)
    start = perf_counter()
    diagnostic = collect_epoch_statistics(model, stats_loader, device=device, tracker=tracker, epoch=1)
    torch.cuda.synchronize(device)
    statistics_seconds = perf_counter() - start
    assert model.training and tracker.tracked_epochs == 1
    assert tracker.we.min() >= 0 and tracker.we.max() <= 1
    assert (tracker.observation_counts == 1).all()
    assert torch.isfinite(torch.from_numpy(tracker.aum)).all()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output_root) / stamp
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = tracker.save_snapshot(directory / "history" / "epoch_001.npz")
    restored = SampleStatistics.load_snapshot(snapshot)
    assert restored.tracked_epochs == tracker.tracked_epochs
    report = {"kind": "step2_synthetic_smoke", "data_source": "generated tensors, not CIFAR-10",
              "seed": settings["seed"], "device": str(device), "gpu": torch.cuda.get_device_name(device),
              "torch": torch.__version__, "images": size, "epochs": 1, "training_batches": len(train_loader),
              "output_classes": int(logits.shape[1]), "train_mean_ce": weighted_loss / size,
              "deterministic_noisy_ce": diagnostic["noisy_ce"], "training_seconds": training_seconds,
              "statistics_seconds": statistics_seconds, "tracked_epochs": tracker.tracked_epochs,
              "we_min": int(tracker.we.min()), "we_max": int(tracker.we.max()),
              "aum_min": float(tracker.aum.min()), "aum_max": float(tracker.aum.max()),
              "observations_per_sample": int(tracker.observation_counts.min()), "snapshot": str(snapshot)}
    (directory / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("Step 2 smoke passed. Synthetic timings are not a full-experiment runtime estimate.")
    return report


def main():
    from config import DEFAULT_CONFIG, load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_smoke(load_config(args.config), args.output_root, args.device)


if __name__ == "__main__":
    main()
