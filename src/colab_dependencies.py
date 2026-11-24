"""Colab GPU dependency setup. Safe to import on Windows without torch.

No models, datasets, training or methodology live here. All torch imports and
CUDA checks run in a disposable subprocess so failed native imports cannot
poison the notebook kernel. Never call this installer from local development.
"""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys

# One matched pair from the official PyTorch installation instructions:
# https://pytorch.org/get-started/previous-versions/#v2141
REPAIR_PAIR = ("torch==2.14.1", "torchvision==0.29.1")
REPAIR_INDEX = "https://download.pytorch.org/whl/cu126"
PROBE_PREFIX = "IDO_GPU_PROBE="


class RestartRequired(RuntimeError):
    """Installation changed native libraries; restart the notebook kernel."""


GPU_PROBE = r"""
import importlib.metadata as metadata
import json
result = {"ok": False}
try:
    from packaging.requirements import Requirement
    import torch
    import torchvision
    torch_requirements = [Requirement(line) for line in (metadata.requires('torchvision') or [])
                          if Requirement(line).name.lower() == 'torch']
    active = [req for req in torch_requirements if not req.marker or req.marker.evaluate()]
    if not active or not all(req.specifier.contains(metadata.version('torch'), prereleases=True) for req in active):
        raise RuntimeError('torch does not satisfy the installed torchvision dependency metadata')
    if not torch.cuda.is_available():
        raise RuntimeError('Imported stack cannot access CUDA')
    # Tiny compiled CUDA operator check: no dataset download or training.
    boxes = torch.tensor([[0., 0., 1., 1.]], device='cuda')
    scores = torch.tensor([1.], device='cuda')
    torchvision.ops.nms(boxes, scores, 0.5)
    torch.cuda.synchronize()
    result['pair_ok'] = True
    # Import every direct dependency used by the project/notebook.
    import numpy, pandas, matplotlib, scipy, sklearn, yaml, PIL, IPython
    names = ['torch', 'torchvision', 'numpy', 'pandas', 'matplotlib',
             'scipy', 'scikit-learn', 'PyYAML', 'Pillow', 'ipython', 'packaging']
    result.update(ok=True, versions={name: metadata.version(name) for name in names},
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
except Exception as error:
    result['error'] = type(error).__name__ + ': ' + str(error)
print('IDO_GPU_PROBE=' + json.dumps(result))
"""


def _require_gpu_runtime(runner):
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("Dependency installation is Colab GPU-only; do not install torch or CUDA locally.")
    try:
        device = runner(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True, text=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("GPU unavailable. Select a Colab GPU runtime before installation.") from error
    if device.returncode != 0 or not device.stdout.strip():
        raise RuntimeError("GPU unavailable. Select Runtime > Change runtime type > GPU and reconnect before installing.")
    print("Assigned GPU:", device.stdout.strip())


def requirement_lines(path):
    """Read our explicit, flat requirements file; never run shell fragments."""
    lines = [line.partition("#")[0].strip() for line in Path(path).read_text().splitlines()]
    lines = [line for line in lines if line]
    if any(not re.match(r"^[A-Za-z0-9]", line) for line in lines):
        raise ValueError("Colab requirements must be explicit package requirements, not pip options/includes")
    return lines


def _name(requirement):
    return re.match(r"[A-Za-z0-9._-]+", requirement).group().lower().replace("_", "-")


def probe_gpu_stack(runner):
    process = runner([sys.executable, "-c", GPU_PROBE], check=False, capture_output=True, text=True)
    for line in reversed(process.stdout.splitlines()):
        if line.startswith(PROBE_PREFIX):
            return json.loads(line[len(PROBE_PREFIX):])
    return {"ok": False, "error": "GPU import probe failed: " + process.stderr[-2000:]}


def ensure_colab_dependencies(repo_root, *, runner=None):
    """Keep a working pair, or repair both together and require a restart.

    Runtime selection is checked before ANY installation. Missing/broken pairs
    use a documented, matched CUDA wheel pair, never independent version guesses.
    The post-repair probe must succeed; setup always stops after torch changes.
    """
    runner = runner or subprocess.run
    _require_gpu_runtime(runner)
    requirements = requirement_lines(Path(repo_root) / "requirements-colab.txt")
    names = {_name(line) for line in requirements}
    if not {"torch", "torchvision"} <= names:
        raise ValueError("Colab requirements must explicitly include torch and torchvision")
    installed = {}
    for name in ("torch", "torchvision"):
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
    print("Installed PyTorch distributions (None means missing):", installed)
    # Common packages cannot replace torch. No --upgrade: preserve satisfied packages.
    common = [line for line in requirements if _name(line) not in {"torch", "torchvision"}]
    if common:
        runner([sys.executable, "-m", "pip", "install", *common], check=True)
    report = probe_gpu_stack(runner)
    if report["ok"]:
        print("Keeping the existing importable, compatible CUDA-enabled torch/torchvision pair.")
        print(json.dumps(report, indent=2))
        return report
    if report.get("pair_ok"):
        raise RuntimeError("The GPU pair is healthy but a common dependency cannot import: " + report.get("error", "unknown error")
                           + ". Fix that dependency; torch/torchvision were preserved.")
    print("GPU stack needs repair:", report.get("error"))
    print("Installing the official matched pair:", REPAIR_PAIR, "from", REPAIR_INDEX)
    runner([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall",
            *REPAIR_PAIR, "--index-url", REPAIR_INDEX], check=True)
    repaired = probe_gpu_stack(runner)
    if not repaired["ok"]:
        raise RuntimeError("Matched-pair repair did not pass CUDA/import checks: " + repaired.get("error", "unknown error")
                           + ". Stop here. Restart or select a fresh Colab GPU runtime and review driver/package compatibility; do not launch training.")
    raise RestartRequired("PyTorch/torchvision changed successfully. Stop here and select Runtime > Restart session "
                          "(Restart runtime in some UIs), then rerun setup cells with the same RUN_ID. "
                          "Do not continue in a kernel that may have cached old native libraries. "
                          "The next setup pass will preserve the now-working pair.")
