"""Dependency-management tests use fake subprocesses; never install/import real torch."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from src import colab_dependencies as dependencies

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_are_valid_and_windows_never_pulls_torch():
    def read_including_includes(path):
        found = []
        for line in path.read_text().splitlines():
            line = line.partition("#")[0].strip()
            if not line:
                continue
            if line.startswith("-r "):
                found.extend(read_including_includes(path.parent / line[3:].strip()))
            else:
                found.append(Requirement(line).name.lower())
        return found
    local = set(read_including_includes(ROOT / "requirements-dev.txt"))
    colab = {Requirement(line).name.lower() for line in dependencies.requirement_lines(ROOT / "requirements-colab.txt")}
    assert not {"torch", "torchvision", "torchaudio"} & local
    assert not any("cuda" in name or name.startswith("nvidia-") for name in local)
    assert {"pyyaml", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn"} <= local
    assert {"torch", "torchvision", "pyyaml", "numpy", "pandas", "matplotlib", "scipy", "scikit-learn", "pillow", "ipython", "packaging"} <= colab


class FakeRunner:
    def __init__(self, reports):
        self.reports = iter(reports)
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[1:2] == ["-c"]:
            return SimpleNamespace(returncode=0, stdout=dependencies.PROBE_PREFIX + json.dumps(next(self.reports)), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")


def pip_calls(runner):
    return [args for args, _ in runner.calls if args[1:4] == ["-m", "pip", "install"]]


def test_working_gpu_pair_is_preserved(monkeypatch):
    monkeypatch.setattr(dependencies, "_require_gpu_runtime", lambda runner: None)
    healthy = {"ok": True, "versions": {"torch": "2.13.0", "torchvision": "0.28.0"}, "gpu": "mock", "cuda": "12.6"}
    runner = FakeRunner([healthy])
    assert dependencies.ensure_colab_dependencies(ROOT, runner=runner) == healthy
    for args in pip_calls(runner):
        assert "torch" not in args and "torchvision" not in args
        assert "--upgrade" not in args and "--force-reinstall" not in args


@pytest.mark.parametrize("error", ["ModuleNotFoundError: torch", "version mismatch", "Imported stack cannot access CUDA", "broken compiled torchvision operator"])
def test_bad_gpu_pair_is_repaired_together_then_requires_restart(monkeypatch, error):
    monkeypatch.setattr(dependencies, "_require_gpu_runtime", lambda runner: None)
    runner = FakeRunner([{"ok": False, "error": error}, {"ok": True}])
    with pytest.raises(dependencies.RestartRequired, match="Restart session"):
        dependencies.ensure_colab_dependencies(ROOT, runner=runner)
    repair = pip_calls(runner)[-1]
    assert "torch==2.14.1" in repair and "torchvision==0.29.1" in repair
    assert repair[-2:] == ["--index-url", "https://download.pytorch.org/whl/cu126"]


def test_failed_repair_never_continues(monkeypatch):
    monkeypatch.setattr(dependencies, "_require_gpu_runtime", lambda runner: None)
    runner = FakeRunner([{"ok": False, "error": "broken"}, {"ok": False, "error": "driver incompatible"}])
    with pytest.raises(RuntimeError, match="Stop here"):
        dependencies.ensure_colab_dependencies(ROOT, runner=runner)


def test_broken_common_import_does_not_reinstall_working_torch(monkeypatch):
    monkeypatch.setattr(dependencies, "_require_gpu_runtime", lambda runner: None)
    runner = FakeRunner([{"ok": False, "pair_ok": True, "error": "scipy import failed"}])
    with pytest.raises(RuntimeError, match="torch/torchvision were preserved"):
        dependencies.ensure_colab_dependencies(ROOT, runner=runner)
    assert len(pip_calls(runner)) == 1 and "--force-reinstall" not in pip_calls(runner)[0]


def test_windows_guard_runs_before_any_install(monkeypatch):
    monkeypatch.setattr(dependencies.sys, "platform", "win32")
    runner = FakeRunner([])
    with pytest.raises(RuntimeError, match="Colab GPU-only"):
        dependencies.ensure_colab_dependencies(ROOT, runner=runner)
    assert runner.calls == []


def test_unassigned_gpu_does_not_trigger_install(monkeypatch):
    monkeypatch.setattr(dependencies.sys, "platform", "linux")
    monkeypatch.setattr(dependencies.Path, "is_dir", lambda path: True)
    calls = []
    def no_gpu(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout="", stderr="no device")
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        dependencies.ensure_colab_dependencies(ROOT, runner=no_gpu)
    assert len(calls) == 1 and calls[0][0] == "nvidia-smi"


@pytest.mark.parametrize("torch_version,expected", [("2.13.0", "dependency metadata"), ("2.14.1", "cannot access CUDA")])
def test_real_probe_logic_rejects_mismatched_or_cpu_only_mock_pair(monkeypatch, capsys, torch_version, expected):
    # Execute the actual probe against fake torch modules, not local PyTorch.
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace())
    monkeypatch.setattr(dependencies.importlib.metadata, "requires", lambda name: ["torch==2.14.1"])
    monkeypatch.setattr(dependencies.importlib.metadata, "version", lambda name: torch_version)
    exec(dependencies.GPU_PROBE, {})
    result = json.loads(capsys.readouterr().out.split(dependencies.PROBE_PREFIX)[-1])
    assert result["ok"] is False and expected in result["error"]


def test_bootstrap_imports_without_site_packages_or_torch():
    script = "import sys; import config; import src.colab_dependencies; assert 'torch' not in sys.modules; assert 'torchvision' not in sys.modules"
    subprocess.run([sys.executable, "-S", "-c", script], cwd=ROOT, check=True)
