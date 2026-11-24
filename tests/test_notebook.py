"""Validate the notebook without Google Drive, CUDA or research modules."""
import ast
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import nbformat
import pytest
from config import load_config

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "colab_runner.ipynb"


def sources():
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    return notebook, [cell.source for cell in notebook.cells if cell.cell_type == "code"]


def test_notebook_schema_syntax_and_cleared_outputs():
    notebook, code = sources()
    nbformat.validate(notebook)
    for source in code:
        ast.parse(source)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is None and cell.outputs == []
    assert any("--smoke" in source for source in code)
    assert any("--resume" in source for source in code)


def test_checked_module_calls_preserve_paths_with_spaces_and_reject_missing_modules(tmp_path):
    repo = tmp_path / "source checkout"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "train.py").touch()
    calls = []
    env = {"REPO_DIR": repo, "sys": sys, "subprocess": SimpleNamespace(run=lambda args, **kwargs: calls.append((args, kwargs)))}
    helper = next(source for source in sources()[1] if source.startswith("def run_module"))
    exec(helper, env)
    target = tmp_path / "Drive outputs with spaces"
    env["run_module"]("src.train", "--output-root", target, "--resume")
    args, kwargs = calls[0]
    assert args == [sys.executable, "-m", "src.train", "--output-root", str(target), "--resume"]
    assert kwargs == {"cwd": repo, "check": True}
    with pytest.raises(RuntimeError, match="not implemented yet"):
        env["run_module"]("src.evaluate")


def test_manifest_freeze_is_valid_and_preserves_existing_configuration(tmp_path):
    bench = tmp_path / "benchmark"
    bench.mkdir()
    (bench / "experiment_config.json").write_text(json.dumps(load_config()))
    env = {"FROZEN_CONFIG": tmp_path / "experiment_config.json", "MANIFEST_PATH": tmp_path / "manifest.json",
           "BENCHMARK_ROOT": bench, "EXPERIMENT_ROOT": tmp_path, "RUN_ID": "test", "GIT_SHA": "test-sha",
           "ENVIRONMENT": {"gpu": "mock"}, "load_config": load_config, "hashlib": hashlib, "json": json}
    freeze = next(source for source in sources()[1] if "FREEZE_APPROVED = False" in source)
    with pytest.raises(RuntimeError, match="Review the benchmark"):
        exec(freeze, env)
    assert not env["FROZEN_CONFIG"].exists()
    freeze = freeze.replace("FREEZE_APPROVED = False", "FREEZE_APPROVED = True")
    exec(freeze, env)
    before = env["FROZEN_CONFIG"].read_bytes()
    manifest = json.loads(env["MANIFEST_PATH"].read_text())
    assert manifest["config_sha256"] == hashlib.sha256(before).hexdigest()
    (bench / "experiment_config.json").write_text("invalid replacement")
    exec(freeze, env)
    assert env["FROZEN_CONFIG"].read_bytes() == before


def test_setup_stops_before_full_cpu_run(monkeypatch):
    import src.colab_dependencies as dependencies
    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU unavailable")
    monkeypatch.setattr(dependencies, "ensure_colab_dependencies", unavailable)
    gpu_check = next(source for source in sources()[1] if "dependency_status = ensure_colab_dependencies" in source)
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        exec(gpu_check, {"REPO_DIR": NOTEBOOK_PATH.parents[1], "sys": sys})



def test_private_checkout_authenticates_clone_and_fetch_without_persisting_token(tmp_path, monkeypatch):
    import base64
    import getpass
    import os
    token = "fake-read-only-token"
    encoded = base64.b64encode(("x-access-token:" + token).encode()).decode()
    monkeypatch.setattr(getpass, "getpass", lambda prompt: token)
    repo = tmp_path / "source checkout"
    url = "https://github.com/Avilas01/ido-aum-lite.git"
    calls = []
    def run(args, **kwargs):
        calls.append((args, {**kwargs, "env": kwargs["env"].copy()}))
        command = args[1:]
        if command[0] == "clone":
            (repo / ".git").mkdir(parents=True)
        output = ""
        if command == ["remote", "get-url", "origin"]:
            output = url
        if command[0] == "rev-parse":
            output = "test-sha"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")
    env = {"REPO_DIR": repo, "REPO_URL": url, "TARGET_REF": "main", "saved_manifest": None,
           "os": os, "subprocess": SimpleNamespace(run=run)}
    source = next(source for source in sources()[1] if source.startswith("def checkout_source"))
    previous = dict(os.environ)
    exec(source, env)
    assert env["GIT_SHA"] == "test-sha" and dict(os.environ) == previous
    assert any(args[1] == "clone" for args, _ in calls)
    assert any(args[1] == "fetch" for args, _ in calls)
    for args, kwargs in calls:
        assert token not in str(args) and encoded not in str(args)
        headers = [value for key, value in kwargs["env"].items() if key.startswith("GIT_CONFIG_VALUE_")]
        assert "Authorization: Basic " + encoded in headers
    assert "token" not in env and not (repo / ".git" / "config").exists()


def test_checkout_errors_show_git_reason_without_exposing_token(tmp_path, monkeypatch):
    import base64
    import getpass
    import os
    token = "fake-private-token"
    encoded = base64.b64encode(("x-access-token:" + token).encode()).decode()
    monkeypatch.setattr(getpass, "getpass", lambda prompt: token)
    def failed(args, **kwargs):
        return SimpleNamespace(returncode=128, stdout="", stderr="Repository not found " + token + " " + encoded)
    env = {"REPO_DIR": tmp_path / "checkout", "REPO_URL": "https://github.com/Avilas01/ido-aum-lite.git",
           "TARGET_REF": "main", "saved_manifest": None, "os": os, "subprocess": SimpleNamespace(run=failed)}
    source = next(source for source in sources()[1] if source.startswith("def checkout_source"))
    with pytest.raises(RuntimeError, match="Repository not found") as error:
        exec(source, env)
    assert token not in str(error.value) and encoded not in str(error.value)
    assert "[REDACTED]" in str(error.value) and "Read-only" in str(error.value)
