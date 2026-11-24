"""Tiny synthetic archive fixtures only; never download CIFAR or import torch."""
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from src import archive_cache

PAYLOAD = b"tiny fixture, not CIFAR"
CHECKSUM = hashlib.md5(PAYLOAD).hexdigest()


def ensure(local, persistent, **kwargs):
    return archive_cache.ensure_archive(local, persistent, checksum=CHECKSUM, **kwargs)


def downloader(calls, payload=PAYLOAD):
    def download(url, root, filename, md5):
        calls.append(url)
        (Path(root) / filename).write_bytes(payload)
    return download


def no_network(*args, **kwargs):
    raise AssertionError("A valid cache must not use the internet")


def test_download_persists_then_new_vm_restores_without_network(tmp_path):
    calls = []
    cache = tmp_path / "Drive cache"
    first = ensure(tmp_path / "first VM", cache, downloader=downloader(calls))
    assert first.read_bytes() == PAYLOAD and len(calls) == 1
    persistent = cache / archive_cache.CIFAR_FILENAME
    modified = persistent.stat().st_mtime_ns
    second = ensure(tmp_path / "second VM", cache, download=False, downloader=no_network)
    assert second.read_bytes() == PAYLOAD
    assert persistent.stat().st_mtime_ns == modified


def test_valid_local_archive_is_backed_up_without_downloading(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    (local / archive_cache.CIFAR_FILENAME).write_bytes(PAYLOAD)
    ensure(local, tmp_path / "Drive", downloader=no_network)
    assert (tmp_path / "Drive" / archive_cache.CIFAR_FILENAME).read_bytes() == PAYLOAD


def test_corrupt_persistent_cache_is_repaired_only_from_verified_data(tmp_path):
    cache = tmp_path / "Drive"
    cache.mkdir()
    saved = cache / archive_cache.CIFAR_FILENAME
    saved.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="No verified"):
        ensure(tmp_path / "local", cache, download=False, downloader=no_network)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        ensure(tmp_path / "local", cache, downloader=downloader([], b"bad download"))
    assert saved.read_bytes() == b"corrupt"
    ensure(tmp_path / "local", cache, downloader=downloader([]))
    assert saved.read_bytes() == PAYLOAD


def test_copy_failure_never_publishes_partial_cache(tmp_path, monkeypatch):
    local = tmp_path / "local"
    local.mkdir()
    original = local / archive_cache.CIFAR_FILENAME
    original.write_bytes(PAYLOAD)
    def fail(source, destination):
        Path(destination).write_bytes(b"partial")
        raise OSError("Drive write failed")
    monkeypatch.setattr(archive_cache.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="Drive write failed"):
        ensure(local, tmp_path / "Drive", downloader=no_network)
    assert not (tmp_path / "Drive" / archive_cache.CIFAR_FILENAME).exists()
    assert list((tmp_path / "Drive").iterdir()) == []
    assert original.read_bytes() == PAYLOAD


def test_checksum_failure_during_copy_does_not_replace_existing_cache(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(PAYLOAD)
    destination = tmp_path / "saved"
    destination.write_bytes(b"old")
    monkeypatch.setattr(archive_cache.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"bad copy"))
    with pytest.raises(RuntimeError, match="copy checksum failed"):
        archive_cache._publish_copy(source, destination, CHECKSUM)
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp"))


def test_archive_module_cli_help_has_no_torch_imports():
    result = subprocess.run([sys.executable, "-c", "import sys; from src import archive_cache; assert 'torch' not in sys.modules; archive_cache.main()", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--archive-root" in result.stdout


def test_cpu_download_reports_progress_and_publishes_only_verified_content(tmp_path, monkeypatch):
    from io import BytesIO
    class Response(BytesIO):
        headers = {"Content-Length": str(len(PAYLOAD))}
    monkeypatch.setattr(archive_cache, "urlopen", lambda url, timeout: Response(PAYLOAD))
    archive_cache._download("fixture", tmp_path, "fixture.tar.gz", CHECKSUM)
    assert (tmp_path / "fixture.tar.gz").read_bytes() == PAYLOAD
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        archive_cache._download("fixture", tmp_path, "other.tar.gz", "0" * 32)
    assert not (tmp_path / "other.tar.gz").exists() and not list(tmp_path.glob("*.part"))
