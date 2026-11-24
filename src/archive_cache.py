"""Verified CIFAR archive persistence; no torch or Colab imports."""
from contextlib import nullcontext
import hashlib
from pathlib import Path
import shutil
from time import monotonic
from urllib.request import urlopen
from uuid import uuid4

from src.preparation_progress import PreparationProgress

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_FILENAME = "cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


def verified(path, checksum):
    path = Path(path)
    if not path.is_file():
        return False
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == checksum


def _publish_copy(source, destination, checksum):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        if not verified(temporary, checksum):
            raise RuntimeError(f"Archive copy checksum failed: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _download(url, root, filename, md5):
    """CPU-only download with bounded socket waits and visible byte progress."""
    root = Path(root)
    temporary = root / (filename + "." + uuid4().hex + ".part")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            total = response.headers.get("Content-Length", "unknown")
            count, last = 0, monotonic()
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                count += len(chunk)
                if monotonic() - last >= 5:
                    print(f"Downloaded {count:,} / {total} bytes", flush=True)
                    last = monotonic()
        if not verified(temporary, md5):
            raise RuntimeError("Downloaded CIFAR archive checksum mismatch; retry the cache block.")
        temporary.replace(root / filename)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_archive(data_root, archive_root=None, *, download=True, progress=None,
                   url=CIFAR_URL, filename=CIFAR_FILENAME, checksum=CIFAR_MD5,
                   downloader=None):
    """Use local -> persistent -> network; publish only fully verified archives.

    Extracted images always remain local. Interrupted temporary files are never
    treated as valid caches. No HTTP partial-download resume is assumed.
    """
    manager = nullcontext(progress) if progress is not None else PreparationProgress()
    with manager as reporter:
        local_root = Path(data_root).expanduser().resolve()
        local_root.mkdir(parents=True, exist_ok=True)
        local = local_root / filename
        persistent = Path(archive_root).expanduser().resolve() / filename if archive_root is not None else None
        with reporter.stage("Verify local CIFAR archive checksum"):
            local_ok = verified(local, checksum)
        persistent_ok = False
        if persistent is not None and persistent != local:
            with reporter.stage("Verify persistent CIFAR archive checksum"):
                persistent_ok = verified(persistent, checksum)
        if not local_ok and persistent_ok:
            with reporter.stage("Restore verified CIFAR archive from persistent storage"):
                _publish_copy(persistent, local, checksum)
            local_ok = True
        if not local_ok:
            if not download:
                raise RuntimeError("No verified CIFAR archive in local or persistent cache; enable download once.")
            with reporter.stage(f"Download/verify CIFAR archive from {url}"):
                (downloader or _download)(url, str(local_root), filename=filename, md5=checksum)
                if not verified(local, checksum):
                    raise RuntimeError("CIFAR archive checksum mismatch; persistent cache was not updated.")
        else:
            reporter.log("Verified CIFAR archive reused; no internet download.")
        if persistent is not None and persistent != local and not persistent_ok:
            with reporter.stage("Save verified CIFAR archive to persistent storage"):
                _publish_copy(local, persistent, checksum)
        reporter.log(f"Archive ready: {local}; persistent archive: {persistent or 'not configured'}")
        return local


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    ensure_archive(args.data_root, args.archive_root, download=not args.no_download)


if __name__ == "__main__":
    main()
