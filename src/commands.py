"""Run project modules with visible output and bounded, interruptible supervision."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
from time import monotonic


def _stop_process(process):
    """Request child/session termination; never wait forever on blocked OS I/O."""
    if process.poll() is not None:
        return True
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=5)
        return True
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)
            return True
        except subprocess.TimeoutExpired:
            # Uninterruptible filesystem I/O can delay even SIGKILL. Report it
            # instead of hanging the notebook while waiting/closing its pipe.
            return False
        except ProcessLookupError:
            return True
    except ProcessLookupError:
        return True


def run_python_module(module, *args, repo_root, timeout_seconds=None, heartbeat_seconds=15, output_stream=None):
    """Stream merged stdout/stderr through Python, not inherited notebook OS FDs.

    A reader thread prevents a quiet or partial output line from blocking timeout
    checks. -u and PYTHONUNBUFFERED make child prints observable immediately.
    No timeout is imposed on future training unless the caller explicitly sets it.
    """
    if not module or not all(part.isidentifier() for part in module.split(".")):
        raise ValueError("Expected a Python module name")
    root = Path(repo_root).resolve()
    if not root.joinpath(*module.split(".")).with_suffix(".py").is_file():
        raise RuntimeError(f"{module} is absent from this checkout; refresh source, or finish the stage if not implemented yet")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive or None")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    stream = output_stream if output_stream is not None else sys.stdout
    def emit(message, end="\n"):
        print(message, end=end, file=stream, flush=True)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONFAULTHANDLER": "1", "PYTHONIOENCODING": "utf-8",
                   "TQDM_MININTERVAL": os.environ.get("TQDM_MININTERVAL", "1")}
    command = [sys.executable, "-u", "-m", module, *map(str, args)]
    started = monotonic()
    emit(f"[run {module}] Starting from {root}; timeout={timeout_seconds or 'none'}s")
    process = subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                               bufsize=1, start_new_session=(os.name == "posix"))
    messages = queue.Queue()
    tail = deque(maxlen=40)
    def read_output():
        try:
            for line in process.stdout:
                messages.put(line)
        except (OSError, ValueError):
            pass
        finally:
            messages.put(None)
    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    last_output = monotonic()
    finished_output = False
    try:
        while not (finished_output and process.poll() is not None):
            now = monotonic()
            if timeout_seconds is not None and now - started >= timeout_seconds:
                raise TimeoutError(f"{module} exceeded {timeout_seconds}s; inspect the last preparation stage below")
            try:
                line = messages.get(timeout=0.2)
                if line is None:
                    finished_output = True
                else:
                    emit(line, end="")
                    tail.append(line.rstrip())
                    last_output = monotonic()
            except queue.Empty:
                pass
            if monotonic() - last_output >= heartbeat_seconds:
                emit(f"[run {module} pid={process.pid}] Still running after {monotonic()-started:.1f}s; no new child line yet")
                last_output = monotonic()
        result = process.wait()
        elapsed = monotonic() - started
        if result:
            raise RuntimeError(f"{module} exited with code {result} after {elapsed:.1f}s. Last output:\n" + "\n".join(tail))
        emit(f"[run {module}] Finished successfully in {elapsed:.2f}s")
        return {"module": module, "pid": process.pid, "returncode": result, "seconds": elapsed}
    except KeyboardInterrupt:
        stopped = _stop_process(process)
        status = "stopped" if stopped else "has not exited; use a fresh runtime before retrying"
        emit(f"[run {module}] Interrupted; child {process.pid} {status}. Cache/persistent files were not deleted.")
        raise
    except BaseException as error:
        if not _stop_process(process):
            emit(f"[run {module}] Child {process.pid} has not exited after termination; it may be blocked in OS I/O. Use a fresh runtime before retrying.")
        if isinstance(error, TimeoutError):
            raise TimeoutError(str(error) + "\nLast output:\n" + "\n".join(tail)) from error
        raise
    finally:
        reader.join(timeout=1)
        # Closing a TextIOWrapper while another thread is blocked reading it can
        # itself wait indefinitely. Let a stuck daemon reader finish with EOF.
        if process.stdout is not None and not reader.is_alive():
            process.stdout.close()
