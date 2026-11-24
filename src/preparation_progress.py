"""Flushed stage timings and one liveness heartbeat for data preparation."""
from contextlib import contextmanager
import threading
from time import monotonic


class PreparationProgress:
    def __init__(self, heartbeat_seconds=15):
        self.started = monotonic()
        self.heartbeat_seconds = heartbeat_seconds
        self.events = []
        self._active = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def log(self, text):
        print(f"[data +{monotonic()-self.started:.1f}s] {text}", flush=True)

    def __enter__(self):
        def heartbeat():
            while not self._stop.wait(self.heartbeat_seconds):
                with self._lock:
                    active = self._active[-1] if self._active else None
                if active:
                    self.log(f"Still working: {active[0]} ({monotonic()-active[1]:.1f}s in this stage)")
        self._thread = threading.Thread(target=heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)

    @contextmanager
    def stage(self, name):
        started = monotonic()
        entry = (name, started)
        with self._lock:
            self._active.append(entry)
        self.log("START " + name)
        status = "ok"
        try:
            yield
        except BaseException as error:
            status = "failed"
            self.log(f"FAILED {name}: {type(error).__name__}: {error}")
            raise
        finally:
            elapsed = monotonic() - started
            self.events.append({"stage": name, "seconds": elapsed, "status": status})
            with self._lock:
                self._active.remove(entry)
            self.log(f"END {name}: {elapsed:.3f}s ({status})")
