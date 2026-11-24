"""Real tiny subprocesses prove observable output and cleanup; no ML packages."""
import io
from pathlib import Path
import textwrap

import pytest
from src import commands


def test_output_is_delivered_before_child_exits_and_stderr_is_visible(tmp_path):
    root = tmp_path / "repo with spaces"
    root.mkdir()
    (root / "probe.py").write_text(textwrap.dedent("""
        import os, sys, time
        from pathlib import Path
        assert os.environ['PYTHONUNBUFFERED'] == '1'
        ack = Path(sys.argv[1])
        print('READY', flush=True)
        print('visible stderr', file=sys.stderr, flush=True)
        deadline = time.monotonic() + 4
        while not ack.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not ack.exists():
            raise SystemExit(9)
        print('ACK RECEIVED BEFORE EXIT', flush=True)
    """))
    ack = root / "parent observed output.txt"
    class LiveSink(io.StringIO):
        def write(self, text):
            if "READY" in text:
                ack.touch()
            return super().write(text)
    sink = LiveSink()
    report = commands.run_python_module("probe", ack, repo_root=root, timeout_seconds=6, output_stream=sink)
    assert report["returncode"] == 0
    assert "visible stderr" in sink.getvalue() and "ACK RECEIVED BEFORE EXIT" in sink.getvalue()


def test_child_failure_includes_exit_code_and_last_output(tmp_path):
    (tmp_path / "bad.py").write_text("import sys\nprint('failed download stage',file=sys.stderr,flush=True)\nsys.exit(7)\n")
    with pytest.raises(RuntimeError, match="code 7") as error:
        commands.run_python_module("bad", repo_root=tmp_path, output_stream=io.StringIO())
    assert "failed download stage" in str(error.value)


@pytest.mark.parametrize("kind", ["timeout", "interrupt"])
def test_timeout_or_interrupt_stops_the_child(tmp_path, monkeypatch, kind):
    (tmp_path / "blocked.py").write_text("import time\nprint('BLOCKED',flush=True)\ntime.sleep(30)\n")
    launched = []
    original = commands.subprocess.Popen
    def launch(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process)
        return process
    monkeypatch.setattr(commands.subprocess, "Popen", launch)
    class InterruptSink(io.StringIO):
        def write(self, text):
            if kind == "interrupt" and "BLOCKED" in text:
                raise KeyboardInterrupt()
            return super().write(text)
    expected = TimeoutError if kind == "timeout" else KeyboardInterrupt
    with pytest.raises(expected):
        commands.run_python_module("blocked", repo_root=tmp_path, timeout_seconds=.5 if kind == "timeout" else 5,
                                   output_stream=InterruptSink())
    assert len(launched) == 1 and launched[0].poll() is not None


def test_missing_module_fails_without_launching_a_child(tmp_path):
    with pytest.raises(RuntimeError, match="absent from this checkout"):
        commands.run_python_module("src.missing", repo_root=tmp_path)



def test_uninterruptible_child_cleanup_is_bounded(monkeypatch):
    class StuckProcess:
        pid = 123456789
        def poll(self): return None
        def terminate(self): pass
        def kill(self): pass
        def wait(self, timeout):
            assert timeout == 5
            raise commands.subprocess.TimeoutExpired("fixture", timeout)
    if commands.os.name == "posix":
        monkeypatch.setattr(commands.os, "killpg", lambda pid, sig: None)
    assert commands._stop_process(StuckProcess()) is False
