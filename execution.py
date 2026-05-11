# Windows-compatible execution.py
# Uses multiprocessing.Process with a Queue for result passing.
# set_start_method('spawn') is called explicitly to avoid WinError issues.

from typing import Optional, Dict
import contextlib
import io
import os
import tempfile
import multiprocessing


def _worker(program: str, result_queue):
    """Runs in a separate process. Puts result string into queue."""
    with _create_tempdir():
        try:
            exec_globals = {}
            with _swallow_io():
                exec(program, exec_globals)
            result_queue.put("passed")
        except BaseException as e:
            result_queue.put(f"failed: {e}")


def check_correctness(program: str, timeout: float,
                      completion_id: Optional[int] = None) -> Dict:
    """
    Spawns a fresh process to execute the program.
    The process is killed if it exceeds the timeout.
    Uses a multiprocessing.Queue (not Manager) to avoid WinError 6.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    p = ctx.Process(target=_worker, args=(program, result_queue))
    p.start()
    p.join(timeout=timeout)

    if p.is_alive():
        p.kill()
        p.join()
        outcome = "timed out"
    else:
        outcome = result_queue.get() if not result_queue.empty() else "timed out"

    return dict(
        passed=outcome == "passed",
        result=outcome,
        completion_id=completion_id,
    )


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


@contextlib.contextmanager
def _create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with _chdir(dirname):
            yield dirname


class _WriteOnlyStringIO(io.StringIO):
    def read(self, *args, **kwargs):
        raise IOError
    def readline(self, *args, **kwargs):
        raise IOError
    def readlines(self, *args, **kwargs):
        raise IOError
    def readable(self, *args, **kwargs):
        return False


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def _chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)