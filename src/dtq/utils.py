"""
Small helpers that every script needs.

Replaces: the benchmark() function that was duplicated across notebooks.
"""
import contextlib


def seed_everything(seed: int) -> None:
    """
    TODO: seed python `random`, numpy, torch (+cuda).
    Question to answer for yourself: is torch.manual_seed enough for the DiT
    pipeline, or should you pass an explicit torch.Generator into pipe(...)?
    (Hint: the old notebooks did both inconsistently.)
    """
    raise NotImplementedError


@contextlib.contextmanager
def cuda_timer(name: str = ""):
    """
    TODO: time a block with torch.cuda.synchronize() before/after (or CUDA events).
    Yield a dict and fill dict["seconds"] on exit.
    """
    raise NotImplementedError
    yield


def peak_memory_gb() -> float:
    """TODO: torch.cuda.max_memory_allocated() in GB. Remember to reset_peak_memory_stats() before a run."""
    raise NotImplementedError


def make_out_dir(path: str) -> str:
    """TODO: mkdir -p and return the path. Optionally append a timestamp."""
    raise NotImplementedError


def save_json(obj, path: str) -> None:
    raise NotImplementedError


def load_json(path: str):
    raise NotImplementedError
