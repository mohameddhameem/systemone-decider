"""Run records: what ran, where, with which code and model revisions.

Each command appends JSON lines to `<run>/provenance.jsonl`, so a run directory carries the
record the experiment protocol asks for (command, seed, git commit, packages, GPU, checkpoints).
"""

import importlib.metadata
import json
import platform
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

PACKAGES = ("torch", "transformers", "vllm", "bitsandbytes", "huggingface_hub", "datasets", "numpy", "pandas",
            "scikit-learn")
REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _gpus() -> list[str]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = _git("status", "--porcelain")
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions,
        "gpus": _gpus(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }


def hub_revision(name: str, revision: str | None = None) -> str | None:
    """Commit hash of a cached Hub model (no network), "local" for a folder, None if unknown."""
    if Path(name).is_dir():
        return "local"
    name, _, at = name.partition("@")
    revision = revision or at or "main"
    if re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(name, "config.json", revision=revision)
    m = re.search(r"snapshots[/\\]([0-9a-f]{40})", str(path)) if isinstance(path, str) else None
    return m.group(1) if m else None


def record(run: Path, event: str, **fields) -> None:
    """Append one event to <run>/provenance.jsonl."""
    run.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, "argv": sys.argv, **fields}
    with (run / "provenance.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
