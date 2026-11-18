"""Append-only experiment log: results/registry.jsonl (one JSON object per line).

Each entry records the time, git commit (and whether the working tree had
uncommitted changes), a payload describing the run, and SHA-256 hashes of any
files passed in (for example the frozen split file), so a result can later be
tied to the exact code and split that produced it.
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*args):
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=15)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def log_event(kind, payload, files=(), path="results/registry.jsonl"):
    rec = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "kind": kind,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "payload": payload,
        "file_sha256": {str(p): sha256_file(p) for p in files if Path(p).exists()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec
