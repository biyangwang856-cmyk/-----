from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .domain import require


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with tmp.open("wb") as f:
        f.write((canonical(value) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def safe_id(value):
    require(isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value), "INVALID_ID", value)
    require(value.upper() not in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}, "INVALID_ID", value)
    return value


def contained(root, path, *, existing=False):
    root = Path(root).resolve()
    candidate = Path(path)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    require(candidate.is_relative_to(root) and candidate != root, "PATH_SCOPE", "Path must stay inside the managed directory")
    if existing:
        require(candidate.is_file(), "FILE_NOT_FOUND", str(candidate), status=404)
    return candidate


def files_manifest(root, *, exclude=()):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "SOURCE_ROOT", root)
    result = []
    for p in sorted(root.rglob("*")):
        require(not p.is_symlink() and not (hasattr(p, "is_junction") and p.is_junction()), "LINK_NOT_ALLOWED", p)
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if rel not in exclude:
                result.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p)})
    return result

