"""Run directories and metadata for reproducible experiments."""

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_dir(base_dir: Path, run_id: str | None = None) -> Path:
    """Create a unique run directory without overwriting previous results."""

    if run_id is None:
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = base_dir / run_id
    suffix = 1
    while path.exists():
        path = base_dir / f"{run_id}_{suffix:02d}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def write_run_metadata(
    run_dir: Path,
    config: dict[str, Any],
    *,
    data_paths: list[Path] | None = None,
) -> None:
    """Write resolved config and environment/data fingerprints."""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            json.loads(json.dumps(config, default=str)),
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "data_sha256": {},
    }
    for path in data_paths or []:
        if path.exists():
            metadata["data_sha256"][str(path)] = sha256_file(path)
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
