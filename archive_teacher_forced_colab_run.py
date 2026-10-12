"""Freeze a completed Colab run with its code bundle and checkpoint metadata."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil

import torch


DEFAULT_BUNDLE = Path(
    "/content/drive/MyDrive/GTR/gator_convergent_colab_bundle.zip"
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_summary(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    return {
        "file": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "step": checkpoint.get("step"),
        "format": checkpoint.get("format"),
        "parameter_tensors_numel": sum(
            int(value.numel()) for value in state.values()
        ),
        "model_config": checkpoint.get("model_config"),
        "task_config": checkpoint.get("task_config"),
        "data_config": checkpoint.get("data_config"),
        "validation": checkpoint.get("validation"),
        "training_evaluation": checkpoint.get("training_evaluation"),
    }


def archive_run(run_dir: Path, source_bundle: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    source_bundle = source_bundle.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {run_dir}")
    if not source_bundle.is_file():
        raise FileNotFoundError(f"source bundle does not exist: {source_bundle}")

    checkpoints = []
    for filename in ("best.pt", "latest.pt", "best_train.pt"):
        path = run_dir / filename
        if path.is_file():
            checkpoints.append(checkpoint_summary(path))
    if not checkpoints:
        raise FileNotFoundError(
            f"no best/latest checkpoint found in {run_dir}"
        )

    frozen_bundle = run_dir / "SOURCE_BUNDLE.zip"
    if frozen_bundle.exists():
        if file_sha256(frozen_bundle) != file_sha256(source_bundle):
            raise FileExistsError(
                "SOURCE_BUNDLE.zip already exists with different bytes"
            )
    else:
        shutil.copy2(source_bundle, frozen_bundle)

    manifest = {
        "experiment": "teacher-forced Gator language capacity control",
        "archived_utc": datetime.now(timezone.utc).isoformat(),
        "run_directory": str(run_dir),
        "source_bundle": {
            "file": frozen_bundle.name,
            "sha256": file_sha256(frozen_bundle),
            "bytes": frozen_bundle.stat().st_size,
        },
        "console_log": (
            "train.log"
            if (run_dir / "train.log").is_file()
            else "not captured by the original launcher; preserve Colab output separately"
        ),
        "checkpoints": checkpoints,
    }
    manifest_path = run_dir / "ARCHIVE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    manifest = archive_run(args.run_dir, args.source_bundle)
    print(f"archived_run={args.run_dir}", flush=True)
    print(f"manifest={manifest}", flush=True)
    print(manifest.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
