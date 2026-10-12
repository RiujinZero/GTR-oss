"""Download the pinned official TinyStories train/validation text to Drive."""

from __future__ import annotations

import argparse
from pathlib import Path


REPOSITORY = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
FILES = ("TinyStories-train.txt", "TinyStories-valid.txt")
DEFAULT_OUTPUT = Path("/content/drive/MyDrive/GTR/training_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is required; run "
            "`pip install -q huggingface_hub` once in the Colab runtime"
        ) from error

    print(
        f"Preparing {REPOSITORY}@{REVISION} in {output_dir}",
        flush=True,
    )
    for filename in FILES:
        destination = output_dir / filename
        if destination.is_file() and destination.stat().st_size > 0:
            print(
                f"ready file={destination} "
                f"size_gb={destination.stat().st_size / 1e9:.3f}",
                flush=True,
            )
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=REPOSITORY,
                filename=filename,
                repo_type="dataset",
                revision=REVISION,
                local_dir=output_dir,
            )
        )
        print(
            f"downloaded file={downloaded} "
            f"size_gb={downloaded.stat().st_size / 1e9:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
