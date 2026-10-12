"""Build the self-contained Gator source/test bundle used by Colab."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "gator_colab_bundle.zip"
INCLUDED_DIRECTORIES = (
    "gator",
    "gator_path",
    "gator_linear",
    "tests",
    "Papers",
)
ROOT_SUFFIXES = {".py", ".md", ".ipynb"}
DIRECTORY_SUFFIXES = {".py", ".md"}
FIXED_ZIP_TIME = (2026, 7, 17, 0, 0, 0)


def bundle_files(
    extra_files: tuple[Path, ...] = (),
) -> list[Path]:
    """Return the deterministic source manifest relative to the repository."""

    files = [
        path
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in ROOT_SUFFIXES
    ]
    for directory_name in INCLUDED_DIRECTORIES:
        directory = ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(f"required bundle directory is missing: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in DIRECTORY_SUFFIXES
        )
    for extra_file in extra_files:
        resolved = extra_file.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(
                f"extra bundle file is missing: {resolved}"
            )
        if not resolved.is_relative_to(ROOT):
            raise ValueError(
                f"extra bundle file must be inside {ROOT}: {resolved}"
            )
        files.append(resolved)
    return sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())


def _zip_info(relative: Path) -> ZipInfo:
    info = ZipInfo(relative.as_posix(), date_time=FIXED_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def build_bundle(
    output: Path = DEFAULT_OUTPUT,
    *,
    extra_files: tuple[Path, ...] = (),
) -> tuple[int, str]:
    """Write the bundle atomically and return its file count and SHA-256."""

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files = bundle_files(extra_files)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with ZipFile(
            temporary_path,
            mode="w",
            compression=ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in files:
                relative = path.relative_to(ROOT)
                archive.writestr(_zip_info(relative), path.read_bytes())
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    digest = sha256(output.read_bytes()).hexdigest()
    return len(files), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="archive path; defaults to gator_colab_bundle.zip",
    )
    parser.add_argument(
        "--include",
        type=Path,
        action="append",
        default=[],
        help="additional repository-local file to include (repeatable)",
    )
    args = parser.parse_args()
    count, digest = build_bundle(
        args.output,
        extra_files=tuple(args.include),
    )
    print(
        f"bundle={args.output.expanduser().resolve()} "
        f"files={count} sha256={digest}"
    )


if __name__ == "__main__":
    main()
