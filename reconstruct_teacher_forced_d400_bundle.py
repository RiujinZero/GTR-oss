"""Reconstruct the exact source ZIP used by the active D400 control.

The live bundle was updated after the D400 run started.  Drive retains the
old revision, but raw revision downloads are not consistently available.
Because the bundle builder is deterministic, this script removes only the
subsequent archival additions and verifies the historical SHA-256.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from prepare_gator_colab_bundle import (
    FIXED_ZIP_TIME,
    ROOT,
    bundle_files,
)


EXPECTED_SHA256 = (
    "21c0e0a710a5122a3bb2c793809bdcdb280f6e3c91a02b1c8d7766c6dc8f9a1e"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experimental_checkpoints"
    / "teacherforced_d400_source_bundle_sha21c0.zip"
)
POST_RUN_FILES = {
    Path("archive_teacher_forced_colab_run.py"),
    Path("run_teacher_forced_language_colab.py"),
    Path("reconstruct_teacher_forced_d400_bundle.py"),
    Path("Papers/GATOR_TEACHER_FORCED_LANGUAGE_CONTROL.md"),
    Path("tests/test_archive_teacher_forced_run.py"),
    Path("tests/test_teacher_forced_colab_runner.py"),
}
CONFIG_LOG_MARKER = b"\n### 2026-07-19: Teacher-forced language control\n"


def _zip_info(relative: Path) -> ZipInfo:
    info = ZipInfo(relative.as_posix(), date_time=FIXED_ZIP_TIME)
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def historical_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if path.relative_to(ROOT) == Path("Papers/GATOR_CONFIGURATION_LOG.md"):
        if CONFIG_LOG_MARKER not in data:
            raise RuntimeError("teacher-forced configuration-log marker missing")
        data = data.split(CONFIG_LOG_MARKER, 1)[0]
    return data


def reconstruct(output: Path = DEFAULT_OUTPUT) -> tuple[Path, str]:
    output = output.expanduser().resolve()
    files = [
        path
        for path in bundle_files()
        if path.relative_to(ROOT) not in POST_RUN_FILES
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
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
                archive.writestr(_zip_info(relative), historical_bytes(path))
        digest = sha256(temporary_path.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                "historical bundle reconstruction mismatch: "
                f"expected {EXPECTED_SHA256}, found {digest}"
            )
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output, EXPECTED_SHA256


if __name__ == "__main__":
    path, digest = reconstruct()
    print(f"bundle={path} sha256={digest} bytes={path.stat().st_size}")
