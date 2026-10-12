"""Thin Colab launcher for configurable GTR-L EvoToken training.

Upload ``gtrl_evotoken_bundle.zip`` to
``MyDrive/gtrl_evotoken_relay/gtrl_evotoken_bundle.zip``, paste this file into
one Colab cell, and run it. All available settings are documented by:

    main(["--help"])

The cell enables the current 130k BF16/compiled block-causal relay smoke run.
Add or change arguments in ``RUN_ARGS`` for slot sweeps, paths, model shape,
optimisation, checkpointing, evaluation, or monitoring.
"""

from pathlib import Path
import sys

from google.colab import drive  # type: ignore[import-not-found]


drive.mount("/content/drive")

BUNDLE = Path(
    "/content/drive/MyDrive/gtrl_evotoken_relay/"
    "gtrl_evotoken_bundle.zip"
)
if not BUNDLE.is_file():
    raise FileNotFoundError(f"Upload the EvoToken bundle to {BUNDLE}")

# Import the launcher directly from the zip; it extracts that same zip before
# starting the detached worker.
sys.path.insert(0, str(BUNDLE))
from gtrl_evotoken_colab import main  # noqa: E402


RUN_ARGS = [
    "--bundle",
    str(BUNDLE),
    "--output-slots",
    "4",
    "--experiment-mode",
    "relay",
    "--block-causal",
    "--interval-seconds",
    "900",
]

main(RUN_ARGS)
