from __future__ import annotations

import unittest
from pathlib import Path

from run_gator_100m_nodes_colab import (
    EXPECTED_PARAMETERS,
    build_command,
    preflight_parameter_count,
)


class Gator100MRunnerTests(unittest.TestCase):
    def test_frozen_parameter_count(self) -> None:
        self.assertEqual(preflight_parameter_count(), EXPECTED_PARAMETERS)
        self.assertEqual(EXPECTED_PARAMETERS, 99_979_407)

    def test_command_freezes_node_heavy_dense_control(self) -> None:
        checkpoint = Path("/content/local/gator100m")
        command = build_command(
            steps=100,
            batch_size=4,
            eval_examples=8,
            log_every=10,
            checkpoint_dir=checkpoint,
            compile_model=True,
            python_executable="python",
        )
        joined = " ".join(command)
        self.assertIn("--vocab-size 32768", joined)
        self.assertIn("--embedding-dim 256", joined)
        self.assertIn("--d-model 256", joined)
        self.assertIn("--nodes 1337", joined)
        self.assertIn("--lanes 10", joined)
        self.assertIn("--relays 2", joined)
        self.assertIn("--active-node-top-k 32", joined)
        self.assertIn("--route-backend dense", joined)
        self.assertIn("--packed-memory-backend matrix", joined)
        self.assertIn("--batch-size 4", joined)
        self.assertEqual(command[-1], "--compile")


if __name__ == "__main__":
    unittest.main()
