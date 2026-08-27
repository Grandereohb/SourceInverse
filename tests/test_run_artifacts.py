import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "pinn_source"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from run_artifacts import save_reproducibility_bundle, sha256_file  # noqa: E402


class ReproducibilityArtifactTests(unittest.TestCase):
    def test_bundle_saves_loadable_checkpoint_and_hashed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            input_path = output_dir / "input.txt"
            input_path.write_text("source data\n", encoding="utf-8")
            model = torch.nn.Linear(2, 1)
            q_table = pd.DataFrame(
                {"time": ["2026-01-01 00:00:00"], "Q": [1.25]}
            )
            config = SimpleNamespace(TEST_SETTING=7, lower_setting="excluded")

            artifacts = save_reproducibility_bundle(
                output_dir=output_dir,
                model=model,
                config_module=config,
                best_epoch=12,
                best_raw_loss=3.5,
                random_seed=4,
                source={"x_norm": 0.1, "y_norm": -0.2},
                normalization={"length_m": 1000.0},
                final_transport={"effective_diffusivity_m2s": 2.0},
                recurrent_context={"t_values_norm": [0.0, 1.0]},
                runtime={
                    "training_epochs": 12,
                    "completed_epochs": 10,
                    "training_wall_time_s": 1.5,
                },
                q_time_series=q_table,
                copied_input_paths={"input": input_path},
                repo_root=REPOSITORY_ROOT,
            )

            checkpoint_path = Path(artifacts["checkpoint"])
            manifest_path = Path(artifacts["manifest"])
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(checkpoint["best_epoch"], 12)
            self.assertIn("model_state_dict", checkpoint)
            self.assertEqual(manifest["config"], {"TEST_SETTING": 7})
            self.assertEqual(manifest["runtime"]["training_epochs"], 12)
            self.assertEqual(
                artifacts["checkpoint_sha256"], sha256_file(checkpoint_path)
            )
            self.assertEqual(artifacts["manifest_sha256"], sha256_file(manifest_path))
            self.assertEqual(manifest["inputs"]["input"]["sha256"], sha256_file(input_path))


if __name__ == "__main__":
    unittest.main()
