"""test_ladder.py -- Unit and regression tests for run_ladder and export_hf."""

import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_ladder


class TestLadder(unittest.TestCase):
    def test_ladder_help(self):
        res = subprocess.run([sys.executable, str(ROOT / "run_ladder.py"), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Multi-seed decider training", res.stdout)

    def test_export_hf_help(self):
        res = subprocess.run([sys.executable, str(ROOT / "export_hf.py"), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("Export decider-v2 checkpoint", res.stdout)

    def test_ladder_dryrun(self):
        cmd = [
            sys.executable, str(ROOT / "run_ladder.py"),
            "--seeds", "0", "1", "2",
            "--mode", "smoke",
            "--dry_run",
            "--work_dir", str(ROOT / "runs" / "_test_ladder")
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(res.returncode, 0)
        self.assertIn("[DRY RUN] Finished planned steps", res.stdout)
        self.assertIn("baseline seed 0", res.stdout)
        self.assertIn("v2 seed 2", res.stdout)

    def test_results_markdown_generation(self):
        mock_result = {
            "per_seed": [
                {"seed": 0, "baseline_acc_pp": 70.0, "v2_acc_pp": 74.0, "delta_pp": 4.0,
                 "ci_low_pp": 2.5, "ci_high_pp": 5.5, "excludes_zero": True},
                {"seed": 1, "baseline_acc_pp": 70.5, "v2_acc_pp": 74.2, "delta_pp": 3.7,
                 "ci_low_pp": 2.2, "ci_high_pp": 5.2, "excludes_zero": True},
                {"seed": 2, "baseline_acc_pp": 70.2, "v2_acc_pp": 74.4, "delta_pp": 4.2,
                 "ci_low_pp": 2.8, "ci_high_pp": 5.8, "excludes_zero": True},
            ],
            "delta_pp": 3.97,
            "seed_spread_pp": 0.5,
            "n_seeds": 3,
            "bootstrap": {
                "n_questions": 4000,
                "n_resamples": 10000,
                "ci_low_pp": 2.5,
                "ci_high_pp": 5.5,
                "excludes_zero": True,
            },
            "verdict": "SEPARABLE",
            "checks": [("a", True), ("b", True), ("c", True), ("d", True)],
        }
        md = run_ladder.generate_results_markdown(mock_result, "baseline", "v2", "smoke")
        self.assertIn("Verdict: SEPARABLE", md)
        self.assertIn("ALL CONDITIONS PASSED", md)
        self.assertIn("Seeds >= 3", md)


if __name__ == "__main__":
    unittest.main()
