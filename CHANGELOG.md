# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-30

### Added
- **Multi-Seed Ladder Orchestrator (`run_ladder.py`)**:
  - Automated multi-seed training and evaluation pipeline running across seeds `[0, 1, 2]` for both `baseline` and `v2` arms.
  - Automatically pipes prediction CSVs to `decide.py` for paired bootstrap resampling (10,000 resamples) and updates `RESULTS.md`.
  - Supports `--dry_run` and `--mode smoke` for instant local pipeline validation.
- **Hugging Face Release Packager (`export_hf.py`)**:
  - Exports trained checkpoints into standard Hugging Face format (`config.json`, weights in `safetensors`, tokenizer configs, and model card `README.md`).
- **Regression Test Suite (`tests/test_ladder.py`)**:
  - Full test coverage for ladder CLI flags, dry-run orchestrator commands, and Markdown report generation.
- **Design Documentation**:
  - Added formal YC Office Hours design record at `docs/designs/decider-v2-evaluation-rigor.md`.
- **Root Repository Documentation**:
  - Comprehensive root `README.md` with architecture diagrams, benchmark comparisons against published baselines, and quickstart workflows.

### Fixed
- **Parameter Counter (`scripts/paramcount.py`)**:
  - Fixed `IndexError` when expanding `layer_types` for 24-layer backbone configurations.
  - Added support for loading local configuration files directly via `Qwen3Config.from_json_file`.
- **Train Loop Budget Footgun (`train.py`)**:
  - Enforced exact budget compliance: runs iterate epochs until `total_steps` is met; exiting off-budget now raises a hard error.
- **Notebook Checklist Validation**:
  - Hardened SageMaker notebook parameter validation against AWS SDK v3 incompatibilities.

### Changed
- **Repository Hygiene**:
  - Removed all temporary dry-run logs (`_dryrun_*.txt`) and test artifacts.
  - Updated root `.gitignore` to prevent tracking of local virtual environments (`.venv-271/`, `.venv-gpu/`), cache folders (`.pytest_cache/`, `__pycache__/`), and agent configurations.

---

## [0.1.0] - 2026-10-20

### Added
- Flat repository restructuring into `decider-train/` in the style of `avbiswas/bev-train`.
- Packaging for Python 3.12 and PyTorch 2.7.1 container parity.
- Configuration ladder: `smoke.yaml`, `baseline_replica.yaml`, `decider_v2.yaml`, and 5 Tier-2 ablations.
- Core structural gate tests (`tests/test_core.py`) and numerical parity check (`tests/golden_check.py`).
