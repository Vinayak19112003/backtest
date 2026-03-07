"""
Phase 4: Regenerate analysis reports in sequence with corrected metrics.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"


def run_script(script_name: str) -> None:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    print(f"Running {script_name}...")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(BASE_DIR), check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {script_name} (exit={result.returncode})")
    print(f"Completed {script_name}")


def main() -> None:
    # Regenerate exhaustive outputs, then enforce corrected metrics, then rebuild institutional reports.
    run_script("12_generate_exhaustive_filter_analysis.py")
    run_script("18_recalculate_metrics_correct.py")
    run_script("13_generate_institutional_reports_all_filters.py")

    print("All reports regenerated with corrected metrics.")


if __name__ == "__main__":
    main()
