"""
Pull ROME results from the cloud Volume to your local directory.
Run from this directory: modal run pull_rome_results.py

Use this after run_rome_grid.py or run_rome_final_eval.py completes (or if your
laptop disconnected during a run). Results are persisted to Modal's Volume,
so they survive disconnects.
"""
from __future__ import annotations

from pathlib import Path

import modal

# Same Volume used by run_rome_grid.py and run_rome_final_eval.py
ROME_RESULTS_VOLUME = modal.Volume.from_name("rome-results", create_if_missing=True)
RESULTS_DIR = "/results"

# Minimal image (no GPU needed)
image = modal.Image.debian_slim(python_version="3.10")

app = modal.App("rome-pull-results")


@app.function(
    image=image,
    volumes={RESULTS_DIR: ROME_RESULTS_VOLUME},
)
def pull_from_volume() -> dict[str, str]:
    """
    Read all result files from the Volume and return as {filename: content}.
    """
    ROME_RESULTS_VOLUME.reload()
    out_dir = Path(RESULTS_DIR)
    files: dict[str, str] = {}

    for name in [
        "tuning_indices_used.json",
        "rome_tuning_results.csv",
        "rome_grid_metadata.json",
        "rome_final_baseline_metrics.csv",
        "rome_final_eval_metadata.json",
    ]:
        p = out_dir / name
        if p.exists():
            files[name] = p.read_text()
    return files


@app.local_entrypoint()
def main():
    print("Pulling ROME results from cloud Volume...")
    files = pull_from_volume.remote()

    if not files:
        print("No result files found. Run run_rome_grid.py or run_rome_final_eval.py first.")
        return

    out_dir = Path.cwd()
    for name, content in files.items():
        path = out_dir / name
        path.write_text(content)
        print(f"  Wrote {path}")
    print(f"Pulled {len(files)} file(s) to {out_dir}")
