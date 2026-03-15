"""
Phase 1: ROME hyperparameter grid search on Qwen2.5-7B (Modal).
Run from this directory: modal run run_rome_grid.py

Uses rome_utils for seeds, dataset loading, metrics, and composite score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Modal app and image (EasyEdit PINNED to prevent repo drift)
# -----------------------------------------------------------------------------

app = modal.App("rome-grid-search")

MOUNT_PATH = "/workspace"
# Pin to a specific commit for reproducibility. Update when intentionally upgrading.
EASYEDIT_COMMIT = "41937637c2171b9cf1f929c143231d45a79f7787"

def get_modal_mounts():
    return [modal.Mount.from_local_dir(".", remote_path=MOUNT_PATH)]

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "pandas",
        "pyyaml",
        "tqdm",
        f"git+https://github.com/zjunlp/EasyEdit.git@{EASYEDIT_COMMIT}",
    )
)

# -----------------------------------------------------------------------------
# Remote function: grid search
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.A100(),
    mounts=get_modal_mounts(),
    timeout=86400,
)
def run_rome_grid_search() -> tuple[list[dict], list[int]]:
    """
    Load known indices, sample 150 for tuning, run ROME grid, return results
    and tuning_indices_used. Uses rome_utils for reproducibility and metrics.
    """
    from pathlib import Path
    import random
    from easyeditor import ROMEHyperParams, BaseEditor
    sys.path.insert(0, MOUNT_PATH)
    import rome_utils

    rome_utils.set_seeds(16)

    workspace = Path(MOUNT_PATH)
    with open(workspace / "qwen_known_indices.json", "r") as f:
        known_indices = [int(i) for i in json.load(f)]

    # Canonical data path (must exist to avoid dataset drift)
    data_path = workspace / "data" / "counterfact" / "counterfact-train.json"
    if not data_path.exists():
        data_path = workspace / "counterfact-train.json"
    if not data_path.exists():
        raise FileNotFoundError(
            "CounterFact data not found. Place counterfact-train.json at "
            "data/counterfact/counterfact-train.json or counterfact-train.json in repo root."
        )
    data_path = str(data_path)

    # Sample exactly 150 indices (seeded); need dataset size for valid range
    with open(data_path, "r") as f:
        n_total = len(json.load(f))
    in_range = list({i for i in known_indices if 0 <= i < n_total})
    if len(in_range) < 150:
        raise ValueError(f"Need at least 150 known indices in range; got {len(in_range)}.")
    tuning_indices_used = list(random.sample(in_range, 150))

    records = rome_utils.load_and_filter_dataset(tuning_indices_used, data_path)
    if len(records) != 150:
        raise ValueError(f"Expected 150 records after filter; got {len(records)}.")
    requests = [rome_utils.record_to_request(r) for r in records]

    base_config = {
        "alg_name": "ROME",
        "model_name": "Qwen/Qwen2.5-7B",
        "stats_dir": "/tmp/rome_stats",
        "device": 0,
        "fact_token": "subject_last",
        "v_loss_layer": 27,
        "v_weight_decay": 1e-3,
        "clamp_norm_factor": 4,
        "kl_factor": 0.0625,
        "mom2_adjustment": False,
        "context_template_length_params": [[5, 10], [10, 10]],
        "rewrite_module_tmp": "model.layers.{}.mlp.down_proj",
        "layer_module_tmp": "model.layers.{}",
        "mlp_module_tmp": "model.layers.{}.mlp",
        "attn_module_tmp": "model.layers.{}.self_attn",
        "ln_f_module": "model.norm",
        "lm_head_module": "lm_head",
        "mom2_dataset": "wikipedia",
        "mom2_n_samples": 100000,
        "mom2_dtype": "float32",
        "model_parallel": False,
        "fp16": False,
        "max_length": 40,
    }

    grid = [
        {"layers": [L], "v_lr": lr, "v_num_grad_steps": steps}
        for L in [15, 20, 24, 27]
        for lr in [5e-1, 1e-1, 5e-2]
        for steps in [20, 30, 40]
    ]

    results = []
    for override in grid:
        config = {**base_config, **override}
        hparams = ROMEHyperParams(**config)
        editor = BaseEditor.from_hparams(hparams)
        all_metrics, _, _ = editor.edit_requests(
            requests,
            sequential_edit=False,
            verbose=False,
            test_generation=False,
        )
        agg = rome_utils.extract_metrics(all_metrics)
        composite = rome_utils.calculate_composite_score(
            agg["Efficacy"], agg["Generalization"], agg["Locality"]
        )
        results.append({
            "layers": override["layers"][0],
            "v_lr": override["v_lr"],
            "v_num_grad_steps": override["v_num_grad_steps"],
            "Efficacy": agg["Efficacy"],
            "Generalization": agg["Generalization"],
            "Locality": agg["Locality"],
            "n_efficacy": agg["n_efficacy"],
            "n_generalization": agg["n_generalization"],
            "n_locality": agg["n_locality"],
            "composite_score": composite,
        })
        del editor
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results, tuning_indices_used


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Running ROME grid search on Modal (A100)...")
    results, tuning_indices_used = run_rome_grid_search.remote()
    print(f"Completed {len(results)} configurations. Tuning indices used: {len(tuning_indices_used)}")

    with open("tuning_indices_used.json", "w") as f:
        json.dump(tuning_indices_used, f, indent=2)
    print("Wrote tuning_indices_used.json")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("rome_tuning_results.csv", index=False)
    print("Wrote rome_tuning_results.csv (includes composite_score for objective selection).")
    print(df.to_string())
