"""
Phase 2: Definitive ROME baseline evaluation on holdout set (Modal).
Run from this directory: modal run run_rome_final_eval.py

Set BEST_LAYER, BEST_V_LR, BEST_V_STEPS from rome_tuning_results.csv (e.g. by composite_score).
Uses same pinned image and rome_utils; validates tuning ⊆ known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# -----------------------------------------------------------------------------
# Modal app and image (same pin as Phase 1)
# -----------------------------------------------------------------------------

app = modal.App("rome-final-eval")

MOUNT_PATH = "/workspace" 
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
# Remote function: holdout evaluation
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu=modal.gpu.A100(),
    mounts=get_modal_mounts(),
    timeout=86400,
)
def run_rome_final_eval() -> tuple[list[dict], dict]:
    """
    Build holdout = known_indices \\ tuning_indices, run ROME with best params,
    return per-edit metrics and aggregate (with n_efficacy, n_generalization, n_locality).
    """
    from pathlib import Path
    from easyeditor import ROMEHyperParams, BaseEditor
    sys.path.insert(0, MOUNT_PATH)
    import rome_utils

    rome_utils.set_seeds(16)

    # ---------- Best hyperparameters: set from grid search (e.g. by composite_score) ----------
    BEST_LAYER = 20
    BEST_V_LR = 5e-1
    BEST_V_STEPS = 30
    # ------------------------------------------------------------------------------------------

    workspace = Path(MOUNT_PATH)
    with open(workspace / "qwen_known_indices.json", "r") as f:
        known_indices = set(int(i) for i in json.load(f))
    with open(workspace / "tuning_indices_used.json", "r") as f:
        tuning_indices = set(int(i) for i in json.load(f))

    assert set(tuning_indices).issubset(set(known_indices)), (
        "Integrity Error: Tuning indices are not a subset of known indices!"
    )

    holdout_indices = sorted(set(known_indices) - set(tuning_indices))

    data_path = workspace / "data" / "counterfact" / "counterfact-train.json"
    if not data_path.exists():
        data_path = workspace / "counterfact-train.json"
    if not data_path.exists():
        raise FileNotFoundError(
            "CounterFact data not found. Place counterfact-train.json at "
            "data/counterfact/counterfact-train.json or counterfact-train.json in repo root."
        )
    data_path = str(data_path)

    records = rome_utils.load_and_filter_dataset(holdout_indices, data_path)
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
        "layers": [BEST_LAYER],
        "v_lr": BEST_V_LR,
        "v_num_grad_steps": BEST_V_STEPS,
    }

    hparams = ROMEHyperParams(**base_config)
    editor = BaseEditor.from_hparams(hparams)
    all_metrics, _, _ = editor.edit_requests(
        requests,
        sequential_edit=False,
        verbose=False,
        test_generation=False,
    )

    agg = rome_utils.extract_metrics(all_metrics)
    per_edit = []
    for idx, (orig_idx, m) in enumerate(zip(holdout_indices, all_metrics)):
        post = m.get("post", {})
        row = {"edit_index": idx, "original_index": orig_idx}
        if "rewrite_acc" in post:
            v = post["rewrite_acc"]
            row["Efficacy"] = float(v) if not isinstance(v, (list, tuple)) else sum(v) / len(v)
        else:
            row["Efficacy"] = None
        if "rephrase_acc" in post:
            v = post["rephrase_acc"]
            row["Generalization"] = float(v) if not isinstance(v, (list, tuple)) else sum(v) / len(v)
        else:
            row["Generalization"] = None
        if "locality" in post and post["locality"]:
            loc_vals = []
            for k, v in post["locality"].items():
                if k.endswith("_acc"):
                    if isinstance(v, (list, tuple)):
                        loc_vals.extend(v)
                    else:
                        loc_vals.append(v)
            row["Locality"] = sum(loc_vals) / len(loc_vals) if loc_vals else None
        else:
            row["Locality"] = None
        per_edit.append(row)

    summary = {
        "Efficacy": agg["Efficacy"],
        "Generalization": agg["Generalization"],
        "Locality": agg["Locality"],
        "n_efficacy": agg["n_efficacy"],
        "n_generalization": agg["n_generalization"],
        "n_locality": agg["n_locality"],
        "n_holdout": len(holdout_indices),
    }

    return per_edit, summary


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Running ROME final baseline evaluation on holdout set (A100)...")
    per_edit, summary = run_rome_final_eval.remote()

    print(f"Holdout size: {summary['n_holdout']}")
    print(
        f"Efficacy:  {summary['Efficacy']}  (N = {summary['n_efficacy']})"
    )
    print(
        f"Generalization: {summary['Generalization']}  (N = {summary['n_generalization']})"
    )
    print(
        f"Locality: {summary['Locality']}  (N = {summary['n_locality']})"
    )

    import pandas as pd
    df = pd.DataFrame(per_edit)
    cols = ["edit_index", "original_index", "Efficacy", "Generalization", "Locality"]
    df = df[[c for c in cols if c in df.columns]]
    mean_row = pd.DataFrame([{
        "edit_index": "MEAN",
        "original_index": "",
        "Efficacy": summary["Efficacy"],
        "Generalization": summary["Generalization"],
        "Locality": summary["Locality"],
    }])
    n_row = pd.DataFrame([{
        "edit_index": "N",
        "original_index": "",
        "Efficacy": summary["n_efficacy"],
        "Generalization": summary["n_generalization"],
        "Locality": summary["n_locality"],
    }])
    df = pd.concat([df, mean_row, n_row], ignore_index=True)
    df.to_csv("rome_final_baseline_metrics.csv", index=False)
    print("Wrote rome_final_baseline_metrics.csv (per-edit + MEAN row + N row for sample sizes).")
