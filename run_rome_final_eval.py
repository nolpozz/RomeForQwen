"""
Phase 2: Definitive ROME baseline evaluation on holdout set (Modal).
Run from this directory: modal run run_rome_final_eval.py

Set BEST_LAYER, BEST_V_LR, BEST_V_STEPS from rome_tuning_results.csv (select by max S).
Uses same pinned image and rome_utils; validates tuning ⊆ known.

Evaluation mode: default ``prob_compare`` (legacy EasyEdit). Set env ``ROME_EVAL_METRIC=paper_rome``
for official kmeng01/rome CounterFact scoring (see RomeForQwen/METRICS_PARITY.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import modal

# -----------------------------------------------------------------------------
# Modal app and image (same pin as Phase 1)
# -----------------------------------------------------------------------------

app = modal.App("rome-final-eval")

# Same cloud Volume as run_rome_grid.py (read tuning indices, write final metrics)
ROME_RESULTS_VOLUME = modal.Volume.from_name("rome-results", create_if_missing=True)
RESULTS_DIR = "/results"

PROJECT_ROOT = Path(__file__).resolve().parent
EASYEDIT_SRC = PROJECT_ROOT.parent / "EasyEdit"
COUNTERFACT_REVISION = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9"

# Cap neighborhood locality prompts sent to EasyEdit (first N; CounterFact lists are length 10).
MAX_NB_PROMPTS = 10

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "huggingface_hub>=0.34.0,<1.0",
        "higher",
        "einops",
        "gpustat",
        "hydra-core",
        "importlib-metadata",
        "matplotlib",
        "nltk",
        "omegaconf",
        "scikit-learn",
        "scipy",
        "sentence-transformers",
        "openai",
        "peft",
        "timm",
        "iopath",
        "opencv-python",
        "av",
        "zhipuai",
        "sentencepiece",
        "rouge",
        "torch",
        "transformers==4.40.2",
        "datasets>=2.14.0",
        "accelerate==0.21.0",
        "pandas",
        "pyyaml",
        "tqdm",
        "fairscale",
        "regex",
    )
    .run_commands("python -c \"import nltk; nltk.download('punkt_tab')\"")
    .add_local_dir(str(EASYEDIT_SRC), "/root/EasyEdit")
    .add_local_dir(str(PROJECT_ROOT), "/root/project")
)

# -----------------------------------------------------------------------------
# Remote function: holdout evaluation
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=86400,
    volumes={RESULTS_DIR: ROME_RESULTS_VOLUME},
)
def run_rome_final_eval() -> tuple[list[dict], dict]:
    """
    Build holdout = known_indices \\ tuning_indices, run ROME with best params.
    Returns per-edit metrics and aggregate using paper names: ES, PS, NS, GE, S.
    """
    project_dir = Path("/root/project")
    easyedit_dir = Path("/root/EasyEdit")
    sys.path.insert(0, str(easyedit_dir))
    sys.path.insert(0, str(project_dir))
    # Apply Qwen2–ROME compatibility patch before loading model (see IMPLEMENTATION_NOTES.md)
    import os

    import qwen2_rome_compat
    qwen2_rome_compat.apply_qwen2_rome_compat_patch()
    from easyeditor import ROMEHyperParams, BaseEditor
    from easyeditor.evaluate.evaluate import compute_rewrite_or_rephrase_quality
    import rome_utils

    # Default to paper-aligned scoring because Modal may not forward local env vars.
    # prob_compare relies on broadcast fix for list paraphrases; paper_rome batches separately (official ROME).
    eval_metric = os.environ.get("ROME_EVAL_METRIC", "paper_rome")
    print(f"[ROME final eval] Using eval_metric={eval_metric!r} (set via ROME_EVAL_METRIC).")
    if eval_metric not in {"prob_compare", "paper_rome"}:
        raise ValueError(
            f"Unknown ROME_EVAL_METRIC={eval_metric!r}. Expected 'prob_compare' or 'paper_rome'."
        )
    if eval_metric != "paper_rome":
        import inspect

        _src = inspect.getsource(compute_rewrite_or_rephrase_quality)
        if "norm_targets = [norm_targets] * len(norm_prompts)" not in _src:
            raise RuntimeError(
                "EasyEdit evaluate.py missing Generalization broadcast fix. "
                "Update EasyEdit/easyeditor/evaluate/evaluate.py in workspace."
            )

    rome_utils.set_seeds(16)

    # ---------- Best hyperparameters: set from grid search (max S) ----------
    # From rome_tuning_results.csv: layers=15, v_lr=0.1, steps=40 -> S highest
    BEST_LAYER = 15
    BEST_V_LR = 1e-1
    BEST_V_STEPS = 40
    # ------------------------------------------------------------------------------------------

    workspace = project_dir
    known_indices = set(rome_utils.load_indices_file(str(workspace / "qwen_known_indices.json")))

    # Read tuning indices from cloud Volume (written by run_rome_grid.py)
    ROME_RESULTS_VOLUME.reload()
    tuning_path = Path(RESULTS_DIR) / "tuning_indices_used.json"
    if not tuning_path.exists():
        raise FileNotFoundError(
            f"{tuning_path} not found. Run run_rome_grid.py first (Phase 1) and ensure "
            "the Volume contains tuning_indices_used.json before running Phase 2."
        )
    with open(tuning_path, "r") as f:
        tuning_indices = set(int(i) for i in json.load(f))

    assert set(tuning_indices).issubset(set(known_indices)), (
        "Integrity Error: Tuning indices are not a subset of known indices!"
    )

    holdout_indices = sorted(set(known_indices) - set(tuning_indices))
    if not holdout_indices:
        raise ValueError(
            "Holdout set is empty (known_indices equals tuning_indices). "
            "Ensure qwen_known_indices.json contains more than 150 IDs."
        )

    dataset_records = rome_utils.download_counterfact_dataset(
        split="test",
        revision=COUNTERFACT_REVISION,
    )
    records = rome_utils.load_and_filter_dataset(
        holdout_indices,
        dataset_records=dataset_records,
        id_field="case_id",
    )
    nb_stats = rome_utils.neighborhood_prompt_count_stats(
        records,
        max_nb_prompts=MAX_NB_PROMPTS,
    )
    print(
        "Neighborhood prompts (holdout): "
        f"raw min/mean/max = {nb_stats['raw_min']} / {nb_stats['raw_mean']:.4f} / {nb_stats['raw_max']}; "
        f"after cap max_nb_prompts={MAX_NB_PROMPTS}: "
        f"min/mean/max = {nb_stats['after_cap_min']} / {nb_stats['after_cap_mean']:.4f} / {nb_stats['after_cap_max']} "
        f"(N records with neighborhood = {nb_stats['n_records_with_neighborhood']})"
    )
    requests = [
        rome_utils.record_to_request(r, max_nb_prompts=MAX_NB_PROMPTS) for r in records
    ]
    if not requests:
        raise ValueError(
            f"No holdout records loaded (requested {len(holdout_indices)}). "
            "Some case_ids may be missing from the dataset. Check qwen_known_indices.json."
        )

    if not any("rephrase_prompt" in req for req in requests):
        raise ValueError(
            "No requests contain `rephrase_prompt`; cannot compute Generalization. "
            "Fix `rome_utils.record_to_request()` mapping from CounterFact."
        )
    if not any(isinstance(req.get("locality"), dict) and "neighborhood" in req["locality"] for req in requests):
        raise ValueError(
            "No requests contain `locality.neighborhood`; cannot compute Locality. "
            "Fix `rome_utils.record_to_request()` mapping from CounterFact."
        )

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
        test_generation=True,  # Enable Fluency (GE) via n-gram entropy
        eval_metric=eval_metric,
    )

    agg = rome_utils.extract_metrics(all_metrics)
    per_edit = []
    for idx, (record, m) in enumerate(zip(records, all_metrics)):
        orig_idx = record.get("case_id", idx)
        post = m.get("post", {})
        row = {"edit_index": idx, "original_index": orig_idx}
        if "rewrite_acc" in post:
            v = post["rewrite_acc"]
            row["ES"] = float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
        else:
            row["ES"] = None
        if "rephrase_acc" in post:
            v = post["rephrase_acc"]
            row["PS"] = float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v)
        else:
            row["PS"] = None
        if "locality" in post and post["locality"]:
            loc_vals = []
            for k, v in post["locality"].items():
                if k.endswith("_acc"):
                    if isinstance(v, (list, tuple)):
                        loc_vals.extend(v)
                    else:
                        loc_vals.append(v)
            row["NS"] = sum(loc_vals) / len(loc_vals) if loc_vals else None
        else:
            row["NS"] = None
        if "fluency" in post and isinstance(post["fluency"], dict) and "ngram_entropy" in post["fluency"]:
            row["GE"] = float(post["fluency"]["ngram_entropy"])
        else:
            row["GE"] = None
        if "portability" in post and post["portability"]:
            por_vals = []
            for k, v in post["portability"].items():
                if k.endswith("_acc"):
                    if isinstance(v, (list, tuple)):
                        por_vals.extend(v)
                    else:
                        por_vals.append(v)
            row["Portability"] = sum(por_vals) / len(por_vals) if por_vals else None
        else:
            row["Portability"] = None
        per_edit.append(row)

    s_score = agg["S"]
    summary = {
        "ES": agg["ES"],
        "PS": agg["PS"],
        "NS": agg["NS"],
        "GE": agg["GE"],
        "S": s_score,
        "Portability": agg.get("Portability"),
        "n_ES": agg["n_ES"],
        "n_PS": agg["n_PS"],
        "n_NS": agg["n_NS"],
        "n_GE": agg["n_GE"],
        "n_portability": agg.get("n_portability", 0),
        "n_holdout": len(holdout_indices),
    }

    import pandas as pd
    out_dir = Path(RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(per_edit)
    cols = ["edit_index", "original_index", "ES", "PS", "NS", "GE", "Portability"]
    df = df[[c for c in cols if c in df.columns]]
    mean_data = {
        "edit_index": "MEAN", "original_index": "",
        "ES": summary["ES"], "PS": summary["PS"], "NS": summary["NS"],
        "GE": summary["GE"], "Portability": summary.get("Portability"),
        "S": summary["S"],
    }
    mean_row = pd.DataFrame([{k: v for k, v in mean_data.items() if k in df.columns or k == "S"}])
    n_data = {
        "edit_index": "N", "original_index": "",
        "ES": summary["n_ES"], "PS": summary["n_PS"], "NS": summary["n_NS"],
        "GE": summary["n_GE"], "Portability": summary.get("n_portability", 0),
    }
    n_row = pd.DataFrame([{k: v for k, v in n_data.items() if k in df.columns}])
    df = pd.concat([df, mean_row, n_row], ignore_index=True)
    df.to_csv(out_dir / "rome_final_baseline_metrics.csv", index=False)

    # Self-describing metadata for reproducibility / audit trail
    with open(out_dir / "rome_final_eval_metadata.json", "w") as f:
        json.dump(
            {
                "phase": "final_eval",
                "seed": 16,
                "dataset": {"name": "azhx/counterfact", "split": "test", "revision": COUNTERFACT_REVISION},
                "model_name": base_config["model_name"],
                "algorithm": base_config["alg_name"],
                "best_hparams": {
                    "layers": BEST_LAYER,
                    "v_lr": BEST_V_LR,
                    "v_num_grad_steps": BEST_V_STEPS,
                },
                "n_known": len(known_indices),
                "n_tuning": len(tuning_indices),
                "n_holdout": len(holdout_indices),
                "max_nb_prompts": MAX_NB_PROMPTS,
                "eval_metric": eval_metric,
                "neighborhood_prompt_count_stats": nb_stats,
                "summary": summary,
            },
            f,
            indent=2,
        )

    ROME_RESULTS_VOLUME.commit()
    print(f"Wrote results to cloud Volume at {RESULTS_DIR}/")

    return per_edit, summary


# -----------------------------------------------------------------------------
# Local entrypoint
# -----------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    print("Running ROME final baseline evaluation on holdout set (A100)...")
    per_edit, summary = run_rome_final_eval.remote()

    print(f"Holdout size: {summary['n_holdout']}")
    print(f"ES (Efficacy):        {summary['ES']}  (N = {summary['n_ES']})")
    print(f"PS (Paraphrase):      {summary['PS']}  (N = {summary['n_PS']})")
    print(f"NS (Neighborhood):    {summary['NS']}  (N = {summary['n_NS']})")
    if summary.get("GE") is not None:
        print(f"GE (Fluency):         {summary['GE']}  (N = {summary['n_GE']})")
    if summary.get("Portability") is not None:
        print(f"Portability:          {summary['Portability']}  (N = {summary.get('n_portability', 0)})")
    if summary.get("S") is not None:
        print(f"S (Composite):        {summary['S']}")

    print("Results saved to cloud Volume (pull anytime with: modal run pull_rome_results.py)")
