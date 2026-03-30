#!/usr/bin/env python3
"""
Test script: run one ROME edit and verify ES, PS, NS, GE metrics.

Run locally (requires GPU + EasyEdit deps):
  cd RomeForQwen && python3 test_one_edit.py --local

Run via Modal (no local GPU):
  cd RomeForQwen && modal run test_one_edit.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
EASYEDIT_SRC = PROJECT_ROOT.parent / "EasyEdit"
COUNTERFACT_REVISION = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9"


def _run_one_edit(project_dir: Path, easyedit_dir: Path):
    import os

    sys.path.insert(0, str(easyedit_dir))
    sys.path.insert(0, str(project_dir))
    import qwen2_rome_compat
    qwen2_rome_compat.apply_qwen2_rome_compat_patch()
    from easyeditor import ROMEHyperParams, BaseEditor
    from easyeditor.evaluate.evaluate import compute_rewrite_or_rephrase_quality
    import rome_utils

    # Match Modal scripts: default to paper-aligned scoring.
    eval_metric = os.environ.get("ROME_EVAL_METRIC", "paper_rome")
    if eval_metric != "paper_rome":
        _src = inspect.getsource(compute_rewrite_or_rephrase_quality)
        if "norm_targets = [norm_targets] * len(norm_prompts)" not in _src:
            raise RuntimeError("EasyEdit evaluate.py missing Generalization broadcast fix.")

    rome_utils.set_seeds(16)

    # Load one record from CounterFact (use first known case_id for reproducibility)
    known_indices = rome_utils.load_indices_file(str(project_dir / "qwen_known_indices.json"))
    case_id = known_indices[0]

    dataset_records = rome_utils.download_counterfact_dataset(
        split="test",
        revision=COUNTERFACT_REVISION,
    )
    records = rome_utils.load_and_filter_dataset([case_id], dataset_records=dataset_records, id_field="case_id")
    if not records:
        print(f"ERROR: case_id {case_id} not found in dataset")
        sys.exit(1)

    request = rome_utils.record_to_request(records[0])
    print(f"Testing edit: case_id={case_id}")
    print(f"  prompt: {request['prompt'][:80]}...")
    print(f"  target_new: {request['target_new']}")
    print(f"  ground_truth: {request['ground_truth']}")
    print(f"  rephrase_prompts: {len(request.get('rephrase_prompt', []) or [])} prompts")
    print(f"  locality.neighborhood: {len(request.get('locality', {}).get('neighborhood', {}).get('prompt', []))} prompts")
    print(f"  generation_prompts: {len(request.get('generation_prompts', []) or [])} prompts")

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
        "layers": [15],
        "v_lr": 1e-1,
        "v_num_grad_steps": 40,
    }

    hparams = ROMEHyperParams(**base_config)
    editor = BaseEditor.from_hparams(hparams)

    print(f"\nRunning one edit with eval_metric={eval_metric!r}, test_generation=True...")
    all_metrics, _, _ = editor.edit_requests(
        [request],
        sequential_edit=False,
        verbose=True,
        test_generation=True,
        eval_metric=eval_metric,
    )

    m = all_metrics[0]
    post = m.get("post", {})

    print("\n--- Results ---")
    es = post.get("rewrite_acc")
    ps = post.get("rephrase_acc")
    ns = post.get("locality", {}).get("neighborhood_acc") if post.get("locality") else None
    ge = post.get("fluency", {}).get("ngram_entropy") if isinstance(post.get("fluency"), dict) else None

    print(f"ES (rewrite_acc):     {es}")
    print(f"PS (rephrase_acc):    {ps}")
    print(f"NS (neighborhood_acc): {ns}")
    print(f"GE (fluency):         {ge}")

    agg = rome_utils.extract_metrics(all_metrics)
    s = rome_utils.calculate_composite_score(agg["ES"], agg["PS"], agg["NS"])
    print(f"\nAggregate: ES={agg['ES']}, PS={agg['PS']}, NS={agg['NS']}, S={s}")

    # Sanity checks
    ok = True
    if es is None:
        print("FAIL: ES (rewrite_acc) is None")
        ok = False
    if ps is None:
        print("FAIL: PS (rephrase_acc) is None")
        ok = False
    if ns is None:
        print("FAIL: NS (neighborhood_acc) is None")
        ok = False
    if ge is None and (request.get("generation_prompts") or request.get("prompt")):
        print("WARN: GE (fluency) is None - generation_prompts may be empty")
    elif ge is not None:
        print(f"OK: GE = {ge}")

    if ok:
        print("\nAll core metrics (ES, PS, NS) computed successfully.")
    else:
        print("\nSome metrics failed.")
        sys.exit(1)


def main():
    _run_one_edit(PROJECT_ROOT, EASYEDIT_SRC)


# Modal app at module level so `modal run test_one_edit.py` discovers it
try:
    import modal
    app = modal.App("rome-test-one-edit")
    image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
        .pip_install(
            "torch", "transformers==4.40.2", "datasets>=2.14.0", "accelerate==0.21.0",
            "huggingface_hub", "numpy", "scipy", "nltk", "pandas", "tqdm", "omegaconf",
            "hydra-core", "einops", "higher", "fairscale", "matplotlib",
            "scikit-learn", "openai", "regex", "pyyaml", "timm", "iopath",
        )
        .run_commands("python -c \"import nltk; nltk.download('punkt_tab')\"")
        .add_local_dir(str(EASYEDIT_SRC), "/root/EasyEdit")
        .add_local_dir(str(PROJECT_ROOT), "/root/project")
    )

    @app.function(image=image, gpu="A100-40GB", timeout=3600)
    def run_test():
        _run_one_edit(Path("/root/project"), Path("/root/EasyEdit"))

    @app.local_entrypoint()
    def main_modal():
        print("Running one-edit test on Modal (A100)...")
        run_test.remote()
        print("Done.")

    _HAS_MODAL_APP = True
except ImportError:
    app = None
    _HAS_MODAL_APP = False


if __name__ == "__main__":
    if "--local" in sys.argv:
        main()
    elif _HAS_MODAL_APP:
        print("Modal app defined. Run: modal run test_one_edit.py")
    else:
        print("Modal not installed. Running locally (requires GPU)...")
        main()
