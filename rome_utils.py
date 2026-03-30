"""
Shared utilities for the ROME 2-phase evaluation pipeline (Qwen2.5-7B).
Ensures reproducibility, consistent dataset handling, and objective metric aggregation.
"""
from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _coerce_text(x: Any) -> str:
    """
    Robustly coerce various CounterFact field variants into plain text.
    Handles strings, lists, dicts, and falls back to str(x).
    """
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        for v in x:
            if v:
                return _coerce_text(v)
        return ""
    if isinstance(x, dict):
        for key in ("str", "text", "value"):
            if key in x and x[key]:
                return _coerce_text(x[key])
        for v in x.values():
            if v:
                return _coerce_text(v)
        return ""
    return str(x)


def set_seeds(seed: int = 16) -> None:
    """Set explicit seeds for torch, numpy, and random to guarantee reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def download_counterfact_dataset(
    *,
    split: str = "test",
    revision: str = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9",
) -> list[dict]:
    """
    Download CounterFact from HuggingFace (`azhx/counterfact`) with a pinned revision.

    This is the canonical data source used by the Modal scripts to avoid local file drift.
    The pinned `revision` must be an immutable commit hash.
    """
    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split=split, revision=revision)
    return [dict(r) for r in ds]


def load_and_filter_dataset(
    indices_to_keep: list[int],
    *,
    dataset_records: list[dict],
    id_field: str = "case_id",
) -> list[dict]:
    """
    Filter CounterFact records to match `indices_to_keep` exactly.

    - If `id_field` exists in records, treats `indices_to_keep` as *IDs* (e.g. case_id) and returns
      records in the order of `indices_to_keep` (dropping IDs not present).
    - Otherwise, treats `indices_to_keep` as *positional indices* into `dataset_records` (dropping out-of-range).

    This prevents dataset order/length drift because selection is done explicitly rather than by
    relying on implicit ordering across different dataset sources.
    """
    if not dataset_records:
        return []

    if isinstance(dataset_records[0], dict) and id_field in dataset_records[0]:
        by_id: dict[int, dict] = {}
        for r in dataset_records:
            try:
                rid = int(r.get(id_field))
            except Exception:
                continue
            by_id[rid] = r
        return [by_id[i] for i in indices_to_keep if i in by_id]

    n = len(dataset_records)
    valid = [i for i in indices_to_keep if 0 <= i < n]
    return [dataset_records[i] for i in valid]


def _get_all_acc_keys(dict_list: list[dict]) -> set:
    acc_keys = set()
    def recurse(d: dict) -> None:
        for k, v in d.items():
            if k.endswith("acc"):
                acc_keys.add(k)
            if isinstance(v, dict):
                recurse(v)
    for d in dict_list:
        recurse(d)
    return acc_keys


def extract_metrics(metrics_list: list[dict]) -> dict[str, Any]:
    """
    Parse EasyEdit per-edit output (list of dicts with 'post' key).
    Expects the same ``rewrite_acc`` / ``rephrase_acc`` / ``locality.*_acc`` keys for
    both ``eval_metric="prob_compare"`` (EasyEdit) and ``eval_metric="paper_rome"``
    (official ROME release scoring); only the per-prompt binary definitions differ.

    Returns means and sample sizes for ROME paper metrics:
    - ES (Efficacy Score): rewrite success
    - PS (Paraphrase Score): generalization to paraphrases
    - NS (Neighborhood Score): locality / specificity
    - GE (Fluency): n-gram entropy of generations (higher = more fluent)
    - S: harmonic mean of ES, PS, NS
    """
    if not metrics_list:
        return {
            "ES": None, "PS": None, "NS": None, "GE": None, "S": None,
            "n_ES": 0, "n_PS": 0, "n_NS": 0, "n_GE": 0,
        }
    es_vals = []
    ps_vals = []
    ns_vals = []
    ge_vals = []
    for m in metrics_list:
        post = m.get("post", {})
        if "rewrite_acc" in post:
            v = post["rewrite_acc"]
            es_vals.append(float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v))
        if "rephrase_acc" in post:
            v = post["rephrase_acc"]
            ps_vals.append(float(np.mean(v)) if isinstance(v, (list, tuple)) else float(v))
        if "locality" in post and post["locality"]:
            loc_flat = []
            for lkey in _get_all_acc_keys([post]):
                if lkey in post["locality"]:
                    v = post["locality"][lkey]
                    if isinstance(v, (list, tuple)):
                        loc_flat.extend(v)
                    else:
                        loc_flat.append(v)
            if loc_flat:
                ns_vals.append(float(np.mean(loc_flat)))
        if "fluency" in post and isinstance(post["fluency"], dict) and "ngram_entropy" in post["fluency"]:
            ge_vals.append(float(post["fluency"]["ngram_entropy"]))
    # Portability: optional, when dataset provides one-hop etc.
    portability_vals = []
    for m in metrics_list:
        post = m.get("post", {})
        if "portability" in post and post["portability"]:
            por_flat = []
            for pkey in _get_all_acc_keys([post]):
                if pkey in post["portability"]:
                    v = post["portability"][pkey]
                    if isinstance(v, (list, tuple)):
                        por_flat.extend(v)
                    else:
                        por_flat.append(v)
            if por_flat:
                portability_vals.append(float(np.mean(por_flat)))
    s_val = calculate_composite_score(
        float(np.mean(es_vals)) if es_vals else None,
        float(np.mean(ps_vals)) if ps_vals else None,
        float(np.mean(ns_vals)) if ns_vals else None,
    )
    return {
        "ES": float(np.mean(es_vals)) if es_vals else None,
        "PS": float(np.mean(ps_vals)) if ps_vals else None,
        "NS": float(np.mean(ns_vals)) if ns_vals else None,
        "GE": float(np.mean(ge_vals)) if ge_vals else None,
        "S": s_val,
        "Portability": float(np.mean(portability_vals)) if portability_vals else None,
        "n_ES": len(es_vals),
        "n_PS": len(ps_vals),
        "n_NS": len(ns_vals),
        "n_GE": len(ge_vals),
        "n_portability": len(portability_vals),
    }


def calculate_composite_score(
    efficacy: float | None,
    generalization: float | None,
    locality: float | None,
) -> float | None:
    """
    Harmonic mean of ES, PS, NS (Score S per ROME paper).
    Used for hyperparameter selection and aggregate reporting.
    Returns None if any metric is missing or non-positive.
    """
    if efficacy is None or generalization is None or locality is None:
        return None
    if efficacy <= 0 or generalization <= 0 or locality <= 0:
        return None
    return 3.0 / (1.0 / efficacy + 1.0 / generalization + 1.0 / locality)


def resolve_neighborhood_ground_truths(
    record: dict,
    loc_prompts: list[str],
    rw: dict | str | None,
) -> tuple[list[str] | None, str | None]:
    """
    Decide neighborhood ``ground_truth`` labels (parallel to ``loc_prompts``).

    Returns ``(truths, source_tag)`` where ``source_tag`` is one of:
    ``parallel_record_field``, ``locality_ground_truth_list``,
    ``locality_ground_truth_scalar_broadcast``, ``requested_rewrite_target_true_broadcast``,
    or ``missing`` if no labels can be resolved.
    """
    n_nb = len(loc_prompts)
    for key in ("neighborhood_ground_truths", "neighborhood_targets", "neighborhood_answers"):
        raw = record.get(key)
        if isinstance(raw, (list, tuple)) and n_nb > 0 and len(raw) >= n_nb:
            return [_coerce_text(x) for x in raw[:n_nb]], "parallel_record_field"

    top_lg = record.get("locality_ground_truth")
    if isinstance(top_lg, (list, tuple)) and n_nb > 0 and len(top_lg) >= n_nb:
        return [_coerce_text(x) for x in top_lg[:n_nb]], "locality_ground_truth_list"
    if top_lg is not None:
        return [_coerce_text(top_lg)] * n_nb, "locality_ground_truth_scalar_broadcast"

    tt = None
    if rw and isinstance(rw, dict):
        tt = rw.get("target_true") or rw.get("ground_truth")
    if tt is not None:
        if isinstance(tt, dict) and "str" in tt:
            tt = tt["str"]
        tt_text = _coerce_text(tt)
        return [tt_text] * n_nb, "requested_rewrite_target_true_broadcast"

    return None, "missing"


def raw_neighborhood_prompt_count(record: dict) -> int:
    """
    Number of neighborhood locality prompts in ``record`` before ``max_nb_prompts``
    truncation (CounterFact: ``neighborhood_prompts`` or ``locality_prompt``).
    """
    loc_prompt = record.get("locality_prompt")
    if loc_prompt is None and "neighborhood_prompts" in record:
        nprompts = record.get("neighborhood_prompts")
        if isinstance(nprompts, (list, tuple)) and nprompts:
            return len(nprompts)
    if loc_prompt is None:
        return 0
    if isinstance(loc_prompt, (list, tuple)):
        return len(loc_prompt)
    return 1


def neighborhood_prompt_count_stats(
    records: list[dict],
    *,
    max_nb_prompts: int = 10,
) -> dict[str, Any]:
    """
    Min / mean / max neighborhood prompt counts: raw dataset vs after ``max_nb_prompts`` cap.
    Only includes records that have at least one neighborhood prompt.
    """
    raw_counts: list[int] = []
    eval_counts: list[int] = []
    for r in records:
        c = raw_neighborhood_prompt_count(r)
        if c <= 0:
            continue
        raw_counts.append(c)
        cap = max_nb_prompts if max_nb_prompts > 0 else c
        eval_counts.append(min(c, cap))
    if not raw_counts:
        return {
            "n_records_with_neighborhood": 0,
            "raw_min": 0,
            "raw_mean": 0.0,
            "raw_max": 0,
            "after_cap_min": 0,
            "after_cap_mean": 0.0,
            "after_cap_max": 0,
            "max_nb_prompts_cap": max_nb_prompts,
        }
    arr_raw = np.array(raw_counts, dtype=np.float64)
    arr_ev = np.array(eval_counts, dtype=np.float64)
    return {
        "n_records_with_neighborhood": len(raw_counts),
        "raw_min": int(arr_raw.min()),
        "raw_mean": float(arr_raw.mean()),
        "raw_max": int(arr_raw.max()),
        "after_cap_min": int(arr_ev.min()),
        "after_cap_mean": float(arr_ev.mean()),
        "after_cap_max": int(arr_ev.max()),
        "max_nb_prompts_cap": max_nb_prompts,
    }


def record_to_request(record: dict, *, max_nb_prompts: int = 10) -> dict:
    """Convert a CounterFact-style record to EasyEdit editor request format (with locality dict).

    This function is robust to slight schema differences between CounterFact
    dumps. It prefers top-level `prompt` / `target_new` / `ground_truth`
    fields, but will fall back to common nested structures (e.g. the
    `requested_rewrite` dict used in some variants) when those are absent.

    :param max_nb_prompts: Cap on neighborhood locality prompts (first *N* from the list).
        Default ``10`` matches typical CounterFact list length; use a larger value to disable
        effectively. If ``<= 0``, no cap is applied (all prompts retained).
    """
    # Primary, flat schema (EasyEdit expects these keys for evaluation):
    # - prompt, target_new, ground_truth
    # - rephrase_prompt
    # - locality_prompt, locality_ground_truth
    prompt = record.get("prompt")
    target_new = record.get("target_new")
    ground_truth = record.get("ground_truth")
    subject = record.get("subject")
    rephrase = record.get("rephrase_prompt")
    loc_prompt = record.get("locality_prompt")

    # Fallback: nested requested_rewrite-style schema
    rw = record.get("requested_rewrite") or record.get("edit", {}).get("requested_rewrite")
    if rw and isinstance(rw, dict):
        if subject is None:
            subject = rw.get("subject")
        if prompt is None:
            prompt = rw.get("prompt")
        if target_new is None:
            target_new = rw.get("target_new")
        if ground_truth is None:
            ground_truth = rw.get("ground_truth") or rw.get("target_true")

    # HuggingFace `azhx/counterfact` schema stores prompts differently:
    # - paraphrase_prompts: list[str] (generalization)
    # - neighborhood_prompts: list[str] (locality)
    # - requested_rewrite.target_true: {"str": "..."} (ground truth)
    if rephrase is None and "paraphrase_prompts" in record:
        pp = record.get("paraphrase_prompts")
        if isinstance(pp, (list, tuple)) and pp:
            # Use ALL paraphrases for a more stable Generalization estimate.
            rephrase = list(pp)
    if loc_prompt is None and "neighborhood_prompts" in record:
        nprompts = record.get("neighborhood_prompts")
        if isinstance(nprompts, (list, tuple)) and nprompts:
            # Full list from CounterFact; optional cap below for EasyEdit cost / comparability.
            loc_prompt = list(nprompts)

    if (
        loc_prompt is not None
        and isinstance(loc_prompt, (list, tuple))
        and max_nb_prompts > 0
    ):
        loc_prompt = list(loc_prompt)[:max_nb_prompts]

    if isinstance(target_new, dict) and "str" in target_new:
        target_new = target_new["str"]
    if isinstance(ground_truth, dict) and "str" in ground_truth:
        ground_truth = ground_truth["str"]

    if prompt is None or target_new is None:
        raise KeyError(
            f"record_to_request could not find prompt/target_new in record: keys={list(record.keys())}"
        )
    prompt = _coerce_text(prompt)
    target_new = _coerce_text(target_new)
    if ground_truth is None:
        # Prefer CounterFact's original answer if present; otherwise fall back.
        ground_truth = "<|endoftext|>"
    ground_truth = _coerce_text(ground_truth)

    if not subject:
        if rw and isinstance(rw, dict) and rw.get("subject"):
            subject = rw.get("subject")
        elif "," in str(prompt):
            subject = str(prompt).split(",")[0].strip()
        else:
            words = str(prompt).split()[:2]
            subject = " ".join(words) if words else "unknown"
    subject = _coerce_text(subject)

    # Fill CounterFact templates like "{} is located in"
    if "{}" in prompt:
        prompt = prompt.replace("{}", subject)

    locality: dict[str, Any] = {}
    if loc_prompt is not None:
        if isinstance(loc_prompt, (list, tuple)):
            loc_prompts = [
                _coerce_text(p).replace("{}", subject) if "{}" in _coerce_text(p) else _coerce_text(p)
                for p in loc_prompt
            ]
        else:
            one = _coerce_text(loc_prompt)
            loc_prompts = [one.replace("{}", subject) if "{}" in one else one]

        loc_ground_truths, nb_src = resolve_neighborhood_ground_truths(record, loc_prompts, rw)
        if nb_src == "requested_rewrite_target_true_broadcast":
            tt_text = loc_ground_truths[0] if loc_ground_truths else ""
            case_id = record.get("case_id")
            cid = f"case_id={case_id}" if case_id is not None else f"keys={list(record.keys())}"
            logger.warning(
                "%s: neighborhood locality uses `requested_rewrite.target_true` (%r) for all "
                "%d neighborhood prompts. azhx/counterfact has no per-prompt neighborhood "
                "answers in the schema (only `neighborhood_prompts`); neighbors refer to other "
                "subjects and may have different correct objects. Supply "
                "`neighborhood_ground_truths` (list, same length as prompts) if you have them.",
                cid,
                tt_text,
                len(loc_prompts),
            )

        if loc_ground_truths is not None:
            locality["neighborhood"] = {
                "prompt": loc_prompts,
                "ground_truth": loc_ground_truths,
            }

    req: dict[str, Any] = {
        "prompt": prompt,
        "target_new": target_new,
        "ground_truth": ground_truth,
        "subject": subject,
        "portability": {},
        # EasyEdit editor expects this key to exist (even if empty).
        "locality": locality,
    }

    if rephrase is not None:
        if isinstance(rephrase, (list, tuple)):
            req["rephrase_prompt"] = [
                _coerce_text(p).replace("{}", subject) if "{}" in _coerce_text(p) else _coerce_text(p)
                for p in rephrase
            ]
        else:
            rp = _coerce_text(rephrase)
            req["rephrase_prompt"] = rp.replace("{}", subject) if "{}" in rp else rp

    # generation_prompts: for Fluency (GE) - n-gram entropy of edited model's generations
    if "generation_prompts" in record:
        gp = record.get("generation_prompts")
        if isinstance(gp, (list, tuple)) and gp:
            req["generation_prompts"] = [
                _coerce_text(p).replace("{}", subject) if "{}" in _coerce_text(p) else _coerce_text(p)
                for p in gp
            ]

    return req


def load_indices_file(path: str) -> list[int]:
    """
    Load an indices JSON file.

    Supported formats:
    - A JSON list of integers: [1, 2, 3, ...]
    - A JSON object containing 'case_ids': {"case_ids": [1, 2, ...], ...}
    """
    import json

    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [int(i) for i in data]

    if isinstance(data, dict):
        if "case_ids" in data and isinstance(data["case_ids"], list):
            return [int(i) for i in data["case_ids"]]
        if "indices" in data and isinstance(data["indices"], list):
            return [int(i) for i in data["indices"]]

    raise ValueError(f"Unsupported indices file format at {path}")
