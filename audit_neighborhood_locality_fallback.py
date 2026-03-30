#!/usr/bin/env python3
"""
Measure how often neighborhood locality relies on `requested_rewrite.target_true`
broadcast (ROME/EasyEdit convention) vs explicit per-prompt labels.

Uses the same CounterFact revision and index split as the Modal pipeline when
`tuning_indices_used.json` is present next to `qwen_known_indices.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run from RomeForQwen/
PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))

import rome_utils  # noqa: E402

COUNTERFACT_REVISION = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9"


def _subject_for_record(record: dict, prompt: str, rw: dict | None) -> str:
    subject = record.get("subject")
    if not subject and rw and isinstance(rw, dict):
        subject = rw.get("subject")
    if not subject:
        if "," in str(prompt):
            subject = str(prompt).split(",")[0].strip()
        else:
            words = str(prompt).split()[:2]
            subject = " ".join(words) if words else "unknown"
    return rome_utils._coerce_text(subject)


def neighborhood_resolution_for_record(record: dict) -> tuple[bool, str | None, int]:
    """
    Returns (has_neighborhood, source_tag, n_prompts) using the same prompt list
    as ``record_to_request`` (after ``{}`` substitution).
    """
    prompt = record.get("prompt")
    target_new = record.get("target_new")
    ground_truth = record.get("ground_truth")
    subject = record.get("subject")
    loc_prompt = record.get("locality_prompt")
    rephrase = record.get("rephrase_prompt")

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

    if rephrase is None and "paraphrase_prompts" in record:
        pp = record.get("paraphrase_prompts")
        if isinstance(pp, (list, tuple)) and pp:
            rephrase = list(pp)
    if loc_prompt is None and "neighborhood_prompts" in record:
        nprompts = record.get("neighborhood_prompts")
        if isinstance(nprompts, (list, tuple)) and nprompts:
            loc_prompt = list(nprompts)

    if isinstance(target_new, dict) and "str" in target_new:
        target_new = target_new["str"]
    if isinstance(ground_truth, dict) and "str" in ground_truth:
        ground_truth = ground_truth["str"]

    if prompt is None or target_new is None:
        return False, None, 0
    prompt = rome_utils._coerce_text(prompt)
    subject = _subject_for_record(record, prompt, rw if isinstance(rw, dict) else None)
    if "{}" in prompt:
        prompt = prompt.replace("{}", subject)

    if loc_prompt is None:
        return False, None, 0

    if isinstance(loc_prompt, (list, tuple)):
        loc_prompts = [
            rome_utils._coerce_text(p).replace("{}", subject) if "{}" in rome_utils._coerce_text(p) else rome_utils._coerce_text(p)
            for p in loc_prompt
        ]
    else:
        one = rome_utils._coerce_text(loc_prompt)
        loc_prompts = [one.replace("{}", subject) if "{}" in one else one]

    _truths, src = rome_utils.resolve_neighborhood_ground_truths(record, loc_prompts, rw)
    return True, src, len(loc_prompts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--known-json",
        type=Path,
        default=PROJECT_ROOT / "qwen_known_indices.json",
        help="Path to qwen_known_indices.json",
    )
    parser.add_argument(
        "--tuning-json",
        type=Path,
        default=PROJECT_ROOT / "tuning_indices_used.json",
        help="Tuning case IDs (holdout = known \ tuning). If missing, uses full known set.",
    )
    parser.add_argument(
        "--revision",
        default=COUNTERFACT_REVISION,
        help="HuggingFace dataset revision (commit hash)",
    )
    args = parser.parse_args()

    known = rome_utils.load_indices_file(str(args.known_json))
    tuning_path = args.tuning_json
    if tuning_path.is_file():
        tuning = set(rome_utils.load_indices_file(str(tuning_path)))
        eval_ids = sorted(set(known) - tuning)
        split_label = "holdout (known \\ tuning)"
    else:
        eval_ids = list(known)
        split_label = "full known set (no tuning_indices_used.json)"

    from datasets import load_dataset

    ds = load_dataset("azhx/counterfact", split="test", revision=args.revision)
    by_id = {int(r["case_id"]): dict(r) for r in ds}

    counts: dict[str, int] = {}
    n_eval = 0
    for cid in eval_ids:
        rec = by_id.get(int(cid))
        if rec is None:
            continue
        n_eval += 1
        has_nb, src, _n = neighborhood_resolution_for_record(rec)
        if not has_nb:
            counts["_no_neighborhood"] = counts.get("_no_neighborhood", 0) + 1
            continue
        assert src is not None
        counts[src] = counts.get(src, 0) + 1

    broadcast = counts.get("requested_rewrite_target_true_broadcast", 0)
    pct = 100.0 * broadcast / n_eval if n_eval else 0.0

    print(f"Split: {split_label}")
    print(f"Records evaluated: {n_eval} (ids requested: {len(eval_ids)})")
    print("Resolution source counts (among records with neighborhood prompts):")
    for k in sorted(counts, key=lambda x: (-counts[x], x)):
        print(f"  {k}: {counts[k]}")
    print()
    print(
        f"requested_rewrite_target_true_broadcast: {broadcast} / {n_eval} = {pct:.1f}% "
        "(warned in logs when record_to_request runs)"
    )
    hf_note = (
        "HuggingFace `azhx/counterfact` features include `neighborhood_prompts` (list[str]) only — "
        "no parallel `neighborhood_ground_truths`. The official ROME `counterfact.json` matches "
        "this shape; EasyEdit examples broadcast `target_true.str`."
    )
    print()
    print(hf_note)
    if pct > 5.0:
        print()
        print(
            "Broadcast rate exceeds ~5%: consider augmenting records with "
            "`neighborhood_ground_truths` (same length as `neighborhood_prompts`) from an "
            "external KB if you need per-neighbor correctness."
        )


if __name__ == "__main__":
    main()
