#!/usr/bin/env python3
"""
Check how many edits in the holdout/tuning sets have non-empty generation_prompts.
Run: python check_generation_prompts.py

Does not require GPU or model download.

  cd RomeForQwen && python3 check_generation_prompts.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
COUNTERFACT_REVISION = "c01c413f856ee38f5c080c9fc5e87aff478e2ff9"


def _coerce_to_list(x) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rome_utils

    # Load known indices
    known_path = PROJECT_ROOT / "qwen_known_indices.json"
    known_indices = set(rome_utils.load_indices_file(str(known_path)))

    # Load dataset
    dataset_records = rome_utils.download_counterfact_dataset(
        split="test",
        revision=COUNTERFACT_REVISION,
    )

    # Build by case_id
    by_id = {}
    for r in dataset_records:
        try:
            cid = int(r.get("case_id"))
        except (TypeError, ValueError):
            continue
        by_id[cid] = r

    # Split: tuning (first 150) vs holdout
    known_list = sorted(known_indices)
    tuning_ids = set(known_list[:150])
    holdout_ids = set(known_list[150:])

    def count_with_gp(indices: set):
        total = 0
        has_key = 0
        non_empty = 0
        for cid in indices:
            if cid not in by_id:
                continue
            total += 1
            r = by_id[cid]
            gp = r.get("generation_prompts")
            gp_list = _coerce_to_list(gp)
            if gp is not None:
                has_key += 1
            if len(gp_list) > 0:
                non_empty += 1
        return total, has_key, non_empty

    t_tune, k_tune, e_tune = count_with_gp(tuning_ids)
    t_hold, k_hold, e_hold = count_with_gp(holdout_ids)
    t_all, k_all, e_all = count_with_gp(known_indices)

    print("Generation prompts in CounterFact (azhx/counterfact):")
    print()
    print("                    Total  Has key  Non-empty")
    print("-" * 50)
    print(f"Tuning (150):       {t_tune:5}  {k_tune:7}  {e_tune}")
    print(f"Holdout (rest):     {t_hold:5}  {k_hold:7}  {e_hold}")
    print(f"All known:          {t_all:5}  {k_all:7}  {e_all}")
    print()
    pct_tune = 100 * e_tune / t_tune if t_tune else 0
    pct_hold = 100 * e_hold / t_hold if t_hold else 0
    pct_all = 100 * e_all / t_all if t_all else 0
    print(f"Fraction with non-empty generation_prompts: tuning={pct_tune:.1f}%, holdout={pct_hold:.1f}%, all={pct_all:.1f}%")
    print()
    if e_all == 0:
        print("NOTE: No records have non-empty generation_prompts. Fluency (GE) will use")
        print("      the rewrite prompt as fallback per evaluate.py.")
    else:
        print("Fluency (GE) will use generation_prompts when present, else rewrite prompt.")


if __name__ == "__main__":
    main()
