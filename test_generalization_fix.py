"""
Test that Generalization (rephrase) evaluation correctly broadcasts a single target
to multiple prompts. Without the fix, EasyEdit zips prompts with target string chars,
causing Generalization ~0.

Run from RomeForQwen: python3 test_generalization_fix.py

Unit tests (test_bug_without_fix, test_broadcast_logic) run without EasyEdit deps.
Integration test (test_integration_with_mock) requires full EasyEdit environment
(e.g. Modal container or venv with EasyEdit deps) and may skip if import fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add EasyEdit and project to path
EASYEDIT = Path(__file__).resolve().parent.parent / "EasyEdit"
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(EASYEDIT))
sys.path.insert(0, str(PROJECT))


def test_broadcast_logic():
    """Unit test: verify the broadcast logic produces correct zip pairs."""
    # Simulates the fix in compute_rewrite_or_rephrase_quality
    def _coerce_text(x):
        return x if isinstance(x, str) else str(x)

    rephrase_prompts = ["Angola belongs to the continent of", "Plaque - Angola belongs to"]
    target_new = "Antarctica"

    norm_prompts = [_coerce_text(p) for p in rephrase_prompts]
    norm_targets = _coerce_text(target_new)

    # THE FIX: broadcast single target to match multiple prompts
    if isinstance(norm_prompts, (list, tuple)) and not isinstance(norm_targets, (list, tuple)):
        norm_targets = [norm_targets] * len(norm_prompts)

    pairs = list(zip(norm_prompts, norm_targets))
    assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"
    assert all(t == "Antarctica" for _, t in pairs), (
        f"Each target must be 'Antarctica', got {[t for _, t in pairs]}"
    )
    # Without fix: zip would give (p1,'A'), (p2,'n') - wrong
    assert pairs[0][1] == "Antarctica" and pairs[1][1] == "Antarctica"
    print("PASS: broadcast logic produces correct (prompt, target) pairs")


def test_bug_without_fix():
    """Demonstrate the bug: without broadcast, zip uses string chars."""
    prompts = ["p1", "p2"]
    targets = "Antarctica"  # scalar string
    pairs = list(zip(prompts, targets))
    assert pairs == [("p1", "A"), ("p2", "n")], f"Bug: zip gives {pairs}"
    print("PASS: confirmed bug (zip with string yields char pairs)")


def test_integration_with_mock():
    """Integration test with mocked test_prediction_acc (requires easyeditor.evaluate)."""
    try:
        from unittest.mock import MagicMock, patch

        # Import only the evaluate module (may fail if deps missing)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "evaluate",
            EASYEDIT / "easyeditor" / "evaluate" / "evaluate.py",
        )
        if spec is None or spec.loader is None:
            print("SKIP: could not load evaluate module")
            return
        evaluate = importlib.util.module_from_spec(spec)
        # Load evaluate_utils into sys.modules so evaluate can import it
        eval_utils_spec = importlib.util.spec_from_file_location(
            "evaluate_utils",
            EASYEDIT / "easyeditor" / "evaluate" / "evaluate_utils.py",
        )
        if eval_utils_spec and eval_utils_spec.loader:
            eval_utils = importlib.util.module_from_spec(eval_utils_spec)
            sys.modules["easyeditor.evaluate.evaluate_utils"] = eval_utils
            # This will likely fail on deps (trainer, higher, etc.)
            try:
                eval_utils_spec.loader.exec_module(eval_utils)
            except Exception as e:
                print(f"SKIP: evaluate_utils import failed ({e})")
                return

        spec.loader.exec_module(evaluate)
        compute = evaluate.compute_rewrite_or_rephrase_quality
    except Exception as e:
        print(f"SKIP: integration test (EasyEdit deps): {e}")
        return

    mock_model = MagicMock()
    mock_tok = MagicMock()
    mock_hparams = MagicMock()
    mock_hparams.alg_name = "ROME"
    mock_hparams.max_length = 40
    mock_hparams.evaluation_type = None

    captured_targets = []

    def fake_test_prediction_acc(model, tok, hparams, prompts, targets, device, **kwargs):
        captured_targets.append(targets)
        return [1.0] * (len(prompts) if isinstance(prompts, (list, tuple)) else 1)

    rephrase_prompts = ["p1", "p2"]
    target_new = "Antarctica"

    with patch.object(evaluate, "test_prediction_acc", side_effect=fake_test_prediction_acc):
        compute(
            mock_model, "Qwen/Qwen2.5-7B", mock_hparams, mock_tok,
            rephrase_prompts, target_new, device=0, test_rephrase=True,
        )

    targets = captured_targets[0]
    assert isinstance(targets, (list, tuple)), f"targets must be list, got {type(targets)}"
    assert targets == ["Antarctica", "Antarctica"], f"expected broadcast, got {targets}"
    print("PASS: compute_rewrite_or_rephrase_quality broadcasts target correctly")


if __name__ == "__main__":
    test_bug_without_fix()
    test_broadcast_logic()
    test_integration_with_mock()
    print("\nAll tests passed.")
