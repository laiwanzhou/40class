from __future__ import annotations

from pathlib import Path


def test_trainer_has_no_obsolete_fixed_frame_or_cache_stages() -> None:
    source = (Path(__file__).parents[1] / "src/train_ir_primary_depth_residual_fullseq.py").read_text(
        encoding="utf-8",
    )
    for forbidden in ("stage_a", "stage_b", "full_sequence_frames", "extract_cache", "cached_epoch"):
        assert forbidden not in source
    assert "FrameBudgetBatchSampler" in source
    assert "competition_test_read" in source
    assert '"skeleton_connected"] = False' in source
    assert 'choices=("raw", "relative", "raw+relative")' in source
    assert 'parser.add_argument("--epochs"' in source
    assert 'parser.add_argument("--patience"' in source


def test_history_row_keeps_shared_plot_columns() -> None:
    source = (Path(__file__).parents[1] / "src/train_ir_primary_depth_residual_fullseq.py").read_text(
        encoding="utf-8",
    )
    for column in (
        '"accuracy": val["accuracy"]',
        '"macro_f1": val["macro_f1"]',
        '"zero_f1_class_count": val["zero_f1_class_count"]',
        '"never_predicted_class_count": val["never_predicted_class_count"]',
        '"number_of_predicted_classes": val["number_of_predicted_classes"]',
    ):
        assert column in source
