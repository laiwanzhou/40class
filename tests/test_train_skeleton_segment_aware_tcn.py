from __future__ import annotations

from pathlib import Path

import yaml

from src.train_skeleton_segment_aware_tcn_strict_oof import model_for


def test_t1_v1_config_matches_c1_contract() -> None:
    c1 = yaml.safe_load(Path("configs/experiments/skeleton_c0_c1_strict_oof.yaml").read_text(encoding="utf-8"))
    t1 = yaml.safe_load(
        Path("configs/experiments/skeleton_segment_aware_tcn_strict_oof.yaml").read_text(encoding="utf-8")
    )

    for key in (
        "data_root", "oof_folds", "strict_views_root", "sequence_length", "embedding_dim",
        "tcn_channels", "batch_size", "epochs", "learning_rate", "weight_decay", "dropout",
        "early_stopping_patience", "gradient_clip", "amp", "num_workers", "seed", "num_classes",
    ):
        assert t1[key] == c1[key]
    assert t1["scale_policy"] == "per_frame"
    assert t1["input_features"] == 102
    assert sum(parameter.numel() for parameter in model_for(t1).parameters()) == 172_776
