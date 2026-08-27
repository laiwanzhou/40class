from __future__ import annotations

from pathlib import Path

import numpy as np

from src.train_motionbert_lite_skeleton_expert import (
    run_motionbert_b1,
    run_motionbert_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motionbert_lite_skeleton_expert_p6b.yaml"


def test_smoke_loads_pretraining_and_updates_head_only(tmp_path: Path) -> None:
    report = run_motionbert_smoke(CONFIG, output_root=tmp_path / "smoke")

    assert report["status"] == "smoke_passed"
    assert report["pretrained_element_coverage"] >= 0.99
    assert report["finite_forward_backward"] is True
    assert report["changed_parameter_groups"] == ["head"]
    assert report["pretrained_random_embedding_max_abs_delta"] > 0
    assert set(report["gradient_user_ids"]).isdisjoint({"user6", "user7"})
    assert report["validation_forward_rows"] == 2
    assert report["peak_cuda_mib"] < 8151
    assert report["head_reload_exact"] is True


def test_b1_cache_and_predictions_preserve_fixed_split(tmp_path: Path) -> None:
    cache = tmp_path / "embeddings.npz"
    visual = tmp_path / "visual_predictions.npz"
    train_rows, validation_rows = 8, 4
    embeddings = np.zeros((train_rows + validation_rows, 512), dtype=np.float32)
    labels = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
    embeddings[np.arange(len(labels)), labels] = 4.0
    np.savez_compressed(
        cache,
        embeddings=embeddings,
        available=np.ones(len(labels), dtype=bool),
        labels=labels,
        sample_ids=np.asarray(
            [f"train_{i}" for i in range(train_rows)]
            + [f"validation_{i}" for i in range(validation_rows)]
        ),
        user_ids=np.asarray(
            ["user1"] * 4
            + ["user2"] * 4
            + ["user6"] * 2
            + ["user7"] * 2
        ),
        partition=np.asarray(
            ["train"] * train_rows + ["validation"] * validation_rows
        ),
        quality=np.ones((len(labels), 3), dtype=np.float32),
    )
    visual_logits = np.zeros((validation_rows, 40), dtype=np.float32)
    visual_logits[:, 39] = 1.0
    np.savez_compressed(
        visual,
        sample_ids=np.asarray(
            [f"validation_{i}" for i in range(validation_rows)]
        ),
        user_ids=np.asarray(["user6", "user6", "user7", "user7"]),
        labels=np.asarray([0, 1, 2, 3]),
        logits=visual_logits,
    )

    result = run_motionbert_b1(
        CONFIG,
        output_root=tmp_path / "b1",
        cache_path=cache,
        visual_predictions_path=visual,
        expected_train_samples=train_rows,
        expected_validation_samples=validation_rows,
    )

    assert result["epochs_completed"] == 20
    assert result["validation_evaluation_count"] == 1
    assert len(result["train_sample_ids"]) == train_rows
    assert len(result["validation_sample_ids"]) == validation_rows
    assert result["validation_users_entered_training"] is False
    assert result["cache"]["embedding_shape"] == [12, 512]
    assert "unique_rescues" in result["visual_comparison"]
    assert (tmp_path / "b1/b1_decision.json").is_file()
