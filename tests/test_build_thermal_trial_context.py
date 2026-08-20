from __future__ import annotations

from pathlib import Path

from PIL import Image

from scripts.build_thermal_trial_context import build_context_records


def save_frame(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 240), (value, value, value)).save(path)


def test_builder_preserves_missing_trial_and_builds_fixed_context(tmp_path: Path) -> None:
    present_dir = tmp_path / "Thermal" / "0_Wash_face" / "user1" / "trial-a"
    for index in range(10):
        save_frame(present_dir / f"frame_{index:03d}.jpg", index)
    records = [
        {
            "sample_id": "train__c00__user1__trial-a",
            "class_id": 0,
            "action_name": "Wash_face",
            "user_id": "user1",
            "trial_id": "trial-a",
            "development_split": "train12",
            "directory_present": True,
            "usable": True,
            "decodable_frame_count": 10,
            "duration_bucket": "9_to_32",
        },
        {
            "sample_id": "train__c01__user6__trial-b",
            "class_id": 1,
            "action_name": "Brush_teeth",
            "user_id": "user6",
            "trial_id": "trial-b",
            "development_split": "val_user6_user7",
            "directory_present": False,
            "usable": False,
            "decodable_frame_count": 0,
            "duration_bucket": "missing_or_unusable",
        },
    ]

    def predictor(paths: list[Path]) -> list[list[tuple[float, float, float, float, float]]]:
        return [[(50.0, 20.0, 180.0, 220.0, 0.8)] for _ in paths]

    output = build_context_records(records, data_root=tmp_path, predictor=predictor, batch_size=4)

    assert len(output) == 2
    assert output[0]["context_available"]
    assert output[0]["bbox_xyxy"] is not None
    assert output[0]["probe_indices"] == [0, 1, 3, 4, 5, 6, 8, 9]
    assert not output[1]["context_available"]
    assert output[1]["fallback_reason"] == "thermal_unavailable"
