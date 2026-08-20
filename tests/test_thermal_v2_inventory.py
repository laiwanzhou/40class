from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.thermal_v2_inventory import (
    load_canonical_thermal_records,
    resolve_thermal_trial_path,
)


def write_t0_report(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps({"thermal_data_audit": {"canonical_trial_records": records}}),
        encoding="utf-8",
    )


def test_canonical_inventory_preserves_missing_thermal_trial(tmp_path: Path) -> None:
    report = tmp_path / "t0.json"
    records = [
        {
            "sample_id": "train__c00__user1__trial-a",
            "class_id": 0,
            "action_name": "Wash_face",
            "user_id": "user1",
            "trial_id": "trial-a",
            "development_split": "train12",
            "directory_present": False,
            "usable": False,
            "decodable_frame_count": 0,
        },
        {
            "sample_id": "train__c01__user6__trial-b",
            "class_id": 1,
            "action_name": "Brush_teeth",
            "user_id": "user6",
            "trial_id": "trial-b",
            "development_split": "val_user6_user7",
            "directory_present": True,
            "usable": True,
            "decodable_frame_count": 10,
        },
    ]
    write_t0_report(report, records)

    loaded = load_canonical_thermal_records(report)

    assert loaded == records
    assert not loaded[0]["directory_present"]


def test_inventory_rejects_non_development_population(tmp_path: Path) -> None:
    report = tmp_path / "t0.json"
    write_t0_report(
        report,
        [
            {
                "sample_id": "sealed",
                "development_split": "heldout4",
                "user_id": "user4",
            }
        ],
    )

    with pytest.raises(ValueError, match="outside the fixed development population"):
        load_canonical_thermal_records(report)


def test_trial_path_uses_thermal_directory_only(tmp_path: Path) -> None:
    record = {
        "class_id": 0,
        "action_name": "Wash_face",
        "user_id": "user1",
        "trial_id": "trial-a",
    }

    path = resolve_thermal_trial_path(tmp_path, record)

    assert path == tmp_path / "Thermal" / "0_Wash_face" / "user1" / "trial-a"
