from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_thermal_v2_pose_cache.py"


def load_module():
    spec = importlib.util.spec_from_file_location("thermal_pose_cache", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakePredictor:
    def __call__(self, paths: list[Path]):
        output = []
        for path in paths:
            index = int(path.stem)
            if index == 0:
                output.append((None, None))
                continue
            keypoints = torch.ones(17, 3)
            keypoints[:, 0] *= 10
            keypoints[:, 1] *= 20
            output.append((keypoints, torch.tensor([1.0, 2.0, 30.0, 40.0, 0.8])))
        return output


def test_pose_cache_is_complete_keyed_and_preserves_failed_detections(tmp_path: Path) -> None:
    module = load_module()
    trial = tmp_path / "Thermal/0_Action/user1/1-1-1"
    trial.mkdir(parents=True)
    for index in range(4):
        assert cv2.imwrite(str(trial / f"{index}.jpg"), np.zeros((48, 64, 3), np.uint8))
    records = [
        {
            "sample_id": "train__c00__user1__1-1-1",
            "development_split": "train12",
            "thermal_relative_path": "Thermal/0_Action/user1/1-1-1",
            "usable": True,
        },
        {
            "sample_id": "train__c01__user6__missing",
            "development_split": "val_user6_user7",
            "thermal_relative_path": "Thermal/1_Action/user6/missing",
            "usable": False,
        },
    ]

    arrays, summary = module.build_pose_cache_arrays(
        records, data_root=tmp_path, predictor=FakePredictor(), batch_size=3
    )

    assert summary["canonical_trials"] == 2
    assert summary["usable_trials"] == 1
    assert summary["cached_unique_frames"] == 4
    assert arrays["pose"].shape == (4, 56)
    assert arrays["valid"].tolist() == [False, True, True, True]
    assert len(set(arrays["keys"].tolist())) == 4
    assert all(key.startswith("train__c00__user1__1-1-1|") for key in arrays["keys"])


def test_pose_cache_rejects_records_outside_fixed_development_population(tmp_path: Path) -> None:
    module = load_module()
    records = [{"sample_id": "sealed", "development_split": "heldout", "usable": False}]

    try:
        module.build_pose_cache_arrays(
            records, data_root=tmp_path, predictor=FakePredictor(), batch_size=1
        )
    except ValueError as error:
        assert "outside fixed development population" in str(error)
    else:
        raise AssertionError("heldout record was accepted")
