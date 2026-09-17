from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from imu_data import read_index
from train_p27_imu_event_forest_oof import dense_logits, drop_one_device, feature_vector


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT = PROJECT_DIR / "runs/p89_composite_imu_test_v1"


def main() -> None:
    teacher = np.load(
        PROJECT_DIR / "runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"
    )
    train_ids = teacher["oof_sample_ids"].astype(str)
    train_labels = teacher["oof_labels"].astype(np.int64)
    p3_test = np.load(PROJECT_DIR / "runs/p3_sd_imu_rf_full18/test_logits.npz")
    test_ids = p3_test["sample_ids"].astype(str)

    cached = np.load(
        PROJECT_DIR / "runs/p86_imu_event_singlefold_v1/event_features.npz"
    )
    cached_lookup = {
        value: index for index, value in enumerate(cached["sample_ids"].astype(str))
    }
    index_rows = read_index(PROJECT_DIR / "cache/imu_32/index.csv")
    index_lookup = {row.sample_id: row for row in index_rows}
    values = np.load(
        PROJECT_DIR / "cache/imu_32/imu_float32.npy", mmap_mode="r", allow_pickle=False
    )
    time_mask = np.load(
        PROJECT_DIR / "cache/imu_32/time_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    device_mask = np.load(
        PROJECT_DIR / "cache/imu_32/device_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    feature_width = int(cached["features"].shape[1])

    def extract(sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = []
        masks = []
        usable = []
        started = time.perf_counter()
        for index, sample_id in enumerate(sample_ids.astype(str), start=1):
            cached_row = cached_lookup.get(sample_id)
            row = index_lookup.get(sample_id)
            if cached_row is not None:
                feature = cached["features"][cached_row]
                mask = cached["masks"][cached_row]
                present = bool(cached["usable"][cached_row])
            elif row is not None and row.usable:
                feature, mask = feature_vector(
                    values[row.cache_index],
                    time_mask[row.cache_index],
                    device_mask[row.cache_index],
                )
                present = True
            else:
                feature = np.zeros(feature_width, dtype=np.float32)
                mask = np.zeros(10, dtype=np.float32)
                present = False
            features.append(np.asarray(feature, dtype=np.float32))
            masks.append(np.asarray(mask, dtype=np.float32))
            usable.append(present)
            if index % 250 == 0:
                print(
                    f"features={index}/{len(sample_ids)} elapsed={time.perf_counter()-started:.1f}s",
                    flush=True,
                )
        return np.stack(features), np.stack(masks), np.asarray(usable, dtype=bool)

    train_features, train_masks, train_usable = extract(train_ids)
    test_features, _test_masks, test_usable = extract(test_ids)
    fit_features = train_features[train_usable]
    fit_masks = train_masks[train_usable]
    fit_labels = train_labels[train_usable]
    dropped = drop_one_device(
        fit_features, fit_masks, np.random.default_rng(20260811)
    )
    model = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=20260811,
    )
    model.fit(
        np.concatenate((fit_features, dropped)),
        np.concatenate((fit_labels, fit_labels)),
    )
    event_logits = np.full((len(test_ids), 40), np.log(1.0 / 40.0), dtype=np.float32)
    event_logits[test_usable] = dense_logits(model, test_features[test_usable])
    p3_lookup = {
        value: index for index, value in enumerate(p3_test["sample_ids"].astype(str))
    }
    p3_rows = np.asarray([p3_lookup[value] for value in test_ids], dtype=np.int64)
    p3_logits = np.asarray(p3_test["imu_logits"], dtype=np.float64)[p3_rows]
    composite = np.logaddexp(
        event_logits + np.log(0.45), p3_logits + np.log(0.55)
    ).astype(np.float32)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT / "composite_imu_test.npz",
        sample_ids=test_ids,
        imu_logits=composite,
        event_logits=event_logits,
        p3_logits=p3_logits.astype(np.float32),
        valid=test_usable,
    )
    report = {
        "stage": "P89_full18_refit_composite_IMU_Test_v1",
        "protocol": (
            "Reuse cached event features, extract only missing train/Test rows, "
            "refit the frozen P86 ExtraTrees design on all 18 labeled users, and "
            "combine event and P3 Test posteriors with frozen 0.45/0.55 weights."
        ),
        "train_rows": int(len(train_ids)),
        "train_usable": int(np.sum(train_usable)),
        "test_rows": int(len(test_ids)),
        "test_usable": int(np.sum(test_usable)),
        "feature_width": feature_width,
        "event_weight": 0.45,
        "p3_weight": 0.55,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
