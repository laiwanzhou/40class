from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

from imu_data import read_index
from train_p27_imu_event_forest_oof import (
    FEATURES_PER_DEVICE,
    dense_logits,
    drop_one_device,
    feature_vector,
)
from train_p86_mobind_pretrain import PERMANENT_USERS, PROXY_USERS, TRAIN_USERS
from train_p86_visual_student_oof import metric_dict


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = PROJECT_DIR / "cache/imu_32"
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_imu_event_singlefold_v1"
DEFAULT_MOTION = PROJECT_DIR / "runs/p86_motion_window_cache_t16_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit richer IMU event features on the fixed P86 single fold."
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--motion-cache", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = args.cache_dir.resolve()
    imu_rows = [row for row in read_index(cache / "index.csv") if row.split == "train"]
    imu_by_id = {row.sample_id: row for row in imu_rows}
    with (args.motion_cache.resolve() / "rows.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        p86_rows = list(csv.DictReader(handle))
    train_rows = [row for row in p86_rows if row["user_id"] in TRAIN_USERS]
    proxy_rows = [row for row in p86_rows if row["user_id"] in PROXY_USERS]
    permanent_rows = [row for row in p86_rows if row["user_id"] in PERMANENT_USERS]
    if (len(train_rows), len(proxy_rows), len(permanent_rows)) != (1497, 973, 444):
        raise RuntimeError("P86 fixed subject split changed")

    values = np.load(cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(
        cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    device_mask = np.load(
        cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    selected_rows = train_rows + proxy_rows
    features: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    started = time.perf_counter()
    feature_width = FEATURES_PER_DEVICE * 5 + 10 + 20
    for row_index, row in enumerate(selected_rows, start=1):
        imu_row = imu_by_id.get(row["sample_id"])
        if imu_row is not None and imu_row.usable:
            feature, mask = feature_vector(
                values[imu_row.cache_index],
                time_mask[imu_row.cache_index],
                device_mask[imu_row.cache_index],
            )
        else:
            feature = np.zeros(feature_width, dtype=np.float32)
            mask = np.zeros(10, dtype=np.float32)
        features.append(feature)
        masks.append(mask)
        if row_index % 250 == 0:
            print(f"features={row_index}/{len(selected_rows)}", flush=True)
    feature_array = np.stack(features)
    mask_array = np.stack(masks)
    labels = np.asarray([int(row["class_id"]) for row in selected_rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in selected_rows])
    usable = np.asarray(
        [
            row["sample_id"] in imu_by_id and imu_by_id[row["sample_id"]].usable
            for row in selected_rows
        ],
        dtype=bool,
    )
    sample_ids = np.asarray([row["sample_id"] for row in selected_rows])
    np.savez_compressed(
        output / "event_features.npz",
        sample_ids=sample_ids,
        labels=labels,
        users=users,
        usable=usable,
        features=feature_array,
        masks=mask_array,
    )

    train_count = len(train_rows)
    fit_mask = usable[:train_count]
    proxy_mask = usable[train_count:]
    fit_features = feature_array[:train_count][fit_mask]
    fit_masks = mask_array[:train_count][fit_mask]
    fit_labels = labels[:train_count][fit_mask]
    dropped = drop_one_device(
        fit_features, fit_masks, np.random.default_rng(args.seed)
    )
    source = np.concatenate((fit_features, dropped))
    source_labels = np.concatenate((fit_labels, fit_labels))
    model = ExtraTreesClassifier(
        n_estimators=600,
        max_depth=20,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    fit_started = time.perf_counter()
    model.fit(source, source_labels)
    fit_seconds = time.perf_counter() - fit_started
    joblib.dump(model, output / "event_teacher.joblib", compress=3)

    proxy_logits = np.full((len(proxy_rows), 40), np.log(1.0 / 40.0), np.float32)
    proxy_logits[proxy_mask] = dense_logits(
        model, feature_array[train_count:][proxy_mask]
    )
    proxy_predictions = proxy_logits.argmax(axis=1)
    proxy_labels = labels[train_count:]
    proxy_users = users[train_count:]
    metrics_all = metric_dict(proxy_labels, proxy_predictions, proxy_users.tolist())
    metrics_present = metric_dict(
        proxy_labels[proxy_mask],
        proxy_predictions[proxy_mask],
        proxy_users[proxy_mask].tolist(),
    )
    np.savez_compressed(
        output / "proxy_logits.npz",
        sample_ids=sample_ids[train_count:],
        labels=proxy_labels,
        users=proxy_users,
        logits=proxy_logits,
        present=proxy_mask,
    )
    summary = {
        "protocol": "P86 fixed single-fold richer IMU event audit",
        "counts": {
            "train": len(train_rows),
            "train_usable": int(fit_mask.sum()),
            "proxy": len(proxy_rows),
            "proxy_usable": int(proxy_mask.sum()),
            "permanent_untouched": len(permanent_rows),
        },
        "feature_width": int(feature_array.shape[1]),
        "feature_seconds": time.perf_counter() - started - fit_seconds,
        "fit_seconds": fit_seconds,
        "proxy_metrics_all": metrics_all,
        "proxy_metrics_present": metrics_present,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
