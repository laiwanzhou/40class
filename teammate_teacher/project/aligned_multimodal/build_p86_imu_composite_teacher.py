from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from train_p86_mobind_pretrain import PROXY_USERS, TRAIN_USERS
from train_p86_visual_student_oof import metric_dict


PROJECT_DIR = Path(__file__).resolve().parent
RUNS = PROJECT_DIR / "runs"
OUTPUT = RUNS / "p86_imu_composite_teacher_v1"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    output = OUTPUT.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_rows(RUNS / "p86_motion_window_cache_t16_v1/rows.csv")
    selected = [
        row
        for row in rows
        if row["user_id"] in TRAIN_USERS or row["user_id"] in PROXY_USERS
    ]
    if len(selected) != 2470:
        raise RuntimeError("P86 fixed train/proxy universe changed")
    allowed_ids = {row["sample_id"] for row in selected}

    event: dict[str, np.ndarray] = {}
    event_valid: dict[str, bool] = {}
    for fold in range(3):
        with np.load(
            RUNS / f"p27_strong_inner/imu_event_forest/fold_{fold}_logits.npz",
            allow_pickle=False,
        ) as data:
            for sample_id, logits, present in zip(
                np.asarray(data["sample_ids"]).astype(str),
                np.asarray(data["logits"], dtype=np.float32),
                np.asarray(data["present"], dtype=bool),
            ):
                if sample_id in allowed_ids and sample_id not in event:
                    event[sample_id] = logits
                    event_valid[sample_id] = bool(present)
    with np.load(
        RUNS / "p86_imu_event_singlefold_v1/proxy_logits.npz",
        allow_pickle=False,
    ) as data:
        for sample_id, logits, present in zip(
            np.asarray(data["sample_ids"]).astype(str),
            np.asarray(data["logits"], dtype=np.float32),
            np.asarray(data["present"], dtype=bool),
        ):
            if sample_id in allowed_ids:
                event[sample_id] = logits
                event_valid[sample_id] = bool(present)

    p3: dict[str, np.ndarray] = {}
    with np.load(
        RUNS / "p3_imu_oof/stat_random_forest_device_dropout_aligned_oof.npz",
        allow_pickle=False,
    ) as data:
        for sample_id, logits in zip(
            np.asarray(data["sample_ids"]).astype(str),
            np.asarray(data["imu_logits"], dtype=np.float32),
        ):
            if sample_id in allowed_ids:
                p3[sample_id] = logits

    event_weight = 0.45
    p3_weight = 1.0 - event_weight
    logits_rows = []
    valid_rows = []
    source_rows = []
    for row in selected:
        sample_id = row["sample_id"]
        has_event = event_valid.get(sample_id, False)
        has_p3 = sample_id in p3
        if has_event and has_p3:
            logits = np.logaddexp(
                event[sample_id] + np.log(event_weight),
                p3[sample_id] + np.log(p3_weight),
            )
            source = "event+p3"
        elif has_event:
            logits = event[sample_id]
            source = "event"
        elif has_p3:
            logits = p3[sample_id]
            source = "p3"
        else:
            logits = np.full(40, np.log(1.0 / 40.0), dtype=np.float32)
            source = "missing"
        logits_rows.append(np.asarray(logits, dtype=np.float32))
        valid_rows.append(has_event or has_p3)
        source_rows.append(source)

    sample_ids = np.asarray([row["sample_id"] for row in selected])
    labels = np.asarray([int(row["class_id"]) for row in selected], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in selected])
    logits = np.stack(logits_rows)
    valid = np.asarray(valid_rows, dtype=bool)
    sources = np.asarray(source_rows)
    np.savez_compressed(
        output / "composite_imu_teacher.npz",
        sample_ids=sample_ids,
        imu_logits=logits,
        valid=valid,
        sources=sources,
    )
    split_metrics = {}
    for name, user_set in (("train", TRAIN_USERS), ("proxy", PROXY_USERS)):
        mask = np.isin(users, sorted(user_set))
        split_metrics[name] = metric_dict(
            labels[mask], logits[mask].argmax(axis=1), users[mask].tolist()
        )
    summary = {
        "protocol": (
            "Historical subject-disjoint event OOF for fixed training subjects; "
            "fixed-train event model for proxy subjects; no permanent samples."
        ),
        "event_weight": event_weight,
        "p3_weight": p3_weight,
        "counts": {
            "total": len(selected),
            "valid": int(valid.sum()),
            "missing": int((~valid).sum()),
            "by_source": {
                source: int(np.sum(sources == source))
                for source in sorted(set(source_rows))
            },
            "permanent_written": 0,
        },
        "metrics": split_metrics,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
