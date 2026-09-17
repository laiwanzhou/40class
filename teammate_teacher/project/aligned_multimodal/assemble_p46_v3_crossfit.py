from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join P46 train-user OOF logits and frozen target-user logits."
    )
    parser.add_argument(
        "--train-oof",
        type=Path,
        default=PROJECT_DIR / "runs/p46_v3_train_oof_v1/complete_oof.npz",
    )
    parser.add_argument(
        "--target-logits",
        type=Path,
        default=PROJECT_DIR / "runs/p46_v3_train_oof_v1/target_validation_logits.npz",
    )
    parser.add_argument(
        "--manifest", type=Path, default=PROJECT_DIR / "data/p46_single_split.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "runs/p46_v3_train_oof_v1/crossfit_detail21.npz",
    )
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        required = {"sample_ids", "source_ids", "labels", "users", "logits"}
        missing = required - set(data.files)
        if missing:
            raise RuntimeError(f"{path} misses {sorted(missing)}")
        return {key: np.asarray(data[key]) for key in required} | {
            "folds": np.asarray(data["folds"], dtype=np.int64)
            if "folds" in data.files
            else np.full(len(data["sample_ids"]), 3, dtype=np.int64)
        }


def main() -> None:
    args = parse_args()
    train = load(args.train_oof)
    target = load(args.target_logits)
    if len(train["sample_ids"]) != 1094 or len(target["sample_ids"]) != 290:
        raise RuntimeError("P46 cross-fit input counts changed")
    if set(train["users"].astype(str)) & set(target["users"].astype(str)):
        raise RuntimeError("P46 train OOF and target-user logits overlap by user")
    arrays = {
        key: np.concatenate((train[key], target[key]), axis=0)
        for key in ("sample_ids", "source_ids", "labels", "users", "logits", "folds")
    }
    if len(set(arrays["sample_ids"].astype(str))) != 1384:
        raise RuntimeError("P46 cross-fit contains duplicate sample IDs")
    with args.manifest.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["detail_selected"] == "1"]
    expected = {row["sample_id"]: row for row in rows}
    if set(arrays["sample_ids"].astype(str)) != set(expected):
        raise RuntimeError("P46 cross-fit does not exactly cover frozen Detail21")
    for sample_id, label, user in zip(
        arrays["sample_ids"].astype(str),
        arrays["labels"].astype(int),
        arrays["users"].astype(str),
    ):
        row = expected[sample_id]
        if int(row["class_id"]) != label or row["user_id"] != user:
            raise RuntimeError(f"P46 cross-fit provenance mismatch: {sample_id}")
    if arrays["logits"].shape != (1384, 21) or not np.isfinite(arrays["logits"]).all():
        raise RuntimeError("P46 cross-fit logits are invalid")
    order = np.argsort(arrays["sample_ids"].astype(str))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.building")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{key: value[order] for key, value in arrays.items()},
        )
    temporary.replace(output)
    summary = {
        "protocol": (
            "P46-v3 cross-fit: three fixed-final OOF models cover the 14 P46 training "
            "users; the original 14-user checkpoint covers the four target users"
        ),
        "samples": 1384,
        "train_oof_samples": 1094,
        "target_samples": 290,
        "target_users": sorted(set(target["users"].astype(str))),
        "folds": sorted(set(arrays["folds"].astype(int).tolist())),
        "output": str(output),
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
