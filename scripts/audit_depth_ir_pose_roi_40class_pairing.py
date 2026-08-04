from __future__ import annotations

import json
from pathlib import Path
import re

from PIL import Image
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
MANIFEST_PATH = PROJECT_ROOT / "metadata/manifest.csv"
FOLD_PATH = PROJECT_ROOT / "metadata/splits/fold_0.json"
CSV_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pairing.csv"
REPORT_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_pairing.md"
CLASS_MAP_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv"
FRAME_PATTERN = re.compile(
    r"^(?P<modality>Depth|IR)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<frame_id>\d+)(?:_Color)?$"
)


def parse_directory(path: Path, modality: str) -> tuple[dict[tuple[str, int], Path], list[str], list[str], list[str]]:
    pairs: dict[tuple[str, int], Path] = {}
    duplicate_pairs: list[str] = []
    duplicate_timestamps: list[str] = []
    unparsed: list[str] = []
    seen_timestamps: set[str] = set()
    for image_path in sorted(path.glob("*.png")):
        match = FRAME_PATTERN.fullmatch(image_path.stem)
        if match is None or match.group("modality") != modality:
            unparsed.append(image_path.name)
            continue
        timestamp = match.group("timestamp")
        key = timestamp, int(match.group("frame_id"))
        if key in pairs:
            duplicate_pairs.append(image_path.name)
        else:
            pairs[key] = image_path
        if timestamp in seen_timestamps:
            duplicate_timestamps.append(image_path.name)
        seen_timestamps.add(timestamp)
    return pairs, duplicate_pairs, duplicate_timestamps, unparsed


def unreadable_files(paths: list[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:  # noqa: BLE001 - the audit must capture every decoder failure
            failures.append(f"{path.name}:{type(error).__name__}")
    return failures


def main() -> None:
    manifest = pd.read_csv(MANIFEST_PATH, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(FOLD_PATH.read_text(encoding="utf-8"))
    split_map = {user: "train" for user in fold["train_users"]} | {user: "validation" for user in fold["val_users"]}
    in_fold = manifest[manifest["user_id"].isin(split_map)].copy()
    both = in_fold[
        in_fold["depth_color_path"].fillna("").astype(str).str.strip().ne("")
        & in_fold["ir_path"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    rows: list[dict[str, object]] = []
    for row in both.itertuples():
        depth, depth_pair_duplicates, depth_time_duplicates, depth_unparsed = parse_directory(
            DATA_ROOT / row.depth_color_path, "Depth"
        )
        ir, ir_pair_duplicates, ir_time_duplicates, ir_unparsed = parse_directory(DATA_ROOT / row.ir_path, "IR")
        reasons: list[str] = []
        for name, values in (
            ("depth_duplicate_pair_keys", depth_pair_duplicates),
            ("ir_duplicate_pair_keys", ir_pair_duplicates),
            ("depth_duplicate_timestamps", depth_time_duplicates),
            ("ir_duplicate_timestamps", ir_time_duplicates),
            ("depth_unparsed", depth_unparsed),
            ("ir_unparsed", ir_unparsed),
        ):
            if values:
                reasons.append(f"{name}={len(values)}")
        depth_times = {key[0] for key in depth}
        ir_times = {key[0] for key in ir}
        depth_ids = {key[1] for key in depth}
        ir_ids = {key[1] for key in ir}
        if depth_times != ir_times:
            reasons.append(f"timestamp_set_mismatch=depth_only:{len(depth_times-ir_times)},ir_only:{len(ir_times-depth_times)}")
        if depth_ids != ir_ids:
            reasons.append(f"frame_id_set_mismatch=depth_only:{len(depth_ids-ir_ids)},ir_only:{len(ir_ids-depth_ids)}")
        if depth.keys() != ir.keys():
            reasons.append(f"pair_key_mismatch=depth_only:{len(depth.keys()-ir.keys())},ir_only:{len(ir.keys()-depth.keys())}")
        unreadable = unreadable_files(list(depth.values()) + list(ir.values()))
        if unreadable:
            reasons.append(f"unreadable_images={len(unreadable)}")
        rows.append(
            {
                "sample_id": row.sample_id,
                "split": split_map[row.user_id],
                "class_id": int(row.class_id),
                "action_name": row.action_name,
                "user_id": row.user_id,
                "trial_id": row.trial_id,
                "depth_frames": len(depth),
                "ir_frames": len(ir),
                "paired_frames": len(depth.keys() & ir.keys()),
                "depth_unparsed": len(depth_unparsed),
                "ir_unparsed": len(ir_unparsed),
                "duplicate_pair_keys": len(depth_pair_duplicates) + len(ir_pair_duplicates),
                "duplicate_timestamps": len(depth_time_duplicates) + len(ir_time_duplicates),
                "unreadable_images": len(unreadable),
                "complete_pairing": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["split", "class_id", "sample_id"])
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    complete = audit[audit["complete_pairing"]]
    exceptions = audit[~audit["complete_pairing"]]
    support = (
        complete.groupby(["class_id", "action_name", "split"]).size().unstack(fill_value=0).reset_index()
        .rename(columns={"train": "train_support", "validation": "val_support"})
        .sort_values("class_id")
    )
    for column in ("train_support", "val_support"):
        if column not in support:
            support[column] = 0
    support[["class_id", "action_name", "train_support", "val_support"]].to_csv(
        CLASS_MAP_PATH, index=False, encoding="utf-8-sig"
    )
    train_count = int((complete["split"] == "train").sum())
    val_count = int((complete["split"] == "validation").sum())
    lines = [
        "# Full 40-class Depth_Color / IR pairing audit",
        "",
        "Pairing uses parsed `(absolute timestamp, frame ID)` keys. Sorted array positions are never used.",
        "",
        f"- Manifest samples: {len(manifest)}.",
        f"- Fold samples before modality filtering: train {sum(in_fold['user_id'].isin(fold['train_users']))}, validation {sum(in_fold['user_id'].isin(fold['val_users']))}.",
        f"- Samples with both Depth_Color and IR paths: {len(both)}.",
        f"- Completely paired and readable samples: {len(complete)}.",
        f"- Exceptional samples: {len(exceptions)}.",
        f"- Final usable samples: train {train_count}, validation {val_count}.",
        f"- Strictly paired frames: {int(complete['paired_frames'].sum())}.",
        "",
        "## Class support",
        "",
        "| class_id | action_name | train | validation |",
        "| ---: | --- | ---: | ---: |",
    ]
    lines.extend(
        f"| {row.class_id} | {row.action_name} | {row.train_support} | {row.val_support} |"
        for row in support.itertuples()
    )
    lines.extend(["", "## Exceptions", ""])
    if exceptions.empty:
        lines.append("None.")
    else:
        lines.extend(["| sample_id | split | class_id | action_name | reason |", "| --- | --- | ---: | --- | --- |"])
        lines.extend(
            f"| {row.sample_id} | {row.split} | {row.class_id} | {row.action_name} | {row.reason} |"
            for row in exceptions.itertuples()
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(support) != 40 or (support[["train_support", "val_support"]] == 0).any().any():
        raise RuntimeError("At least one of the 40 classes is absent from the final train or validation set.")
    print(json.dumps({"manifest": len(manifest), "both": len(both), "complete": len(complete), "exceptions": len(exceptions), "train": train_count, "validation": val_count, "frames": int(complete['paired_frames'].sum())}))


if __name__ == "__main__":
    main()
