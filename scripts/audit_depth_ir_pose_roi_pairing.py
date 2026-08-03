from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/depth_pose_roi_expert.yaml"
CSV_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_pairing.csv"
REPORT_PATH = PROJECT_ROOT / "reports/depth_ir_pose_roi_pairing.md"
FRAME_PATTERN = re.compile(
    r"^(?P<modality>Depth|IR)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<frame_id>\d+)(?:_Color)?$"
)


def parse_directory(path: Path, expected_modality: str) -> tuple[dict[tuple[str, int], Path], set[str], set[int], list[str], list[str]]:
    pairs: dict[tuple[str, int], Path] = {}
    timestamps: set[str] = set()
    frame_ids: set[int] = set()
    duplicates: list[str] = []
    unparsed: list[str] = []
    for image_path in sorted(path.glob("*.png")):
        match = FRAME_PATTERN.fullmatch(image_path.stem)
        if match is None or match.group("modality") != expected_modality:
            unparsed.append(image_path.name)
            continue
        timestamp = match.group("timestamp")
        frame_id = int(match.group("frame_id"))
        key = (timestamp, frame_id)
        if key in pairs:
            duplicates.append(image_path.name)
        else:
            pairs[key] = image_path
        timestamps.add(timestamp)
        frame_ids.add(frame_id)
    return pairs, timestamps, frame_ids, duplicates, unparsed


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = pd.read_csv(PROJECT_ROOT / config["manifest"], encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads((PROJECT_ROOT / config["fold"]).read_text(encoding="utf-8"))
    split_map = {user: "train" for user in fold["train_users"]} | {user: "validation" for user in fold["val_users"]}
    selected = manifest[
        manifest["action_name"].isin(config["hard_actions"])
        & manifest["user_id"].isin(split_map)
        & manifest["depth_color_path"].fillna("").ne("")
        & manifest["ir_path"].fillna("").ne("")
    ].copy()
    data_root = Path(config["data_root"])
    rows: list[dict[str, object]] = []
    for row in selected.itertuples():
        depth, depth_times, depth_ids, depth_duplicates, depth_unparsed = parse_directory(
            data_root / row.depth_color_path, "Depth"
        )
        ir, ir_times, ir_ids, ir_duplicates, ir_unparsed = parse_directory(
            data_root / row.ir_path, "IR"
        )
        reasons: list[str] = []
        if depth_duplicates:
            reasons.append(f"depth_duplicate_keys={len(depth_duplicates)}")
        if ir_duplicates:
            reasons.append(f"ir_duplicate_keys={len(ir_duplicates)}")
        if depth_unparsed:
            reasons.append(f"depth_unparsed={len(depth_unparsed)}")
        if ir_unparsed:
            reasons.append(f"ir_unparsed={len(ir_unparsed)}")
        if depth_times != ir_times:
            reasons.append(f"timestamp_set_mismatch=depth_only:{len(depth_times-ir_times)},ir_only:{len(ir_times-depth_times)}")
        if depth_ids != ir_ids:
            reasons.append(f"frame_id_set_mismatch=depth_only:{len(depth_ids-ir_ids)},ir_only:{len(ir_ids-depth_ids)}")
        if depth.keys() != ir.keys():
            reasons.append(f"pair_key_mismatch=depth_only:{len(depth.keys()-ir.keys())},ir_only:{len(ir.keys()-depth.keys())}")
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
                "depth_duplicate_keys": len(depth_duplicates),
                "ir_duplicate_keys": len(ir_duplicates),
                "depth_unparsed": len(depth_unparsed),
                "ir_unparsed": len(ir_unparsed),
                "complete_pairing": not reasons,
                "reason": "; ".join(reasons),
            }
        )
    audit = pd.DataFrame(rows).sort_values(["split", "class_id", "sample_id"])
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    complete = audit[audit["complete_pairing"]]
    exceptions = audit[~audit["complete_pairing"]]
    lines = [
        "# Hard-subset Depth_Color / IR frame pairing audit",
        "",
        "Pairing key is the parsed `(absolute timestamp, frame ID)` from each filename. Sorted array indices are not used.",
        "",
        f"- Hard-subset samples audited: {len(audit)}.",
        f"- Completely frame-aligned samples: {len(complete)}.",
        f"- Samples with missing, duplicate, unparsed, or misaligned frames: {len(exceptions)}.",
        f"- Final usable train samples: {int((complete['split'] == 'train').sum())}.",
        f"- Final usable validation samples: {int((complete['split'] == 'validation').sum())}.",
        f"- Exact paired frames: {int(audit['paired_frames'].sum())}.",
        "",
        "## Exceptions",
        "",
    ]
    if exceptions.empty:
        lines.append("None. All hard-subset samples have identical timestamp sets, frame-ID sets, and combined pairing-key sets.")
    else:
        lines.extend(["| sample_id | split | reason |", "| --- | --- | --- |"])
        lines.extend(f"| {row.sample_id} | {row.split} | {row.reason} |" for row in exceptions.itertuples())
    train_count = int((complete["split"] == "train").sum())
    val_count = int((complete["split"] == "validation").sum())
    lines.append("")
    if (train_count, val_count) == (596, 148):
        lines.append("The usable set remains 596 train / 148 validation samples, so the new model is directly comparable with E1.")
    else:
        lines.append(
            f"The usable set is {train_count} train / {val_count} validation, not E1's 596/148. "
            "A Depth-only pose-ROI baseline must therefore be retrained on this exact common subset."
        )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(CSV_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
