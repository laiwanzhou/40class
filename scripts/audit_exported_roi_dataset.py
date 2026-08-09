from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


VIEW_NAMES = [
    "ir_context",
    "ir_left",
    "ir_right",
    "ir_relation",
    "depth_context",
    "depth_relation",
]
LOCAL_VIEWS = {"ir_left", "ir_right", "ir_relation", "depth_relation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def image_metrics(item: tuple[int, str, int]) -> dict[str, object]:
    row_id, path, valid = item
    result: dict[str, object] = {"row_id": row_id, "exists": False, "readable": False}
    try:
        stat = os.stat(path)
        result.update({"exists": True, "file_bytes": stat.st_size})
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            return result
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        histogram = np.bincount(thumb.reshape(-1), minlength=256).astype(np.float64)
        probability = histogram[histogram > 0] / histogram.sum()
        entropy = float(-(probability * np.log2(probability)).sum())
        result.update(
            {
                "readable": True,
                "height": int(gray.shape[0]),
                "width": int(gray.shape[1]),
                "channels": 1 if image.ndim == 2 else int(image.shape[2]),
                "mean": float(gray.mean()),
                "std": float(gray.std()),
                "minimum": int(gray.min()),
                "maximum": int(gray.max()),
                "dynamic_range": int(gray.max()) - int(gray.min()),
                "nonzero_fraction": float(np.count_nonzero(gray) / gray.size),
                "entropy_32": entropy,
                "exact_black": bool(np.count_nonzero(gray) == 0),
                "valid_flag": int(valid),
            }
        )
    except (OSError, ValueError, cv2.error):
        pass
    return result


def longest_false_run(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            best = max(best, current)
    return best


def box_iou(left: pd.DataFrame, right: pd.DataFrame) -> np.ndarray:
    ix1 = np.maximum(left.x1.to_numpy(), right.x1.to_numpy())
    iy1 = np.maximum(left.y1.to_numpy(), right.y1.to_numpy())
    ix2 = np.minimum(left.x2.to_numpy(), right.x2.to_numpy())
    iy2 = np.minimum(left.y2.to_numpy(), right.y2.to_numpy())
    intersection = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    left_area = (left.x2 - left.x1).to_numpy() * (left.y2 - left.y1).to_numpy()
    right_area = (right.x2 - right.x1).to_numpy() * (right.y2 - right.y1).to_numpy()
    union = left_area + right_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def main() -> None:
    args = parse_args()
    root = args.data_root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((root / "export_metadata.json").read_text(encoding="utf-8"))
    frames = pd.read_csv(root / "all_frame_inputs.csv")
    audit = pd.read_csv(root / "roi_frame_audit.csv")
    audit.insert(0, "row_id", np.arange(len(audit), dtype=np.int64))

    expected_keys = ["split", "class_id", "sample_id", "source_frame_index"]
    frame_duplicates = int(frames.duplicated(expected_keys).sum())
    audit_duplicates = int(audit.duplicated(expected_keys + ["view_name"]).sum())
    view_counts = audit.groupby(expected_keys).view_name.nunique()
    six_view_frames = int((view_counts == 6).sum())

    pixel_cache = report_dir / "roi640_pixel_metrics.csv"
    if pixel_cache.exists():
        metrics = pd.read_csv(pixel_cache)
        if len(metrics) != len(audit):
            raise RuntimeError("Pixel cache row count does not match the audit manifest")
    else:
        metric_items = [
            (int(row.row_id), str(root / row.output_path), int(row.valid))
            for row in audit.itertuples(index=False)
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            metrics = pd.DataFrame(executor.map(image_metrics, metric_items, chunksize=128))
        metrics.to_csv(pixel_cache, index=False)
    audit = audit.merge(metrics, on="row_id", how="left", validate="one_to_one")

    audit["box_width"] = audit.x2 - audit.x1
    audit["box_height"] = audit.y2 - audit.y1
    audit["box_area_fraction"] = audit.box_width * audit.box_height / (640.0 * 480.0)
    audit["touch_boundary"] = (
        (audit.x1 <= 1.0) | (audit.y1 <= 1.0) | (audit.x2 >= 639.0) | (audit.y2 >= 479.0)
    )
    audit["valid_near_constant"] = (
        (audit.valid == 1)
        & audit.readable.fillna(False)
        & ((audit.dynamic_range <= 8) | (audit["std"] <= 2.0) | (audit.entropy_32 <= 1.0))
    )
    audit["invalid_not_black"] = (
        (audit.valid == 0) & audit.readable.fillna(False) & ~audit.exact_black.fillna(False)
    )
    audit["valid_exact_black"] = (
        (audit.valid == 1) & audit.readable.fillna(False) & audit.exact_black.fillna(False)
    )

    summary = (
        audit.groupby(["split", "class_id", "action_name", "view_name"], as_index=False)
        .agg(
            frames=("row_id", "size"),
            valid_rate=("valid", "mean"),
            missing_files=("exists", lambda x: int((~x.fillna(False)).sum())),
            unreadable=("readable", lambda x: int((~x.fillna(False)).sum())),
            exact_black_rate=("exact_black", "mean"),
            valid_near_constant_rate=("valid_near_constant", "mean"),
            invalid_not_black=("invalid_not_black", "sum"),
            valid_exact_black=("valid_exact_black", "sum"),
            mean_entropy=("entropy_32", "mean"),
            mean_std=("std", "mean"),
            mean_area_fraction=("box_area_fraction", "mean"),
            boundary_touch_rate=("touch_boundary", "mean"),
        )
    )
    summary.to_csv(report_dir / "roi640_quality_by_action_view.csv", index=False, encoding="utf-8-sig")

    # Consecutive invalid runs and box motion are computed within each clip and view.
    temporal_rows: list[dict[str, object]] = []
    sorted_audit = audit.sort_values(["sample_id", "view_name", "source_frame_index"])
    for (sample_id, view_name), group in sorted_audit.groupby(["sample_id", "view_name"], sort=False):
        valid = group.valid.to_numpy(dtype=bool)
        centers = np.column_stack(((group.x1 + group.x2) / 2, (group.y1 + group.y2) / 2))
        deltas = np.linalg.norm(np.diff(centers, axis=0), axis=1) / 800.0 if len(group) > 1 else np.array([])
        consecutive_valid = valid[:-1] & valid[1:]
        valid_deltas = deltas[consecutive_valid]
        temporal_rows.append(
            {
                "split": group.split.iloc[0],
                "class_id": int(group.class_id.iloc[0]),
                "action_name": group.action_name.iloc[0],
                "sample_id": sample_id,
                "view_name": view_name,
                "frames": len(group),
                "valid_rate": float(valid.mean()),
                "longest_invalid_run": longest_false_run(valid),
                "max_center_jump_fraction": float(valid_deltas.max(initial=0.0)),
                "p95_center_jump_fraction": float(np.quantile(valid_deltas, 0.95)) if len(valid_deltas) else 0.0,
            }
        )
    temporal = pd.DataFrame(temporal_rows)
    temporal.to_csv(report_dir / "roi640_temporal_quality_by_sample_view.csv", index=False, encoding="utf-8-sig")

    # Identify suspicious left/right crops by coordinate overlap. Exact duplicates should have been suppressed.
    left = audit[audit.view_name == "ir_left"].set_index(expected_keys).sort_index()
    right = audit[audit.view_name == "ir_right"].set_index(expected_keys).sort_index()
    common = left.index.intersection(right.index)
    left_common = left.loc[common]
    right_common = right.loc[common]
    pair = left_common.reset_index()[expected_keys + ["action_name", "user_id"]].copy()
    pair["sample_id"] = left_common.reset_index().sample_id
    pair["left_valid"] = left_common.valid.to_numpy()
    pair["right_valid"] = right_common.valid.to_numpy()
    pair["box_iou"] = box_iou(left_common, right_common)
    pair["both_valid"] = (pair.left_valid == 1) & (pair.right_valid == 1)
    pair.to_csv(report_dir / "roi640_left_right_overlap.csv", index=False, encoding="utf-8-sig")

    candidates: list[dict[str, object]] = []

    def add_candidate(row: pd.Series, priority: str, reason: str) -> None:
        candidates.append(
            {
                "priority": priority,
                "split": row.get("split"),
                "class_id": int(row.get("class_id")),
                "action_name": row.get("action_name"),
                "sample_id": row.get("sample_id"),
                "user_id": row.get("user_id", ""),
                "source_frame_index": int(row.get("source_frame_index", -1)),
                "view_name": row.get("view_name"),
                "reason": reason,
                "output_path": row.get("output_path", ""),
                "absolute_path": str(root / row.get("output_path", "")),
            }
        )

    for _, row in audit[audit.valid_exact_black].iterrows():
        add_candidate(row, "P0", "valid=1但图片完全为黑图")
    for _, row in audit[~audit.exists.fillna(False) | ~audit.readable.fillna(False)].head(500).iterrows():
        add_candidate(row, "P0", "文件缺失或PNG无法解码")

    near_constant = audit[audit.valid_near_constant].sort_values(["entropy_32", "std"])
    for _, group in near_constant.groupby(["action_name", "view_name"], sort=False):
        for _, row in group.head(3).iterrows():
            add_candidate(row, "P1", f"有效ROI低信息: std={row['std']:.2f}, entropy={row['entropy_32']:.2f}")

    local_boundary = audit[(audit.view_name.isin(LOCAL_VIEWS)) & (audit.valid == 1) & audit.touch_boundary]
    for _, group in local_boundary.groupby(["action_name", "view_name"], sort=False):
        for _, row in group.sort_values("box_area_fraction").head(2).iterrows():
            add_candidate(row, "P1", "局部交互ROI触碰原图边界，可能裁失框外物品")

    long_runs = temporal[(temporal.view_name.isin(LOCAL_VIEWS)) & (temporal.longest_invalid_run >= 8)]
    for _, row in long_runs.sort_values("longest_invalid_run", ascending=False).head(160).iterrows():
        source = audit[
            (audit.sample_id == row.sample_id)
            & (audit.view_name == row.view_name)
            & (audit.valid == 0)
        ].iloc[0]
        add_candidate(
            source,
            "P1",
            f"局部ROI连续失效，最长{int(row.longest_invalid_run)}帧/{int(row.frames)}帧",
        )

    jumps = []
    for (_, _), group in sorted_audit.groupby(["sample_id", "view_name"], sort=False):
        if len(group) < 2:
            continue
        centers = np.column_stack(((group.x1 + group.x2) / 2, (group.y1 + group.y2) / 2))
        delta = np.linalg.norm(np.diff(centers, axis=0), axis=1) / 800.0
        valid = group.valid.to_numpy(dtype=bool)
        jump_mask = (delta >= 0.20) & valid[:-1] & valid[1:]
        for index in np.flatnonzero(jump_mask):
            row = group.iloc[index + 1]
            jumps.append((float(delta[index]), row))
    for jump, row in sorted(jumps, key=lambda item: item[0], reverse=True)[:160]:
        add_candidate(row, "P1", f"相邻帧ROI中心跳变{jump:.1%}原图对角线")

    overlapping = pair[pair.both_valid & (pair.box_iou >= 0.80)].sort_values("box_iou", ascending=False)
    for _, row in overlapping.head(160).iterrows():
        source = audit[
            (audit["split"] == row["split"])
            & (audit.class_id == row.class_id)
            & (audit.sample_id == row.sample_id)
            & (audit.source_frame_index == row.source_frame_index)
            & (audit.view_name == "ir_left")
        ].iloc[0]
        add_candidate(source, "P1", f"左右交互ROI高度重叠，box IoU={row.box_iou:.3f}")

    small_object_actions = {
        "Brush_teeth", "Drink_water", "Eat_food", "Take_and_use_tableware", "Pour_drinks",
        "Stir_drinks", "Peel_fruits", "Wipe_bowls", "Tap_the_keyboard", "Write",
        "Make_a_phone_call", "Check_the_time", "Read_documents", "Turn_pages",
        "Listen_to_music_with_headphones", "Use_a_mobile_phone", "Watch_TV", "Play_games",
        "Take_a_selfie", "Take_medicine", "Take_body_temperature",
    }
    # Even statistically healthy small-object crops require semantic inspection: pixels cannot prove object inclusion.
    small = audit[(audit.action_name.isin(small_object_actions)) & (audit.valid == 1)]
    selected_samples = (
        small[["split", "action_name", "sample_id"]]
        .drop_duplicates()
        .sort_values(["split", "action_name", "sample_id"])
        .groupby(["split", "action_name"], as_index=False)
        .head(1)
    )
    selected_keys = set(zip(selected_samples["split"], selected_samples.action_name, selected_samples.sample_id))
    small = small[
        [key in selected_keys for key in zip(small["split"], small.action_name, small.sample_id)]
    ]
    for (_, sample_id, view_name), group in small.groupby(["action_name", "sample_id", "view_name"], sort=False):
        if view_name not in {"ir_context", "ir_left", "ir_right", "ir_relation", "depth_relation"}:
            continue
        for position in sorted(set([0, len(group) // 2, len(group) - 1])):
            row = group.sort_values("source_frame_index").iloc[position]
            add_candidate(row, "P2", "小物品动作语义抽检：确认手持物与交互对象实际包含且可辨")

    review = pd.DataFrame(candidates).drop_duplicates(
        ["priority", "sample_id", "source_frame_index", "view_name", "reason"]
    )
    priority_order = pd.Categorical(review.priority, categories=["P0", "P1", "P2"], ordered=True)
    review = review.assign(_priority=priority_order).sort_values(
        ["_priority", "action_name", "sample_id", "source_frame_index", "view_name"]
    ).drop(columns="_priority")
    review.to_csv(report_dir / "roi640_manual_review_candidates.csv", index=False, encoding="utf-8-sig")

    result = {
        "metadata": metadata,
        "integrity": {
            "frame_rows": len(frames),
            "audit_rows": len(audit),
            "frame_duplicates": frame_duplicates,
            "audit_duplicates": audit_duplicates,
            "frames_with_exactly_six_views": six_view_frames,
            "missing_files": int((~audit.exists.fillna(False)).sum()),
            "unreadable_files": int((~audit.readable.fillna(False)).sum()),
            "wrong_dimensions": int(((audit.width != 256) | (audit.height != 256)).fillna(True).sum()),
            "valid_exact_black": int(audit.valid_exact_black.sum()),
            "invalid_not_black": int(audit.invalid_not_black.sum()),
        },
        "global": {
            "valid_rate_by_view": audit.groupby("view_name").valid.mean().to_dict(),
            "boundary_touch_rate_by_view": audit.groupby("view_name").touch_boundary.mean().to_dict(),
            "valid_near_constant_by_view": audit.groupby("view_name").valid_near_constant.sum().astype(int).to_dict(),
            "valid_near_constant_rate_by_view": audit.groupby("view_name").valid_near_constant.mean().to_dict(),
            "long_invalid_sample_views_ge8": int((temporal.longest_invalid_run >= 8).sum()),
            "local_long_invalid_sample_views_ge8": int(
                (temporal.view_name.isin(LOCAL_VIEWS) & (temporal.longest_invalid_run >= 8)).sum()
            ),
            "center_jumps_ge20pct": len(jumps),
            "left_right_box_iou_ge80pct": int((pair.both_valid & (pair.box_iou >= 0.80)).sum()),
        },
        "manual_review": {
            "rows": len(review),
            "by_priority": review.priority.value_counts().to_dict(),
            "unique_samples": int(review.sample_id.nunique()),
        },
    }
    (report_dir / "roi640_quality_audit_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
