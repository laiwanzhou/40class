from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_CANDIDATES = (
    PROJECT_ROOT / "outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz",
    PROJECT_ROOT.parent / "40class/outputs/depth_ir_person_crop_40class_fold0/person_crop_pose_tracks.npz",
)
DEFAULT_OUTPUT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\yulan2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose-cache", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.25)
    return parser.parse_args()


def resolve_cache(value: Path | None) -> Path:
    if value is not None:
        if not value.exists():
            raise FileNotFoundError(value)
        return value
    for candidate in DEFAULT_CACHE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Pose cache not found: {DEFAULT_CACHE_CANDIDATES}")


def longest_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def main() -> None:
    args = parse_args()
    cache_path = resolve_cache(args.pose_cache)
    with np.load(cache_path) as data:
        sample_ids = data["sample_ids"].astype(str)
        class_ids = data["class_ids"].astype(np.int64)
        action_names = data["action_names"].astype(str)
        splits = data["splits"].astype(str)
        keypoints = data["keypoints_xy"].astype(np.float32)
        confidence = data["keypoints_confidence"].astype(np.float32)

    finite = np.isfinite(keypoints).all(axis=-1)
    left = (confidence[:, 9] >= args.threshold) & finite[:, 9]
    right = (confidence[:, 10] >= args.threshold) & finite[:, 10]
    rows: list[dict[str, object]] = []
    for sample_id in pd.unique(sample_ids):
        indices = np.flatnonzero(sample_ids == sample_id)
        sample_left = left[indices]
        sample_right = right[indices]
        both = sample_left & sample_right
        single = sample_left ^ sample_right
        neither = ~(sample_left | sample_right)
        frames = len(indices)
        rows.append({
            "sample_id": sample_id,
            "class_id": int(class_ids[indices[0]]),
            "action_name": str(action_names[indices[0]]),
            "split": str(splits[indices[0]]),
            "frames": frames,
            "left_wrist_rate": float(sample_left.mean()),
            "right_wrist_rate": float(sample_right.mean()),
            "both_wrists_rate": float(both.mean()),
            "exactly_one_wrist_rate": float(single.mean()),
            "neither_wrist_rate": float(neither.mean()),
            "single_wrist_majority": bool(single.mean() >= 0.50),
            "single_wrist_dominant": bool(single.mean() >= 0.80),
            "neither_wrist_majority": bool(neither.mean() >= 0.50),
            "both_wrists_rare": bool(both.mean() < 0.20),
            "longest_single_wrist_run": longest_run(single),
            "longest_neither_wrist_run": longest_run(neither),
        })
    samples = pd.DataFrame(rows).sort_values(["class_id", "sample_id"])

    action_rows: list[dict[str, object]] = []
    for (class_id, action_name), group in samples.groupby(["class_id", "action_name"], sort=True):
        total_frames = int(group["frames"].sum())
        weighted = lambda column: float(np.average(group[column], weights=group["frames"]))
        val = group[group["split"] == "validation"]
        action_rows.append({
            "class_id": int(class_id),
            "action_name": action_name,
            "sample_count": len(group),
            "frame_count": total_frames,
            "both_wrists_frame_rate": weighted("both_wrists_rate"),
            "exactly_one_wrist_frame_rate": weighted("exactly_one_wrist_rate"),
            "neither_wrist_frame_rate": weighted("neither_wrist_rate"),
            "single_wrist_majority_sample_count": int(group["single_wrist_majority"].sum()),
            "single_wrist_majority_sample_rate": float(group["single_wrist_majority"].mean()),
            "single_wrist_dominant_sample_count": int(group["single_wrist_dominant"].sum()),
            "single_wrist_dominant_sample_rate": float(group["single_wrist_dominant"].mean()),
            "neither_wrist_majority_sample_count": int(group["neither_wrist_majority"].sum()),
            "both_wrists_rare_sample_count": int(group["both_wrists_rare"].sum()),
            "validation_sample_count": len(val),
            "validation_single_wrist_majority_count": int(val["single_wrist_majority"].sum()),
            "maximum_single_wrist_run": int(group["longest_single_wrist_run"].max()),
            "maximum_neither_wrist_run": int(group["longest_neither_wrist_run"].max()),
        })
    actions = pd.DataFrame(action_rows)
    actions["systematic_single_wrist"] = (
        (actions["exactly_one_wrist_frame_rate"] >= 0.25)
        | (actions["single_wrist_majority_sample_rate"] >= 0.20)
    )
    actions = actions.sort_values(
        ["systematic_single_wrist", "exactly_one_wrist_frame_rate"], ascending=[False, False],
    )

    args.output.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output / "wrist_visibility_by_sample.csv", index=False, encoding="utf-8-sig")
    actions.to_csv(args.output / "wrist_visibility_by_action.csv", index=False, encoding="utf-8-sig")

    affected = actions[actions["systematic_single_wrist"]]
    any_majority = actions[actions["single_wrist_majority_sample_count"] > 0]
    lines = [
        "# 腕部姿态可见性审计", "",
        f"- 姿态缓存：`{cache_path}`。",
        f"- 样本：{len(samples)}；动作：{len(actions)}；帧：{int(samples['frames'].sum())}。",
        "- 只使用固定train/validation数据，未读取test。", "",
        "## 口径", "",
        "- 单腕帧：左右腕关键点中恰好一个置信度达到0.25。",
        "- 单腕占多数样本：一个样本中至少50%的帧只有一个腕部有效。",
        "- 单腕主导样本：一个样本中至少80%的帧只有一个腕部有效。",
        "- 系统性单腕动作：动作单腕帧率至少25%，或至少20%的样本为单腕占多数。", "",
        "## 总体", "",
        f"- 存在至少一个单腕占多数样本的动作：{len(any_majority)}/40。",
        f"- 达到系统性单腕标准的动作：{len(affected)}/40。", "",
        "## 系统性单腕动作", "",
        "| 动作 | 单腕帧率 | 单腕占多数样本 | 单腕主导样本 | 双腕均缺失帧率 |", "|---|---:|---:|---:|---:|",
    ]
    for row in affected.itertuples(index=False):
        lines.append(
            f"| {row.action_name} | {row.exactly_one_wrist_frame_rate:.2%} | "
            f"{row.single_wrist_majority_sample_count}/{row.sample_count} | "
            f"{row.single_wrist_dominant_sample_count}/{row.sample_count} | "
            f"{row.neither_wrist_frame_rate:.2%} |"
        )
    (args.output / "腕部姿态可见性审计.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"samples={len(samples)} actions={len(actions)} frames={int(samples['frames'].sum())} "
        f"any_majority_actions={len(any_majority)} systematic_actions={len(affected)}"
    )


if __name__ == "__main__":
    main()
