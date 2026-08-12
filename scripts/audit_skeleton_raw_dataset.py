from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
FRAME_RE = re.compile(
    r"^Color_(?:(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_)?(?P<frame>\d+)$"
)
H36M_NAMES = (
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip",
    "left_knee", "left_ankle", "spine", "thorax", "neck", "head",
    "left_shoulder", "left_elbow", "left_wrist", "right_shoulder",
    "right_elbow", "right_wrist",
)
H36M_EDGES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only full audit of competition-train Skeleton JSON data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/manifest.csv")
    parser.add_argument("--fold", type=Path, default=PROJECT_ROOT / "metadata/splits/fold_0.json")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports/skeleton_raw_dataset_audit")
    return parser.parse_args()


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.as_posix()))


def describe(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {name: math.nan for name in ("min", "p01", "p05", "median", "mean", "p95", "p99", "max", "std")}
    return {
        "min": float(np.min(finite)), "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)), "median": float(np.median(finite)),
        "mean": float(np.mean(finite)), "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)), "max": float(np.max(finite)),
        "std": float(np.std(finite)),
    }


def split_name(user_id: str, fold: dict) -> str:
    if user_id in fold["train_users"]:
        return "train"
    if user_id in fold["val_users"]:
        return "validation"
    return "outside_fold"


def select_person(people: object) -> tuple[dict | None, int, str | None]:
    if not isinstance(people, list):
        return None, 0, "top_level_not_list"
    candidates = [person for person in people if isinstance(person, dict) and "keypoints" in person]
    if not candidates:
        return None, 0, "empty_people" if not people else "no_keypoint_person"

    def quality(person: dict) -> float:
        scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float64)
        return float(np.nanmean(scores)) if scores.size and np.isfinite(scores).any() else -math.inf

    return max(candidates, key=quality), len(candidates), None


def markdown_table(frame: pd.DataFrame, columns: list[str], floatfmt: str = ".4f") -> str:
    return frame.loc[:, columns].to_markdown(index=False, floatfmt=floatfmt)


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(args.fold.read_text(encoding="utf-8"))
    skeleton_present = manifest["skeleton_path"].fillna("").astype(str).str.strip().ne("")

    all_points: list[np.ndarray] = []
    all_unique_points: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    trial_rows: list[dict[str, object]] = []
    anomaly_rows: list[dict[str, object]] = []
    schema_counter: Counter[str] = Counter()
    people_counter: Counter[int] = Counter()
    keypoint_shape_counter: Counter[str] = Counter()
    score_shape_counter: Counter[str] = Counter()
    person_key_counter: Counter[str] = Counter()
    all_person_key_counter: Counter[str] = Counter()
    all_person_shape_counter: Counter[str] = Counter()
    all_person_score_shape_counter: Counter[str] = Counter()
    all_person_score_counter: Counter[float] = Counter()
    invalid_person_records = 0
    total_person_records = 0
    timestamp_intervals: list[float] = []

    for ordinal, row in enumerate(manifest.itertuples(), 1):
        path_value = str(row.skeleton_path) if skeleton_present.iloc[ordinal - 1] else ""
        trial_path = args.data_root / path_value if path_value else None
        files = (
            sorted(trial_path.rglob("*.json"), key=lambda item: natural_key(item.relative_to(trial_path)))
            if trial_path is not None and trial_path.is_dir() else []
        )
        valid_frames = 0
        invalid_frames = 0
        empty_frames = 0
        malformed_frames = 0
        multi_person_frames = 0
        frames_with_nonfinite_points = 0
        frames_with_nonpositive_scores = 0
        frames_with_whole_zero_joint = 0
        frame_ids: list[int] = []
        frame_times: list[datetime] = []
        points_by_frame_id: dict[int, list[np.ndarray]] = {}
        filenames_parsed = 0

        for file_index, path in enumerate(files):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                invalid_frames += 1
                malformed_frames += 1
                anomaly_rows.append({
                    "sample_id": row.sample_id, "frame_path": str(path), "category": "json_read_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                })
                continue
            schema_counter[type(payload).__name__] += 1
            if isinstance(payload, list):
                for candidate in payload:
                    if not isinstance(candidate, dict):
                        invalid_person_records += 1
                        continue
                    total_person_records += 1
                    all_person_key_counter["|".join(sorted(candidate))] += 1
                    try:
                        candidate_points = np.asarray(candidate.get("keypoints", []), dtype=np.float64)
                        candidate_scores = np.asarray(candidate.get("keypoint_scores", []), dtype=np.float64)
                    except (TypeError, ValueError):
                        invalid_person_records += 1
                        continue
                    all_person_shape_counter[str(candidate_points.shape)] += 1
                    all_person_score_shape_counter[str(candidate_scores.shape)] += 1
                    if (
                        candidate_points.shape != (17, 3) or candidate_scores.shape != (17,)
                        or not np.isfinite(candidate_points).all() or not np.isfinite(candidate_scores).all()
                    ):
                        invalid_person_records += 1
                    else:
                        values, counts = np.unique(candidate_scores, return_counts=True)
                        for value, count in zip(values, counts, strict=True):
                            all_person_score_counter[float(value)] += int(count)
            person, people_count, selection_error = select_person(payload)
            people_counter[people_count] += 1
            if people_count > 1:
                multi_person_frames += 1
            if person is None:
                invalid_frames += 1
                empty_frames += selection_error == "empty_people"
                malformed_frames += selection_error != "empty_people"
                anomaly_rows.append({
                    "sample_id": row.sample_id, "frame_path": str(path), "category": selection_error,
                    "detail": "no usable person selected",
                })
                continue
            person_key_counter["|".join(sorted(person))] += 1
            try:
                points = np.asarray(person["keypoints"], dtype=np.float64)
                scores = np.asarray(person.get("keypoint_scores", []), dtype=np.float64)
            except (TypeError, ValueError) as exc:
                invalid_frames += 1
                malformed_frames += 1
                anomaly_rows.append({
                    "sample_id": row.sample_id, "frame_path": str(path), "category": "nonnumeric_values",
                    "detail": str(exc),
                })
                continue
            keypoint_shape_counter[str(points.shape)] += 1
            score_shape_counter[str(scores.shape)] += 1
            if points.shape != (17, 3) or scores.shape != (17,):
                invalid_frames += 1
                malformed_frames += 1
                anomaly_rows.append({
                    "sample_id": row.sample_id, "frame_path": str(path), "category": "unexpected_shape",
                    "detail": f"keypoints={points.shape}; scores={scores.shape}",
                })
                continue
            valid_frames += 1
            nonfinite_points = ~np.isfinite(points).all(axis=1)
            nonpositive_scores = ~np.isfinite(scores) | (scores <= 0)
            whole_zero = np.isclose(points, 0.0, atol=0.0).all(axis=1)
            frames_with_nonfinite_points += bool(nonfinite_points.any())
            frames_with_nonpositive_scores += bool(nonpositive_scores.any())
            frames_with_whole_zero_joint += bool(whole_zero.any())
            if nonfinite_points.any() or nonpositive_scores.any() or whole_zero.any():
                anomaly_rows.append({
                    "sample_id": row.sample_id, "frame_path": str(path), "category": "joint_value_anomaly",
                    "detail": (
                        f"nonfinite_joints={np.flatnonzero(nonfinite_points).tolist()};"
                        f" nonpositive_score_joints={np.flatnonzero(nonpositive_scores).tolist()};"
                        f" whole_zero_joints={np.flatnonzero(whole_zero).tolist()}"
                    ),
                })
            all_points.append(points)
            all_scores.append(scores)

            match = FRAME_RE.fullmatch(path.stem)
            if match:
                filenames_parsed += 1
                frame_id = int(match.group("frame"))
                frame_ids.append(frame_id)
                points_by_frame_id.setdefault(frame_id, []).append(points)
                if match.group("timestamp"):
                    frame_times.append(datetime.strptime(match.group("timestamp"), "%Y-%m-%d_%H-%M-%S.%f"))

        if len(frame_times) >= 2:
            timestamp_intervals.extend(
                (second - first).total_seconds() * 1000.0 for first, second in zip(frame_times, frame_times[1:])
            )
        unique_frame_ids = sorted(set(frame_ids))
        unique_trial_points = [points_by_frame_id[frame_id][0] for frame_id in unique_frame_ids]
        all_unique_points.extend(unique_trial_points)
        duplicate_frame_ids = sum(max(0, len(values) - 1) for values in points_by_frame_id.values())
        duplicate_frame_ids_exact = sum(
            max(0, len(values) - 1)
            for values in points_by_frame_id.values()
            if len(values) > 1 and all(np.array_equal(values[0], value) for value in values[1:])
        )
        missing_frame_id_slots = sum(
            max(0, second - first - 1) for first, second in zip(unique_frame_ids, unique_frame_ids[1:])
        )
        frame_id_strict = len(frame_ids) == len(unique_frame_ids) and all(
            second > first for first, second in zip(frame_ids, frame_ids[1:])
        )
        timestamp_strict = all(second > first for first, second in zip(frame_times, frame_times[1:]))
        mean_motion = math.nan
        p95_motion = math.nan
        if len(unique_trial_points) >= 2:
            motion = np.linalg.norm(np.diff(np.stack(unique_trial_points), axis=0), axis=2)
            mean_motion = float(np.nanmean(motion))
            p95_motion = float(np.nanquantile(motion, 0.95))
        trial_rows.append({
            "sample_id": row.sample_id, "class_id": int(row.class_id), "action_name": row.action_name,
            "user_id": row.user_id, "trial_id": row.trial_id, "fold_split": split_name(row.user_id, fold),
            "manifest_has_skeleton_path": bool(path_value), "trial_directory_exists": bool(trial_path and trial_path.is_dir()),
            "json_frames": len(files), "valid_frames": valid_frames, "invalid_frames": invalid_frames,
            "unique_frame_ids": len(unique_frame_ids), "duplicate_frame_ids": duplicate_frame_ids,
            "duplicate_frame_ids_exact": duplicate_frame_ids_exact, "missing_frame_id_slots": missing_frame_id_slots,
            "empty_frames": empty_frames, "malformed_frames": malformed_frames,
            "multi_person_frames": multi_person_frames, "frames_with_nonfinite_points": frames_with_nonfinite_points,
            "frames_with_nonpositive_scores": frames_with_nonpositive_scores,
            "frames_with_whole_zero_joint": frames_with_whole_zero_joint,
            "parsed_filenames": filenames_parsed, "frame_ids_strictly_increasing": frame_id_strict,
            "timestamps_strictly_increasing": timestamp_strict, "mean_raw_joint_motion": mean_motion,
            "p95_raw_joint_motion": p95_motion,
        })
        if ordinal % 250 == 0 or ordinal == len(manifest):
            print(f"audited trials {ordinal}/{len(manifest)}; valid frames={len(all_points)}", flush=True)

    trials = pd.DataFrame(trial_rows)
    anomalies = pd.DataFrame(anomaly_rows, columns=["sample_id", "frame_path", "category", "detail"])
    trials.to_csv(report_dir / "trial_inventory.csv", index=False, encoding="utf-8-sig")
    anomalies.to_csv(report_dir / "frame_anomalies.csv", index=False, encoding="utf-8-sig")
    file_points = np.stack(all_points)
    points = np.stack(all_unique_points)
    scores = np.stack(all_scores)

    joint_rows = []
    for joint, name in enumerate(H36M_NAMES):
        record: dict[str, object] = {
            "joint_index": joint, "inferred_h36m_name": name,
            "inference_status": "topology_and_coordinate_signature; no organizer joint-name metadata",
            "whole_joint_exact_zero_rate": float(np.all(points[:, joint] == 0.0, axis=1).mean()),
            "nonfinite_rate": float((~np.isfinite(points[:, joint]).all(axis=1)).mean()),
            "score_nonpositive_rate": float((~np.isfinite(scores[:, joint]) | (scores[:, joint] <= 0)).mean()),
            "score_min": float(np.nanmin(scores[:, joint])), "score_max": float(np.nanmax(scores[:, joint])),
        }
        for axis, axis_name in enumerate("xyz"):
            axis_values = points[:, joint, axis]
            axis_stats = describe(axis_values)
            record.update({f"{axis_name}_{key}": value for key, value in axis_stats.items()})
            record[f"{axis_name}_exact_zero_rate"] = float((axis_values == 0.0).mean())
        joint_rows.append(record)
    joints = pd.DataFrame(joint_rows)
    joints.to_csv(report_dir / "joint_coordinate_statistics.csv", index=False, encoding="utf-8-sig")

    edge_rows = []
    for parent, child in H36M_EDGES:
        vectors = points[:, child] - points[:, parent]
        lengths = np.linalg.norm(vectors, axis=1)
        stats = describe(lengths)
        edge_rows.append({
            "parent_index": parent, "parent_name": H36M_NAMES[parent], "child_index": child,
            "child_name": H36M_NAMES[child], **{f"length_{key}": value for key, value in stats.items()},
            "length_cv": float(np.std(lengths) / np.mean(lengths)),
            "median_abs_dx": float(np.median(np.abs(vectors[:, 0]))),
            "median_abs_dy": float(np.median(np.abs(vectors[:, 1]))),
            "median_abs_dz": float(np.median(np.abs(vectors[:, 2]))),
        })
    edges = pd.DataFrame(edge_rows)
    edges.to_csv(report_dir / "inferred_h36m_bone_statistics.csv", index=False, encoding="utf-8-sig")

    length_frame = trials[trials["unique_frame_ids"] > 0]
    length_summary_rows = []
    for split, group in [("all", length_frame), *length_frame.groupby("fold_split")]:
        stats = describe(group["unique_frame_ids"].to_numpy())
        length_summary_rows.append({"group_type": "fold", "group": split, "trials": len(group), **stats})
    for action, group in length_frame.groupby("action_name"):
        stats = describe(group["unique_frame_ids"].to_numpy())
        length_summary_rows.append({"group_type": "action", "group": action, "trials": len(group), **stats})
    for user, group in length_frame.groupby("user_id"):
        stats = describe(group["unique_frame_ids"].to_numpy())
        length_summary_rows.append({"group_type": "user", "group": user, "trials": len(group), **stats})
    length_summary = pd.DataFrame(length_summary_rows)
    length_summary.to_csv(report_dir / "sequence_length_summary.csv", index=False, encoding="utf-8-sig")

    length_bucket_labels = ("1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-95", "96-127", "128+")
    length_bucket_rows = []
    for split, group in [("all", length_frame), *length_frame.groupby("fold_split")]:
        buckets = pd.cut(
            group["unique_frame_ids"], bins=[0, 1, 3, 7, 15, 31, 63, 95, 127, np.inf],
            labels=length_bucket_labels, include_lowest=True,
        ).value_counts(sort=False)
        for bucket, count in buckets.items():
            length_bucket_rows.append({
                "fold_split": split, "length_bucket": str(bucket), "trials": int(count),
                "rate": float(count / len(group)),
            })
    length_buckets = pd.DataFrame(length_bucket_rows)
    length_buckets.to_csv(report_dir / "sequence_length_buckets.csv", index=False, encoding="utf-8-sig")

    availability_rows = []
    manifest_with_split = manifest.copy()
    manifest_with_split["fold_split"] = manifest_with_split["user_id"].map(lambda user: split_name(user, fold))
    manifest_with_split["skeleton_available"] = skeleton_present.to_numpy()
    group_specs = [
        ("fold", ["fold_split"]), ("action", ["action_name"]), ("user", ["user_id"]),
    ]
    for group_type, columns in group_specs:
        for key, group in manifest_with_split.groupby(columns[0], dropna=False):
            present = int(group["skeleton_available"].sum())
            availability_rows.append({
                "group_type": group_type, "group": key, "manifest_trials": len(group),
                "skeleton_present_trials": present, "skeleton_missing_trials": len(group) - present,
                "coverage_rate": float(present / len(group)),
            })
    availability = pd.DataFrame(availability_rows)
    availability.to_csv(report_dir / "skeleton_availability_summary.csv", index=False, encoding="utf-8-sig")

    schema_rows = []
    for category, counter in (
        ("top_level_type", schema_counter), ("selected_person_keys", person_key_counter),
        ("people_count", people_counter), ("keypoints_shape", keypoint_shape_counter),
        ("keypoint_scores_shape", score_shape_counter), ("all_person_keys", all_person_key_counter),
        ("all_person_keypoints_shape", all_person_shape_counter),
        ("all_person_scores_shape", all_person_score_shape_counter),
        ("all_person_score_value", all_person_score_counter),
    ):
        for value, count in sorted(counter.items(), key=lambda item: str(item[0])):
            schema_rows.append({"category": category, "value": value, "frames": count})
    schema = pd.DataFrame(schema_rows)
    schema.to_csv(report_dir / "schema_summary.csv", index=False, encoding="utf-8-sig")

    min_z_joint_counter = Counter(np.argmin(points[:, :, 2], axis=1).tolist())
    min_z_rows = pd.DataFrame([
        {"joint_index": joint, "inferred_h36m_name": H36M_NAMES[joint], "frames_as_min_z": min_z_joint_counter[joint],
         "rate": min_z_joint_counter[joint] / len(points)}
        for joint in range(17)
    ])
    min_z_rows.to_csv(report_dir / "minimum_z_joint_distribution.csv", index=False, encoding="utf-8-sig")

    overall_lengths = describe(length_frame["unique_frame_ids"].to_numpy())
    interval_stats = describe(np.asarray(timestamp_intervals))
    manifest_missing = int((~skeleton_present).sum())
    missing_dirs = int((skeleton_present & ~trials["trial_directory_exists"]).sum())
    zero_frame_trials = int((trials["json_frames"] == 0).sum())
    invalid_frame_count = int(trials["invalid_frames"].sum())
    multi_count = int(trials["multi_person_frames"].sum())
    score_unique = np.unique(scores)
    joint0_xy_zero = float(np.all(points[:, 0, :2] == 0.0, axis=1).mean())
    min_z_zero = float((points[:, :, 2].min(axis=1) == 0.0).mean())
    max_abs = np.max(np.abs(points), axis=(0, 1))

    overall_length_table = pd.DataFrame([{"scope": "Skeleton-present trials", "trials": len(length_frame), **overall_lengths}])
    missing_table = pd.DataFrame([{
        "manifest_rows": len(manifest), "manifest_without_skeleton_path": manifest_missing,
        "declared_path_missing_directory": missing_dirs, "zero_JSON_trials": zero_frame_trials,
        "JSON_files_seen": int(trials["json_frames"].sum()), "valid_17x3_file_records": len(file_points),
        "unique_time_steps": int(trials["unique_frame_ids"].sum()),
        "duplicate_frame_id_files": int(trials["duplicate_frame_ids"].sum()),
        "exact_duplicate_frame_id_files": int(trials["duplicate_frame_ids_exact"].sum()),
        "missing_frame_id_slots": int(trials["missing_frame_id_slots"].sum()),
        "invalid_frames": invalid_frame_count, "multi_person_frames": multi_count,
        "frames_with_nonfinite_points": int(trials["frames_with_nonfinite_points"].sum()),
        "frames_with_nonpositive_scores": int(trials["frames_with_nonpositive_scores"].sum()),
        "frames_with_whole_zero_joint": int(trials["frames_with_whole_zero_joint"].sum()),
    }])
    schema_table = schema[schema["category"].isin(["top_level_type", "selected_person_keys", "keypoints_shape", "keypoint_scores_shape"])]
    anchor_table = pd.DataFrame([
        {"invariant": "joint 0 x=y exactly zero", "rate": joint0_xy_zero},
        {"invariant": "per-frame minimum z exactly zero", "rate": min_z_zero},
    ])
    overall_bucket_table = length_buckets[length_buckets["fold_split"] == "all"]

    report = f"""# Skeleton 原始数据全量审计

## 审计边界

- 数据根：`{(args.data_root / 'Skeleton').resolve()}`。
- 只读取 competition train 数据根下的 Skeleton；未读取 competition test。
- 清单覆盖全部 {len(manifest)} 个 competition-train trial，并按现有 fold 标注 train/validation，仅用于分组统计，不改变划分。
- 扫描脚本：`scripts/audit_skeleton_raw_dataset.py`。

## 1. 原始结构

{markdown_table(missing_table, list(missing_table.columns), '.0f')}

{markdown_table(schema_table, ['category', 'value', 'frames'], '.0f')}

每个有效帧文件的直接可证实结构是：顶层 `list`，其中人物记录包含且仅包含 `keypoints` 与 `keypoint_scores`。全部 {total_person_records} 个人物记录均为 `keypoints[17][3]` 和 `keypoint_scores[17]`，不合格人物记录 {invalid_person_records}。顶层 list 是人物维，不能把它当时间维；时间维由一个 trial 的多个 JSON 文件组成。

全部人物的 `keypoint_scores` 唯一值为 `{sorted(all_person_score_counter)}`。因此它在这份导出中不是有信息量的逐关节置信度或可见性信号；不能据此识别遮挡或 missing joint。

人物数分布：1 人 {people_counter[1]} 帧，2 人 {people_counter[2]} 帧，3 人 {people_counter[3]} 帧，4 人 {people_counter[4]} 帧。共有 {sum(count for people, count in people_counter.items() if people > 1)} 个多人物帧。JSON 不含 track ID，且所有 score 相同；旧 loader 的“最高平均 score”选择在多人物帧上实际退化为取列表第一个人，不能保证跨帧身份连续。

## 2. 关节数量、顺序与字段含义

实际关节数是 **17**。数据文件没有 joint-name 元数据。根据 17 点骨链排列、左右肢体连续性以及坐标锚点，顺序与 H36M-17 一致：

| Index | 推断语义 | Index | 推断语义 |
|---:|---|---:|---|
| 0 | pelvis | 9 | neck |
| 1 | right_hip | 10 | head |
| 2 | right_knee | 11 | left_shoulder |
| 3 | right_ankle | 12 | left_elbow |
| 4 | left_hip | 13 | left_wrist |
| 5 | left_knee | 14 | right_shoulder |
| 6 | left_ankle | 15 | right_elbow |
| 7 | spine | 16 | right_wrist |
| 8 | thorax |  |  |

该命名是由拓扑和数值签名建立的强推断，不是 organizer JSON 中明示的字段。逐关节坐标范围、零值率和 score 统计见 `joint_coordinate_statistics.csv`，骨链长度统计见 `inferred_h36m_bone_statistics.csv`。

每个 joint 的三个数值字段只能可靠解释为导出的 `(x, y, z)` 三维姿态坐标；它们不带单位、相机内参、外参、原点或轴定义元数据。

文件 schema、H36M-17 关节顺序和“最低关节高度归零”行为与 MMPose 官方 `human3d`/MotionBERT inferencer 文档高度一致。官方文档也说明预测输出按人物给出 `keypoints`、`keypoint_scores`，并提供关闭高度 rebasing 的选项。这是生成来源的强证据，但数据集没有保存模型配置、版本和执行命令，因此不能把具体生成器视为已被文件本身完全证明。来源：[MMPose inference 文档](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/inference.md)。

## 3. 坐标系诊断

{markdown_table(anchor_table, ['invariant', 'rate'])}

- 全局绝对范围：x=`[{points[..., 0].min():.6f}, {points[..., 0].max():.6f}]`，y=`[{points[..., 1].min():.6f}, {points[..., 1].max():.6f}]`，z=`[{points[..., 2].min():.6f}, {points[..., 2].max():.6f}]`；全轴最大绝对值 `{max_abs.tolist()}`。
- joint 0 的 x/y 被逐帧强制锚定到 0；每帧至少一个 joint 的 z 被强制锚定到 0。最低 z 关节分布见 `minimum_z_joint_distribution.csv`。
- 从身体拓扑看，z 随 pelvis→spine→thorax→neck/head 上升，并在 ankle 附近取零，因此这里的 z 更像**竖直高度轴**，不是常规相机坐标中“离相机的深度 z”。x/y 是以 pelvis 为原点的另外两个轴。
- 结论：这是经过逐帧平移规范化的、单位与尺度未知的 body/world-like 3D pose 表示。它不是图像像素坐标，不是保留人体全局平移的 camera-global XYZ，也不能可靠解释为米或毫米。没有生成器元数据时，x/y 的精确朝向和单位不能再从文件本身唯一恢复。

时间戳相邻间隔统计（ms）：median={interval_stats['median']:.3f}，p01={interval_stats['p01']:.3f}，p99={interval_stats['p99']:.3f}，min={interval_stats['min']:.3f}，max={interval_stats['max']:.3f}。

## 4. Sequence 长度

{markdown_table(overall_length_table, ['scope', 'trials', 'min', 'p01', 'p05', 'median', 'mean', 'p95', 'p99', 'max'])}

{markdown_table(overall_bucket_table, ['length_bucket', 'trials', 'rate'])}

- 这里按 trial 内唯一 `frame_id` 计数。磁盘共有 {int(trials['json_frames'].sum())} 个 JSON，但 4 个 trial 同时保存纯帧号和时间戳命名的相同数据，形成 {int(trials['duplicate_frame_ids'].sum())} 个逐数组完全一致的重复文件；实际唯一时间步为 {int(trials['unique_frame_ids'].sum())}。
- 完整逐 trial、动作、用户和 fold 分布：`trial_inventory.csv`、`sequence_length_summary.csv` 和 `sequence_length_buckets.csv`。
- 原始 sequence 没有固定 64 帧；64 是旧 loader 的在线线性重采样目标。

## 5. Missing joint 与 invalid frame

- 原始 JSON 没有 joint-valid mask、occlusion 字段或逐关节有效性字段。
- `keypoint_scores` 全量恒定，不能当 missingness mask。
- 非有限坐标、非正 score、整 joint 三轴全零和 malformed frame 的全量统计见上表；逐异常文件见 `frame_anomalies.csv`。
- 清单没有 Skeleton 路径的 trial 与“有目录但 frame 异常”是两种不同缺失，已分别统计。
- 105 个 trial 在 manifest 中没有 Skeleton 路径；已声明的 2931 个 Skeleton 目录全部存在。按 fold、动作和用户的覆盖率见 `skeleton_availability_summary.csv`。唯一 frame-id 序列内部还缺少 {int(trials['missing_frame_id_slots'].sum())} 个编号位置，这些是时间序列缺帧而不是 missing joint。
- 旧 loader 遇到空人物列表或没有 `keypoints` 的帧会制造 `17×3` 全零姿态，并继续参与中心化、缩放和插值；它没有把该帧标成 temporal invalid。这是潜在的数据语义问题。

## 6. 旧 `64×102` 的精确组成

旧实现位于 `src/data/skeleton_dataset.py`，处理顺序如下：

```text
raw poses                       [T, 17, 3]
root = mean(joint 11, joint 12) [T, 1, 3]
centered = poses - root          [T, 17, 3]
scale = RMS(centered over 17×3)  [T, 1, 1]
normalized_pose = centered/scale [T, 17, 3]
velocity[t] = pose[t]-pose[t-1]  [T, 17, 3], velocity[0]=0
concat per joint [x,y,z,vx,vy,vz] [T,17,6]
joint-major flatten             [T,102]
per-channel linear interpolation [64,102]
train-set featurewise z-score    [64,102]
temporal_mask                    [64], 全 True
```

102 的准确来源是 `17 × (3 normalized coordinates + 3 first-order velocities)`。它不包含 bone vector、joint angle、score、valid mask、timestamp 或原始 sequence length。

关键纠正：在 H36M-17 顺序下 joint 11=`left_shoulder`，joint 12=`left_elbow`。所以旧代码的 `(11+12)/2` **不是 hip midpoint**，而是左上肢中点。旧特征实际上围绕左肩/左肘中心化，并且每帧独立 RMS 缩放。随后对 train split 的 102 个通道计算均值/标准差，validation 使用同一组统计。

## 7. 审计结论

1. 原始模态是逐帧 JSON 的单/多人物容器，主 schema 为一个人物、17 个 H36M 风格关节、每关节 3 坐标加一个恒定 score。
2. 坐标已经逐帧去除 x/y pelvis 平移并把最低 z 移到 0；尺度和轴定义未提供，不能作为度量 camera-space Skeleton。
3. 原始长度是可变长，旧 `64` 来自强制插值，不是采集长度。
4. 原始格式没有可信 missing-joint 指示；必须从 schema、有限性和显式规则建立 mask，不能使用恒定 score。多人物帧还需要显式身份跟踪。
5. 4 个 trial 的重复命名文件会被旧 loader 当成额外时间步读入，制造整段重复与连接处伪速度。
6. 旧 `64×102` 的最大语义错误是按 COCO 索引用 11/12 做 hip root。未来 Skeleton 专家应先修正拓扑和 root，再决定是否重采样；不能把旧 baseline 的预处理直接视为正确的 H36M 特征管线。
"""
    (report_dir / "skeleton_raw_dataset_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "manifest_rows": len(manifest), "skeleton_trials": len(length_frame),
        "json_files": int(trials["json_frames"].sum()), "valid_file_records": len(file_points),
        "unique_time_steps": len(points),
        "invalid_frames": invalid_frame_count, "joint0_xy_zero_rate": joint0_xy_zero,
        "min_z_zero_rate": min_z_zero, "sequence_length": overall_lengths,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
