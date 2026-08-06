from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.pose_skeleton_matching_audit.audit_core import (
    COMMON_JOINT_MAP,
    COMMON_YOLO_INDICES,
    JOINT_NAMES,
    SKELETON_JOINT_NAMES,
    NormalizedTrial,
    Trial,
    choose_main_person,
    draw_overlay,
    fit_linear_projection,
    frame_key,
    left_right_swap,
    map_skeleton_to_yolo_order,
    pair_metrics,
    percentile_auc,
    project_skeleton,
    resample,
    resample_masked,
    retrieval_scores,
    retrieval_query_is_valid,
    shift_pair,
    normalize_trial,
    safe_corr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit IR YOLO pose to official 3D Skeleton correspondence.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/pose_skeleton_matching_audit.yaml",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def path_map(root: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = frame_key(path)
        if key in result:
            raise ValueError(f"Duplicate frame key below {root}: {key}")
        result[key] = path
    return result


def load_trials(config: dict) -> tuple[list[Trial], pd.DataFrame, dict[str, int]]:
    manifest = pd.read_csv(resolve(config["manifest"]), encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(resolve(config["fold"]).read_text(encoding="utf-8"))
    audit = pd.read_csv(resolve(config["pairing_audit"]), encoding="utf-8-sig")
    allowed = set(audit.loc[(audit["split"] == "train") & audit["complete_pairing"], "sample_id"].astype(str))
    selected = manifest[
        manifest["user_id"].isin(fold["train_users"]) & manifest["sample_id"].isin(allowed)
    ].sort_values(["user_id", "class_id", "sample_id"])

    cache_path = resolve(config["pose_cache"])
    with np.load(cache_path) as cache:
        sample_ids = cache["sample_ids"].astype(str)
        cache_keys = cache["frame_keys"].astype(str)
        cache_xy = cache["keypoints_xy"].astype(np.float64)
        cache_confidence = cache["keypoints_confidence"].astype(np.float64)
        widths = cache["original_width"].astype(int)
        heights = cache["original_height"].astype(int)
    cache_groups: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        if sample_id in allowed:
            cache_groups.setdefault(sample_id, []).append(index)

    data_root = Path(config["data_root"])
    trials: list[Trial] = []
    exclusions: list[dict[str, object]] = []
    counts = {"manifest_train": int(manifest["user_id"].isin(fold["train_users"]).sum()), "eligible": len(selected)}
    for ordinal, row in enumerate(selected.itertuples(), 1):
        sample_id = str(row.sample_id)
        try:
            indices = cache_groups[sample_id]
            cache_lookup = {cache_keys[index]: index for index in indices}
            skeleton_lookup = path_map(data_root / str(row.skeleton_path), {".json"})
            depth_lookup = path_map(data_root / str(row.depth_color_path), {".png", ".jpg", ".jpeg"})
            ir_lookup = path_map(data_root / str(row.ir_path), {".png", ".jpg", ".jpeg"})
            keys = sorted(set(cache_lookup) & set(skeleton_lookup) & set(depth_lookup) & set(ir_lookup))
            if len(keys) < 8:
                raise ValueError(f"only {len(keys)} common frames")
            skeleton = [choose_main_person(skeleton_lookup[key]) for key in keys]
            valid = [index for index, points in enumerate(skeleton) if points is not None]
            if len(valid) < 8:
                raise ValueError(f"only {len(valid)} valid Skeleton frames")
            keys = [keys[index] for index in valid]
            skeleton_array = np.stack([map_skeleton_to_yolo_order(skeleton[index]) for index in valid])
            cache_indices = [cache_lookup[key] for key in keys]
            size = {(int(widths[index]), int(heights[index])) for index in cache_indices}
            if len(size) != 1:
                raise ValueError(f"changing image sizes: {sorted(size)}")
            trials.append(Trial(
                sample_id=sample_id,
                class_id=int(row.class_id),
                action_name=str(row.action_name),
                user_id=str(row.user_id),
                trial_id=str(row.trial_id),
                frame_keys=keys,
                yolo_xy=cache_xy[cache_indices],
                yolo_confidence=cache_confidence[cache_indices],
                skeleton_xyz=skeleton_array,
                depth_paths=[depth_lookup[key] for key in keys],
                ir_paths=[ir_lookup[key] for key in keys],
                image_size=next(iter(size)),
            ))
        except (KeyError, OSError, ValueError) as exc:
            exclusions.append({"sample_id": sample_id, "reason": str(exc)})
        if ordinal % 250 == 0 or ordinal == len(selected):
            print(f"loaded trial metadata {ordinal}/{len(selected)}; retained={len(trials)}", flush=True)
    counts.update({"retained": len(trials), "excluded": len(exclusions), "frames": sum(len(item.frame_keys) for item in trials)})
    return trials, pd.DataFrame(exclusions, columns=["sample_id", "reason"]), counts


def aggregate_rows(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = [
        "frames", "coverage", "rmse", "pck", "angle_correlation", "velocity_correlation",
        "matching_score", "retrieval_top1", "retrieval_top5", "reciprocal_rank",
    ]
    aggregations = {name: "mean" for name in metrics if name in frame}
    result = frame.groupby(groups, as_index=False).agg(aggregations)
    counts = frame.groupby(groups, as_index=False).size().rename(columns={"size": "trials"})
    return counts.merge(result, on=groups)


def joint_metrics(
    item: NormalizedTrial, projected: np.ndarray, mode: str, shift: int, pck_threshold: float
) -> list[dict[str, object]]:
    observed = item.yolo_2d if mode == "2d" else item.yolo_25d
    mask = item.yolo_2d_mask if mode == "2d" else item.yolo_25d_mask
    target, target_mask, estimate = shift_pair(observed, mask, projected, shift)
    rows = []
    for joint in COMMON_YOLO_INDICES:
        name = JOINT_NAMES[joint]
        valid = target_mask[:, joint] & np.isfinite(estimate[:, joint]).all(axis=1)
        distance = np.linalg.norm(target[:, joint] - estimate[:, joint], axis=1)
        velocity_valid = valid[1:] & valid[:-1]
        target_velocity = np.linalg.norm(np.diff(target[:, joint], axis=0), axis=1)
        estimate_velocity = np.linalg.norm(np.diff(estimate[:, joint], axis=0), axis=1)
        rows.append({
            "mode": mode, "sample_id": item.metadata.sample_id, "joint_index": joint, "joint_name": name,
            "coverage": float(valid.sum() / max(1, len(valid))),
            "pck": float(np.mean(distance[valid] <= pck_threshold)) if valid.any() else math.nan,
            "rmse": float(np.sqrt(np.mean(np.square(distance[valid])))) if valid.any() else math.nan,
            "velocity_correlation": safe_corr(np.where(velocity_valid, target_velocity, np.nan), estimate_velocity),
        })
    return rows


def summarize_controls(distributions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (mode, control), group in distributions.groupby(["mode", "control"]):
        values = group["score"].to_numpy(float)
        rows.append({
            "mode": mode, "control": control, "n": len(values), "mean": np.mean(values),
            "std": np.std(values), "p05": np.quantile(values, 0.05), "median": np.median(values),
            "p95": np.quantile(values, 0.95),
        })
    return pd.DataFrame(rows)


def build_report(
    config: dict, counts: dict[str, int], overall: pd.DataFrame, retrieval: pd.DataFrame,
    controls: pd.DataFrame, user_summary: pd.DataFrame, action_summary: pd.DataFrame,
    failures: pd.DataFrame, decision: str,
) -> str:
    def table(frame: pd.DataFrame, columns: list[str]) -> str:
        return frame[columns].to_markdown(index=False, floatfmt=".4f")

    overall_table = table(overall, [
        "mode", "trials", "frames", "coverage", "pck", "rmse", "angle_correlation",
        "velocity_correlation", "matching_score",
    ])
    retrieval_table = table(retrieval, [
        "mode", "queries", "mean_gallery_size", "top1", "top5", "mrr", "positive_vs_all_controls_auc",
    ])
    control_table = table(controls, ["mode", "control", "n", "mean", "std", "p05", "median", "p95"])
    weakest_users = user_summary.sort_values("matching_score").groupby("mode").head(3)
    weakest_actions = action_summary.sort_values("matching_score").groupby("mode").head(5)
    weak_user_table = table(weakest_users, ["mode", "user_id", "trials", "coverage", "pck", "matching_score", "retrieval_top1", "reciprocal_rank"])
    weak_action_table = table(weakest_actions, ["mode", "class_id", "action_name", "trials", "coverage", "pck", "matching_score", "retrieval_top1", "reciprocal_rank"])
    failure_table = table(failures.head(12), ["mode", "sample_id", "action_name", "user_id", "coverage", "pck", "matching_score", "rank", "failure_reason"])
    return f"""# 图像姿态—三维 Skeleton 匹配可行性诊断

## 范围与防泄漏约束

- 只读取 `fold_0` 的 14 个 train 用户；未读取 validation 用户或 competition test。
- 清单 train trial：{counts['manifest_train']}；具备既有 Depth/IR 严格配对与 YOLO 缓存：{counts['eligible']}；最终纳入：{counts['retained']}；排除：{counts['excluded']}；纳入帧：{counts['frames']}。
- 官方 Skeleton 的 17 点拓扑由连续骨链确认为 H36M 风格顺序，而 YOLO 为 COCO-17。实验只映射两者可明确对应的 12 个肩、肘、腕、髋、膝、踝关节；YOLO 面部点以及 Skeleton 的 hip-center/spine/thorax/neck/head 不强行配对。
- YOLO 置信度阈值为 {config['confidence_threshold']}；低置信度关节保持缺失，未插值、未人工补全。
- 每个被评估用户使用其余 13 个 train 用户拟合一套固定线性弱透视映射；映射不按帧拟合。每个正/负匹配只在 ±{config['time_shift_frames']} 帧内搜索时移。
- 2.5D 的 z 来自 `Depth_Color` Jet 伪彩图在 YOLO 点周围的相对色序，**不是毫米深度或相机坐标**。因此该分支只能回答现有缓存条件下伪彩相对深度是否增加对应证据。

## 总体匹配指标

PCK 使用躯干归一化距离阈值 {config['pck_threshold']}；coverage 的分母包含所有帧与全部 12 个公共关节。

{overall_table}

## 同 trial 检索

具备至少 {config['min_retrieval_joint_observations']} 个有效“重采样帧×关节”观测的 IR/YOLO query，在同一用户的全部官方 Skeleton trial 中检索；低于门槛的 query 不进入检索分母。这样不会借用用户身份完成检索；gallery 同时包含同动作其他 trial 和不同动作 trial。

{retrieval_table}

## 正匹配与负对照

分数越高越相似。`same_action_other_trial`、`different_action`、`time_shuffle`、`left_right_swap` 均使用与正样本相同的映射和时移搜索。

{control_table}

## 薄弱用户与动作

### 用户

{weak_user_table}

### 动作

{weak_action_table}

完整的 40 类和 14 用户结果分别见 `per_action_metrics.csv` 与 `per_user_metrics.csv`；逐关节结果见 `per_joint_metrics.csv`。

## 代表失败案例

{failure_table}

成功/失败叠图在 `outputs/pose_skeleton_matching_audit/visualizations/`，青色为 IR YOLO，红色为固定映射后的 Skeleton。图片作为本地诊断产物未纳入 Git；索引见 `representative_cases.csv`。

## 最终判断

**{decision}**

判定规则预先固定：帧级融合要求 PCK≥0.60、角度相关≥0.50、速度相关≥0.40、Top-1≥0.50 且正负 AUC≥0.80；序列级融合要求 Top-1≥0.30、MRR≥0.50 且 AUC≥0.75。若只有 2.5D 达标，仍因伪彩深度非度量而降一级解释。

本实验不训练动作分类或融合模型，也没有修改 B2、数据划分和已有缓存。结论只针对现有 IR YOLO 缓存、官方 Skeleton 表示及当前可用的伪彩 Depth。
"""


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_dir, report_dir = resolve(config["output_dir"]), resolve(config["report_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    trials, exclusions, counts = load_trials(config)
    exclusions.to_csv(report_dir / "excluded_trials.csv", index=False, encoding="utf-8-sig")
    if not trials:
        raise RuntimeError("No train-fold trials survived input validation")

    normalized: list[NormalizedTrial] = []
    for index, trial in enumerate(trials, 1):
        normalized.append(normalize_trial(
            trial, float(config["confidence_threshold"]), int(config["depth_patch_radius"]),
            float(config["depth_min_saturation"]),
        ))
        if index % 100 == 0 or index == len(trials):
            print(f"decoded and normalized trials {index}/{len(trials)}", flush=True)

    users = sorted({item.metadata.user_id for item in normalized})
    matrices: dict[tuple[str, str], np.ndarray] = {}
    for user in users:
        calibration = [item for item in normalized if item.metadata.user_id != user]
        for mode in ("2d", "25d"):
            matrices[(user, mode)] = fit_linear_projection(calibration, mode, float(config["ridge"]))
            print(f"fit leave-one-user-out projection user={user} mode={mode}", flush=True)

    trial_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []
    projected_lookup: dict[tuple[str, str], np.ndarray] = {}
    for index, item in enumerate(normalized, 1):
        for mode in ("2d", "25d"):
            projected = project_skeleton(item.skeleton, matrices[(item.metadata.user_id, mode)])
            projected_lookup[(item.metadata.sample_id, mode)] = projected
            observed = item.yolo_2d if mode == "2d" else item.yolo_25d
            mask = item.yolo_2d_mask if mode == "2d" else item.yolo_25d_mask
            metrics, _, shift = pair_metrics(
                observed, mask, projected, int(config["time_shift_frames"]), float(config["pck_threshold"])
            )
            trial_rows.append({
                "mode": mode, "sample_id": item.metadata.sample_id, "class_id": item.metadata.class_id,
                "action_name": item.metadata.action_name, "user_id": item.metadata.user_id,
                "trial_id": item.metadata.trial_id, "frames": len(item.metadata.frame_keys),
                "best_shift_frames": shift, **metrics,
            })
            joint_rows.extend(joint_metrics(item, projected, mode, shift, float(config["pck_threshold"])))
        if index % 250 == 0 or index == len(normalized):
            print(f"computed positive metrics {index}/{len(normalized)}", flush=True)

    trial_frame = pd.DataFrame(trial_rows)
    rng = np.random.default_rng(int(config["seed"]))
    retrieval_length = int(config["retrieval_length"])
    max_shift = int(config["time_shift_frames"])
    distribution_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    by_user = {user: [item for item in normalized if item.metadata.user_id == user] for user in users}
    for user, items in by_user.items():
        skeleton_resampled = np.stack([resample(item.skeleton, retrieval_length) for item in items])
        for mode in ("2d", "25d"):
            gallery = skeleton_resampled @ matrices[(user, mode)]
            for query_index, item in enumerate(items):
                observed = item.yolo_2d if mode == "2d" else item.yolo_25d
                mask = item.yolo_2d_mask if mode == "2d" else item.yolo_25d_mask
                query, query_mask = resample_masked(observed, mask, retrieval_length)
                if not retrieval_query_is_valid(query_mask, int(config["min_retrieval_joint_observations"])):
                    continue
                scores = retrieval_scores(query, query_mask, gallery, max_shift)
                positive_score = float(scores[query_index])
                rank = 1 + int(np.sum(scores > positive_score))
                retrieval_rows.append({
                    "mode": mode, "sample_id": item.metadata.sample_id, "class_id": item.metadata.class_id,
                    "action_name": item.metadata.action_name, "user_id": user, "gallery_size": len(items),
                    "rank": rank, "retrieval_top1": float(rank == 1), "retrieval_top5": float(rank <= 5),
                    "reciprocal_rank": 1.0 / rank,
                })
                distribution_rows.append({"mode": mode, "sample_id": item.metadata.sample_id, "control": "positive_same_trial", "score": positive_score})
                same_action = [i for i, candidate in enumerate(items) if i != query_index and candidate.metadata.class_id == item.metadata.class_id]
                other_action = [i for i, candidate in enumerate(items) if candidate.metadata.class_id != item.metadata.class_id]
                if same_action:
                    selected = int(rng.choice(same_action))
                    distribution_rows.append({"mode": mode, "sample_id": item.metadata.sample_id, "control": "same_action_other_trial", "score": float(scores[selected])})
                if other_action:
                    selected = int(rng.choice(other_action))
                    distribution_rows.append({"mode": mode, "sample_id": item.metadata.sample_id, "control": "different_action", "score": float(scores[selected])})
                order = rng.permutation(retrieval_length)
                shuffled = gallery[query_index, order][None]
                swapped = left_right_swap(gallery[query_index])[None]
                distribution_rows.append({"mode": mode, "sample_id": item.metadata.sample_id, "control": "time_shuffle", "score": float(retrieval_scores(query, query_mask, shuffled, max_shift)[0])})
                distribution_rows.append({"mode": mode, "sample_id": item.metadata.sample_id, "control": "left_right_swap", "score": float(retrieval_scores(query, query_mask, swapped, max_shift)[0])})
        print(f"completed retrieval user={user} trials={len(items)}", flush=True)

    retrieval_frame = pd.DataFrame(retrieval_rows)
    trial_frame = trial_frame.merge(
        retrieval_frame[["mode", "sample_id", "rank", "gallery_size", "retrieval_top1", "retrieval_top5", "reciprocal_rank"]],
        on=["mode", "sample_id"], how="left", validate="one_to_one",
    )
    distribution_frame = pd.DataFrame(distribution_rows)
    control_summary = summarize_controls(distribution_frame)

    retrieval_summary_rows = []
    for mode, group in retrieval_frame.groupby("mode"):
        positive = distribution_frame[(distribution_frame["mode"] == mode) & (distribution_frame["control"] == "positive_same_trial")]["score"].to_numpy()
        negative = distribution_frame[(distribution_frame["mode"] == mode) & (distribution_frame["control"] != "positive_same_trial")]["score"].to_numpy()
        retrieval_summary_rows.append({
            "mode": mode, "queries": len(group), "mean_gallery_size": group["gallery_size"].mean(),
            "top1": group["retrieval_top1"].mean(), "top5": group["retrieval_top5"].mean(),
            "mrr": group["reciprocal_rank"].mean(),
            "positive_vs_all_controls_auc": percentile_auc(positive, negative),
        })
    retrieval_summary = pd.DataFrame(retrieval_summary_rows)

    overall = aggregate_rows(trial_frame, ["mode"])
    per_action = aggregate_rows(trial_frame, ["mode", "class_id", "action_name"])
    per_user = aggregate_rows(trial_frame, ["mode", "user_id"])
    per_joint_trial = pd.DataFrame(joint_rows)
    per_joint = per_joint_trial.groupby(["mode", "joint_index", "joint_name"], as_index=False).agg(
        trials=("sample_id", "count"), coverage=("coverage", "mean"), pck=("pck", "mean"),
        rmse=("rmse", "mean"), velocity_correlation=("velocity_correlation", "mean"),
    )

    trial_frame.to_csv(report_dir / "per_trial_metrics.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(report_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")
    retrieval_summary.to_csv(report_dir / "retrieval_metrics.csv", index=False, encoding="utf-8-sig")
    per_action.to_csv(report_dir / "per_action_metrics.csv", index=False, encoding="utf-8-sig")
    per_user.to_csv(report_dir / "per_user_metrics.csv", index=False, encoding="utf-8-sig")
    per_joint.to_csv(report_dir / "per_joint_metrics.csv", index=False, encoding="utf-8-sig")
    distribution_frame.to_csv(report_dir / "score_distributions.csv", index=False, encoding="utf-8-sig")
    control_summary.to_csv(report_dir / "control_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "yolo_joint_index": yolo_index, "yolo_joint_name": JOINT_NAMES[yolo_index],
            "skeleton_joint_index": skeleton_index, "skeleton_joint_name": SKELETON_JOINT_NAMES[skeleton_index],
            "mapping_status": "topology-established H36M-style semantic mapping",
        }
        for yolo_index, skeleton_index in COMMON_JOINT_MAP.items()
    ]).to_csv(report_dir / "common_joint_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"mode": mode, "shift_frames": shift, "trials": count}
        for (mode, shift), count in Counter(zip(trial_frame["mode"], trial_frame["best_shift_frames"], strict=True)).items()
    ]).sort_values(["mode", "shift_frames"]).to_csv(report_dir / "alignment_summary.csv", index=False, encoding="utf-8-sig")

    failures = trial_frame.sort_values(["mode", "matching_score"]).groupby("mode").head(10).copy()
    failures["failure_reason"] = np.select(
        [failures["coverage"] < 0.35, failures["pck"] < 0.30, failures["rank"] > 5],
        ["low_joint_coverage", "low_geometric_agreement", "same_trial_not_in_top5"], default="combined_metric_failure",
    )
    failures.to_csv(report_dir / "failure_cases.csv", index=False, encoding="utf-8-sig")

    item_lookup = {item.metadata.sample_id: item for item in normalized}
    representative_rows = []
    for mode in ("2d", "25d"):
        ordered = trial_frame[trial_frame["mode"] == mode].sort_values("matching_score")
        case_count = int(config["visualization_cases_per_mode"])
        selections = [("failure", row) for _, row in ordered.head(case_count).iterrows()]
        success_pool = ordered.tail(max(2, case_count)).head(case_count)
        selections += [("success", row) for _, row in success_pool.iterrows()]
        for case_type, row in selections:
            item = item_lookup[row.sample_id]
            mask = item.yolo_2d_mask if mode == "2d" else item.yolo_25d_mask
            valid_frames = np.flatnonzero(mask[:, [5, 6, 11, 12]].all(axis=1))
            frame_index = int(valid_frames[len(valid_frames) // 2]) if len(valid_frames) else len(mask) // 2
            projected = projected_lookup[(row.sample_id, mode)]
            output = output_dir / "visualizations" / f"{mode}_{case_type}_{row.sample_id}.png"
            draw_overlay(
                item.metadata.ir_paths[frame_index], item.metadata.yolo_xy[frame_index],
                item.yolo_2d_mask[frame_index], projected[frame_index], item.yolo_root_2d[frame_index],
                float(item.yolo_scale_2d[frame_index]), output,
                f"{mode} {case_type} score={row.matching_score:.3f} PCK={row.pck:.3f}",
            )
            representative_rows.append({
                "mode": mode, "case_type": case_type, "sample_id": row.sample_id,
                "action_name": row.action_name, "user_id": row.user_id, "matching_score": row.matching_score,
                "pck": row.pck, "rank": row["rank"], "image_path": str(output.resolve()),
            })
    pd.DataFrame(representative_rows).to_csv(report_dir / "representative_cases.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    for (mode, control), group in distribution_frame.groupby(["mode", "control"]):
        if control in {"positive_same_trial", "same_action_other_trial", "different_action", "time_shuffle", "left_right_swap"}:
            plt.hist(group["score"], bins=40, density=True, histtype="step", label=f"{mode}:{control}")
    plt.xlabel("retrieval similarity score")
    plt.ylabel("density")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=150)
    plt.close()

    merged = overall.merge(retrieval_summary, on="mode")
    frame_level = merged[
        (merged["pck"] >= 0.60) & (merged["angle_correlation"] >= 0.50)
        & (merged["velocity_correlation"] >= 0.40) & (merged["top1"] >= 0.50)
        & (merged["positive_vs_all_controls_auc"] >= 0.80)
    ]
    sequence_level = merged[
        (merged["top1"] >= 0.30) & (merged["mrr"] >= 0.50)
        & (merged["positive_vs_all_controls_auc"] >= 0.75)
    ]
    if not frame_level.empty and "2d" in set(frame_level["mode"]):
        decision = "适合帧级融合：2D 分支达到预设几何、动态与检索门槛。"
    elif not frame_level.empty:
        decision = "最多适合序列级融合：只有伪彩 2.5D 达到帧级数值门槛，非度量深度限制使其不能作为可靠帧级相机对应。"
    elif not sequence_level.empty:
        decision = "适合序列级融合，不支持帧级关节直接融合：trial 身份可检索，但逐帧几何/动态一致性未同时达标。"
    else:
        decision = "不具备可靠对应：现有 2D 与伪彩 2.5D 均未达到序列级检索及正负分离门槛。"

    report = build_report(config, counts, overall, retrieval_summary, control_summary, per_user, per_action, failures, decision)
    (report_dir / "pose_skeleton_matching_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({"counts": counts, "decision": decision}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
