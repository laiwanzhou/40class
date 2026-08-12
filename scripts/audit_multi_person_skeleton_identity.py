from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_MODULE_DIR = PROJECT_ROOT / "scripts/experiments/pose_skeleton_matching_audit"
sys.path.insert(0, str(AUDIT_MODULE_DIR))

from audit_core import (  # noqa: E402
    COCO_EDGES,
    COMMON_YOLO_INDICES,
    map_skeleton_to_yolo_order,
    normalize_pose,
)


FRAME_RE = re.compile(
    r"(?:Color|Depth|IR)_(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_"
    r"(?P<frame>\d+)(?:_Color)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Identify multi-person Skeleton candidates using IR YOLO pose.")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs/experiments/pose_skeleton_matching_audit.yaml",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_multi_person_identity",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "outputs/skeleton_multi_person_identity",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def frame_key(path: Path) -> str | None:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        return None
    return f"{match.group('timestamp')}_{match.group('frame')}"


def path_map(root: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        key = frame_key(path)
        if key is not None:
            result[key] = path
    return result


def load_people(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = []
    if not isinstance(payload, list):
        return result
    for person in payload:
        if not isinstance(person, dict) or "keypoints" not in person:
            continue
        points = np.asarray(person["keypoints"], dtype=np.float64)
        if points.shape == (17, 3) and np.isfinite(points).all():
            result.append(map_skeleton_to_yolo_order(points))
    return result


def normalize_yolo(points: np.ndarray, confidence: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    common = np.zeros(17, dtype=bool)
    common[COMMON_YOLO_INDICES] = True
    mask = (confidence >= threshold) & np.isfinite(points).all(axis=1) & common
    normalized, normalized_mask, roots, scales = normalize_pose(points[None], mask[None])
    return normalized[0], normalized_mask[0], roots[0], float(scales[0])


def normalize_candidate(points: np.ndarray) -> np.ndarray | None:
    mask = np.isfinite(points).all(axis=1)
    normalized, normalized_mask, _, _ = normalize_pose(points[None], mask[None])
    required = normalized_mask[0, COMMON_YOLO_INDICES]
    return normalized[0] if required.sum() >= 6 else None


def candidate_rmse(
    observed: np.ndarray, observed_mask: np.ndarray, candidate: np.ndarray, projection: np.ndarray
) -> tuple[float, int, np.ndarray]:
    projected = candidate @ projection
    valid = observed_mask & np.isfinite(candidate).all(axis=1) & np.isfinite(projected).all(axis=1)
    if valid.sum() < 6:
        return np.nan, int(valid.sum()), projected
    error = np.linalg.norm(observed[valid] - projected[valid], axis=1)
    return float(np.sqrt(np.mean(np.square(error)))), int(valid.sum()), projected


def render_case(
    image_path: Path, output_path: Path, observed_xy: np.ndarray, observed_mask: np.ndarray,
    projections: list[np.ndarray], root: np.ndarray, scale: float, rmses: list[float], selected: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [(255, 70, 50), (30, 220, 90), (40, 140, 255), (255, 190, 20)]
    for left, right in COCO_EDGES:
        if observed_mask[left] and observed_mask[right]:
            draw.line([tuple(observed_xy[left]), tuple(observed_xy[right])], fill=(255, 255, 255), width=3)
    for index, projection in enumerate(projections):
        xy = projection * scale + root
        color = colors[index % len(colors)]
        width = 4 if index == selected else 2
        for left, right in COCO_EDGES:
            if np.isfinite(xy[[left, right]]).all():
                draw.line([tuple(xy[left]), tuple(xy[right])], fill=color, width=width)
        draw.text((8, 8 + 18 * index), f"candidate {index}: rmse={rmses[index]:.3f}", fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    report_dir = resolve(args.report_dir)
    output_dir = resolve(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(resolve(config["manifest"]), encoding="utf-8-sig", dtype={"user_id": str})
    fold = json.loads(resolve(config["fold"]).read_text(encoding="utf-8"))
    pairing = pd.read_csv(resolve(config["pairing_audit"]), encoding="utf-8-sig")
    allowed = set(pairing.loc[(pairing["split"] == "train") & pairing["complete_pairing"], "sample_id"].astype(str))
    selected = manifest[
        manifest["user_id"].isin(fold["train_users"]) & manifest["sample_id"].isin(allowed)
    ].sort_values(["user_id", "class_id", "sample_id"])

    with np.load(resolve(config["pose_cache"])) as cache:
        cache_ids = cache["sample_ids"].astype(str)
        cache_keys = cache["frame_keys"].astype(str)
        cache_xy = cache["keypoints_xy"].astype(np.float64)
        cache_conf = cache["keypoints_confidence"].astype(np.float64)
    cache_groups: dict[str, dict[str, int]] = {}
    for index, sample_id in enumerate(cache_ids):
        if sample_id in allowed:
            cache_groups.setdefault(sample_id, {})[cache_keys[index]] = index

    threshold = float(config["confidence_threshold"])
    frame_records: list[dict[str, object]] = []
    singleton_pool: dict[str, list[dict[str, object]]] = {}
    data_root = Path(config["data_root"])

    for ordinal, row in enumerate(selected.itertuples(), 1):
        sample_id = str(row.sample_id)
        skeleton_lookup = path_map(data_root / str(row.skeleton_path), {".json"})
        ir_lookup = path_map(data_root / str(row.ir_path), {".png", ".jpg", ".jpeg"})
        for key in sorted(set(cache_groups.get(sample_id, {})) & set(skeleton_lookup) & set(ir_lookup)):
            cache_index = cache_groups[sample_id][key]
            observed, observed_mask, root, scale = normalize_yolo(
                cache_xy[cache_index], cache_conf[cache_index], threshold
            )
            if observed_mask.sum() < 6:
                continue
            people = load_people(skeleton_lookup[key])
            normalized_people = [normalize_candidate(person) for person in people]
            normalized_people = [person for person in normalized_people if person is not None]
            if not normalized_people:
                continue
            base = {
                "sample_id": sample_id, "class_id": int(row.class_id), "action_name": str(row.action_name),
                "user_id": str(row.user_id), "frame_key": key, "skeleton_path": str(skeleton_lookup[key]),
                "ir_path": str(ir_lookup[key]), "observed": observed, "observed_mask": observed_mask,
                "root": root, "scale": scale, "people": normalized_people,
            }
            if len(normalized_people) == 1:
                singleton_pool.setdefault(str(row.action_name), []).append(base)
            else:
                frame_records.append(base)
        if ordinal % 250 == 0 or ordinal == len(selected):
            print(f"loaded {ordinal}/{len(selected)} trials; multi frames={len(frame_records)}", flush=True)

    total_gram = np.zeros((3, 3), dtype=np.float64)
    total_cross = np.zeros((3, 2), dtype=np.float64)
    user_grams: dict[str, np.ndarray] = {}
    user_crosses: dict[str, np.ndarray] = {}
    for pool in singleton_pool.values():
        for record in pool:
            candidate = record["people"][0]
            valid = record["observed_mask"] & np.isfinite(candidate).all(axis=1)
            source = candidate[valid]
            target = record["observed"][valid]
            gram = source.T @ source
            cross = source.T @ target
            user = str(record["user_id"])
            total_gram += gram
            total_cross += cross
            user_grams[user] = user_grams.get(user, np.zeros((3, 3), dtype=np.float64)) + gram
            user_crosses[user] = user_crosses.get(user, np.zeros((3, 2), dtype=np.float64)) + cross
    ridge = float(config["ridge"]) * np.eye(3)
    projection = np.linalg.solve(total_gram + ridge, total_cross)
    user_heldout_projections = {
        user: np.linalg.solve(total_gram - user_grams[user] + ridge, total_cross - user_crosses[user])
        for user in user_grams
    }

    decision_rows: list[dict[str, object]] = []
    render_payloads: list[dict[str, object]] = []
    for record in frame_records:
        results = [
            candidate_rmse(record["observed"], record["observed_mask"], candidate, projection)
            for candidate in record["people"]
        ]
        rmses = [item[0] for item in results]
        order = np.argsort(np.asarray(rmses))
        best, second = int(order[0]), int(order[1])
        relative_margin = float((rmses[second] - rmses[best]) / max(rmses[second], 1e-12))
        decision_rows.append({
            "sample_id": record["sample_id"], "class_id": record["class_id"],
            "action_name": record["action_name"], "user_id": record["user_id"],
            "frame_key": record["frame_key"], "people_count": len(rmses), "selected_candidate": best,
            "best_rmse": rmses[best], "second_rmse": rmses[second], "relative_margin": relative_margin,
            "confident_10pct": relative_margin >= 0.10, "confident_20pct": relative_margin >= 0.20,
            "selected_is_json_first": best == 0, "valid_common_joints": results[best][1],
            "candidate_rmses": "|".join(f"{value:.8f}" for value in rmses),
            "skeleton_path": record["skeleton_path"], "ir_path": record["ir_path"],
        })
        render_payloads.append({**record, "rmses": rmses, "projections": [item[2] for item in results], "selected": best})

    decisions = pd.DataFrame(decision_rows)
    decisions.to_csv(report_dir / "multi_person_candidate_decisions.csv", index=False, encoding="utf-8-sig")
    trial_summary = decisions.groupby(
        ["sample_id", "class_id", "action_name", "user_id"], as_index=False
    ).agg(
        frames=("frame_key", "size"), mean_best_rmse=("best_rmse", "mean"),
        median_relative_margin=("relative_margin", "median"), confident_10pct_rate=("confident_10pct", "mean"),
        confident_20pct_rate=("confident_20pct", "mean"), json_first_rate=("selected_is_json_first", "mean"),
    )
    trial_summary.to_csv(report_dir / "multi_person_trial_summary.csv", index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(args.seed)
    benchmark_rows = []
    for action, pool in singleton_pool.items():
        if len(pool) < 2:
            continue
        sample_indexes = rng.choice(len(pool), size=min(200, len(pool)), replace=False)
        for index in sample_indexes:
            target_record = pool[int(index)]
            alternatives = [item for item in pool if item["sample_id"] != target_record["sample_id"]]
            if not alternatives:
                continue
            distractor = alternatives[int(rng.integers(len(alternatives)))]
            benchmark_projection = user_heldout_projections[str(target_record["user_id"])]
            true_rmse, _, _ = candidate_rmse(
                target_record["observed"], target_record["observed_mask"], target_record["people"][0],
                benchmark_projection,
            )
            distractor_rmse, _, _ = candidate_rmse(
                target_record["observed"], target_record["observed_mask"], distractor["people"][0],
                benchmark_projection,
            )
            benchmark_rows.append({
                "action_name": action, "target_sample_id": target_record["sample_id"],
                "distractor_sample_id": distractor["sample_id"], "true_rmse": true_rmse,
                "distractor_rmse": distractor_rmse, "correct": true_rmse < distractor_rmse,
                "relative_margin": (distractor_rmse - true_rmse) / max(distractor_rmse, 1e-12),
                "winner_relative_margin": abs(distractor_rmse - true_rmse) / max(true_rmse, distractor_rmse, 1e-12),
            })
    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(report_dir / "singleton_same_action_distractor_benchmark.csv", index=False, encoding="utf-8-sig")

    non_first = decisions[~decisions["selected_is_json_first"]]
    render_order = (
        list(decisions.nlargest(2, "relative_margin").index)
        + list(non_first.nlargest(2, "relative_margin").index)
        + list(decisions.nsmallest(2, "relative_margin").index)
    )
    for rank, index in enumerate(render_order):
        record = render_payloads[index]
        render_case(
            Path(record["ir_path"]), output_dir / f"case_{rank + 1}_{record['sample_id']}_{record['frame_key']}.png",
            cache_xy[cache_groups[record["sample_id"]][record["frame_key"]]], record["observed_mask"],
            record["projections"], record["root"], record["scale"], record["rmses"], record["selected"],
        )

    candidate_counts = Counter(decisions["people_count"])
    benchmark_10 = benchmark[benchmark["winner_relative_margin"] >= 0.10]
    benchmark_20 = benchmark[benchmark["winner_relative_margin"] >= 0.20]
    raw_inventory_path = PROJECT_ROOT / "reports/skeleton_raw_dataset_audit/trial_inventory.csv"
    raw_train_multi = np.nan
    if raw_inventory_path.is_file():
        raw_inventory = pd.read_csv(raw_inventory_path, encoding="utf-8-sig")
        raw_train_multi = int(raw_inventory.loc[raw_inventory["fold_split"] == "train", "multi_person_frames"].sum())
    raw_coverage = len(decisions) / raw_train_multi if np.isfinite(raw_train_multi) and raw_train_multi else np.nan
    report = f"""# 多候选 Skeleton 的视觉身份选择诊断

## 方法

- 范围限定为 fold-0 的 14 个 train 用户和既有完整配对样本；未读取 held-out validation 用户或 competition test。
- 用单候选帧拟合一个全局、固定的 Skeleton 3D→YOLO 2D 线性投影，不进行逐帧自由拟合。
- 对多候选帧中的每个 Skeleton 分别计算与 IR 主 YOLO pose 的 12 个公共关节 RMSE，选择 RMSE 最低者。
- `relative_margin=(second-best-best)/second-best`。margin 越大，视觉证据越明确；这里只给出可辨识度，不把无标注选择冒充 ground truth accuracy。

## 覆盖与结果

| 指标 | 数值 |
|---|---:|
| 原始审计中的 train-fold 多候选 JSON | {raw_train_multi:.0f} |
| 可参与视觉选择的多候选帧 | {len(decisions)} |
| 对 train-fold 多候选 JSON 的覆盖率 | {raw_coverage:.4%} |
| 涉及 trial | {decisions['sample_id'].nunique()} |
| 2/3/4 候选帧 | {candidate_counts[2]} / {candidate_counts[3]} / {candidate_counts[4]} |
| 选择 JSON 第一个候选的比例 | {decisions['selected_is_json_first'].mean():.4%} |
| 视觉排序改选非第一候选 | {(~decisions['selected_is_json_first']).sum()} |
| margin ≥10% | {decisions['confident_10pct'].mean():.4%} |
| margin ≥20% | {decisions['confident_20pct'].mean():.4%} |
| margin <20%，标记 ambiguous | {(~decisions['confident_20pct']).sum()} |
| median relative margin | {decisions['relative_margin'].median():.6f} |

## 伪干扰验证

单候选帧作为已知正候选，并从同一动作、不同 trial 抽取一个 Skeleton 作为干扰。每个目标用户都使用排除该用户后拟合的投影，避免目标用户泄漏进投影校准：

| 策略 | 覆盖率 | Top-1 |
|---|---:|---:|
| 全部 pair | 100.0000% | {benchmark['correct'].mean():.4%} |
| winner margin ≥10% | {len(benchmark_10) / len(benchmark):.4%} | {benchmark_10['correct'].mean():.4%} |
| winner margin ≥20% | {len(benchmark_20) / len(benchmark):.4%} | {benchmark_20['correct'].mean():.4%} |

这个 benchmark 检验姿态形状是否能排除同动作干扰，但真实多候选经常是镜像、反射或近似同步人物，通常比随机同动作干扰更难，因此不能把该 Top-1 当作真实多人选择准确率。

## 判断

- 可以为每帧候选产生可复现的视觉匹配排序，完整逐帧结果见 `multi_person_candidate_decisions.csv`。
- margin 较高的帧可以自动选人；margin 较低时，原始 Skeleton 已丢失图像位置和平移，单凭规范化姿态无法可靠区分真人、镜像或做相似动作的人。
- 第一版建议只自动接受 margin≥20% 的候选；其余标记为 ambiguous，不补造身份。若要覆盖更多帧，应增加序列级动态规划和人工抽样核验，而不是默认信任 JSON 第一项。
"""
    (report_dir / "skeleton_multi_person_identity.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "multi_frames": len(decisions), "trials": int(decisions["sample_id"].nunique()),
        "json_first_rate": float(decisions["selected_is_json_first"].mean()),
        "margin_10_rate": float(decisions["confident_10pct"].mean()),
        "margin_20_rate": float(decisions["confident_20pct"].mean()),
        "benchmark_pairs": len(benchmark), "benchmark_top1": float(benchmark["correct"].mean()),
    }, indent=2))


if __name__ == "__main__":
    main()
