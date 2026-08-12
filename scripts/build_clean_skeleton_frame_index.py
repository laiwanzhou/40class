from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(r"D:\work\2026.7.14_kaggle\datasets\Small-Model-Track\train")
FRAME_RE = re.compile(
    r"^Color_(?:(?P<timestamp>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d{3})_)?(?P<frame>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a non-destructive cleaned Skeleton frame index.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/manifest.csv")
    parser.add_argument("--fold", type=Path, default=PROJECT_ROOT / "metadata/splits/fold_0.json")
    parser.add_argument(
        "--candidate-decisions", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_multi_person_identity/multi_person_candidate_decisions.csv",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_clean_frame_index",
    )
    parser.add_argument("--margin-threshold", type=float, default=0.20)
    return parser.parse_args()


def split_name(user_id: str, fold: dict[str, list[str]]) -> str:
    if user_id in fold["train_users"]:
        return "train"
    if user_id in fold["val_users"]:
        return "validation"
    return "outside_fold"


def parse_frame(path: Path) -> tuple[int, str | None, str | None]:
    match = FRAME_RE.fullmatch(path.stem)
    if match is None:
        raise ValueError(f"Unparseable Skeleton filename: {path}")
    timestamp = match.group("timestamp")
    frame_id = int(match.group("frame"))
    frame_key = f"{timestamp}_{match.group('frame')}" if timestamp else None
    return frame_id, timestamp, frame_key


def canonical_file(paths: list[Path]) -> Path:
    """Prefer the timestamped filename when duplicate frame IDs exist."""
    return sorted(paths, key=lambda path: (FRAME_RE.fullmatch(path.stem).group("timestamp") is None, str(path)))[0]


def load_people_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return 0
    return sum(isinstance(person, dict) and "keypoints" in person for person in payload)


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def assign_retained_segments(index: pd.DataFrame) -> pd.Series:
    segments = pd.Series(pd.NA, index=index.index, dtype="Int64")
    for _, group in index[index["use_for_frame_training"]].groupby("sample_id", sort=False):
        ordered = group.sort_values("frame_id")
        segment_ids = ordered["frame_id"].diff().ne(1).cumsum().astype("Int64") - 1
        segments.loc[ordered.index] = segment_ids.to_numpy()
    return segments


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"user_id": str, "trial_id": str})
    fold = json.loads(args.fold.read_text(encoding="utf-8"))
    decisions = pd.read_csv(args.candidate_decisions, encoding="utf-8-sig")
    decision_lookup = {
        (str(row.sample_id), str(row.frame_key)): row
        for row in decisions.itertuples()
    }

    rows: list[dict[str, object]] = []
    for ordinal, trial in enumerate(manifest.itertuples(), 1):
        path_value = "" if pd.isna(trial.skeleton_path) else str(trial.skeleton_path).strip()
        if not path_value:
            continue
        trial_path = args.data_root / path_value
        paths_by_frame: dict[int, list[Path]] = {}
        for path in trial_path.rglob("*.json"):
            frame_id, _, _ = parse_frame(path)
            paths_by_frame.setdefault(frame_id, []).append(path)

        for frame_id in sorted(paths_by_frame):
            path = canonical_file(paths_by_frame[frame_id])
            _, timestamp, frame_key = parse_frame(path)
            people_count = load_people_count(path)
            decision = decision_lookup.get((str(trial.sample_id), str(frame_key))) if frame_key else None
            status = "ambiguous"
            reason = "multi_person_visual_decision_unavailable"
            candidate_index: int | None = None
            margin: float | None = None
            if people_count == 1:
                status = "retained"
                reason = "single_person"
                candidate_index = 0
            elif people_count > 1 and decision is not None:
                margin = float(decision.relative_margin)
                if bool_value(decision.confident_20pct) and margin >= args.margin_threshold:
                    candidate_index = int(decision.selected_candidate)
                    if candidate_index >= people_count:
                        raise ValueError(
                            f"Candidate index {candidate_index} exceeds {people_count} people: {path}"
                        )
                    status = "retained"
                    reason = "multi_person_visual_margin_ge_20pct"
                else:
                    reason = "multi_person_visual_margin_lt_20pct"
            elif people_count <= 0:
                reason = "no_valid_person_record"

            rows.append({
                "sample_id": trial.sample_id, "class_id": int(trial.class_id),
                "action_name": trial.action_name, "user_id": trial.user_id, "trial_id": trial.trial_id,
                "fold_split": split_name(str(trial.user_id), fold), "frame_id": frame_id,
                "timestamp": timestamp or "", "frame_key": frame_key or "",
                "skeleton_json_path": path.relative_to(args.data_root).as_posix(),
                "duplicate_json_files_for_frame": len(paths_by_frame[frame_id]),
                "people_count": people_count, "candidate_index": candidate_index,
                "selection_margin": margin, "clean_status": status, "selection_reason": reason,
                "use_for_frame_training": status == "retained",
            })
        if ordinal % 500 == 0 or ordinal == len(manifest):
            print(f"indexed {ordinal}/{len(manifest)} trials; unique frames={len(rows)}", flush=True)

    index = pd.DataFrame(rows)
    index["retained_segment_index"] = assign_retained_segments(index)
    retained = index[index["use_for_frame_training"]].copy()
    ambiguous = index[~index["use_for_frame_training"]].copy()
    retained.to_csv(report_dir / "skeleton_retained_frame_index.csv", index=False, encoding="utf-8-sig")
    ambiguous.to_csv(report_dir / "skeleton_ambiguous_frame_index.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for group_type, column in (("fold", "fold_split"), ("action", "action_name"), ("user", "user_id")):
        for name, group in index.groupby(column):
            kept = int(group["use_for_frame_training"].sum())
            summary_rows.append({
                "group_type": group_type, "group": name, "unique_frames": len(group),
                "retained_frames": kept, "ambiguous_frames": len(group) - kept,
                "retention_rate": kept / len(group),
            })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(report_dir / "skeleton_cleaning_group_summary.csv", index=False, encoding="utf-8-sig")

    trial_summary = index.groupby(
        ["sample_id", "class_id", "action_name", "user_id", "trial_id", "fold_split"], as_index=False
    ).agg(
        unique_frames=("frame_id", "size"), retained_frames=("use_for_frame_training", "sum"),
        single_person_frames=("selection_reason", lambda values: int((values == "single_person").sum())),
        confident_multi_person_frames=(
            "selection_reason", lambda values: int((values == "multi_person_visual_margin_ge_20pct").sum())
        ),
        ambiguous_frames=("clean_status", lambda values: int((values == "ambiguous").sum())),
        retained_segments=("retained_segment_index", lambda values: int(values.dropna().nunique())),
    )
    trial_summary["retention_rate"] = trial_summary["retained_frames"] / trial_summary["unique_frames"]
    trial_summary.to_csv(report_dir / "skeleton_cleaning_trial_summary.csv", index=False, encoding="utf-8-sig")

    reason_counts = index.groupby(["clean_status", "selection_reason"], as_index=False).size()
    reason_counts["rate"] = reason_counts["size"] / len(index)
    reason_counts.to_csv(report_dir / "skeleton_cleaning_reason_summary.csv", index=False, encoding="utf-8-sig")

    singleton_count = int((index["selection_reason"] == "single_person").sum())
    confident_multi_count = int(
        (index["selection_reason"] == "multi_person_visual_margin_ge_20pct").sum()
    )
    report = f"""# Skeleton 帧级清洗索引

## 决策

采用**索引式清洗**，不复制、不移动、不改写原始 JSON。两份互补索引记录规范 JSON 路径、候选人数、选中的 `candidate_index`、视觉 margin、保留状态和排除原因。

训练只读取 `skeleton_retained_frame_index.csv`；`skeleton_ambiguous_frame_index.csv` 仅用于诊断，不用于该帧训练。

## 规则

1. 单候选帧：保留，`candidate_index=0`。
2. 多候选帧且视觉匹配 `margin≥{args.margin_threshold:.0%}`：保留视觉选中的 candidate。
3. 其他多候选帧：标记 `ambiguous`，`use_for_frame_training=false`。
4. 同一 frame_id 有重复命名 JSON 时只索引一份，优先时间戳文件。

## 统计

| 项目 | 帧数 | 占全部唯一帧 |
|---|---:|---:|
| 全部唯一 Skeleton 时间步 | {len(index)} | 100.0000% |
| 保留：单候选 | {singleton_count} | {singleton_count / len(index):.4%} |
| 保留：多候选且 margin≥20% | {confident_multi_count} | {confident_multi_count / len(index):.4%} |
| 总保留 | {len(retained)} | {len(retained) / len(index):.4%} |
| ambiguous / 不用于该帧训练 | {len(ambiguous)} | {len(ambiguous) / len(index):.4%} |

ambiguous 中，{int((index['selection_reason'] == 'multi_person_visual_margin_lt_20pct').sum())} 帧是已有视觉比较但 margin 不足，{int((index['selection_reason'] == 'multi_person_visual_decision_unavailable').sum())} 帧是没有满足当前诊断条件的视觉决策。共有 {(trial_summary['retained_frames'] == 0).sum()} 个 trial 没有保留帧，全部位于 validation；train 中没有整条 trial 被清空。

注意：多候选视觉诊断限定在 fold-0 train 用户与完整跨模态配对帧。因此 validation 多候选帧以及 train 中没有可靠视觉决策的多候选帧均保守标为 ambiguous，不用 held-out validation 调阈值或补做选择。

## 使用契约

- `skeleton_retained_frame_index.csv` 与 `skeleton_ambiguous_frame_index.csv` 是全集的互补分区。
- `skeleton_json_path` 相对于 competition-train 数据根，便于迁移而不绑定绝对盘符。
- `candidate_index` 是原 JSON 顶层 person list 的零基索引；ambiguous 行为空值。
- 下游应按 `sample_id, frame_id` 排序，且只能读取 `use_for_frame_training=true` 的行。
- `retained_segment_index` 在 ambiguous 或原始缺帧位置断开；速度、窗口和插值不得跨 segment 直接连接。
- 原始 JSON 仍是唯一事实来源；该索引只表达当前版本的清洗决策。
"""
    (report_dir / "skeleton_frame_cleaning_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "unique_frames": len(index), "single_person_retained": singleton_count,
        "confident_multi_person_retained": confident_multi_count, "retained": len(retained),
        "ambiguous": len(ambiguous), "retention_rate": len(retained) / len(index),
    }, indent=2))


if __name__ == "__main__":
    main()
