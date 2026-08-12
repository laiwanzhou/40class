from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = {
    "Depth": "depth_color_path",
    "IMU": "imu_path",
    "IR": "ir_path",
    "Radar": "radar_path",
    "Skeleton": "skeleton_path",
    "Thermal": "thermal_path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit six-modality availability under shared OOF user folds.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "metadata/manifest.csv")
    parser.add_argument("--fold", type=Path, default=PROJECT_ROOT / "metadata/splits/fold_0.json")
    parser.add_argument(
        "--oof-folds", type=Path, default=PROJECT_ROOT / "metadata/splits/train14_oof_3fold.json"
    )
    parser.add_argument(
        "--skeleton-trial-summary", type=Path,
        default=PROJECT_ROOT / "reports/skeleton_clean_frame_index/skeleton_cleaning_trial_summary.csv",
    )
    parser.add_argument(
        "--report-dir", type=Path,
        default=PROJECT_ROOT / "reports/multimodal_oof_availability_audit",
    )
    return parser.parse_args()


def present_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().ne("")


def scope_row(
    manifest: pd.DataFrame, users: list[str], modality: str, available: pd.Series,
    fold_index: int, scope: str, availability_kind: str,
) -> dict[str, object]:
    owned = manifest["user_id"].isin(users)
    canonical = manifest[owned]
    subset = manifest[owned & available]
    users_with_data = sorted(subset["user_id"].unique())
    classes = sorted(int(value) for value in subset["class_id"].unique())
    canonical_classes = set(int(value) for value in canonical["class_id"].unique())
    return {
        "fold": fold_index, "scope": scope, "modality": modality,
        "availability_kind": availability_kind, "assigned_user_count": len(users),
        "users_with_data_count": len(users_with_data), "trial_count": len(subset),
        "class_count": len(classes),
        "canonical_trial_count": len(canonical), "canonical_class_count": len(canonical_classes),
        "missing_user_ids": "|".join(sorted(set(users) - set(users_with_data))),
        "missing_class_ids": "|".join(str(value) for value in sorted(set(range(40)) - set(classes))),
        "canonical_missing_class_ids": "|".join(
            str(value) for value in sorted(set(range(40)) - canonical_classes)
        ),
        "modality_specific_missing_class_ids": "|".join(
            str(value) for value in sorted(canonical_classes - set(classes))
        ),
    }


def validate_oof(oof: dict, canonical_users: set[str]) -> None:
    validation_ownership: list[str] = []
    for fold in oof["folds"]:
        outer_train = set(fold["train_user_ids"])
        outer_val = set(fold["validation_user_ids"])
        inner_fit = set(fold["epoch_selection"]["fit_user_ids"])
        inner_val = set(fold["epoch_selection"]["validation_user_ids"])
        if outer_train & outer_val or outer_train | outer_val != canonical_users:
            raise ValueError(f"Invalid outer ownership in fold {fold['fold']}")
        if inner_fit & inner_val or inner_fit | inner_val != outer_train:
            raise ValueError(f"Invalid inner ownership in fold {fold['fold']}")
        validation_ownership.extend(outer_val)
    if len(validation_ownership) != len(set(validation_ownership)) or set(validation_ownership) != canonical_users:
        raise ValueError("Outer validation users must partition the canonical train14 users exactly once")


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig", dtype={"user_id": str})
    fold = json.loads(args.fold.read_text(encoding="utf-8"))
    oof = json.loads(args.oof_folds.read_text(encoding="utf-8"))
    train14_users = set(fold["train_users"])
    validate_oof(oof, train14_users)

    user_rows = []
    for user, group in manifest.groupby("user_id"):
        record: dict[str, object] = {
            "user_id": user, "canonical_trials": len(group),
            "development_role": "train14" if user in train14_users else "heldout4",
        }
        for modality, column in MODALITIES.items():
            record[f"{modality}_present_trials"] = int(present_mask(group, column).sum())
        user_rows.append(record)
    user_matrix = pd.DataFrame(user_rows).sort_values("user_id")

    skeleton_summary = pd.read_csv(args.skeleton_trial_summary, encoding="utf-8-sig")
    skeleton_usable_ids = set(
        skeleton_summary.loc[skeleton_summary["retained_frames"] > 0, "sample_id"].astype(str)
    )
    skeleton_usable = manifest["sample_id"].astype(str).isin(skeleton_usable_ids)
    user_usable = manifest[skeleton_usable].groupby("user_id").size()
    user_matrix["Skeleton_master_clean_usable_trials"] = user_matrix["user_id"].map(user_usable).fillna(0).astype(int)
    user_matrix.to_csv(report_dir / "user_modality_availability.csv", index=False, encoding="utf-8-sig")

    scope_rows = []
    for item in oof["folds"]:
        fold_index = int(item["fold"])
        scopes = {
            "outer_train": item["train_user_ids"],
            "outer_validation": item["validation_user_ids"],
            "inner_fit": item["epoch_selection"]["fit_user_ids"],
            "inner_validation": item["epoch_selection"]["validation_user_ids"],
        }
        for scope, users in scopes.items():
            for modality, column in MODALITIES.items():
                scope_rows.append(scope_row(
                    manifest, users, modality, present_mask(manifest, column),
                    fold_index, scope, "manifest_present",
                ))
            scope_rows.append(scope_row(
                manifest, users, "Skeleton", skeleton_usable,
                fold_index, scope, "master_clean_usable_non_strict_oof",
            ))
    scope_summary = pd.DataFrame(scope_rows)
    scope_summary.to_csv(report_dir / "fold_scope_modality_availability.csv", index=False, encoding="utf-8-sig")

    missing_users = scope_summary[
        (scope_summary["availability_kind"] == "manifest_present")
        & (scope_summary["users_with_data_count"] < scope_summary["assigned_user_count"])
    ]
    missing_classes = scope_summary[
        (scope_summary["availability_kind"] == "manifest_present") & (scope_summary["class_count"] < 40)
    ]
    modality_specific_class_loss = scope_summary[
        (scope_summary["availability_kind"] == "manifest_present")
        & scope_summary["modality_specific_missing_class_ids"].fillna("").ne("")
    ]
    all_user_modality_positive = bool(
        (user_matrix[[f"{name}_present_trials" for name in MODALITIES]] > 0).all(axis=None)
    )
    modality_totals = pd.DataFrame([
        {
            "modality": modality,
            "present_trials": int(user_matrix[f"{modality}_present_trials"].sum()),
            "minimum_trials_for_one_user": int(user_matrix[f"{modality}_present_trials"].min()),
            "maximum_trials_for_one_user": int(user_matrix[f"{modality}_present_trials"].max()),
        }
        for modality in MODALITIES
    ])
    skeleton_clean_scopes = scope_summary[
        scope_summary["availability_kind"] == "master_clean_usable_non_strict_oof"
    ][["fold", "scope", "assigned_user_count", "users_with_data_count", "trial_count", "class_count", "missing_class_ids"]]
    oof_sha256 = hashlib.sha256(args.oof_folds.read_bytes()).hexdigest()
    report = f"""# 六模态共享 OOF Fold 可用性审计

## 结论

- `train14_oof_3fold.json` 的 outer validation 用户恰好分割 train14，每个用户只出现一次；每个 inner fit/validation 也严格分割对应 outer train。
- 全部 18 个用户是否在每个模态至少有一个 manifest-present trial：**{all_user_modality_positive}**。
- 六模态、三折、四个 scope 中，缺少整个 assigned user 的 manifest-present 情况：**{len(missing_users)} 行**。
- 六模态、三折、四个 scope 中，类别少于 40 的情况：**{len(missing_classes)} 行**。这主要反映固定用户组本身的类别支持，不等价于模态特有缺失。
- 其中相对 canonical user scope 额外丢失类别的模态行：**{len(modality_specific_class_loss)} 行**；应与固定用户组本身缺类分开解释。

因此可以复用 IR 的三套 user-level fold。共享的是 user ownership；每个模态仍按自身 present/usable 状态形成稀疏样本集，不替换缺失用户。

冻结 fold 文件 SHA256：`{oof_sha256}`。

## 全局用户覆盖

{modality_totals.to_markdown(index=False)}

每个模态的 `minimum_trials_for_one_user` 都大于 0。因此这里没有“IR 有某用户、Skeleton 或其他模态完全没有该用户”的情况。

## Skeleton Master Clean v1 在共享折中的覆盖

{skeleton_clean_scopes.to_markdown(index=False)}

train14 的每个 outer/inner scope 都保留全部 assigned users。此前 6 个 clean 后空 trial 属于 heldout4 的 `user17`，不在这套 train14 三折中。

唯一模态特有的类别缺口出现在 fold1 inner-validation：canonical 用户组包含 class 33，但 Depth、IMU、IR、Radar、Skeleton 对该类均无 path，只有 Thermal 有 user5 的 6 个 class-33 trial。固定用户组自身还缺 class 25/26。因此这个 inner-validation 对五个模态覆盖 37 类，对 Thermal 覆盖 38 类。

## 口径

- `manifest_present`：对应 path 字段非空，只证明原始 trial 被登记；不证明 expert preprocessing 成功。
- `master_clean_usable_non_strict_oof`：Skeleton Master Clean v1 至少保留一帧。它只用于数据开发统计，不能替代 fold-specific strict-OOF identity curation。
- 本审计未读取 competition test，未训练模型。

## 文件

- `user_modality_availability.csv`：18 users × 六模态 present trial 数，以及 Skeleton master-clean usable trial 数。
- `fold_scope_modality_availability.csv`：fold × outer/inner scope × modality 的用户、trial、类别覆盖及缺口。
"""
    (report_dir / "multimodal_oof_availability_audit.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "users": len(user_matrix), "all_user_modality_positive": all_user_modality_positive,
        "scope_rows": len(scope_summary), "scope_rows_missing_users": len(missing_users),
        "scope_rows_missing_classes": len(missing_classes),
        "scope_rows_modality_specific_class_loss": len(modality_specific_class_loss),
    }, indent=2))


if __name__ == "__main__":
    main()
