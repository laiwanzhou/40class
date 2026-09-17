"""Audit A9 Top-10 hard classes with full-data class-local V/S/I evidence probes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression

from a18_full_teacher_data import A18_SOURCE_USERS, load_a18_data
from audit_p102_session_closure import load_npz, true_rank
from p103_local_feature_data import MASTER_MANIFEST
from p106_forensic_features import _imu_block, _skeleton_block
from train_p104_modality_specialists_oof import (
    PCA_COMPONENTS,
    Projection,
    cyclic_shuffle_source,
    metric_bundle,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_A9_OOF = HERE / "runs/p102_session_closure_v1/session_oof_predictions.npz"
DEFAULT_VJEPA = PROJECT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
DEFAULT_CLASSES = PROJECT / "class_mapping.csv"
DEFAULT_OUTPUT = HERE / "runs/p108_a9_top10_full_evidence_v1"
SEED = 20260824
TOP_HARD_CLASSES = 10
CONFUSER_COUNT = 3

A9_USERS = (
    "user16",
    "user17",
    "user18",
    "user19",
    "user23",
    "user5",
    "user6",
    "user7",
    "user8",
)
A9_USER_SET = frozenset(A9_USERS)
EXPECTED_TOP10 = (8, 37, 22, 26, 34, 39, 24, 7, 10, 20)

# Each fold contains three A9 discovery subjects and three non-A9 subjects.
FULL_OUTER_FOLDS = (
    ("user16", "user19", "user6", "user1", "user3", "user20"),
    ("user17", "user23", "user7", "user2", "user4", "user22"),
    ("user18", "user5", "user8", "user21", "user9", "user24"),
)

BLOCK_MODALITY = {
    "VLIT": "Visual",
    "VHPD": "Visual",
    "VWPD": "Visual",
    "SWT": "Skeleton",
    "IAPD": "IMU",
}
BLOCK_DESCRIPTION = {
    "VLIT": "手—物交互 crop 的 full/early/late/motion-peak 局部视觉",
    "VHPD": "左右手及交互 crop 的 late−early、motion-peak−full 相位差",
    "VWPD": "workspace crop 的 full/early/middle/late 与时间差分",
    "SWT": "腕/肘轨迹、速度、加速度、双手距离与手—头关系",
    "IAPD": "左右臂 IMU 的方向、强度、相位与频率",
}
CANDIDATES = (
    ("VLIT",),
    ("VHPD",),
    ("VWPD",),
    ("SWT",),
    ("IAPD",),
    ("VLIT", "SWT"),
    ("VHPD", "SWT"),
    ("VWPD", "SWT"),
    ("VLIT", "IAPD"),
    ("VHPD", "IAPD"),
    ("VWPD", "IAPD"),
    ("SWT", "IAPD"),
    ("VLIT", "SWT", "IAPD"),
    ("VHPD", "SWT", "IAPD"),
    ("VWPD", "SWT", "IAPD"),
)

# Human-readable operational checks. They interpret, but do not replace, the
# subject-disjoint quantitative probe and still require visual confirmation.
SEMANTIC_RULES = {
    7: (
        "食物或餐具反复从工作区送到口部",
        "确认存在多次进食循环；一次性小物入口优先考虑服药，主要是器具摆放则考虑 class 8。",
    ),
    8: (
        "抓取、摆放或使用碗杯餐具，交互中心在桌面器具",
        "确认没有持续圆周搅拌、容器倾倒或反复送入口部的主阶段。",
    ),
    10: (
        "餐具在相对固定的杯/碗内重复圆周运动",
        "确认运动中心位于容器内部且容器基本固定；容器本身倾斜应判倒饮料。",
    ),
    20: (
        "视线与手腕或明确计时设备发生短时定向检查",
        "确认检查对象和短时查看阶段；持续手机点击浏览应判使用手机。",
    ),
    22: (
        "手指抓取单页并完成抬起—翻转—落下",
        "确认存在离散页面状态切换；文档稳定展开且主要观看应判阅读。",
    ),
    24: (
        "手机在面前被持握并发生滑动或点击浏览",
        "确认手机—手指交互，排除贴耳通话、伸臂自拍与持续双拇指游戏输入。",
    ),
    26: (
        "双拇指或双手对手机/控制器做持续快速输入",
        "确认输入节律连续且双手协同，明显强于普通手机浏览。",
    ),
    34: (
        "身体从站立降低到坐姿，终点躯干仍大致竖直",
        "确认骨盆落到座面且最终为坐姿；相位方向与站起相反，终点也不是卧姿。",
    ),
    37: (
        "先拾取小物再短暂送入口部，常为单次剂量动作",
        "确认小物拾取—入口的因果链；若主证据是杯/瓶持续倾斜则判喝水。",
    ),
    39: (
        "温度计样小设备在腋下、口部或额头等位置持续固定",
        "确认设备—身体的稳定放置阶段，排除短暂看时间、按摩或服药。",
    ),
}


@dataclass(frozen=True)
class EvidenceBlock:
    key: str
    values: np.ndarray
    available: np.ndarray
    audit: dict[str, Any]

    def validate(self, rows: int) -> None:
        if self.key not in BLOCK_MODALITY:
            raise ValueError(f"unknown P108 block {self.key}")
        if self.values.ndim != 2 or self.values.shape[0] != rows:
            raise ValueError(f"invalid P108 {self.key} shape {self.values.shape}")
        if self.available.shape != (rows,):
            raise ValueError(f"invalid P108 {self.key} availability")
        if not np.isfinite(self.values).all():
            raise ValueError(f"non-finite P108 block {self.key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a9-oof", type=Path, default=DEFAULT_A9_OOF)
    parser.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA)
    parser.add_argument("--class-mapping", type=Path, default=DEFAULT_CLASSES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def read_class_names(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        result = {
            int(row["action_id"]): str(row["action_name"])
            for row in csv.DictReader(handle)
        }
    if set(result) != set(range(40)):
        raise RuntimeError("P108 class mapping is not 0..39")
    return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        raise RuntimeError(f"refusing to write empty P108 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def candidate_key(candidate: tuple[str, ...]) -> str:
    return "+".join(candidate)


def candidate_modalities(candidate: tuple[str, ...]) -> tuple[str, ...]:
    present = {BLOCK_MODALITY[value] for value in candidate}
    return tuple(value for value in ("Visual", "Skeleton", "IMU") if value in present)


def candidate_description(candidate: tuple[str, ...]) -> str:
    return " + ".join(BLOCK_DESCRIPTION[value] for value in candidate)


def _flatten(*values: np.ndarray) -> np.ndarray:
    rows = len(values[0])
    return np.concatenate(
        [np.asarray(value, dtype=np.float32).reshape(rows, -1) for value in values],
        axis=1,
    ).astype(np.float32, copy=False)


def _read_manifest() -> tuple[np.ndarray, np.ndarray]:
    with MASTER_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    return (
        np.asarray([row["sample_id"] for row in records], dtype=str),
        np.asarray([row["user_id"] for row in records], dtype=str),
    )


def load_visual_blocks(data: Any, root: Path) -> dict[str, EvidenceBlock]:
    summary = json.loads((root / "cache_summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete") or not summary.get("label_free_extraction"):
        raise RuntimeError("P108 V-JEPA2 cache is not complete label-free extraction")
    view_names = tuple(map(str, summary["view_names"]))
    manifest_ids, manifest_users = _read_manifest()
    lookup = {sample_id: index for index, sample_id in enumerate(manifest_ids.tolist())}
    order = np.asarray([lookup[value] for value in data.sample_ids.astype(str)], dtype=np.int64)
    if not np.array_equal(manifest_users[order], data.users):
        raise RuntimeError("P108 full V-JEPA2 user alignment differs")
    done = np.load(root / "done.npy", mmap_mode="r")
    if not np.asarray(done[order], dtype=bool).all():
        raise RuntimeError("P108 full V-JEPA2 selected rows are incomplete")

    required = (
        "global_full_workspace",
        "global_early_workspace",
        "global_middle_workspace",
        "global_late_workspace",
        "hand_full_left",
        "hand_full_right",
        "hand_full_interaction",
        "hand_early_left",
        "hand_early_right",
        "hand_early_interaction",
        "hand_late_left",
        "hand_late_right",
        "hand_late_interaction",
        "hand_motion_peak_left",
        "hand_motion_peak_right",
        "hand_motion_peak_interaction",
    )
    view_lookup = {name: index for index, name in enumerate(view_names)}
    selected_indices = np.asarray([view_lookup[name] for name in required], dtype=np.int64)
    features_memmap = np.load(root / "features.npy", mmap_mode="r")
    actions_memmap = np.load(root / "ssv2_logits.npy", mmap_mode="r")
    features = np.asarray(
        features_memmap[order[:, None], selected_indices[None, :]], dtype=np.float32
    )
    actions = np.asarray(
        actions_memmap[order[:, None], selected_indices[None, :]], dtype=np.float32
    )
    selected_lookup = {name: index for index, name in enumerate(required)}

    def view(values: np.ndarray, names: list[str]) -> np.ndarray:
        return values[:, [selected_lookup[name] for name in names]]

    interaction_names = [
        "hand_full_interaction",
        "hand_early_interaction",
        "hand_late_interaction",
        "hand_motion_peak_interaction",
    ]
    vlit = _flatten(view(features, interaction_names), view(actions, interaction_names))

    delta_feature = []
    delta_action = []
    for part in ("left", "right", "interaction"):
        late = f"hand_late_{part}"
        early = f"hand_early_{part}"
        peak = f"hand_motion_peak_{part}"
        full = f"hand_full_{part}"
        delta_feature.extend(
            [view(features, [late]) - view(features, [early]), view(features, [peak]) - view(features, [full])]
        )
        delta_action.extend(
            [view(actions, [late]) - view(actions, [early]), view(actions, [peak]) - view(actions, [full])]
        )
    vhpd = _flatten(*delta_feature, *delta_action)

    workspace_names = [
        "global_full_workspace",
        "global_early_workspace",
        "global_middle_workspace",
        "global_late_workspace",
    ]
    vwpd = _flatten(
        view(features, workspace_names),
        view(actions, workspace_names),
        view(features, ["global_late_workspace"]) - view(features, ["global_early_workspace"]),
        view(features, ["global_middle_workspace"]) - view(features, ["global_early_workspace"]),
        view(actions, ["global_late_workspace"]) - view(actions, ["global_early_workspace"]),
        view(actions, ["global_middle_workspace"]) - view(actions, ["global_early_workspace"]),
    )
    available = np.ones(len(data.sample_ids), dtype=bool)
    return {
        "VLIT": EvidenceBlock(
            "VLIT",
            vlit,
            available,
            {"source": "V-JEPA2 dense24", "views": interaction_names},
        ),
        "VHPD": EvidenceBlock(
            "VHPD",
            vhpd,
            available,
            {"source": "V-JEPA2 dense24", "evidence": BLOCK_DESCRIPTION["VHPD"]},
        ),
        "VWPD": EvidenceBlock(
            "VWPD",
            vwpd,
            available,
            {"source": "V-JEPA2 dense24", "views": workspace_names},
        ),
    }


def load_evidence_blocks(data: Any, vjepa_root: Path) -> dict[str, EvidenceBlock]:
    blocks = load_visual_blocks(data, vjepa_root)
    skeleton = _skeleton_block(data)
    imu = _imu_block(data)
    blocks["SWT"] = EvidenceBlock(
        "SWT", skeleton.aligned, skeleton.available, skeleton.audit
    )
    blocks["IAPD"] = EvidenceBlock("IAPD", imu.aligned, imu.available, imu.audit)
    if tuple(blocks) != tuple(BLOCK_MODALITY):
        raise RuntimeError(f"P108 evidence block order changed: {tuple(blocks)}")
    for block in blocks.values():
        block.validate(len(data.sample_ids))
    return blocks


def build_full_fold_ids(users: np.ndarray) -> np.ndarray:
    expected = set(A18_SOURCE_USERS)
    flattened = [user for fold in FULL_OUTER_FOLDS for user in fold]
    if len(flattened) != len(set(flattened)) or set(flattened) != expected:
        raise RuntimeError("P108 full fold user partition changed")
    fold_ids = np.full(len(users), -1, dtype=np.int64)
    for fold, held_users in enumerate(FULL_OUTER_FOLDS):
        fold_ids[np.isin(users, np.asarray(held_users, dtype=str))] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("P108 full fold assignment incomplete")
    return fold_ids


def select_confusers(
    target: int,
    labels: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    users: np.ndarray,
    selected: np.ndarray,
    count: int = CONFUSER_COUNT,
) -> tuple[list[int], list[dict[str, Any]]]:
    target_rows = np.flatnonzero(selected & (labels == target)).astype(np.int64)
    if not len(target_rows):
        raise RuntimeError(f"A9 has no class {target}")
    wrong_rows = target_rows[prediction[target_rows] != target]
    errors = Counter(map(int, prediction[wrong_rows].tolist()))
    error_users: dict[int, set[str]] = defaultdict(set)
    for row in wrong_rows.tolist():
        error_users[int(prediction[row])].add(str(users[row]))
    mean_probability = np.mean(probability[target_rows], axis=0)
    ranked = sorted(
        (class_id for class_id in range(probability.shape[1]) if class_id != target),
        key=lambda class_id: (
            -int(errors[class_id]),
            -len(error_users[class_id]),
            -float(mean_probability[class_id]),
            class_id,
        ),
    )
    return ranked[:count], [
        {
            "class_id": class_id,
            "error_rows": int(errors[class_id]),
            "error_subjects": len(error_users[class_id]),
            "mean_probability": float(mean_probability[class_id]),
            "selected": class_id in ranked[:count],
        }
        for class_id in ranked
    ]


def binary_counts(labels: np.ndarray, prediction: np.ndarray, target: int) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64) == int(target)
    predicted = np.asarray(prediction, dtype=np.int64) == int(target)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum((~truth) & predicted))
    fn = int(np.sum(truth & (~predicted)))
    tn = int(np.sum((~truth) & (~predicted)))
    return finish_binary(tp, fp, fn, tn)


def finish_binary(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def sum_binary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(records)
    return finish_binary(
        *[sum(int(record[key]) for record in values) for key in ("tp", "fp", "fn", "tn")]
    )


def fit_local(x: np.ndarray, labels: np.ndarray, classes: list[int]) -> LogisticRegression:
    actual = set(map(int, np.unique(labels).tolist()))
    if actual != set(classes):
        raise RuntimeError(f"P108 local support expected={classes}, actual={sorted(actual)}")
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    model.fit(np.asarray(x, dtype=np.float32), labels)
    return model


def projected_matrix(
    projected: dict[str, np.ndarray], candidate: tuple[str, ...], rows: np.ndarray
) -> np.ndarray:
    return np.concatenate([projected[key][rows] for key in candidate], axis=1)


def selection_key(result: dict[str, Any]) -> tuple[float, float, float, int, int]:
    candidate = tuple(result["candidate_blocks"])
    return (
        float(result["target_metrics"]["f1"]),
        float(result["family_metrics"]["balanced_accuracy"]),
        float(result["family_metrics"]["macro_f1"]),
        -len(candidate_modalities(candidate)),
        -CANDIDATES.index(candidate),
    )


def evidence_strength(
    exact_folds: int,
    modality_folds: int,
    independent_aligned_f1: float,
    independent_shuffle_f1: float,
    independent_zero_f1: float,
    worst_subject_recall: float,
) -> str:
    delta = min(
        independent_aligned_f1 - independent_shuffle_f1,
        independent_aligned_f1 - independent_zero_f1,
    )
    if (
        exact_folds >= 2
        and independent_aligned_f1 >= 0.70
        and delta >= 0.05
        and worst_subject_recall >= 0.50
    ):
        return "STRONG"
    if modality_folds >= 2 and independent_aligned_f1 >= 0.60 and delta > 0.0:
        return "MODERATE"
    return "WEAK/UNRESOLVED"


def display_classes(classes: Iterable[int], names: dict[int, str]) -> str:
    return "|".join(f"{value}:{names[value]}" for value in classes)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = read_class_names(args.class_mapping.resolve())
    a9 = load_npz(args.a9_oof.resolve())
    a9_sample_ids = a9["sample_ids"].astype(str)
    a9_users = a9["users"].astype(str)
    a9_labels = np.asarray(a9["labels"], dtype=np.int64)
    a9_probability = np.asarray(a9["selected_probability"], dtype=np.float64)
    a9_prediction = a9_probability.argmax(axis=1)
    if str(np.asarray(a9["selected_system"]).item()) != "VS_session":
        raise RuntimeError("P108 A9 discovery input is not P102 VS_session")
    a9_mask = np.isin(a9_users, np.asarray(A9_USERS, dtype=str))
    if int(a9_mask.sum()) != 1497 or set(a9_users[a9_mask].tolist()) != A9_USER_SET:
        raise RuntimeError("P108 canonical A9 nine-subject pool changed")
    a9_error = a9_mask & (a9_prediction != a9_labels)
    error_counts = Counter(map(int, a9_labels[a9_error].tolist()))
    ranked_error_classes = sorted(error_counts, key=lambda value: (-error_counts[value], value))
    hard_classes = ranked_error_classes[:TOP_HARD_CLASSES]
    if tuple(hard_classes) != EXPECTED_TOP10:
        raise RuntimeError(f"P108 A9 Top-10 changed: {hard_classes}")
    if set(hard_classes) != set(SEMANTIC_RULES):
        raise RuntimeError("P108 semantic rules do not match hard classes")
    selected_error_count = sum(error_counts[value] for value in hard_classes)

    families: dict[int, list[int]] = {}
    confuser_audit: dict[int, list[dict[str, Any]]] = {}
    discovery_rows = []
    for rank, target in enumerate(ranked_error_classes, start=1):
        confusers, audit = select_confusers(
            target,
            a9_labels,
            a9_prediction,
            a9_probability,
            a9_users,
            a9_mask,
        )
        if target in hard_classes:
            families[target] = [target, *confusers]
            confuser_audit[target] = audit
        target_rows = a9_mask & (a9_labels == target)
        target_errors = target_rows & (a9_prediction != target)
        confusion = Counter(map(int, a9_prediction[target_errors].tolist()))
        discovery_rows.append(
            {
                "error_rank": rank,
                "class_id": target,
                "class_name": names[target],
                "a9_samples": int(np.sum(target_rows)),
                "a9_top1_errors": int(error_counts[target]),
                "a9_error_rate": float(error_counts[target] / np.sum(target_rows)),
                "a9_error_confusions": "|".join(
                    f"{class_id}:{names[class_id]}({count})"
                    for class_id, count in confusion.most_common()
                ),
                "local_confusers": "|".join(map(str, confusers)),
                "local_confuser_names": display_classes(confusers, names),
                "selected_top10": int(target in hard_classes),
                "verdict": "P108_AUDIT" if target in hard_classes else "BACKLOG",
            }
        )
    write_csv(output / "a9_hard_class_discovery.csv", discovery_rows)

    data = load_a18_data()  # Full 18-source label-free tensor loader only.
    full_fold_ids = build_full_fold_ids(data.users)
    full_lookup = {sample_id: index for index, sample_id in enumerate(data.sample_ids.astype(str))}
    if not set(a9_sample_ids[a9_mask].tolist()) <= set(full_lookup):
        raise RuntimeError("P108 full source pool misses A9 rows")
    blocks = load_evidence_blocks(data, args.vjepa_root.resolve())
    sample_ids = data.sample_ids.astype(str)
    users = data.users.astype(str)
    labels = data.labels.astype(np.int64)

    class_ids = hard_classes[:2] if args.smoke else hard_classes
    outer_folds = [0] if args.smoke else list(range(3))
    candidates = CANDIDATES[:3] if args.smoke else CANDIDATES
    components = 8 if args.smoke else PCA_COMPONENTS
    n = len(labels)
    row_specialist = np.full(n, -1, dtype=np.int64)
    row_shuffle = np.full(n, -1, dtype=np.int64)
    row_zero = np.full(n, -1, dtype=np.int64)
    row_candidate = np.full(n, "", dtype=object)
    row_modalities = np.full(n, "", dtype=object)
    row_confusers = np.full(n, "", dtype=object)
    source_candidate_rows: list[dict[str, Any]] = []
    fold_selected_rows: list[dict[str, Any]] = []
    fold_details: list[dict[str, Any]] = []

    for outer_fold in outer_folds:
        source = full_fold_ids != outer_fold
        held = ~source
        source_rows = np.flatnonzero(source).astype(np.int64)
        held_rows = np.flatnonzero(held).astype(np.int64)
        inner_users = sorted(set(users[source].tolist()))
        if args.smoke:
            inner_users = inner_users[:2]
        inner_covered = np.zeros(n, dtype=bool)
        inner_prediction = np.full(
            (len(class_ids), len(candidates), n), -1, dtype=np.int16
        )
        for inner_user in inner_users:
            inner_train_all = np.flatnonzero(source & (users != inner_user)).astype(np.int64)
            inner_held = source & (users == inner_user)
            projected: dict[str, np.ndarray] = {}
            for key, block in blocks.items():
                projection = Projection.fit(block.values, inner_train_all, components)
                projected[key] = projection.transform(block.values)
            for target_index, target in enumerate(class_ids):
                classes = families[target]
                train_rows = np.flatnonzero(
                    source
                    & (users != inner_user)
                    & np.isin(labels, np.asarray(classes, dtype=np.int64))
                ).astype(np.int64)
                validation_rows = np.flatnonzero(
                    inner_held & np.isin(labels, np.asarray(classes, dtype=np.int64))
                ).astype(np.int64)
                if not len(validation_rows):
                    continue
                for candidate_index, candidate in enumerate(candidates):
                    model = fit_local(
                        projected_matrix(projected, candidate, train_rows),
                        labels[train_rows],
                        classes,
                    )
                    inner_prediction[target_index, candidate_index, validation_rows] = (
                        model.predict(projected_matrix(projected, candidate, validation_rows))
                    )
            inner_covered |= inner_held

        selected_by_target: dict[int, dict[str, Any]] = {}
        for target_index, target in enumerate(class_ids):
            classes = families[target]
            source_family_rows = np.flatnonzero(
                inner_covered & np.isin(labels, np.asarray(classes, dtype=np.int64))
            ).astype(np.int64)
            candidate_results = []
            for candidate_index, candidate in enumerate(candidates):
                predicted = inner_prediction[target_index, candidate_index, source_family_rows]
                if np.any(predicted < 0):
                    raise RuntimeError(
                        f"P108 inner coverage failed fold={outer_fold} class={target} candidate={candidate_key(candidate)}"
                    )
                family_metrics = metric_bundle(
                    labels[source_family_rows], predicted, users[source_family_rows], classes
                )
                target_metrics = binary_counts(labels[source_family_rows], predicted, target)
                result = {
                    "outer_fold": outer_fold,
                    "target_class": target,
                    "family_classes": classes,
                    "candidate": candidate_key(candidate),
                    "candidate_blocks": list(candidate),
                    "modalities": "+".join(candidate_modalities(candidate)),
                    "source_rows": len(source_family_rows),
                    "target_metrics": target_metrics,
                    "family_metrics": family_metrics,
                }
                candidate_results.append(result)
                source_candidate_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "target_class": target,
                        "target_name": names[target],
                        "family_classes": "|".join(map(str, classes)),
                        "candidate": candidate_key(candidate),
                        "modalities": result["modalities"],
                        "source_inner_rows": len(source_family_rows),
                        "target_precision": target_metrics["precision"],
                        "target_recall": target_metrics["recall"],
                        "target_f1": target_metrics["f1"],
                        "family_accuracy": family_metrics["accuracy"],
                        "family_balanced_accuracy": family_metrics["balanced_accuracy"],
                        "family_macro_f1": family_metrics["macro_f1"],
                    }
                )
            selected_by_target[target] = max(candidate_results, key=selection_key)

        shuffle_map = cyclic_shuffle_source(sample_ids, users, held)
        projected_source: dict[str, np.ndarray] = {}
        projected_held: dict[str, dict[str, np.ndarray]] = {}
        projection_audit = {}
        for key, block in blocks.items():
            projection = Projection.fit(block.values, source_rows, components)
            projected_source[key] = projection.transform(block.values[source_rows])
            projected_held[key] = {
                "aligned": projection.transform(block.values[held_rows]),
                "shuffle": projection.transform(block.values[shuffle_map[held_rows]]),
                "zero": projection.zero(len(held_rows)),
            }
            projection_audit[key] = {
                "components": projection.components,
                "explained_variance": float(np.sum(projection.pca.explained_variance_ratio_)),
                "available_source": int(np.sum(block.available[source_rows])),
                "available_held": int(np.sum(block.available[held_rows])),
            }

        source_labels = labels[source_rows]
        held_labels = labels[held_rows]
        held_users = users[held_rows]
        fold_target_details = []
        for target in class_ids:
            selected = selected_by_target[target]
            candidate = tuple(selected["candidate_blocks"])
            classes = families[target]
            source_family = np.isin(source_labels, np.asarray(classes, dtype=np.int64))
            held_family = np.isin(held_labels, np.asarray(classes, dtype=np.int64))
            model = fit_local(
                projected_matrix(projected_source, candidate, np.flatnonzero(source_family)),
                source_labels[source_family],
                classes,
            )
            family_positions = np.flatnonzero(held_family).astype(np.int64)
            family_labels = held_labels[held_family]
            family_users = held_users[held_family]
            variant_predictions = {
                variant: model.predict(
                    projected_matrix(
                        {key: projected_held[key][variant] for key in blocks},
                        candidate,
                        family_positions,
                    )
                ).astype(np.int64)
                for variant in ("aligned", "shuffle", "zero")
            }
            independent = ~np.isin(family_users, np.asarray(A9_USERS, dtype=str))
            variant_results = {}
            for variant, prediction in variant_predictions.items():
                variant_results[variant] = {
                    "target": binary_counts(family_labels, prediction, target),
                    "family": metric_bundle(family_labels, prediction, family_users, classes),
                    "independent_target": binary_counts(
                        family_labels[independent], prediction[independent], target
                    ),
                    "independent_family": metric_bundle(
                        family_labels[independent],
                        prediction[independent],
                        family_users[independent],
                        classes,
                    ),
                }
            target_in_family = family_labels == target
            actual_target_rows = held_rows[family_positions[target_in_family]]
            row_specialist[actual_target_rows] = variant_predictions["aligned"][target_in_family]
            row_shuffle[actual_target_rows] = variant_predictions["shuffle"][target_in_family]
            row_zero[actual_target_rows] = variant_predictions["zero"][target_in_family]
            row_candidate[actual_target_rows] = candidate_key(candidate)
            row_modalities[actual_target_rows] = "+".join(candidate_modalities(candidate))
            row_confusers[actual_target_rows] = "|".join(map(str, classes[1:]))
            record = {
                "outer_fold": outer_fold,
                "held_users": list(FULL_OUTER_FOLDS[outer_fold]),
                "target_class": target,
                "target_name": names[target],
                "family_classes": classes,
                "confusers": classes[1:],
                "candidate": candidate_key(candidate),
                "candidate_blocks": list(candidate),
                "modalities": "+".join(candidate_modalities(candidate)),
                "source_selection": selected,
                "held_family_rows": int(np.sum(held_family)),
                "held_target_rows": int(np.sum(target_in_family)),
                "held_independent_family_rows": int(np.sum(independent)),
                "variants": variant_results,
            }
            fold_target_details.append(record)
            fold_selected_rows.append(
                {
                    "outer_fold": outer_fold,
                    "held_users": "|".join(FULL_OUTER_FOLDS[outer_fold]),
                    "target_class": target,
                    "target_name": names[target],
                    "confusers": "|".join(map(str, classes[1:])),
                    "confuser_names": display_classes(classes[1:], names),
                    "candidate": candidate_key(candidate),
                    "modalities": record["modalities"],
                    "source_target_f1": selected["target_metrics"]["f1"],
                    "source_family_balanced_accuracy": selected["family_metrics"]["balanced_accuracy"],
                    "held_target_f1": variant_results["aligned"]["target"]["f1"],
                    "held_shuffle_target_f1": variant_results["shuffle"]["target"]["f1"],
                    "held_zero_target_f1": variant_results["zero"]["target"]["f1"],
                    "independent_target_f1": variant_results["aligned"]["independent_target"]["f1"],
                    "independent_shuffle_target_f1": variant_results["shuffle"]["independent_target"]["f1"],
                    "independent_zero_target_f1": variant_results["zero"]["independent_target"]["f1"],
                }
            )
        fold_details.append(
            {
                "outer_fold": outer_fold,
                "source_users": sorted(set(users[source].tolist())),
                "held_users": list(FULL_OUTER_FOLDS[outer_fold]),
                "source_rows": len(source_rows),
                "held_rows": len(held_rows),
                "projection_audit": projection_audit,
                "targets": fold_target_details,
            }
        )

    write_csv(output / "source_candidate_metrics.csv", source_candidate_rows)
    write_csv(output / "fold_selected_evidence.csv", fold_selected_rows)
    (output / "fold_details.json").write_text(
        json.dumps(fold_details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.smoke:
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "status": "smoke_complete",
                    "classes": class_ids,
                    "folds": outer_folds,
                    "candidates": [candidate_key(value) for value in candidates],
                    "a18_model_or_prediction_loaded": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"status": "smoke_complete", "output": str(output)}))
        return

    hard_sample_mask = np.isin(labels, np.asarray(hard_classes, dtype=np.int64))
    if (
        np.any(row_specialist[hard_sample_mask] < 0)
        or np.any(row_shuffle[hard_sample_mask] < 0)
        or np.any(row_zero[hard_sample_mask] < 0)
    ):
        raise RuntimeError("P108 full hard-class OOF coverage incomplete")

    records_by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fold in fold_details:
        for record in fold["targets"]:
            records_by_target[int(record["target_class"])].append(record)
    a9_to_full = {
        row: full_lookup[a9_sample_ids[row]] for row in np.flatnonzero(a9_mask).tolist()
    }
    class_rows = []
    class_summary: dict[int, dict[str, Any]] = {}
    for target in hard_classes:
        records = records_by_target[target]
        exact_counter = Counter(record["candidate"] for record in records)
        modality_counter = Counter(record["modalities"] for record in records)
        maximum_exact = max(exact_counter.values())
        primary_exact = min(
            (key for key, count in exact_counter.items() if count == maximum_exact),
            key=lambda key: [candidate_key(value) for value in CANDIDATES].index(key),
        )
        maximum_modality = max(modality_counter.values())
        dominant_modality = min(
            (key for key, count in modality_counter.items() if count == maximum_modality),
            key=lambda key: (len(key.split("+")), key),
        )
        primary_tuple: tuple[str, ...] | None
        if maximum_exact >= 2:
            primary_tuple = next(
                value for value in CANDIDATES if candidate_key(value) == primary_exact
            )
            primary_evidence = primary_exact
            primary_modalities = "+".join(candidate_modalities(primary_tuple))
            primary_description = candidate_description(primary_tuple)
            evidence_consensus = "EXACT_BLOCK_CONSENSUS"
        elif maximum_modality >= 2:
            primary_tuple = None
            primary_evidence = f"{dominant_modality} (exact visual block unstable)"
            primary_modalities = dominant_modality
            primary_description = (
                f"两个 folds 同意 {dominant_modality}，但具体 visual block/组合不一致；"
                "只能下模态级结论。"
            )
            evidence_consensus = "MODALITY_ONLY_CONSENSUS"
        else:
            primary_tuple = None
            primary_evidence = "NO_FOLD_CONSENSUS"
            primary_modalities = "NO_FOLD_CONSENSUS"
            primary_description = "三个 outer folds 选择了不同模态组合，不能指定单一主证据。"
            evidence_consensus = "NO_CONSENSUS"
        full_variant = {
            variant: sum_binary(record["variants"][variant]["target"] for record in records)
            for variant in ("aligned", "shuffle", "zero")
        }
        independent_variant = {
            variant: sum_binary(
                record["variants"][variant]["independent_target"] for record in records
            )
            for variant in ("aligned", "shuffle", "zero")
        }
        target_rows = labels == target
        subject_recall = {
            user: float(np.mean(row_specialist[target_rows & (users == user)] == target))
            for user in sorted(set(users[target_rows].tolist()))
        }
        worst_subject, worst_recall = min(
            subject_recall.items(), key=lambda item: (item[1], item[0])
        )
        strength = evidence_strength(
            maximum_exact,
            maximum_modality,
            independent_variant["aligned"]["f1"],
            independent_variant["shuffle"]["f1"],
            independent_variant["zero"]["f1"],
            worst_recall,
        )
        a9_target_rows = np.flatnonzero(a9_mask & (a9_labels == target)).astype(np.int64)
        full_rows = np.asarray([a9_to_full[int(row)] for row in a9_target_rows], dtype=np.int64)
        a9_correct = a9_prediction[a9_target_rows] == target
        specialist_correct = row_specialist[full_rows] == target
        rescue = int(np.sum((~a9_correct) & specialist_correct))
        harm = int(np.sum(a9_correct & (~specialist_correct)))
        confusion = Counter(map(int, a9_prediction[a9_target_rows[~a9_correct]].tolist()))
        primary_cue, confirmation_rule = SEMANTIC_RULES[target]
        confusers = families[target][1:]
        if strength == "WEAK/UNRESOLVED":
            verdict = "当前 V/S/I 无稳定 sample-specific 主证据；保留人工确认规则，不进入 specialist。"
        else:
            verdict = f"优先检查 {primary_description}，再按语义规则排除主要混淆。"
        record = {
            "class_id": target,
            "class_name": names[target],
            "a9_error_rank": ranked_error_classes.index(target) + 1,
            "a9_sample_count": len(a9_target_rows),
            "a9_top1_errors": int(error_counts[target]),
            "a9_error_confusions": "|".join(
                f"{class_id}:{names[class_id]}({count})"
                for class_id, count in confusion.most_common()
            ),
            "local_confusers": "|".join(map(str, confusers)),
            "local_confuser_names": display_classes(confusers, names),
            "full_18_sample_count": int(np.sum(target_rows)),
            "primary_evidence": primary_evidence,
            "primary_modalities": primary_modalities,
            "primary_evidence_description": primary_description,
            "evidence_consensus": evidence_consensus,
            "exact_selection_folds": maximum_exact,
            "modality_selection_folds": maximum_modality,
            "all_fold_selections": "|".join(
                f"f{record['outer_fold']}:{record['candidate']}" for record in records
            ),
            "full_oof_target_precision": full_variant["aligned"]["precision"],
            "full_oof_target_recall": full_variant["aligned"]["recall"],
            "full_oof_target_f1": full_variant["aligned"]["f1"],
            "non_a9_target_precision": independent_variant["aligned"]["precision"],
            "non_a9_target_recall": independent_variant["aligned"]["recall"],
            "non_a9_target_f1": independent_variant["aligned"]["f1"],
            "non_a9_shuffle_f1": independent_variant["shuffle"]["f1"],
            "non_a9_zero_f1": independent_variant["zero"]["f1"],
            "aligned_minus_shuffle_f1": independent_variant["aligned"]["f1"]
            - independent_variant["shuffle"]["f1"],
            "aligned_minus_zero_f1": independent_variant["aligned"]["f1"]
            - independent_variant["zero"]["f1"],
            "worst_subject": worst_subject,
            "worst_subject_recall": worst_recall,
            "a9_error_rescue": rescue,
            "a9_correct_harm": harm,
            "a9_oracle_class_net": rescue - harm,
            "evidence_strength": strength,
            "semantic_primary_cue": primary_cue,
            "final_confirmation_rule": confirmation_rule,
            "semantic_rule_status": "needs visual confirmation",
            "verdict": verdict,
        }
        class_rows.append(record)
        class_summary[target] = record
    write_csv(output / "class_evidence_atlas.csv", class_rows)

    full_sample_rows = []
    for row in np.flatnonzero(hard_sample_mask).tolist():
        target = int(labels[row])
        full_sample_rows.append(
            {
                "sample_id": sample_ids[row],
                "subject": users[row],
                "discovery_group": "A9" if users[row] in A9_USER_SET else "NON_A9_CONFIRMATION",
                "outer_fold": int(full_fold_ids[row]),
                "true_class": target,
                "true_name": names[target],
                "local_confusers": row_confusers[row],
                "recommended_evidence": row_candidate[row],
                "recommended_modalities": row_modalities[row],
                "specialist_prediction": int(row_specialist[row]),
                "specialist_prediction_name": names[int(row_specialist[row])],
                "aligned_correct": int(row_specialist[row] == target),
                "shuffle_prediction": int(row_shuffle[row]),
                "shuffle_correct": int(row_shuffle[row] == target),
                "zero_prediction": int(row_zero[row]),
                "zero_correct": int(row_zero[row] == target),
                "class_evidence_strength": class_summary[target]["evidence_strength"],
                "semantic_primary_cue": class_summary[target]["semantic_primary_cue"],
                "final_confirmation_rule": class_summary[target]["final_confirmation_rule"],
            }
        )
    write_csv(output / "full_hard_class_samples.csv", full_sample_rows)

    ranks = true_rank(a9_probability, a9_labels)
    top5 = np.argsort(a9_probability, axis=1)[:, ::-1][:, :5]
    a9_selected_errors = np.flatnonzero(a9_error & np.isin(a9_labels, hard_classes)).astype(np.int64)
    a9_error_rows = []
    for row in a9_selected_errors.tolist():
        full_row = full_lookup[a9_sample_ids[row]]
        target = int(a9_labels[row])
        aligned_correct = int(row_specialist[full_row] == target)
        shuffle_correct = int(row_shuffle[full_row] == target)
        zero_correct = int(row_zero[full_row] == target)
        if aligned_correct and not shuffle_correct and not zero_correct:
            resolution = "ALIGNED_ONLY_RESCUE"
        elif aligned_correct:
            resolution = "RESCUED_NOT_SAMPLE_SPECIFIC"
        else:
            resolution = "UNRESOLVED"
        a9_error_rows.append(
            {
                "sample_id": a9_sample_ids[row],
                "subject": a9_users[row],
                "true_class": target,
                "true_name": names[target],
                "a_prediction": int(a9_prediction[row]),
                "a_prediction_name": names[int(a9_prediction[row])],
                "a_confidence": float(np.max(a9_probability[row])),
                "a_true_probability": float(a9_probability[row, target]),
                "a_true_rank": int(ranks[row]),
                "a_top5": "|".join(map(str, top5[row].tolist())),
                "a_top5_names": display_classes(top5[row].tolist(), names),
                "true_in_top5": int(ranks[row] <= 5),
                "local_confusers": row_confusers[full_row],
                "recommended_evidence": row_candidate[full_row],
                "recommended_modalities": row_modalities[full_row],
                "specialist_prediction": int(row_specialist[full_row]),
                "specialist_prediction_name": names[int(row_specialist[full_row])],
                "shuffle_prediction": int(row_shuffle[full_row]),
                "zero_prediction": int(row_zero[full_row]),
                "resolution_status": resolution,
                "class_evidence_strength": class_summary[target]["evidence_strength"],
                "semantic_primary_cue": class_summary[target]["semantic_primary_cue"],
                "final_confirmation_rule": class_summary[target]["final_confirmation_rule"],
            }
        )
    write_csv(output / "a9_top10_error_samples.csv", a9_error_rows)

    full_correct = row_specialist[hard_sample_mask] == labels[hard_sample_mask]
    non_a9_hard = hard_sample_mask & (~np.isin(users, np.asarray(A9_USERS, dtype=str)))
    a9_error_resolution = Counter(row["resolution_status"] for row in a9_error_rows)
    total_rescue = sum(int(row["a9_error_rescue"]) for row in class_rows)
    total_harm = sum(int(row["a9_correct_harm"]) for row in class_rows)
    summary = {
        "status": "complete",
        "protocol": "A9 Top-10 discovery then full-18 subject-disjoint class-local evidence audit",
        "a9_discovery": {
            "users": list(A9_USERS),
            "rows": int(np.sum(a9_mask)),
            "correct": int(np.sum(a9_mask & (a9_prediction == a9_labels))),
            "accuracy": float(np.mean(a9_prediction[a9_mask] == a9_labels[a9_mask])),
            "top1_errors": int(np.sum(a9_error)),
            "selected_hard_classes": hard_classes,
            "selected_error_count": selected_error_count,
            "selected_error_coverage": float(selected_error_count / np.sum(a9_error)),
            "backlog_error_classes": ranked_error_classes[TOP_HARD_CLASSES:],
        },
        "full_18_capability_oof": {
            "rows_all": n,
            "hard_class_rows": int(np.sum(hard_sample_mask)),
            "hard_class_correct": int(np.sum(full_correct)),
            "hard_class_accuracy": float(np.mean(full_correct)),
            "non_a9_hard_rows": int(np.sum(non_a9_hard)),
            "non_a9_hard_correct": int(np.sum(row_specialist[non_a9_hard] == labels[non_a9_hard])),
            "non_a9_hard_accuracy": float(np.mean(row_specialist[non_a9_hard] == labels[non_a9_hard])),
            "scope": "true-class-selected capability audit, not a deployable router/system accuracy",
        },
        "a9_top10_oracle_class_probe": {
            "error_rows": len(a9_error_rows),
            "rescue": total_rescue,
            "harm_on_a9_correct_same_classes": total_harm,
            "net": total_rescue - total_harm,
            "resolution_status": dict(a9_error_resolution),
        },
        "evidence_strength": dict(Counter(row["evidence_strength"] for row in class_rows)),
        "primary_modalities": dict(Counter(row["primary_modalities"] for row in class_rows)),
        "a18_model_or_prediction_loaded": False,
        "old_eight_confusion_families_used_for_selection": False,
        "no_unified_b": True,
        "no_40_class_model": True,
        "negative_results_preserved": True,
        "artifacts": {
            "discovery": "a9_hard_class_discovery.csv",
            "class_atlas": "class_evidence_atlas.csv",
            "full_samples": "full_hard_class_samples.csv",
            "a9_errors": "a9_top10_error_samples.csv",
            "fold_selection": "fold_selected_evidence.csv",
            "source_candidates": "source_candidate_metrics.csv",
            "fold_details": "fold_details.json",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
