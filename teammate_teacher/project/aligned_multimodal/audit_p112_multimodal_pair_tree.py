"""P112 source-safe multimodal pair tree over the frozen P89 Top-2 structure.

This is a tree of pair specialists, not a 40-class replacement model.  Four
label-free modality descriptors are reduced without labels, then pair-specific
logistic heads are cross-fitted by source user.  Modality and call/abstain are
selected independently for each pair using source rows only.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from a18_full_teacher_data import load_a18_data
from p90_crossuser_visual_router import load_splits


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_VJEPA = PROJECT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
DEFAULT_OUTPUT = HERE / "runs/p112_multimodal_pair_tree_v1"
SEED = 20260824
COMPONENTS = 128
OUTER_USERS = {
    "H1_selection": ("user6", "user8", "user17", "user23"),
    "H2_confirmation": ("user5", "user7", "user16", "user18", "user19"),
    "H3_independent_fold0": ("user20", "user22", "user24", "user3", "user4", "user9"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vjepa-root", type=Path, default=DEFAULT_VJEPA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("all", "descriptors", "route"), default="all")
    parser.add_argument("--components", type=int, default=COMPONENTS)
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    records = list(rows)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def masked_temporal_stats(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if mask is None:
        active = np.ones(source.shape, dtype=np.float32)
    else:
        active = np.asarray(mask, dtype=np.float32)
        while active.ndim < source.ndim:
            active = active[..., None]
        active = np.broadcast_to(active, source.shape).astype(np.float32)
    count = np.maximum(active.sum(axis=1), 1.0)
    mean = (source * active).sum(axis=1) / count
    variance = (np.square(source - mean[:, None]) * active).sum(axis=1) / count
    first = source[:, 0]
    last = source[:, -1]
    return np.concatenate(
        [mean.reshape(len(source), -1), np.sqrt(np.maximum(variance, 1e-8)).reshape(len(source), -1),
         first.reshape(len(source), -1), last.reshape(len(source), -1),
         (last - first).reshape(len(source), -1)],
        axis=1,
    ).astype(np.float32)


def reduce_unlabelled(values: np.ndarray, components: int) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.asarray(values, dtype=np.float32)
    scaler = StandardScaler(copy=False)
    standardized = scaler.fit_transform(source)
    actual = min(int(components), len(source) - 1, source.shape[1])
    pca = PCA(
        n_components=actual,
        svd_solver="randomized",
        iterated_power=2,
        random_state=SEED,
    )
    reduced = pca.fit_transform(standardized).astype(np.float32)
    reduced /= np.maximum(np.linalg.norm(reduced, axis=1, keepdims=True), 1e-6)
    return reduced, {
        "input_dim": int(source.shape[1]),
        "output_dim": int(reduced.shape[1]),
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "label_free_fit_rows": int(len(source)),
    }


def build_descriptors(vjepa_root: Path, output: Path, components: int) -> dict[str, Any]:
    data = load_a18_data()
    if len(data.sample_ids) != 2914:
        raise RuntimeError("P112 A18 inventory changed")
    done = np.load(vjepa_root / "done.npy", mmap_mode="r")
    if not np.asarray(done, dtype=bool).all():
        raise RuntimeError("P112 V-JEPA cache incomplete")
    vjepa_features = np.load(vjepa_root / "features.npy", mmap_mode="r")
    vjepa_actions = np.load(vjepa_root / "ssv2_logits.npy", mmap_mode="r")
    manifest_ids = _manifest_sample_ids(HERE / "data/manifest.csv")
    order = _align(manifest_ids, data.sample_ids)
    local_indices = np.asarray([2, 5, 8, 11, *range(12, 24)], dtype=np.int64)
    local_raw = np.concatenate(
        [
            np.asarray(vjepa_features[order[:, None], local_indices[None]], dtype=np.float32).reshape(len(order), -1),
            np.asarray(vjepa_actions[order[:, None], local_indices[None]], dtype=np.float32).reshape(len(order), -1),
        ],
        axis=1,
    )
    global_raw = np.concatenate(
        [
            data.visual_vmae.reshape(len(data.sample_ids), -1),
            data.visual_iv2.reshape(len(data.sample_ids), -1),
            data.visual_vmae_action.reshape(len(data.sample_ids), -1),
            data.visual_iv2_action.reshape(len(data.sample_ids), -1),
        ],
        axis=1,
    ).astype(np.float32)
    skeleton_raw = np.concatenate(
        [
            masked_temporal_stats(data.skeleton_motionbert),
            masked_temporal_stats(data.skeleton_hdgcn.reshape(len(data.sample_ids), 6, -1)),
            masked_temporal_stats(data.skeleton_sequence, data.skeleton_mask),
            data.skeleton_statistics,
        ],
        axis=1,
    ).astype(np.float32)
    imu_mask = np.broadcast_to(
        data.imu_mask[..., None], data.imu_sequence.shape
    ).reshape(len(data.sample_ids), 32, -1)
    imu_raw = np.concatenate(
        [
            masked_temporal_stats(
                data.imu_sequence.reshape(len(data.sample_ids), 32, -1),
                imu_mask,
            ),
            data.imu_statistics,
        ],
        axis=1,
    ).astype(np.float32)
    raw = {
        "GlobalV": global_raw,
        "LocalV": local_raw,
        "Skeleton": skeleton_raw,
        "IMU": imu_raw,
    }
    reduced: dict[str, np.ndarray] = {}
    audit: dict[str, Any] = {}
    for name, values in raw.items():
        print(f"P112 reduce {name}: {values.shape}", flush=True)
        reduced[name], audit[name] = reduce_unlabelled(values, components)
        del values
    np.savez_compressed(
        output / "modality_descriptors.npz",
        sample_ids=data.sample_ids,
        users=data.users,
        labels=data.labels,
        **reduced,
    )
    summary = {
        "status": "complete",
        "label_free_reduction": True,
        "rows": len(data.sample_ids),
        "modalities": audit,
        "local_vjepa_views": local_indices.tolist(),
    }
    (output / "descriptor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def _manifest_sample_ids(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return np.asarray([row["sample_id"] for row in csv.DictReader(handle)], dtype=str)


def _align(source_ids: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(value): index for index, value in enumerate(source_ids)}
    missing = [str(value) for value in target_ids if str(value) not in lookup]
    if missing:
        raise RuntimeError(f"P112 alignment misses {len(missing)} rows")
    return np.asarray([lookup[str(value)] for value in target_ids], dtype=np.int64)


@dataclass
class BinaryHead:
    scaler: StandardScaler
    model: LogisticRegression

    @classmethod
    def fit(cls, values: np.ndarray, labels: np.ndarray) -> "BinaryHead":
        scaler = StandardScaler().fit(values)
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        ).fit(scaler.transform(values), labels)
        return cls(scaler, model)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.model.predict(self.scaler.transform(values)).astype(np.int64)


def modality_sets(base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def joined(*names: str) -> np.ndarray:
        value = np.concatenate([base[name] for name in names], axis=1).astype(np.float32)
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-6)
        return value

    return {
        "GlobalV": base["GlobalV"],
        "LocalV": base["LocalV"],
        "Skeleton": base["Skeleton"],
        "IMU": base["IMU"],
        "GlobalV+LocalV": joined("GlobalV", "LocalV"),
        "LocalV+Skeleton": joined("LocalV", "Skeleton"),
        "LocalV+IMU": joined("LocalV", "IMU"),
        "Skeleton+IMU": joined("Skeleton", "IMU"),
        "GlobalV+Skeleton": joined("GlobalV", "Skeleton"),
        "GlobalV+LocalV+Skeleton": joined("GlobalV", "LocalV", "Skeleton"),
    }


def run_route(descriptor_path: Path, output: Path) -> dict[str, Any]:
    with np.load(descriptor_path, allow_pickle=False) as archive:
        full_ids = archive["sample_ids"].astype(str)
        full_users = archive["users"].astype(str)
        full_labels = archive["labels"].astype(np.int64)
        descriptors = modality_sets(
            {name: archive[name].astype(np.float32) for name in ("GlobalV", "LocalV", "Skeleton", "IMU")}
        )
    splits = load_splits()
    split_names = tuple(OUTER_USERS)
    ids = np.concatenate([splits[name].sample_ids.astype(str) for name in split_names])
    users = np.concatenate([splits[name].users.astype(str) for name in split_names])
    labels = np.concatenate([splits[name].labels.astype(np.int64) for name in split_names])
    safe = np.concatenate([splits[name].safe_prediction.astype(np.int64) for name in split_names])
    probability = np.concatenate([splits[name].safe_probability.astype(np.float64) for name in split_names])
    fold_names = np.concatenate([np.repeat(name, len(splits[name].sample_ids)) for name in split_names]).astype(str)
    order = _align(full_ids, ids)
    canonical = {name: value[order] for name, value in descriptors.items()}
    top2 = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    top2_pair = np.sort(top2, axis=1)
    system = safe.copy()
    pair_audit: list[dict[str, Any]] = []
    held_pair_audit: list[dict[str, Any]] = []
    selection_audit: list[dict[str, Any]] = []
    fold_summary: dict[str, Any] = {}

    for outer in split_names:
        held = fold_names == outer
        source = ~held
        held_users = set(OUTER_USERS[outer])
        opportunities: Counter[tuple[int, int]] = Counter()
        for row in np.flatnonzero(source & (safe != labels)):
            pair = tuple(map(int, top2_pair[row]))
            if int(labels[row]) in pair and int(safe[row]) in pair:
                opportunities[pair] += 1
        candidates = sorted(
            [pair for pair, count in opportunities.items() if count >= 2],
            key=lambda pair: (-opportunities[pair], pair),
        )
        activated: dict[tuple[int, int], tuple[str, BinaryHead]] = {}

        for pair in candidates:
            modality_records: list[dict[str, Any]] = []
            for modality, full_values in descriptors.items():
                values = canonical[modality]
                predictions: dict[int, int] = {}
                user_net: dict[str, int] = {}
                for inner_user in sorted(set(users[source].tolist())):
                    eval_rows = np.flatnonzero(
                        source & (users == inner_user)
                        & np.all(top2_pair == np.asarray(pair)[None], axis=1)
                        & np.isin(safe, pair)
                    )
                    if not len(eval_rows):
                        continue
                    train = (~np.isin(full_users, list(held_users) + [inner_user])) & np.isin(full_labels, pair)
                    if len(set(full_labels[train].tolist())) != 2:
                        continue
                    head = BinaryHead.fit(full_values[train], full_labels[train])
                    prediction = head.predict(values[eval_rows])
                    predictions.update(zip(eval_rows.tolist(), prediction.tolist()))
                    user_net[inner_user] = int(
                        np.sum(prediction == labels[eval_rows]) - np.sum(safe[eval_rows] == labels[eval_rows])
                    )
                evaluated = np.asarray(sorted(predictions), dtype=np.int64)
                prediction = np.asarray([predictions[int(row)] for row in evaluated], dtype=np.int64)
                base_correct = safe[evaluated] == labels[evaluated]
                candidate_correct = prediction == labels[evaluated]
                rescue = int(np.sum((~base_correct) & candidate_correct))
                harm = int(np.sum(base_correct & (~candidate_correct)))
                record = {
                    "outer_fold": outer,
                    "pair": f"{pair[0]}<->{pair[1]}",
                    "modality": modality,
                    "source_opportunities": opportunities[pair],
                    "source_crossfit_routes": len(evaluated),
                    "source_crossfit_rescue": rescue,
                    "source_crossfit_harm": harm,
                    "source_crossfit_net": rescue - harm,
                    "source_crossfit_worst_user_net": min(user_net.values()) if user_net else -999,
                }
                modality_records.append(record)
                pair_audit.append(record)
            selected = max(
                modality_records,
                key=lambda row: (
                    row["source_crossfit_net"], row["source_crossfit_rescue"],
                    -row["source_crossfit_harm"], row["modality"],
                ),
            )
            deploy = (
                selected["source_crossfit_rescue"] >= 4
                and selected["source_crossfit_net"] >= 3
                and selected["source_crossfit_worst_user_net"] >= 0
            )
            selection_audit.append({**selected, "deploy": int(deploy)})
            full_values = descriptors[selected["modality"]]
            train = (~np.isin(full_users, list(held_users))) & np.isin(full_labels, pair)
            final_head = BinaryHead.fit(full_values[train], full_labels[train])
            held_family = held & np.isin(labels, pair)
            held_prediction = final_head.predict(canonical[selected["modality"]][held_family])
            held_pair_audit.append(
                {
                    "outer_fold": outer,
                    "pair": f"{pair[0]}<->{pair[1]}",
                    "selected_modality_source_only": selected["modality"],
                    "held_family_rows": int(held_family.sum()),
                    "held_family_accuracy": float(np.mean(held_prediction == labels[held_family])),
                    "deploy": int(deploy),
                }
            )
            if deploy:
                activated[pair] = (str(selected["modality"]), final_head)

        before = system.copy()
        for pair, (modality, head) in activated.items():
            selected_rows = np.flatnonzero(
                held & np.all(top2_pair == np.asarray(pair)[None], axis=1) & np.isin(safe, pair)
            )
            if len(selected_rows):
                system[selected_rows] = head.predict(canonical[modality][selected_rows])
        base_correct = safe == labels
        final_correct = system == labels
        fold_summary[outer] = {
            "rows": int(held.sum()),
            "p89_correct": int(np.sum(held & base_correct)),
            "system_correct": int(np.sum(held & final_correct)),
            "accuracy": float(np.mean(final_correct[held])),
            "changed": int(np.sum(held & (system != before))),
            "rescue": int(np.sum(held & (~base_correct) & final_correct)),
            "harm": int(np.sum(held & base_correct & (~final_correct))),
            "net": int(np.sum(held & final_correct) - np.sum(held & base_correct)),
            "activated": [f"{pair[0]}<->{pair[1]}:{value[0]}" for pair, value in activated.items()],
        }

    base_correct = safe == labels
    final_correct = system == labels
    sample_rows = [
        {
            "sample_id": ids[row], "subject": users[row], "outer_fold": fold_names[row],
            "true_label": int(labels[row]), "p89_prediction": int(safe[row]),
            "p89_adjusted_top2": f"{int(top2[row, 0])}|{int(top2[row, 1])}",
            "system_prediction": int(system[row]), "changed": int(system[row] != safe[row]),
            "rescued": int((not base_correct[row]) and final_correct[row]),
            "harmed": int(base_correct[row] and (not final_correct[row])),
        }
        for row in range(len(ids))
    ]
    summary = {
        "stage": "P112_multimodal_pair_tree",
        "status": "complete",
        "p89": {"correct": 2117, "rows": 2470, "accuracy": 2117 / 2470},
        "system": {
            "correct": int(final_correct.sum()), "rows": len(labels), "accuracy": float(final_correct.mean()),
            "rescue": int(np.sum((~base_correct) & final_correct)),
            "harm": int(np.sum(base_correct & (~final_correct))),
            "net": int(final_correct.sum() - base_correct.sum()),
            "changed": int(np.sum(system != safe)),
        },
        "folds": fold_summary,
        "modalities": list(descriptors),
        "selection": {
            "pair_candidates": "source P89 errors whose true and safe labels both occur in adjusted Top-2; >=2 opportunities",
            "modality": "maximum source-user crossfit net, then rescue, then minimum harm",
            "deploy": "source rescue>=4, net>=3, worst routed source-user net>=0",
            "held_labels_used": False,
        },
    }
    write_csv(output / "pair_modality_crossfit.csv", pair_audit)
    write_csv(output / "pair_selection.csv", selection_audit)
    write_csv(output / "held_pair_capability.csv", held_pair_audit)
    write_csv(output / "sample_predictions.csv", sample_rows)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    descriptor_path = output / "modality_descriptors.npz"
    if args.stage in ("all", "descriptors"):
        build_descriptors(args.vjepa_root.resolve(), output, args.components)
    if args.stage in ("all", "route"):
        if not descriptor_path.is_file():
            raise FileNotFoundError(descriptor_path)
        run_route(descriptor_path, output)


if __name__ == "__main__":
    main()
