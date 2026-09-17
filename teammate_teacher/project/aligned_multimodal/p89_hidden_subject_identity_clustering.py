from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from audit_p87_sequence_decoder import align_metadata, build_sessions
from audit_yolo11_pose_skeleton import frame_map
from p89_skeleton_identity_tracking_expert import bone_signature, load_people


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p89_hidden_subject_identity_v1"
TRAIN_MANIFEST = PROJECT_DIR / "data/six_modality_audit/train_union_manifest.csv"
TEST_MANIFEST = PROJECT_DIR / "data/p46_test_union_manifest.csv"
TRAIN_METADATA = PROJECT_DIR / "data/p85_recording_metadata/train_recording_metadata.csv"
TEST_METADATA = PROJECT_DIR / "data/p85_recording_metadata/test_recording_metadata.csv"
TRAIN_VIDEO = PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
TEST_VIDEO = PROJECT_DIR / "runs/p46_videomae_large_multiclip_test_v1/complete_features.npz"
SEED = 20260816
COHORTS = {
    "may_early_users1_5": {"users": [f"user{i}" for i in range(1, 6)], "clusters": 5},
    "may_june_users6_9": {"users": [f"user{i}" for i in range(6, 10)], "clusters": 4},
    "june_users16_20": {"users": [f"user{i}" for i in range(16, 21)], "clusters": 5},
    "june_users21_24": {"users": [f"user{i}" for i in range(21, 25)], "clusters": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover anonymous subject identity from action-invariant body proportions "
            "and frozen appearance embeddings; validate on mixed-date train cohorts."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild-features", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def trial_skeleton_identity(path: Path) -> np.ndarray:
    signatures = []
    for frame_path in frame_map(path, "skeleton").values():
        people = load_people(frame_path)
        if people:
            signature = bone_signature(people[0])
            if signature is not None:
                signatures.append(signature)
    if not signatures:
        return np.zeros(48, dtype=np.float32)
    values = np.stack(signatures).astype(np.float64)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    # Absolute log lengths retain body scale; centred lengths retain proportions.
    centered = median - median.mean()
    return np.concatenate((median, centered, mad)).astype(np.float32)


def build_skeleton_features(
    manifest_path: Path, output: Path, split: str, rebuild: bool
) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in read_csv(manifest_path) if row["skeleton_usable"] == "1"]
    sample_ids = np.asarray(
        [row.get("official_sample_id", row["sample_id"]) or row["sample_id"] for row in rows]
    ).astype(str)
    cache = output / f"{split}_skeleton_identity.npz"
    if cache.is_file() and not rebuild:
        with np.load(cache) as data:
            if np.array_equal(data["sample_ids"].astype(str), sample_ids):
                return sample_ids, np.asarray(data["features"], dtype=np.float32)
    features = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        features.append(trial_skeleton_identity(Path(row["skeleton_path"])))
        if (index + 1) % 200 == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "split": split,
                        "processed": index + 1,
                        "total": len(rows),
                        "elapsed_seconds": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    matrix = np.stack(features)
    np.savez_compressed(cache, sample_ids=sample_ids, features=matrix)
    return sample_ids, matrix


def align(source_ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {sample_id: index for index, sample_id in enumerate(source_ids.astype(str))}
    result = np.zeros((len(target_ids), *values.shape[1:]), dtype=values.dtype)
    for index, sample_id in enumerate(target_ids.astype(str)):
        if sample_id in lookup:
            result[index] = values[lookup[sample_id]]
    return result


def video_features(path: Path, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        source_ids = data["sample_ids"].astype(str)
        raw = np.asarray(data["features"], dtype=np.float32)
    # Match the train/test [early,late] x [scene,person,workspace] contract.
    flattened = raw.reshape(len(raw), -1)
    present = np.isin(target_ids, source_ids)
    return align(source_ids, flattened, target_ids), present


def sessions_for_train(sample_ids: np.ndarray):
    metadata = align_metadata(TRAIN_METADATA, sample_ids)
    return metadata, build_sessions(
        np.arange(len(sample_ids)), metadata, 30.0, grouping="known_user"
    )


def sessions_for_test(sample_ids: np.ndarray):
    metadata = align_metadata(TEST_METADATA, sample_ids)
    return metadata, build_sessions(
        np.arange(len(sample_ids)), metadata, 30.0, grouping="anonymous_date"
    )


def aggregate_sessions(
    sessions: list[np.ndarray], skeleton: np.ndarray, video: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skeleton_values = []
    video_values = []
    sizes = []
    for session in sessions:
        skeleton_values.append(np.median(skeleton[session], axis=0))
        video_values.append(np.mean(video[session], axis=0))
        sizes.append(len(session))
    return np.stack(skeleton_values), np.stack(video_values), np.asarray(sizes)


def clustering_accuracy(labels: np.ndarray, clusters: np.ndarray, weights: np.ndarray) -> float:
    true_values = sorted(set(labels.tolist()))
    cluster_values = sorted(set(map(int, clusters.tolist())))
    matrix = np.zeros((len(true_values), len(cluster_values)), dtype=np.float64)
    for true_index, true_value in enumerate(true_values):
        for cluster_index, cluster_value in enumerate(cluster_values):
            rows = (labels == true_value) & (clusters == cluster_value)
            matrix[true_index, cluster_index] = weights[rows].sum()
    row, column = linear_sum_assignment(-matrix)
    return float(matrix[row, column].sum() / weights.sum())


def configurations() -> list[dict[str, object]]:
    result = []
    for method in ("kmeans", "ward"):
        for feature in ("skeleton", "video", "combined"):
            for video_dimensions in ((16, 32, 64) if feature != "skeleton" else (0,)):
                for video_weight in ((0.25, 0.50, 1.0) if feature == "combined" else (1.0,)):
                    result.append(
                        {
                            "method": method,
                            "feature": feature,
                            "video_dimensions": video_dimensions,
                            "video_weight": video_weight,
                        }
                    )
    return result


def cluster(values: np.ndarray, count: int, method: str) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=count, n_init=100, random_state=SEED).fit_predict(values)
    if method == "ward":
        return AgglomerativeClustering(n_clusters=count, linkage="ward").fit_predict(values)
    raise ValueError(method)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_ids, train_skeleton = build_skeleton_features(
        TRAIN_MANIFEST, output, "train", args.rebuild_features
    )
    test_ids, test_skeleton = build_skeleton_features(
        TEST_MANIFEST, output, "test", args.rebuild_features
    )
    # The immutable P85/P87 training and metadata contract contains 2914 rows.
    # Keep the separately audited 17 synchronization repairs out of this branch.
    with np.load(TRAIN_VIDEO) as source:
        contract_ids = source["sample_ids"].astype(str)
        contract_source_ids = source["source_ids"].astype(str)
    train_lookup = {sample_id: index for index, sample_id in enumerate(train_ids)}
    train_skeleton = train_skeleton[
        np.asarray(
            [train_lookup[sample_id] for sample_id in contract_source_ids],
            dtype=np.int64,
        )
    ]
    train_ids = contract_ids
    train_video, train_video_present = video_features(TRAIN_VIDEO, train_ids)
    test_video, test_video_present = video_features(TEST_VIDEO, test_ids)
    train_metadata, train_sessions = sessions_for_train(train_ids)
    test_metadata, test_sessions = sessions_for_test(test_ids)
    train_session_skeleton, train_session_video, train_sizes = aggregate_sessions(
        train_sessions, train_skeleton, train_video
    )
    test_session_skeleton, test_session_video, test_sizes = aggregate_sessions(
        test_sessions, test_skeleton, test_video
    )
    train_session_users = np.asarray(
        [train_metadata.users[session[0]] for session in train_sessions]
    ).astype(str)
    train_session_dates = np.asarray(
        [train_metadata.dates[session[0]] for session in train_sessions]
    ).astype(str)
    test_session_dates = np.asarray(
        [test_metadata.dates[session[0]] for session in test_sessions]
    ).astype(str)

    skeleton_scaler = StandardScaler().fit(train_session_skeleton)
    train_skeleton_scaled = skeleton_scaler.transform(train_session_skeleton)
    test_skeleton_scaled = skeleton_scaler.transform(test_session_skeleton)
    video_norm = np.linalg.norm(train_session_video, axis=1, keepdims=True)
    train_video_normalized = train_session_video / np.maximum(video_norm, 1e-8)
    test_video_normalized = test_session_video / np.maximum(
        np.linalg.norm(test_session_video, axis=1, keepdims=True), 1e-8
    )
    video_scaler = StandardScaler().fit(train_video_normalized)
    train_video_scaled = video_scaler.transform(train_video_normalized)
    test_video_scaled = video_scaler.transform(test_video_normalized)

    records = []
    cached_pca = {}
    for configuration in configurations():
        dimensions = int(configuration["video_dimensions"])
        if dimensions and dimensions not in cached_pca:
            pca = PCA(n_components=dimensions, whiten=True, random_state=SEED)
            cached_pca[dimensions] = (
                pca.fit_transform(train_video_scaled),
                pca.transform(test_video_scaled),
            )
        if configuration["feature"] == "skeleton":
            train_values = train_skeleton_scaled
        elif configuration["feature"] == "video":
            train_values = cached_pca[dimensions][0]
        else:
            train_values = np.concatenate(
                (
                    train_skeleton_scaled,
                    float(configuration["video_weight"]) * cached_pca[dimensions][0],
                ),
                axis=1,
            )
        cohort_results = {}
        for cohort_name, cohort in COHORTS.items():
            rows = np.isin(train_session_users, cohort["users"])
            clusters = cluster(
                train_values[rows], int(cohort["clusters"]), str(configuration["method"])
            )
            labels = train_session_users[rows]
            weights = train_sizes[rows]
            cohort_results[cohort_name] = {
                "sessions": int(np.sum(rows)),
                "rows": int(weights.sum()),
                "adjusted_rand": float(adjusted_rand_score(labels, clusters)),
                "normalized_mutual_information": float(
                    normalized_mutual_info_score(labels, clusters)
                ),
                "weighted_assignment_accuracy": clustering_accuracy(labels, clusters, weights),
            }
        scores = [item["weighted_assignment_accuracy"] for item in cohort_results.values()]
        records.append(
            {
                "configuration": configuration,
                "cohorts": cohort_results,
                "minimum_cohort_accuracy": min(scores),
                "mean_cohort_accuracy": float(np.mean(scores)),
            }
        )
    records.sort(
        key=lambda item: (
            item["minimum_cohort_accuracy"], item["mean_cohort_accuracy"]
        ),
        reverse=True,
    )
    selected = records[0]
    configuration = selected["configuration"]
    dimensions = int(configuration["video_dimensions"])
    if configuration["feature"] == "skeleton":
        test_values = test_skeleton_scaled
    elif configuration["feature"] == "video":
        test_values = cached_pca[dimensions][1]
    else:
        test_values = np.concatenate(
            (
                test_skeleton_scaled,
                float(configuration["video_weight"]) * cached_pca[dimensions][1],
            ),
            axis=1,
        )

    early = np.isin(test_session_dates, ("2025-05-31", "2025-06-01", "2025-06-02"))
    early_clusters = cluster(test_values[early], 6, str(configuration["method"])).astype(np.int64)
    session_clusters = np.full(len(test_sessions), 6, dtype=np.int64)
    session_clusters[early] = early_clusters
    sample_clusters = np.full(len(test_ids), -1, dtype=np.int64)
    for session_index, session in enumerate(test_sessions):
        sample_clusters[session] = session_clusters[session_index]
    missing_metadata = sample_clusters < 0
    sample_clusters[missing_metadata] = 7
    np.savez_compressed(
        output / "test_subject_clusters.npz",
        sample_ids=test_ids,
        subject_cluster=sample_clusters,
        session_count=np.asarray(len(test_sessions)),
        session_cluster=session_clusters,
        session_date=test_session_dates,
    )
    report = {
        "stage": "P89_hidden_subject_identity_clustering_v1",
        "protocol": (
            "Unsupervised clustering uses only median root-centred bone lengths and "
            "frozen VideoMAE embeddings. Configuration selection is based on four "
            "known-user mixed-date training cohorts; Test labels/predictions are absent."
        ),
        "train_samples": len(train_ids),
        "test_samples": len(test_ids),
        "train_sessions": len(train_sessions),
        "test_sessions": len(test_sessions),
        "test_video_present": int(np.sum(test_video_present)),
        "selected": selected,
        "all_configurations": records,
        "test": {
            "early_sessions": int(np.sum(early)),
            "early_clusters": 6,
            "late_sessions": int(np.sum(~early)),
            "late_policy": "one presumed user25 cluster",
            "unassigned_samples": int(np.sum(missing_metadata)),
            "cluster_sizes": {
                str(value): int(np.sum(sample_clusters == value))
                for value in sorted(set(sample_clusters.tolist()))
            },
        },
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "all_configurations"},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
