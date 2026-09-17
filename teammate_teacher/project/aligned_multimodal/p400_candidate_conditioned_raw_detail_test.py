"""Matched unlabeled Test deployment of the P399 raw-detail candidate scorer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import p216_native_p150_test as p216
import p399_candidate_conditioned_raw_detail_head_oof as p399
from p117_transductive_multicandidate_router import load_candidate_splits
from p386_visual_scope_nonvisual_competence_gate import NONVISUAL_TEACHERS, VISUAL_REFERENCES


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "runs/p400_candidate_conditioned_raw_detail_test_v1"
P309 = HERE / "runs/p309_union_repeat_group_test_v1/predictions.npz"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/student_test_targets.npz"
P315 = HERE / "runs/p315_final_kaggle_candidate_v1/submission_p315_compact_student_raw.csv"
OFFICIAL = ROOT / "Testing/test.csv"
TEST_PHYSICAL = (
    ROOT / "runs/p90_videomaev2_distilled_test_v1/complete_features.npz",
    ROOT / "runs/p90_internvideo2_l_k400_test_v1/complete_features.npz",
    HERE / "runs/p232_depth_thermal_test_features_v1/depth_features.npz",
    HERE / "runs/p232_depth_thermal_test_features_v1/thermal_features.npz",
)
TEST_MOTION = HERE / "runs/p87s_test_motion_window_t16_v1"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def align(values, source_ids, target_ids, fill=0.0):
    values = np.asarray(values)
    output = np.full((len(target_ids), *values.shape[1:]), fill, dtype=values.dtype)
    target = {sample_id: index for index, sample_id in enumerate(np.asarray(target_ids).astype(str))}
    for row, sample_id in enumerate(np.asarray(source_ids).astype(str)):
        if sample_id in target:
            output[target[sample_id]] = values[row]
    return output


def physical_test_features(ids):
    blocks = []
    for path in TEST_PHYSICAL:
        archive = np.load(path)
        token = archive["features"].astype(np.float32).reshape(len(archive["features"]), -1, 768)
        token = align(token, archive["sample_ids"], ids, fill=0.0)
        blocks.extend((p399.l2(token.mean(axis=1)), p399.l2(token.std(axis=1))))
    return p399.l2(np.concatenate(blocks, axis=1).astype(np.float32))


def motion_test_features(ids):
    rows = read_csv(TEST_MOTION / "rows.csv")
    source_ids = np.asarray([row["sample_id"] for row in rows])
    position = {sample_id: index for index, sample_id in enumerate(source_ids)}
    order = np.asarray([position[sample_id] for sample_id in ids])
    skeleton = np.asarray(np.load(TEST_MOTION / "skeleton_features.npy", mmap_mode="r")[order], np.float32)
    relation = np.asarray(np.load(TEST_MOTION / "skeleton_relations.npy", mmap_mode="r")[order], np.float32)
    imu = np.asarray(np.load(TEST_MOTION / "imu_bin_statistics.npy", mmap_mode="r")[order], np.float32)
    imu_global = np.asarray(np.load(TEST_MOTION / "imu_global_statistics.npy", mmap_mode="r")[order], np.float32)
    blocks = []
    for value in (skeleton, relation, imu):
        temporal_axes = tuple(range(1, value.ndim - 2))
        blocks.extend(
            (
                p399.l2(value.mean(axis=temporal_axes).reshape(len(value), -1)),
                p399.l2(value.std(axis=temporal_axes).reshape(len(value), -1)),
            )
        )
    blocks.append(p399.l2(imu_global.reshape(len(imu_global), -1)))
    return p399.l2(np.concatenate(blocks, axis=1).astype(np.float32))


def train_source():
    parts = p399.load_data()
    p399.attach_raw(parts)
    return p399.concatenate([parts[cohort] for cohort in p399.COHORTS])


def test_part(ids):
    train = load_candidate_splits(
        full_visual_bank=True,
        structured_bank=True,
        legacy_visual_bank=True,
        hand_object_bank=True,
        vjepa_dense_bank=True,
        nonvisual_bank=True,
        hierarchical_bank=True,
        epic_bank=True,
        expanded_bank=True,
    )
    names = list(next(iter(train.values())).candidates)
    test, _, values = p216.test_lookup(train, names)
    source_ids = test.split.sample_ids.astype(str)
    visual_references = {
        name: np.mean(
            np.stack([align(values[item], source_ids, ids) for item in members], axis=1),
            axis=1,
        ).astype(np.float32)
        for name, members in VISUAL_REFERENCES.items()
    }
    nonvisual = np.stack(
        [align(values[name], source_ids, ids) for name in NONVISUAL_TEACHERS], axis=1
    ).astype(np.float32)
    group = np.load(P309)
    target = np.load(P310)
    base = align(target["emission_prediction"], target["sample_ids"], ids).astype(int)
    probability = align(group["probability"], group["sample_ids"], ids).astype(np.float32)
    return {
        "ids": ids,
        "labels": np.full(len(ids), -1, dtype=int),
        "users": np.full(len(ids), "anonymous"),
        "base": base,
        "group_probability": probability,
        "visual_references": visual_references,
        "nonvisual_probability": nonvisual,
        "physical": physical_test_features(ids),
        "motion": motion_test_features(ids),
    }


def main():
    print(
        "P400 refits the P399 sparse raw-detail scorer on all 2470 source OOF rows and "
        "runs the matched unlabeled 405-row Test path.",
        flush=True,
    )
    source = train_source()
    best = None
    for k in p399.KS:
        for c_value in p399.CS:
            proposal = source["base"].copy()
            margin = np.full(len(proposal), -np.inf, dtype=float)
            for user in np.unique(source["users"]):
                validation = source["users"] == user
                model = p399.fit_model(p399.subset(source, ~validation), k, c_value)
                proposal[validation], margin[validation] = p399.predict(
                    model, p399.subset(source, validation), k
                )
            threshold = p399.choose_threshold(source, proposal, margin)
            key = (
                threshold["minimum_cohort_gain"] >= 0,
                threshold["net"],
                threshold["rescue"],
                -threshold["harm"],
                -threshold["changed"],
                -k,
                -c_value,
            )
            if best is None or key > best[0]:
                best = (key, k, c_value, threshold)
    _, k, c_value, threshold = best
    targets = np.load(P310)
    ids = targets["sample_ids"].astype(str)
    test = test_part(ids)
    model = p399.fit_model(source, k, c_value)
    proposal, margin = p399.predict(model, test, k)
    route = (proposal != test["base"]) & (margin >= float(threshold["threshold"]))
    prediction = test["base"].copy()
    prediction[route] = proposal[route]

    official = read_csv(OFFICIAL)
    official_ids = np.asarray([row["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] for row in official])
    position = {sample_id: index for index, sample_id in enumerate(ids)}
    official_prediction = np.asarray([prediction[position[sample_id]] for sample_id in official_ids])
    p315_rows = read_csv(P315)
    p315_prediction = np.asarray([int(row["prediction"]) for row in p315_rows])
    if not np.array_equal(p315_prediction, np.asarray([test["base"][position[sample_id]] for sample_id in official_ids])):
        raise RuntimeError("P315 CSV differs from P310 Test base")

    OUT.mkdir(parents=True, exist_ok=True)
    submission = OUT / "submission_p400_raw_detail_teacher.csv"
    with submission.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(official, official_prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(prediction)), prediction] = 0.9805
    np.savez_compressed(
        OUT / "student_test_targets.npz",
        sample_ids=ids,
        target_mask=np.ones(len(ids), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(ids), 0.9805, dtype=np.float32),
        emission_prediction=prediction,
        structured_distillation_prediction=prediction,
    )
    classifier = model[0].named_steps["logisticregression"]
    changed = np.flatnonzero(route)
    report = {
        "stage": "P400_candidate_conditioned_raw_detail_Test",
        "status": "candidate" if len(changed) else "no_test_delta",
        "validation": {
            "p399_correct": 2215,
            "rows": 2470,
            "accuracy": 2215 / 2470,
            "net_vs_p310": 4,
            "fold_nets": [0, 2, 2],
            "strict_gate_pass": True,
        },
        "full_source_selection": {
            "k": k,
            "C": c_value,
            "threshold": threshold,
            "nonzero_coefficients": int(np.sum(np.abs(classifier.coef_[0]) > 1e-10)),
        },
        "test": {
            "rows": len(ids),
            "changes_vs_p315": int(len(changed)),
            "changed_rows_zero_based": changed.tolist(),
            "changed_sample_ids": ids[changed].tolist(),
            "changed_pairs": [f"{test['base'][row]}->{prediction[row]}" for row in changed],
            "candidate_margins": [float(margin[row]) for row in changed],
            "frozen_row_collisions": [int(row) for row in changed if int(row) in (77, 283, 328)],
            "submission": str(submission.resolve()),
            "test_labels_read": False,
        },
        "protocol": {
            "visual_role": "agreement lock and candidate scope only",
            "missing_test_teacher_policy": "safe-posterior fallback",
            "corrupt_IR_policy": "zero IR physical tokens; preserve Depth/Thermal/motion",
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: matched P399 all-source refit and unlabeled Test audit.\n"
        + json.dumps(report["test"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
