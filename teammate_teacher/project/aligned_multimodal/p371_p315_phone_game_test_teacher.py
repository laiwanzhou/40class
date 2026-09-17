"""Deploy the frozen P370 phone/game hypothesis to unlabeled Test."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier
from torch.utils.data import DataLoader

from p86_cached_motion_data import P86CachedSequenceMotionDataset, collate_p86_cached_motion
from p87s_test_data import P87STestCachedSequenceMotionDataset, collate_p87s_test
from predict_p87s_test_student import load_student
from train_p86_mobind_fusion_proxy import model_forward


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "runs/p371_p315_phone_game_test_teacher_v1"
CHECKPOINT = HERE / "runs/p310_student_test_adapt_e40_v1/unified_student.pt"
TRAIN_SEQUENCE = HERE / "runs/p87s_mc3_sequence_all2914_v1"
TRAIN_MOTION = HERE / "runs/p86_motion_window_cache_t16_v1"
TRAIN_PIXELS = HERE / "runs/p86_visual_pixel_cache_t16_r160_v12"
TRAIN_VMAE = HERE / "runs/p85_videomae_large_multiclip_full40_v1/complete_features.npz"
TRAIN_VMAE_HEAD = HERE / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
TEST_SEQUENCE = HERE / "runs/p87s_test_mc3_sequence_v1"
TEST_MOTION = HERE / "runs/p87s_test_motion_window_t16_v1"
TEST_PIXELS = HERE / "runs/p87s_test_pixel_cache_t16_r160_v1"
P309 = HERE / "runs/p309_union_repeat_group_test_v1/predictions.npz"
P310_TARGETS = HERE / "runs/p310_union_repeat_precedence_teacher_v1/student_test_targets.npz"
P315_CSV = HERE / "runs/p315_final_kaggle_candidate_v1/submission_p315_compact_student_raw.csv"
OFFICIAL = ROOT / "Testing/test.csv"
PAIR = (24, 26)
ALPHA = 10.0
SPECIALIST_THRESHOLD = 0.12
SUPPORT_GAP = 0.30


def l2(values):
    values = np.asarray(values, np.float32)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-6, None)


def infer(model, loader, device):
    ids = []
    embeddings = []
    logits = []
    motion_logits = []
    model.eval().to(device)
    with torch.inference_mode():
        for batch in loader:
            ids.extend(map(str, batch["sample_id"]))
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                output = model_forward(model, batch)
            embeddings.append(output["visual_embedding"].float().cpu().numpy())
            logits.append(output["logits"].float().cpu().numpy())
            motion_logits.append(output["motion_logits"].float().cpu().numpy())
    return (
        np.asarray(ids),
        np.concatenate(embeddings),
        np.concatenate(logits),
        np.concatenate(motion_logits),
    )


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    print(
        "P371 extracts the final compact Student embedding, fits the frozen 24/26 tiny head, "
        "and audits unlabeled Test changes under the +0.30 teacher-consensus guard.",
        flush=True,
    )
    model, _, stage = load_student(CHECKPOINT)
    train_dataset = P86CachedSequenceMotionDataset(
        TRAIN_SEQUENCE, TRAIN_MOTION, TRAIN_PIXELS, TRAIN_VMAE, TRAIN_VMAE_HEAD
    )
    test_dataset = P87STestCachedSequenceMotionDataset(
        TEST_SEQUENCE, TEST_MOTION, TEST_PIXELS, temporal_augment=False
    )
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_p86_cached_motion
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_p87s_test
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, train_embedding, train_logits, train_motion = infer(model, train_loader, device)
    test_ids, test_embedding, test_logits, test_motion = infer(model, test_loader, device)
    model.to("cpu")
    if len(train_ids) != 2914 or len(test_ids) != 405:
        raise RuntimeError("unexpected compact embedding universe")

    train_labels = np.asarray([int(row["class_id"]) for row in train_dataset.rows], dtype=int)
    train_feature = np.concatenate((l2(train_embedding), l2(train_logits), l2(train_motion)), axis=1)
    test_feature = np.concatenate((l2(test_embedding), l2(test_logits), l2(test_motion)), axis=1)
    selected = np.isin(train_labels, PAIR)
    classifier = RidgeClassifier(
        alpha=ALPHA, class_weight="balanced", solver="lsqr", tol=1e-5, max_iter=5000
    )
    classifier.fit(train_feature[selected], train_labels[selected])
    decision = np.asarray(classifier.decision_function(test_feature), float)
    proposal = np.where(decision >= 0.0, PAIR[1], PAIR[0]).astype(int)
    specialist_score = np.abs(decision)

    p309 = np.load(P309)
    targets = np.load(P310_TARGETS)
    if not np.array_equal(test_ids.astype(str), targets["sample_ids"].astype(str)):
        raise RuntimeError("P315 Test embedding order differs from P310 targets")
    if not np.array_equal(test_ids.astype(str), p309["sample_ids"].astype(str)):
        raise RuntimeError("P315 Test embedding order differs from P309 posterior")
    base = targets["emission_prediction"].astype(int)
    posterior = p309["probability"].astype(float)
    rows = np.arange(len(base))
    support_gap = posterior[rows, proposal] - posterior[rows, base]
    route = (
        np.isin(base, PAIR)
        & (proposal != base)
        & (specialist_score >= SPECIALIST_THRESHOLD)
        & (support_gap >= SUPPORT_GAP)
    )
    prediction = base.copy()
    prediction[route] = proposal[route]

    official = read_csv(OFFICIAL)
    official_ids = np.asarray([row["path"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] for row in official])
    if set(official_ids.tolist()) != set(test_ids.tolist()):
        raise RuntimeError("official Test rows differ from compact embedding rows")
    position = {sample_id: index for index, sample_id in enumerate(test_ids.tolist())}
    official_prediction = np.asarray([prediction[position[sample_id]] for sample_id in official_ids])
    p315_rows = read_csv(P315_CSV)
    p315_prediction = np.asarray([int(row["prediction"]) for row in p315_rows])
    if not np.array_equal(p315_prediction, np.asarray([base[position[sample_id]] for sample_id in official_ids])):
        raise RuntimeError("P315 CSV no longer matches frozen P310 targets")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "compact_train_test_embeddings.npz",
        train_sample_ids=train_ids,
        train_labels=train_labels,
        train_feature=train_feature.astype(np.float16),
        test_sample_ids=test_ids,
        test_feature=test_feature.astype(np.float16),
        test_logits=test_logits.astype(np.float32),
    )
    np.savez_compressed(
        OUT / "phone_game_pair_head.npz",
        classes=classifier.classes_,
        coef=classifier.coef_.astype(np.float32),
        intercept=classifier.intercept_.astype(np.float32),
        alpha=np.asarray(ALPHA),
        specialist_threshold=np.asarray(SPECIALIST_THRESHOLD),
        support_gap=np.asarray(SUPPORT_GAP),
    )
    submission = OUT / "submission_p371_phone_game_teacher.csv"
    with submission.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "prediction"))
        writer.writeheader()
        for row, value in zip(official, official_prediction, strict=True):
            writer.writerow({"path": row["path"], "prediction": int(value)})
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[rows, prediction] = 0.9805
    np.savez_compressed(
        OUT / "student_test_targets.npz",
        sample_ids=test_ids,
        target_mask=np.ones(len(test_ids), dtype=bool),
        emission_probability=probability,
        structured_distillation_probability=probability,
        structured_confidence=np.full(len(test_ids), 0.9805, dtype=np.float32),
        emission_prediction=prediction,
        structured_distillation_prediction=prediction,
    )
    changed = np.flatnonzero(route)
    report = {
        "stage": "P371_P315_phone_game_Test_teacher",
        "status": "complete" if len(changed) else "no_test_delta",
        "protocol": {
            "oof_source": "P370 post-diagnostic [0,+1,+2]",
            "pair": list(PAIR),
            "compact_feature": "P315 visual_embedding + logits + motion_logits",
            "ridge_alpha": ALPHA,
            "specialist_threshold": SPECIALIST_THRESHOLD,
            "p307_support_gap": SUPPORT_GAP,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "compact_checkpoint_stage": stage,
        "test": {
            "rows": len(test_ids),
            "changes_vs_p315": int(len(changed)),
            "changed_rows_zero_based": changed.tolist(),
            "changed_sample_ids": test_ids[changed].tolist(),
            "changed_pairs": [f"{base[index]}->{prediction[index]}" for index in changed],
            "specialist_scores": [float(specialist_score[index]) for index in changed],
            "support_gaps": [float(support_gap[index]) for index in changed],
            "submission": str(submission.resolve()),
            "test_labels_read": False,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: full compact P315 24/26 pair head with frozen +0.30 P307 support guard.\n"
        + json.dumps(report["test"], ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
