from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from audit_p86_videomae_small_frozen_inner import metrics, selection_score
from build_p46_videomae_cache import encode, restore_legacy_attention_biases
from build_p46_videomae_multiclip_cache import prepare_trial
from build_p85_videomae_full40_multiclip_cache import read_rows
from p86_videomae_small_visual_model import DEFAULT_VIDEOMAE_SMALL, resolve_snapshot
from train_p46_videomae_head import l2_normalize, make_model, row_standardize
from train_p85_videomae_full40_head import sample_weights
from train_p86_visual_student_oof import split_universe


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data/p46_single_split.csv"
DEFAULT_P29 = PROJECT_DIR / "runs/p29_dir_multiscale_roi_full"
DEFAULT_TEACHER_LOGITS = (
    PROJECT_DIR / "runs/p85_videomae_large_multiclip_full40_head_v1/candidate_oof_logits.npz"
)
DEFAULT_REUSE = (
    PROJECT_DIR / "runs/p86_videomae_small_native_probe_fold0_v3/inner_native_features.npz"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs/p86_videomae_small_native_singlefold_v4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed 15-subject/3-subject VideoMAE-Small native-geometry probe."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--p29-run", type=Path, default=DEFAULT_P29)
    parser.add_argument("--teacher-logits", type=Path, default=DEFAULT_TEACHER_LOGITS)
    parser.add_argument("--reuse-features", type=Path, default=DEFAULT_REUSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pretrained-model", default=DEFAULT_VIDEOMAE_SMALL)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--trial-batch", type=int, default=8)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.resolve(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    teacher = load_npz(args.teacher_logits)
    split = split_universe(teacher, outer_fold=0, seed=args.seed)
    sample_ids = np.asarray(split["sample_ids"]).astype(str)
    validation_ids = sample_ids[split["inner_dev"]]
    validation_set = set(validation_ids.tolist())
    training_ids = np.asarray(
        [sample_id for sample_id in sample_ids if sample_id not in validation_set]
    )
    if len(training_ids) != 2470 or len(validation_ids) != 444:
        raise RuntimeError("fixed single-fold counts changed")
    selected_ids = np.concatenate((training_ids, validation_ids))
    row_lookup = {row["sample_id"]: row for row in read_rows(args.manifest.resolve())}

    reuse = load_npz(args.reuse_features)
    reuse_ids = np.asarray(reuse["sample_ids"]).astype(str)
    reuse_lookup = {sample_id: index for index, sample_id in enumerate(reuse_ids)}
    feature_lookup = {
        sample_id: np.asarray(reuse["features"][index], dtype=np.float32)
        for sample_id, index in reuse_lookup.items()
    }
    kinetics_lookup = {
        sample_id: np.asarray(reuse["kinetics_logits"][index], dtype=np.float32)
        for sample_id, index in reuse_lookup.items()
    }
    missing_ids = [sample_id for sample_id in selected_ids if sample_id not in feature_lookup]
    if missing_ids:
        snapshot = resolve_snapshot(args.pretrained_model)
        processor = VideoMAEImageProcessor.from_pretrained(snapshot, local_files_only=True)
        model = VideoMAEForVideoClassification.from_pretrained(snapshot, local_files_only=True)
        load_audit = restore_legacy_attention_biases(model, snapshot)
        if (
            int(model.config.hidden_size) != 384
            or int(model.config.num_frames) != 16
            or int(model.config.image_size) != 224
        ):
            raise RuntimeError("official VideoMAE-Small geometry changed")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        started = time.perf_counter()
        peak_cuda_gib = 0.0
        missing_rows = [row_lookup[sample_id] for sample_id in missing_ids]
        for batch_start in range(0, len(missing_rows), args.trial_batch):
            batch_rows = missing_rows[batch_start : batch_start + args.trial_batch]
            videos: list[list[np.ndarray]] = []
            for row in batch_rows:
                row_videos, _ = prepare_trial(row, args.p29_run.resolve())
                videos.extend(row_videos)
            features, kinetics, peak = encode(model, processor, videos, device)
            peak_cuda_gib = max(peak_cuda_gib, peak)
            features = features.reshape(len(batch_rows), 2, 3, 384)
            kinetics = kinetics.reshape(len(batch_rows), 2, 3, 400)
            for index, row in enumerate(batch_rows):
                feature_lookup[row["sample_id"]] = features[index]
                kinetics_lookup[row["sample_id"]] = kinetics[index]
            processed = min(batch_start + len(batch_rows), len(missing_rows))
            if processed == len(missing_rows) or processed % 80 == 0:
                print(
                    json.dumps(
                        {
                            "new_features": processed,
                            "missing_total": len(missing_rows),
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                            "peak_cuda_gib": round(peak_cuda_gib, 3),
                        }
                    ),
                    flush=True,
                )
    else:
        load_audit = {"source": "all rows reused"}
        peak_cuda_gib = 0.0

    features = np.stack([feature_lookup[value] for value in selected_ids]).astype(np.float32)
    kinetics = np.stack([kinetics_lookup[value] for value in selected_ids]).astype(np.float32)
    rows = [row_lookup[value] for value in selected_ids]
    labels = np.asarray([int(row["class_id"]) for row in rows], dtype=np.int64)
    users = np.asarray([row["user_id"] for row in rows]).astype(str)
    training_index = np.arange(len(training_ids))
    validation_index = np.arange(len(training_ids), len(selected_ids))

    # Frozen before this run from the legacy-inner native-geometry diagnostic.
    feature_matrix = l2_normalize(features).reshape(len(features), -1)
    kinetics_matrix = row_standardize(kinetics.reshape(len(kinetics), -1))
    values = np.concatenate((feature_matrix, kinetics_matrix), axis=1)
    head = make_model(alpha=1000.0)
    head.fit(
        values[training_index],
        labels[training_index],
        ridge__sample_weight=sample_weights(labels[training_index], power=0.35),
    )
    prediction = head.predict(values[validation_index]).astype(np.int64)
    validation_metrics = metrics(
        labels[validation_index], prediction, users[validation_index]
    )
    joblib.dump(head, output / "fixed_ridge_head.joblib")
    np.savez_compressed(
        output / "native_features_all2914.npz",
        sample_ids=selected_ids,
        users=users,
        labels=labels,
        split=np.asarray(
            ["candidate_train"] * len(training_ids)
            + ["fixed_validation"] * len(validation_ids)
        ),
        features=features.astype(np.float16),
        kinetics_logits=kinetics.astype(np.float16),
    )
    summary = {
        "stage": "P86_VideoMAE_Small_fixed_singlefold",
        "protocol": (
            "Permanent single subject-disjoint split: train all 15 non-validation subjects and "
            "evaluate fixed user1/user2/user21 once. Probe settings were frozen before adding the "
            "extra 973 training rows; no three-fold OOF or validation retuning."
        ),
        "training_samples": int(len(training_ids)),
        "validation_samples": int(len(validation_ids)),
        "validation_subjects": sorted(set(users[validation_index].tolist())),
        "geometry": {"frames": 16, "resolution": 224, "clips": 6},
        "fixed_probe": {
            "feature_set": "features_plus_kinetics",
            "class_weight_power": 0.35,
            "alpha": 1000.0,
        },
        "newly_extracted_samples": int(len(missing_ids)),
        "pretrained_load_audit": load_audit,
        "peak_cuda_gib": peak_cuda_gib,
        "validation_metrics": validation_metrics,
        "selection_score": selection_score(validation_metrics),
        "three_fold_oof": False,
        "test_used_for_selection": False,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
