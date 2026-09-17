from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from audit_p46_causal_ablation import (
    DEFAULT_RUN,
    SEED,
    atomic_json,
    classify_components,
    metrics,
    state_digest,
    write_csv,
)
from p46_event_data import P46EventDataset, collate_p46_events
from p46_protocol import HARD_CLASS_IDS
from p46_step10_model import P46Step10Model
from train_p46_step10 import FrameBudgetBatchSampler, move_batch


DEFAULT_OUTPUT = DEFAULT_RUN / "causal_alignment_supplement_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P46 causal alignment supplement")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--frame-budget", type=int, default=1024)
    return parser.parse_args()


def time_permute(
    source: dict[str, Any],
    frame_mask: torch.Tensor,
    *,
    fraction: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if (fraction is None) == (seed is None):
        raise ValueError("provide exactly one of fraction or seed")
    output: dict[str, Any] = {}
    batch, steps = frame_mask.shape
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    permutations: list[torch.Tensor] = []
    for row in range(batch):
        count = int(frame_mask[row].sum())
        if fraction is not None:
            offset = max(1, int(round(count * fraction)))
            permutation = torch.roll(torch.arange(count), shifts=offset)
        else:
            permutation = torch.randperm(count, generator=generator)
        permutations.append(permutation.to(frame_mask.device))
    for key, value in source.items():
        if torch.is_tensor(value):
            if value.ndim >= 2 and tuple(value.shape[:2]) == (batch, steps):
                changed = value.clone()
                for row, permutation in enumerate(permutations):
                    changed[row, : len(permutation)] = value[row, permutation]
                output[key] = changed
            else:
                output[key] = value.clone()
        else:
            output[key] = value
    return output


def normalized_time_resample(
    source: dict[str, Any],
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    batch, source_steps = source_mask.shape
    target_steps = target_mask.shape[1]
    for key, value in source.items():
        if not torch.is_tensor(value) or value.ndim < 2 or tuple(value.shape[:2]) != (batch, source_steps):
            output[key] = value.clone() if torch.is_tensor(value) else value
            continue
        changed = torch.zeros(
            (batch, target_steps, *value.shape[2:]),
            dtype=value.dtype,
            device=value.device,
        )
        for row in range(batch):
            source_count = int(source_mask[row].sum())
            target_count = int(target_mask[row].sum())
            if source_count == 0 or target_count == 0:
                continue
            indices = torch.linspace(
                0, source_count - 1, target_count, device=value.device
            ).round().long()
            changed[row, :target_count] = value[row, indices]
        output[key] = changed
    # LocalVisualObjectEncoder deliberately keeps candidate 0 as a learned null
    # visual token even on padded frames.  MultiheadAttention otherwise sees an
    # all-True key-padding mask on those frames and returns NaNs before the frame
    # mask can zero the result.
    if "part_source_mask" in output:
        output["part_source_mask"][:, :, :, 0] = True
    return output


def build_same_class_cross_subject_donors(
    dataset: P46EventDataset,
) -> tuple[dict[str, int], set[str]]:
    donors: dict[str, int] = {}
    supported: set[str] = set()
    for target_index, target in enumerate(dataset.rows):
        candidates = [
            index
            for index, row in enumerate(dataset.rows)
            if int(row["class_id"]) == int(target["class_id"])
            and row["user_id"] != target["user_id"]
        ]
        if not candidates:
            donors[target["source_id"]] = target_index
            continue
        donor = min(
            candidates,
            key=lambda index: (
                abs(dataset.frame_lengths[index] - dataset.frame_lengths[target_index]),
                dataset.rows[index]["source_id"],
            ),
        )
        donors[target["source_id"]] = donor
        supported.add(target["source_id"])
    return donors, supported


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "frozen_config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        run / "best_macro_f1.pt", map_location="cpu", weights_only=False
    )
    validation = P46EventDataset(
        config["event_run"], config["context_run"], split="val"
    )
    donors, donor_supported = build_same_class_cross_subject_donors(validation)
    sampler = FrameBudgetBatchSampler(
        validation.frame_lengths, args.batch_size, args.frame_budget
    )
    loader = DataLoader(
        validation,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_p46_events,
    )
    device = torch.device(args.device)
    model = P46Step10Model(subjects=len(config["train_subjects"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    digest_before = state_digest(model)
    conditions = (
        "baseline",
        "visual_shift_quarter",
        "visual_shift_third",
        "visual_shift_half",
        "visual_shuffle_seed_1",
        "visual_shuffle_seed_2",
        "visual_shuffle_seed_3",
        "same_class_cross_subject_visual",
        "same_class_cross_subject_motion",
    )
    stored: dict[str, list[np.ndarray]] = {name: [] for name in conditions}
    labels_all: list[np.ndarray] = []
    sources_all: list[str] = []
    users_all: list[str] = []
    dummy_generator = torch.Generator(device="cpu")
    dummy_generator.manual_seed(SEED)
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            donor_items = [validation[donors[source]] for source in raw_batch["source_id"]]
            raw_donor = collate_p46_events(donor_items)
            batch = move_batch(raw_batch, device)
            donor_batch = move_batch(raw_donor, device)
            with torch.autocast("cuda", dtype=torch.float16):
                motion = model.encoder.motion(batch)
                visual = model.encoder.visual(batch)
                donor_motion = model.encoder.motion(donor_batch)
                donor_visual = model.encoder.visual(donor_batch)
                logits: dict[str, torch.Tensor] = {}
                logits["baseline"] = classify_components(
                    model, motion, visual, batch["frame_mask"], None, dummy_generator
                )
                for name, fraction in (
                    ("visual_shift_quarter", 0.25),
                    ("visual_shift_third", 1.0 / 3.0),
                    ("visual_shift_half", 0.50),
                ):
                    changed = time_permute(
                        visual, batch["frame_mask"], fraction=fraction
                    )
                    logits[name] = classify_components(
                        model, motion, changed, batch["frame_mask"], None, dummy_generator
                    )
                for offset in (1, 2, 3):
                    name = f"visual_shuffle_seed_{offset}"
                    changed = time_permute(
                        visual, batch["frame_mask"], seed=SEED + offset + batch_index * 1000
                    )
                    logits[name] = classify_components(
                        model, motion, changed, batch["frame_mask"], None, dummy_generator
                    )
                donor_visual = normalized_time_resample(
                    donor_visual, donor_batch["frame_mask"], batch["frame_mask"]
                )
                donor_motion = normalized_time_resample(
                    donor_motion, donor_batch["frame_mask"], batch["frame_mask"]
                )
                logits["same_class_cross_subject_visual"] = classify_components(
                    model,
                    motion,
                    donor_visual,
                    batch["frame_mask"],
                    None,
                    dummy_generator,
                )
                logits["same_class_cross_subject_motion"] = classify_components(
                    model,
                    donor_motion,
                    visual,
                    batch["frame_mask"],
                    None,
                    dummy_generator,
                )
            for name in conditions:
                stored[name].append(logits[name].float().cpu().numpy())
            labels_all.append(batch["detail_index"].cpu().numpy())
            sources_all.extend(batch["source_id"])
            users_all.extend(batch["user_id"])
            print(
                json.dumps(
                    {
                        "stage": "alignment_batch",
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "samples_seen": len(sources_all),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    labels = np.concatenate(labels_all)
    arrays = {name: np.concatenate(values) for name, values in stored.items()}
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            count = int((~np.isfinite(value)).sum())
            raise RuntimeError(f"non-finite logits in {name}: {count}")
    baseline_prediction = arrays["baseline"].argmax(1)
    baseline_correct = baseline_prediction == labels
    saved: dict[str, dict[str, str]] = {}
    with (run / "best_macro_f1_predictions.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            saved[row["source_id"]] = row
    saved_prediction = np.asarray(
        [int(saved[source]["predicted_detail_index"]) for source in sources_all]
    )
    if not np.array_equal(saved_prediction, baseline_prediction):
        raise RuntimeError("supplement baseline does not reproduce saved predictions")
    baseline_metrics = metrics(labels, arrays["baseline"])
    supported_mask = np.asarray(
        [source in donor_supported for source in sources_all], dtype=bool
    )
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for name in conditions:
        prediction = arrays[name].argmax(1)
        correct = prediction == labels
        current = metrics(labels, arrays[name])
        changed = prediction != baseline_prediction
        rows.append(
            {
                "condition": name,
                **current,
                "accuracy_delta_pp": 100 * (current["accuracy"] - baseline_metrics["accuracy"]),
                "balanced_accuracy_delta_pp": 100
                * (current["balanced_accuracy"] - baseline_metrics["balanced_accuracy"]),
                "macro_f1_delta_pp": 100 * (current["macro_f1"] - baseline_metrics["macro_f1"]),
                "prediction_changes": int(changed.sum()),
                "baseline_correct_to_wrong": int((baseline_correct & ~correct).sum()),
                "baseline_wrong_to_correct": int((~baseline_correct & correct).sum()),
                "mean_absolute_logit_change": float(
                    np.abs(arrays[name] - arrays["baseline"]).mean()
                ),
                "supported_samples": int(
                    supported_mask.sum()
                    if name.startswith("same_class_cross_subject")
                    else len(labels)
                ),
                "supported_accuracy": float(
                    correct[supported_mask].mean()
                    if name.startswith("same_class_cross_subject")
                    else correct.mean()
                ),
                "supported_baseline_accuracy": float(
                    baseline_correct[supported_mask].mean()
                    if name.startswith("same_class_cross_subject")
                    else baseline_correct.mean()
                ),
                "supported_accuracy_delta_pp": float(
                    100
                    * (
                        correct[supported_mask].mean()
                        - baseline_correct[supported_mask].mean()
                    )
                    if name.startswith("same_class_cross_subject")
                    else 100 * (correct.mean() - baseline_correct.mean())
                ),
            }
        )
        for index, source in enumerate(sources_all):
            prediction_rows.append(
                {
                    "condition": name,
                    "source_id": source,
                    "user_id": users_all[index],
                    "true_class_id": int(HARD_CLASS_IDS[int(labels[index])]),
                    "predicted_class_id": int(HARD_CLASS_IDS[int(prediction[index])]),
                    "correct": int(correct[index]),
                    "baseline_prediction_changed": int(changed[index]),
                }
            )
    digest_after = state_digest(model)
    if digest_before != digest_after:
        raise RuntimeError("weights changed during alignment supplement")
    summary = {
        "stage": "P46_zero_training_alignment_supplement_v1",
        "training_performed": False,
        "human_labels_used": False,
        "validation_samples": len(labels),
        "same_class_cross_subject_donor_coverage": len(donor_supported),
        "same_class_cross_subject_donor_fraction": len(donor_supported)
        / len(validation),
        "baseline": baseline_metrics,
        "conditions": rows,
        "weights_unchanged": True,
        "weight_digest": digest_after,
        "seed": SEED,
    }
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "predictions.csv", prediction_rows)
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"stage": "complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
