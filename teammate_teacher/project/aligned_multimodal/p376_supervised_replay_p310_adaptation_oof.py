"""Strict outer-fold P310 pseudo adaptation with source-label replay."""
from __future__ import annotations

import csv
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from adapt_p87s_structured_student import (
    LabelFreePseudoDataset,
    configure_adaptation_parameters,
    make_loader,
    model_build_args,
    seed_all,
    tempered_probability,
    to_device,
)
from p86_cached_motion_data import P86CachedSequenceMotionDataset, collate_p86_cached_motion
from p117_transductive_multicandidate_router import load_candidate_splits
from train_p86_mobind_fusion_proxy import build_model, evaluate, model_forward


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "runs/p376_supervised_replay_p310_adaptation_oof_v1"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/oof_predictions.npz"
COHORTS = ("H1_selection", "H2_confirmation", "H3_independent_fold0")
BASE_RUNS = (
    HERE / "runs/p87s_fusion_holdout1_c0_v1",
    HERE / "runs/p87s_fusion_holdout2_c0_v1",
    HERE / "runs/p87s_fusion_holdout3_c0_v1",
)
EPOCHS = 40
BATCH_SIZE = 64
FUSION_LR = 1e-4
VISUAL_LR = 5e-5
MINIMUM_LR = 2e-6
WEIGHT_DECAY = 0.02
REPLAY_WEIGHT = 0.10
HARD_CLASSES = (7, 20, 24, 26, 37, 39)
HARD_CLASS_WEIGHT = 2.0
SEED = 20260903
TTA_PASSES = 7


def resolve(path_value):
    path = Path(str(path_value))
    return path if path.is_absolute() else (ROOT / path).resolve()


def dataset_args(summary):
    config = summary["config"]
    return {
        "sequence_cache": resolve(config["sequence_cache"]),
        "motion_cache": resolve(config["motion_cache"]),
        "pixel_cache": resolve(config["pixel_cache"]),
        "teacher_features": resolve(config["teacher_features"]),
        "teacher_logits": resolve(config["teacher_logits"]),
        "imu_teacher_logits": resolve(config["imu_teacher_logits"]) if config.get("imu_teacher_logits") else None,
        "imu_event_features": resolve(config["imu_event_features"]) if config.get("imu_event_features") else None,
    }


def smoothed_targets(prediction):
    probability = np.full((len(prediction), 40), 0.0005, dtype=np.float32)
    probability[np.arange(len(prediction)), prediction] = 0.9805
    return probability


def train_replay(model, pseudo_loader, source_loader, device):
    fusion, visual = configure_adaptation_parameters(model, "heads_motion_encoder")
    optimizer = torch.optim.AdamW(
        (
            {"params": fusion, "lr": FUSION_LR, "initial_lr": FUSION_LR},
            {"params": visual, "lr": VISUAL_LR, "initial_lr": VISUAL_LR},
        ),
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    source_iterator = itertools.cycle(source_loader)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        cosine = 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / max(EPOCHS - 1, 1)))
        for group in optimizer.param_groups:
            group["lr"] = MINIMUM_LR + (float(group["initial_lr"]) - MINIMUM_LR) * cosine
        pseudo_loss_sum = replay_loss_sum = 0.0
        agreement = samples = 0
        started = time.perf_counter()
        for pseudo_batch in pseudo_loader:
            source_batch = next(source_iterator)
            if "label" in pseudo_batch or "label" not in source_batch:
                raise RuntimeError("pseudo/source label boundary violated")
            pseudo_batch = to_device(pseudo_batch, device)
            source_batch = to_device(source_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pseudo_output = model_forward(model, pseudo_batch)
                pseudo_target = tempered_probability(pseudo_batch["pseudo_probability"].float(), 1.0)
                pseudo_loss = F.kl_div(
                    F.log_softmax(pseudo_output["logits"].float(), dim=1),
                    pseudo_target,
                    reduction="batchmean",
                )
                source_output = model_forward(model, source_batch)
                source_labels = source_batch["label"].long()
                per_source = F.cross_entropy(
                    source_output["logits"].float(), source_labels, reduction="none", label_smoothing=0.05
                )
                hard = torch.zeros_like(source_labels, dtype=torch.bool)
                for class_id in HARD_CLASSES:
                    hard |= source_labels == class_id
                sample_weight = torch.where(
                    hard,
                    torch.full_like(per_source, HARD_CLASS_WEIGHT),
                    torch.ones_like(per_source),
                )
                replay_loss = (per_source * sample_weight).sum() / sample_weight.sum()
                loss = pseudo_loss + REPLAY_WEIGHT * replay_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()
            count = len(pseudo_batch["sample_id"])
            samples += count
            pseudo_loss_sum += float(pseudo_loss.detach()) * count
            replay_loss_sum += float(replay_loss.detach()) * count
            agreement += int(
                pseudo_output["logits"].argmax(1).eq(pseudo_target.argmax(1)).sum().detach()
            )
        record = {
            "epoch": epoch,
            "pseudo_kl": pseudo_loss_sum / samples,
            "source_replay_ce": replay_loss_sum / samples,
            "pseudo_agreement": agreement / samples,
            "seconds": time.perf_counter() - started,
        }
        history.append(record)
        if epoch in (1, EPOCHS) or epoch % 10 == 0:
            print(json.dumps(record), flush=True)
    return history


def infer_tta_logits(model, dataset, device, fold_index):
    members = []
    model.eval()
    for pass_index in range(TTA_PASSES):
        np.random.seed(SEED + 1000 * fold_index + pass_index)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_p86_cached_motion,
        )
        rows = []
        with torch.inference_mode():
            for batch in loader:
                batch = to_device(batch, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    rows.append(model_forward(model, batch)["logits"].float().cpu().numpy())
        members.append(np.concatenate(rows))
    return np.mean(np.stack(members), axis=0).astype(np.float32)


def main():
    print(
        "P376 tests source-label replay during P310 pseudo adaptation. Held labels are "
        "excluded from training and read only after the fixed 40 epochs.",
        flush=True,
    )
    seed_all(SEED)
    candidates = load_candidate_splits()
    current = np.load(P310)
    report = {
        "stage": "P376_supervised_replay_P310_adaptation_OOF",
        "status": "complete",
        "protocol": {
            "teacher": "P310/P315 label-identical OOF hard target",
            "pseudo_rows": "held cohort only",
            "supervised_replay": "all non-held users from the corresponding base model space",
            "replay_weight": REPLAY_WEIGHT,
            "hard_classes": list(HARD_CLASSES),
            "hard_class_weight": HARD_CLASS_WEIGHT,
            "temporal_tta_passes_plus_canonical": TTA_PASSES + 1,
            "epochs": EPOCHS,
            "adaptation_scope": "heads_motion_encoder",
            "held_labels_used_for_training_or_epoch_selection": False,
            "test_rows_loaded": 0,
            "test_labels_read": False,
            "user_id_used_as_feature": False,
        },
        "cohorts": {},
    }
    all_predictions = []
    all_logits = []
    all_tta_predictions = []
    all_tta_logits = []
    offset = 0
    for fold_index, (cohort, base_run) in enumerate(zip(COHORTS, BASE_RUNS, strict=True)):
        base_path = base_run / "unified_student.pt"
        summary = json.loads((base_run / "summary.json").read_text(encoding="utf-8"))
        checkpoint = torch.load(base_path, map_location="cpu", weights_only=False)
        build_args = model_build_args(base_path, summary)
        model, _, _ = build_model(build_args)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        common = dataset_args(summary)
        full = P86CachedSequenceMotionDataset(**common)
        split = candidates[cohort].split
        rows = len(split.labels)
        teacher_prediction = current["prediction"][offset : offset + rows].astype(int)
        if not np.array_equal(current["labels"][offset : offset + rows], split.labels.astype(int)):
            raise RuntimeError(f"P310 alignment failed: {cohort}")
        held_indices = np.asarray([full.index_lookup[sample_id] for sample_id in split.sample_ids.astype(str)])
        held_users = set(split.users.astype(str).tolist())
        source_indices = np.asarray(
            [index for index, row in enumerate(full.rows) if row["user_id"] not in held_users],
            dtype=np.int64,
        )
        if len(source_indices) != int(summary["counts"]["train"]):
            raise RuntimeError(f"source replay universe mismatch: {cohort}")
        pseudo_base = P86CachedSequenceMotionDataset(
            **common, indices=held_indices, temporal_augment=True
        )
        probability = smoothed_targets(teacher_prediction)
        ids = split.sample_ids.astype(str)
        pseudo = LabelFreePseudoDataset(
            pseudo_base,
            dict(zip(ids, probability, strict=True)),
            {sample_id: 0.9805 for sample_id in ids},
        )
        source = P86CachedSequenceMotionDataset(
            **common, indices=source_indices, temporal_augment=True
        )
        evaluation = P86CachedSequenceMotionDataset(
            **common, indices=held_indices, temporal_augment=False
        )
        tta_evaluation = P86CachedSequenceMotionDataset(
            **common, indices=held_indices, temporal_augment=True
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        history = train_replay(
            model,
            make_loader(pseudo, BATCH_SIZE, 0, shuffle=True),
            DataLoader(
                source,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_p86_cached_motion,
                drop_last=False,
            ),
            device,
        )
        metrics, evaluation_rows, logits = evaluate(
            model,
            DataLoader(
                evaluation,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_p86_cached_motion,
            ),
            device,
            max_batches=0,
        )
        prediction = logits.argmax(axis=1).astype(int)
        model.to(device)
        augmented_logits = infer_tta_logits(model, tta_evaluation, device, fold_index)
        tta_logits = (logits.astype(np.float32) + TTA_PASSES * augmented_logits) / (TTA_PASSES + 1)
        tta_prediction = tta_logits.argmax(axis=1).astype(int)
        model.to("cpu")
        labels = split.labels.astype(int)
        teacher_correct = teacher_prediction == labels
        student_correct = prediction == labels
        report["cohorts"][cohort] = {
            "base_run": str(base_run),
            "source_replay_rows": len(source_indices),
            "pseudo_rows": len(held_indices),
            "teacher_correct": int(teacher_correct.sum()),
            "student_correct": int(student_correct.sum()),
            "net_vs_teacher": int(student_correct.sum() - teacher_correct.sum()),
            "teacher_agreement": float(np.mean(prediction == teacher_prediction)),
            "student_changes": int(np.sum(prediction != teacher_prediction)),
            "teacher_error_rescue": int(np.sum(~teacher_correct & student_correct)),
            "teacher_correct_harm": int(np.sum(teacher_correct & ~student_correct)),
            "tta_student_correct": int(np.sum(tta_prediction == labels)),
            "tta_net_vs_teacher": int(np.sum(tta_prediction == labels) - teacher_correct.sum()),
            "tta_teacher_agreement": float(np.mean(tta_prediction == teacher_prediction)),
            "tta_changes": int(np.sum(tta_prediction != teacher_prediction)),
            "tta_rescue": int(np.sum(~teacher_correct & (tta_prediction == labels))),
            "tta_harm": int(np.sum(teacher_correct & (tta_prediction != labels))),
            "final_pseudo_agreement_during_training": history[-1]["pseudo_agreement"],
            "metrics": metrics,
        }
        all_predictions.append(prediction)
        all_logits.append(logits.astype(np.float32))
        all_tta_predictions.append(tta_prediction)
        all_tta_logits.append(tta_logits)
        fold_dir = OUT / cohort
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps({"cohort": cohort, **report["cohorts"][cohort]}, ensure_ascii=False), flush=True)
        offset += rows

    labels = current["labels"].astype(int)
    teacher = current["prediction"].astype(int)
    prediction = np.concatenate(all_predictions)
    logits = np.concatenate(all_logits)
    tta_prediction = np.concatenate(all_tta_predictions)
    tta_logits = np.concatenate(all_tta_logits)
    fold_nets = [report["cohorts"][cohort]["net_vs_teacher"] for cohort in COHORTS]
    strict_pass = bool(
        all(net > 0 for net in fold_nets)
        or (sum(net > 0 for net in fold_nets) >= 2 and min(fold_nets) >= -1)
    )
    report["aggregate"] = {
        "rows": len(labels),
        "teacher_correct": int(np.sum(teacher == labels)),
        "student_correct": int(np.sum(prediction == labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "net_vs_p310": int(np.sum(prediction == labels) - np.sum(teacher == labels)),
        "teacher_agreement": float(np.mean(prediction == teacher)),
        "changes": int(np.sum(prediction != teacher)),
        "rescue": int(np.sum((teacher != labels) & (prediction == labels))),
        "harm": int(np.sum((teacher == labels) & (prediction != labels))),
        "fold_nets": fold_nets,
        "strict_gate_pass": strict_pass,
        "decision": "eligible_for_full_train_test" if strict_pass else "reject_before_test",
    }
    tta_fold_nets = [report["cohorts"][cohort]["tta_net_vs_teacher"] for cohort in COHORTS]
    tta_strict_pass = bool(
        all(net > 0 for net in tta_fold_nets)
        or (sum(net > 0 for net in tta_fold_nets) >= 2 and min(tta_fold_nets) >= -1)
    )
    report["tta_aggregate"] = {
        "rows": len(labels),
        "teacher_correct": int(np.sum(teacher == labels)),
        "student_correct": int(np.sum(tta_prediction == labels)),
        "accuracy": float(np.mean(tta_prediction == labels)),
        "net_vs_p310": int(np.sum(tta_prediction == labels) - np.sum(teacher == labels)),
        "teacher_agreement": float(np.mean(tta_prediction == teacher)),
        "changes": int(np.sum(tta_prediction != teacher)),
        "rescue": int(np.sum((teacher != labels) & (tta_prediction == labels))),
        "harm": int(np.sum((teacher == labels) & (tta_prediction != labels))),
        "fold_nets": tta_fold_nets,
        "strict_gate_pass": tta_strict_pass,
        "decision": "eligible_for_full_train_test" if tta_strict_pass else "reject_before_test",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "oof_predictions.npz",
        labels=labels,
        teacher_prediction=teacher,
        prediction=prediction,
        logits=logits,
        tta_prediction=tta_prediction,
        tta_logits=tta_logits,
    )
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: fixed 0.10 supervised replay with 2x hard-class weighting during P310 pseudo adaptation.\n"
        + json.dumps({"canonical": report["aggregate"], "tta": report["tta_aggregate"]}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))
    print(json.dumps(report["tta_aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
