"""Subject-safe revalidation of the fixed A18 Visual+Skeleton Teacher.

This runner reuses the canonical P90 three-fold subject split.  It records train
fit only as a diagnostic; checkpoint selection is based exclusively on complete
held subjects.  The fixed P102 Session recipe is fitted on outer-train labels and
then applied to outer-held emissions without using held labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from a18_full_teacher_data import A18_ROWS, A18_SOURCE_USER_SET, load_a18_data
from audit_p102_session_closure import (
    classification_metrics,
    comparison,
    log_softmax,
)
from audit_p87_sequence_decoder import DecoderConfig, align_metadata
from build_p87s_structured_targets import (
    backed_off_structured_probability,
    build_targets,
)
from p100a_global_teacher_data import (
    FoldNormalizer,
    P100ADataset,
    class_user_sample_weights,
    within_subject_permutation,
)
from p100a_global_teacher_model import P100AGlobalTeacher, P100AModelConfig
from p90_teacher_common import load_protocol
from train_p100a_global_teacher_oof import (
    cosine_with_warmup,
    evaluate_model,
    make_loader,
    move_batch,
    set_seed,
    softmax_numpy,
)


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = HERE / "configs/a18_subject_safe_revalidation.json"
DEFAULT_OUTPUT = PROJECT / "runs/a18_subject_safe_revalidation"
DEFAULT_METADATA = HERE / "data/p85_recording_metadata/train_recording_metadata.csv"
MODALITIES = ("visual", "skeleton")
NUM_CLASSES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--folds", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def subject_mean_top1(
    probability: np.ndarray, labels: np.ndarray, users: np.ndarray
) -> float:
    prediction = np.asarray(probability).argmax(axis=1)
    values = [
        float(np.mean(prediction[users == user] == labels[users == user]))
        for user in sorted(set(users.tolist()))
    ]
    return float(np.mean(values))


def checkpoint_selection_key(
    held_metrics: dict[str, Any], mean_subject_top1: float, epoch: int
) -> tuple[float, float, float, float, int]:
    """Frozen key; deliberately has no train-loss or train-accuracy input."""

    return (
        float(mean_subject_top1),
        float(held_metrics["top1"]),
        float(held_metrics["macro_f1"]),
        -float(held_metrics["nll"]),
        -int(epoch),
    )


def protocol_fold_ids(sample_ids: np.ndarray, users: np.ndarray) -> np.ndarray:
    protocol = load_protocol()
    if not np.array_equal(protocol.sample_ids.astype(str), sample_ids.astype(str)):
        raise RuntimeError("A18 sample order differs from the frozen P90 protocol")
    if not np.array_equal(protocol.users.astype(str), users.astype(str)):
        raise RuntimeError("A18 subjects differ from the frozen P90 protocol")
    folds = np.asarray(protocol.fold_id, dtype=np.int64)
    if sorted(np.unique(folds).tolist()) != [0, 1, 2]:
        raise RuntimeError("P90 outer folds changed")
    for fold in range(3):
        held_users = set(users[folds == fold].tolist())
        train_users = set(users[folds != fold].tolist())
        if len(held_users) != 6 or len(train_users) != 12:
            raise RuntimeError(f"fold {fold} is not the canonical 12/6 subject split")
        if held_users & train_users:
            raise RuntimeError(f"subject leakage in fold {fold}")
    return folds


def checkpoint_payload(
    model: P100AGlobalTeacher,
    model_config: P100AModelConfig,
    normalizer: FoldNormalizer,
    fold: int,
    epoch: int,
    train_users: list[str],
    held_users: list[str],
    selection: dict[str, Any],
) -> dict[str, Any]:
    return {
        "state_dict": model.state_dict(),
        "model_config": model_config.__dict__,
        "normalizer_means": normalizer.means,
        "normalizer_stds": normalizer.stds,
        "fold": int(fold),
        "epoch": int(epoch),
        "train_users": train_users,
        "held_users": held_users,
        "selection": selection,
        "train_fit_used_for_selection": False,
    }


def evaluate_logits(
    model: P100AGlobalTeacher,
    data,
    indices: np.ndarray,
    normalizer: FoldNormalizer,
    batch_size: int,
    device: torch.device,
    seed: int,
    *,
    zero_skeleton: bool = False,
    shuffle_skeleton: bool = False,
) -> np.ndarray:
    skeleton_source = None
    zero_modalities: tuple[str, ...] = ()
    if zero_skeleton:
        zero_modalities = ("skeleton",)
    if shuffle_skeleton:
        skeleton_source = within_subject_permutation(
            data, indices, "skeleton", seed + 10_000
        )
    dataset = P100ADataset(
        data,
        indices,
        normalizer,
        MODALITIES,
        skeleton_source=skeleton_source,
        zero_modalities=zero_modalities,
        cross_available=not shuffle_skeleton,
    )
    result = evaluate_model(
        model,
        make_loader(dataset, batch_size, shuffle=False, seed=seed),
        device,
    )
    if not np.array_equal(np.asarray(result["rows"]), indices):
        raise RuntimeError("evaluation row order changed")
    return np.asarray(result["logits"], dtype=np.float32)


def load_checkpoint_into(
    path: Path, model: P100AGlobalTeacher, device: torch.device
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return checkpoint


def run_fold(
    data,
    fold_ids: np.ndarray,
    fold: int,
    config: dict[str, Any],
    output: Path,
    device: torch.device,
    resume: bool,
    smoke: bool,
) -> dict[str, Any]:
    fold_output = output / f"fold_{fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    result_path = fold_output / "fold_result.json"
    prediction_path = fold_output / "predictions.npz"
    best_checkpoint_path = fold_output / "best_checkpoint.pt"
    final_checkpoint_path = fold_output / "final_epoch_checkpoint.pt"
    if (
        resume
        and result_path.exists()
        and prediction_path.exists()
        and best_checkpoint_path.exists()
        and final_checkpoint_path.exists()
    ):
        with np.load(prediction_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        return {
            "record": json.loads(result_path.read_text(encoding="utf-8")),
            "arrays": arrays,
        }

    train_indices = np.flatnonzero(fold_ids != fold).astype(np.int64)
    held_indices = np.flatnonzero(fold_ids == fold).astype(np.int64)
    train_users = sorted(set(data.users[train_indices].tolist()))
    held_users = sorted(set(data.users[held_indices].tolist()))
    if set(train_users) & set(held_users):
        raise RuntimeError("subject leakage before training")
    normalizer = FoldNormalizer.fit(data, train_indices)
    weights = class_user_sample_weights(data, train_indices)
    training = config["training"]
    epochs = 2 if smoke else int(training["epochs"])
    eligible_max = 1 if smoke else int(
        config["checkpoint_selection"]["eligible_epoch_max"]
    )
    if eligible_max >= epochs:
        raise ValueError("best-checkpoint epoch range must exclude the final control epoch")
    seed = int(training["seed"]) + fold
    set_seed(seed)
    loader = make_loader(
        P100ADataset(
            data,
            train_indices,
            normalizer,
            MODALITIES,
            sample_weights=weights,
        ),
        int(training["batch_size"]),
        shuffle=True,
        seed=seed,
    )
    model_config = P100AModelConfig(modalities=MODALITIES, **config["model"])
    model = P100AGlobalTeacher(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        betas=(0.9, 0.98),
    )
    total_steps = epochs * len(loader)
    warmup_steps = int(total_steps * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_with_warmup(step, total_steps, warmup_steps),
    )
    use_amp = device.type == "cuda" and bool(training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_key: tuple[float, float, float, float, int] | None = None
    best_epoch = -1
    best_held_logits: np.ndarray | None = None
    final_held_logits: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    eval_batch_size = int(training["eval_batch_size"])

    print(
        json.dumps(
            {
                "stage": "A18_subject_safe_fold",
                "fold": fold,
                "train_subjects": train_users,
                "held_subjects": held_users,
                "train_rows": len(train_indices),
                "held_rows": len(held_indices),
                "epochs": epochs,
                "selection_uses_train_fit": False,
            }
        ),
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        train_correct = 0
        train_rows = 0
        started = time.perf_counter()
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output_values = model(batch)
                losses = F.cross_entropy(
                    output_values["logits"],
                    batch["label"],
                    reduction="none",
                    label_smoothing=float(training["label_smoothing"]),
                )
                loss = (losses * batch["weight"]).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch["label"])
            train_correct += int(
                (output_values["logits"].argmax(dim=1) == batch["label"]).sum()
            )
            train_rows += len(batch["label"])

        held_logits = evaluate_logits(
            model,
            data,
            held_indices,
            normalizer,
            eval_batch_size,
            device,
            seed,
        )
        held_probability = softmax_numpy(held_logits)
        held_metrics = classification_metrics(
            held_probability, data.labels[held_indices], data.users[held_indices]
        )
        mean_subject = subject_mean_top1(
            held_probability, data.labels[held_indices], data.users[held_indices]
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(train_rows, 1),
            "train_accuracy_diagnostic_only": train_correct / max(train_rows, 1),
            "held_top1": held_metrics["top1"],
            "held_top5": held_metrics["top5"],
            "held_macro_f1": held_metrics["macro_f1"],
            "held_nll": held_metrics["nll"],
            "mean_held_subject_top1": mean_subject,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "seconds": time.perf_counter() - started,
            "eligible_for_checkpoint_selection": epoch <= eligible_max,
        }
        history.append(record)
        print(json.dumps({"fold": fold, **record}), flush=True)
        if epoch <= eligible_max:
            key = checkpoint_selection_key(held_metrics, mean_subject, epoch)
            if best_key is None or key > best_key:
                best_key = key
                best_epoch = epoch
                best_held_logits = held_logits.copy()
                torch.save(
                    checkpoint_payload(
                        model,
                        model_config,
                        normalizer,
                        fold,
                        epoch,
                        train_users,
                        held_users,
                        {
                            "primary": "mean_held_subject_top1",
                            "key": list(key),
                            "held_metrics": held_metrics,
                        },
                    ),
                    best_checkpoint_path,
                )
        if epoch == epochs:
            final_held_logits = held_logits.copy()
            torch.save(
                checkpoint_payload(
                    model,
                    model_config,
                    normalizer,
                    fold,
                    epoch,
                    train_users,
                    held_users,
                    {
                        "role": "fixed_final_epoch_control_only",
                        "used_for_selection": False,
                    },
                ),
                final_checkpoint_path,
            )

    if best_held_logits is None or final_held_logits is None or best_epoch < 1:
        raise RuntimeError("fold training did not produce both best and final checkpoints")

    load_checkpoint_into(best_checkpoint_path, model, device)
    best_train_logits = evaluate_logits(
        model,
        data,
        train_indices,
        normalizer,
        eval_batch_size,
        device,
        seed,
    )
    best_zero_logits = evaluate_logits(
        model,
        data,
        held_indices,
        normalizer,
        eval_batch_size,
        device,
        seed,
        zero_skeleton=True,
    )
    best_shuffle_logits = evaluate_logits(
        model,
        data,
        held_indices,
        normalizer,
        eval_batch_size,
        device,
        seed,
        shuffle_skeleton=True,
    )
    load_checkpoint_into(final_checkpoint_path, model, device)
    final_train_logits = evaluate_logits(
        model,
        data,
        train_indices,
        normalizer,
        eval_batch_size,
        device,
        seed,
    )
    best_train_probability = softmax_numpy(best_train_logits)
    final_train_probability = softmax_numpy(final_train_logits)
    best_held_probability = softmax_numpy(best_held_logits)
    final_held_probability = softmax_numpy(final_held_logits)
    arrays = {
        "held_indices": held_indices,
        "best_logits": best_held_logits,
        "final_logits": final_held_logits,
        "best_zero_skeleton_logits": best_zero_logits,
        "best_shuffle_skeleton_logits": best_shuffle_logits,
    }
    np.savez_compressed(prediction_path, **arrays)
    fold_record = {
        "fold": fold,
        "train_subjects": train_users,
        "held_subjects": held_users,
        "train_rows": int(len(train_indices)),
        "held_rows": int(len(held_indices)),
        "best_epoch": int(best_epoch),
        "best_checkpoint_path": portable_path(best_checkpoint_path),
        "final_checkpoint_path": portable_path(final_checkpoint_path),
        "best_checkpoint_sha256": sha256(best_checkpoint_path),
        "final_checkpoint_sha256": sha256(final_checkpoint_path),
        "selection_key": list(best_key or ()),
        "selection_uses_train_fit": False,
        "best_train_fit_diagnostic": classification_metrics(
            best_train_probability,
            data.labels[train_indices],
            data.users[train_indices],
        ),
        "final_train_fit_diagnostic": classification_metrics(
            final_train_probability,
            data.labels[train_indices],
            data.users[train_indices],
        ),
        "best_held_raw": classification_metrics(
            best_held_probability,
            data.labels[held_indices],
            data.users[held_indices],
        ),
        "final_held_raw": classification_metrics(
            final_held_probability,
            data.labels[held_indices],
            data.users[held_indices],
        ),
        "epoch_history": history,
    }
    result_path.write_text(
        json.dumps(fold_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"record": fold_record, "arrays": arrays}


def apply_source_safe_session(
    logits: np.ndarray,
    labels: np.ndarray,
    users: np.ndarray,
    fold_ids: np.ndarray,
    metadata,
    recipe: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    raw_probability = softmax_numpy(logits).astype(np.float64)
    session_probability = np.zeros_like(raw_probability)
    decoder = DecoderConfig(
        gap_seconds=float(recipe["gap_seconds"]),
        transition_weight=float(recipe["transition_weight"]),
        trigram_backoff=float(recipe["trigram_backoff"]),
        beam_width=int(recipe["beam_width"]),
    )
    fold_records: list[dict[str, Any]] = []
    for fold in range(3):
        train = np.flatnonzero(fold_ids != fold).astype(np.int64)
        held = np.flatnonzero(fold_ids == fold).astype(np.int64)
        if set(users[train].tolist()) & set(users[held].tolist()):
            raise RuntimeError("subject leakage before Session fitting")
        masked_labels = labels.copy()
        masked_labels[held] = -10_000
        full_log_probability = np.full(
            (len(labels), NUM_CLASSES), -math.log(NUM_CLASSES), dtype=np.float64
        )
        full_log_probability[held] = log_softmax(logits[held])
        target = build_targets(
            full_log_probability,
            masked_labels,
            train,
            held,
            metadata,
            decoder,
            posterior_temperature=float(recipe["posterior_temperature"]),
        )
        structured = np.asarray(target["structured_probability"], dtype=np.float64)
        selected, structured_weight = backed_off_structured_probability(
            raw_probability, structured, beam_width=decoder.beam_width
        )
        session_probability[held] = selected[held]
        fold_records.append(
            {
                "fold": fold,
                "train_subjects": sorted(set(users[train].tolist())),
                "held_subjects": sorted(set(users[held].tolist())),
                "train_sessions": int(target["train_session_count"]),
                "held_sessions": int(target["holdout_session_count"]),
                "decoded_held_rows": int(
                    np.sum(np.asarray(target["session_id"])[held] >= 0)
                ),
                "mean_structured_weight": float(structured_weight[held].mean()),
            }
        )
    if not np.isfinite(session_probability).all():
        raise RuntimeError("Session OOF probability lacks complete finite coverage")
    return session_probability, fold_records


def aggregate_train_fit(
    fold_records: list[dict[str, Any]], key: str
) -> dict[str, Any]:
    rows = sum(int(record[key]["rows"]) for record in fold_records)
    correct = sum(int(record[key]["top1_correct"]) for record in fold_records)
    return {
        "scope": "three outer-train fits; rows are repeated across folds",
        "rows_with_repeats": rows,
        "top1_correct": correct,
        "top1": correct / max(rows, 1),
        "used_for_checkpoint_selection": False,
    }


def load_p89_safe_baseline(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = PROJECT / str(config["baseline"]["path"])
    with np.load(path, allow_pickle=False) as archive:
        pieces = []
        for cohort in (
            "H1_selection",
            "H2_confirmation",
            "H3_independent_fold0",
        ):
            pieces.append(
                (
                    np.asarray(archive[f"{cohort}_sample_ids"]).astype(str),
                    np.asarray(archive[f"{cohort}_labels"], dtype=np.int64),
                    np.asarray(archive[f"{cohort}_safe_prediction"], dtype=np.int64),
                )
            )
    sample_ids = np.concatenate([piece[0] for piece in pieces])
    labels = np.concatenate([piece[1] for piece in pieces])
    prediction = np.concatenate([piece[2] for piece in pieces])
    if len(sample_ids) != int(config["baseline"]["scope_rows"]):
        raise RuntimeError("P89 safe baseline row count changed")
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise RuntimeError("P89 safe baseline contains duplicate samples")
    correct = int(np.sum(prediction == labels))
    if correct != int(config["baseline"]["expected_correct"]):
        raise RuntimeError("P89 safe baseline accuracy changed")
    return {
        "path": np.asarray(portable_path(path)),
        "sample_ids": sample_ids,
        "labels": labels,
        "prediction": prediction,
    }


def write_confusion_artifacts(
    output: Path,
    labels: np.ndarray,
    probability: np.ndarray,
    stem: str,
) -> None:
    matrix = confusion_matrix(
        labels, probability.argmax(axis=1), labels=np.arange(NUM_CLASSES)
    )
    np.savetxt(
        output / f"{stem}.csv", matrix, fmt="%d", delimiter=","
    )
    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    axis.set_title(stem.replace("_", " "))
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(NUM_CLASSES))
    axis.set_yticks(np.arange(NUM_CLASSES))
    axis.tick_params(axis="both", labelsize=6)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output / f"{stem}.png", dpi=180)
    plt.close(figure)


def write_per_subject(
    output: Path,
    users: np.ndarray,
    fold_ids: np.ndarray,
    systems: dict[str, dict[str, Any]],
) -> None:
    names = list(systems)
    with (output / "per_subject_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = ["subject", "fold", "rows"] + [
            f"{name}_top1" for name in names
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for user in sorted(set(users.tolist())):
            indices = np.flatnonzero(users == user)
            folds = np.unique(fold_ids[indices])
            if len(folds) != 1:
                raise RuntimeError(f"subject {user} spans multiple outer folds")
            row: dict[str, Any] = {
                "subject": user,
                "fold": int(folds[0]),
                "rows": int(len(indices)),
            }
            for name in names:
                row[f"{name}_top1"] = systems[name]["metrics"]["per_subject"][
                    user
                ]["top1"]
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    metadata_path = args.metadata.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_bytes = config_path.read_bytes()
    config: dict[str, Any] = json.loads(config_bytes.decode("utf-8"))
    if args.seed is not None:
        config["training"]["seed"] = int(args.seed)
    if config.get("variant") != "VS":
        raise ValueError("A18 revalidation is fixed to Visual+Skeleton")
    if bool(config["checkpoint_selection"].get("train_fit_used", True)):
        raise RuntimeError("train fit must not be used for checkpoint selection")
    if int(config["training"]["epochs"]) != int(
        config["checkpoint_selection"]["final_epoch_is_control_only"]
    ):
        raise RuntimeError("final epoch control contract changed")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    torch.set_float32_matmul_precision("high")
    data = load_a18_data()
    if len(data.sample_ids) != A18_ROWS or set(data.users.tolist()) != A18_SOURCE_USER_SET:
        raise RuntimeError("A18 source contract changed")
    fold_ids = protocol_fold_ids(data.sample_ids, data.users)
    folds = (
        list(range(3))
        if args.folds == "all"
        else [int(value) for value in args.folds.split(",")]
    )
    if folds != list(range(3)) and not args.smoke:
        raise ValueError("formal revalidation must cover all three P90 folds")

    manifest = {
        "status": "smoke_started" if args.smoke else "formal_started",
        "stage": "A18_subject_safe_revalidation",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "config": portable_path(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "output": portable_path(output),
        "device": str(device),
        "folds": folds,
        "test_data_loaded": False,
        "submission_generation_authorized": False,
        "train_fit_used_for_selection": False,
        "b_router_specialist_used": False,
        "sweep_used": False,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fold_outputs = [
        run_fold(
            data,
            fold_ids,
            fold,
            config,
            output,
            device,
            args.resume,
            args.smoke,
        )
        for fold in folds
    ]
    if args.smoke:
        print(json.dumps({"status": "smoke_complete", "folds": folds}), flush=True)
        return

    best_logits = np.full((A18_ROWS, NUM_CLASSES), np.nan, dtype=np.float32)
    final_logits = np.full_like(best_logits, np.nan)
    zero_logits = np.full_like(best_logits, np.nan)
    shuffle_logits = np.full_like(best_logits, np.nan)
    fold_records: list[dict[str, Any]] = []
    for item in fold_outputs:
        record = item["record"]
        arrays = item["arrays"]
        for key in ("best_checkpoint_path", "final_checkpoint_path"):
            record[key] = portable_path(Path(record[key]))
        (output / f"fold_{record['fold']}" / "fold_result.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        held = np.asarray(arrays["held_indices"], dtype=np.int64)
        best_logits[held] = arrays["best_logits"]
        final_logits[held] = arrays["final_logits"]
        zero_logits[held] = arrays["best_zero_skeleton_logits"]
        shuffle_logits[held] = arrays["best_shuffle_skeleton_logits"]
        fold_records.append(record)
    for name, values in (
        ("best", best_logits),
        ("final", final_logits),
        ("zero_skeleton", zero_logits),
        ("shuffle_skeleton", shuffle_logits),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} OOF lacks complete coverage")

    metadata = align_metadata(metadata_path, data.sample_ids)
    if not np.array_equal(metadata.users, data.users):
        raise RuntimeError("recording metadata subject alignment changed")
    best_raw_probability = softmax_numpy(best_logits).astype(np.float64)
    final_raw_probability = softmax_numpy(final_logits).astype(np.float64)
    zero_probability = softmax_numpy(zero_logits).astype(np.float64)
    shuffle_probability = softmax_numpy(shuffle_logits).astype(np.float64)
    best_session_probability, best_session_folds = apply_source_safe_session(
        best_logits,
        data.labels,
        data.users,
        fold_ids,
        metadata,
        config["session"],
    )
    final_session_probability, final_session_folds = apply_source_safe_session(
        final_logits,
        data.labels,
        data.users,
        fold_ids,
        metadata,
        config["session"],
    )

    systems = {
        "best_raw_oof": {
            "metrics": classification_metrics(
                best_raw_probability, data.labels, data.users
            )
        },
        "best_session_oof": {
            "metrics": classification_metrics(
                best_session_probability, data.labels, data.users
            ),
            "vs_raw": comparison(
                data.labels,
                data.users,
                best_raw_probability,
                best_session_probability,
            ),
        },
        "final_raw_oof": {
            "metrics": classification_metrics(
                final_raw_probability, data.labels, data.users
            )
        },
        "final_session_oof": {
            "metrics": classification_metrics(
                final_session_probability, data.labels, data.users
            ),
            "vs_raw": comparison(
                data.labels,
                data.users,
                final_raw_probability,
                final_session_probability,
            ),
        },
        "best_zero_skeleton_oof": {
            "metrics": classification_metrics(
                zero_probability, data.labels, data.users
            )
        },
        "best_shuffle_skeleton_oof": {
            "metrics": classification_metrics(
                shuffle_probability, data.labels, data.users
            )
        },
    }
    write_per_subject(output, data.users, fold_ids, systems)
    write_confusion_artifacts(
        output, data.labels, best_raw_probability, "confusion_matrix_best_raw_oof"
    )
    write_confusion_artifacts(
        output,
        data.labels,
        best_session_probability,
        "confusion_matrix_best_session_oof",
    )

    baseline = load_p89_safe_baseline(config)
    a18_lookup = {str(sample_id): index for index, sample_id in enumerate(data.sample_ids)}
    baseline_rows = np.asarray(
        [a18_lookup[str(sample_id)] for sample_id in baseline["sample_ids"]],
        dtype=np.int64,
    )
    if not np.array_equal(data.labels[baseline_rows], baseline["labels"]):
        raise RuntimeError("P89 baseline label alignment changed")
    baseline_prediction = np.asarray(baseline["prediction"], dtype=np.int64)
    baseline_accuracy = float(
        np.mean(baseline_prediction == np.asarray(baseline["labels"]))
    )
    same_scope = {
        "rows": int(len(baseline_rows)),
        "p89_safe": baseline_accuracy,
        "a18_best_raw": float(
            np.mean(best_raw_probability[baseline_rows].argmax(axis=1) == data.labels[baseline_rows])
        ),
        "a18_best_session": float(
            np.mean(best_session_probability[baseline_rows].argmax(axis=1) == data.labels[baseline_rows])
        ),
        "a18_final_raw": float(
            np.mean(final_raw_probability[baseline_rows].argmax(axis=1) == data.labels[baseline_rows])
        ),
        "a18_final_session": float(
            np.mean(final_session_probability[baseline_rows].argmax(axis=1) == data.labels[baseline_rows])
        ),
    }
    exceeds_baseline = bool(same_scope["a18_best_session"] > baseline_accuracy)
    best_paths = [record["best_checkpoint_path"] for record in fold_records]
    (output / "best_checkpoint_paths.json").write_text(
        json.dumps(
            {
                "role": "one validation-selected checkpoint per outer fold",
                "paths": best_paths,
                "epochs": [record["best_epoch"] for record in fold_records],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    best_train_fit = aggregate_train_fit(
        fold_records, "best_train_fit_diagnostic"
    )
    final_train_fit = aggregate_train_fit(
        fold_records, "final_train_fit_diagnostic"
    )
    attribution = {
        "epoch_overfit_best_minus_final_raw_pp": 100.0
        * (
            systems["best_raw_oof"]["metrics"]["top1"]
            - systems["final_raw_oof"]["metrics"]["top1"]
        ),
        "session_best_minus_raw_pp": 100.0
        * (
            systems["best_session_oof"]["metrics"]["top1"]
            - systems["best_raw_oof"]["metrics"]["top1"]
        ),
        "aligned_skeleton_minus_zero_pp": 100.0
        * (
            systems["best_raw_oof"]["metrics"]["top1"]
            - systems["best_zero_skeleton_oof"]["metrics"]["top1"]
        ),
        "aligned_skeleton_minus_shuffle_pp": 100.0
        * (
            systems["best_raw_oof"]["metrics"]["top1"]
            - systems["best_shuffle_skeleton_oof"]["metrics"]["top1"]
        ),
        "best_session_vs_p89_same_scope_pp": 100.0
        * (same_scope["a18_best_session"] - baseline_accuracy),
    }
    metrics = {
        "status": "complete",
        "stage": "A18_subject_safe_revalidation",
        "protocol": {
            "outer_split": "frozen P90 three subject-disjoint folds",
            "rows": A18_ROWS,
            "subjects": 18,
            "folds": 3,
            "checkpoint_selection": config["checkpoint_selection"],
            "selection_warning": (
                "Each best checkpoint is selected on its outer held fold. This is "
                "subject-disjoint validation-selected OOF, not an untouched nested "
                "confirmation estimate."
            ),
            "session": config["session"],
            "test_data_loaded": False,
            "submission_created": False,
            "train_fit_used_for_selection": False,
            "b_router_specialist_used": False,
            "sweep_used": False,
        },
        "inputs": {
            "config": {"path": portable_path(config_path), "sha256": sha256(config_path)},
            "metadata": {"path": portable_path(metadata_path), "sha256": sha256(metadata_path)},
            "p90_fold_dir": portable_path(HERE / "data/subject_folds"),
            "p89_baseline": str(np.asarray(baseline["path"]).item()),
        },
        "folds": fold_records,
        "session_folds": {
            "best": best_session_folds,
            "final": final_session_folds,
        },
        "systems": systems,
        "train_fit_diagnostic": {
            "best_checkpoint": best_train_fit,
            "final_epoch": final_train_fit,
            "selection_use": "forbidden and not used",
        },
        "p89_baseline": {
            "scope": "canonical P89 safe H1+H2+H3 rows",
            "rows": int(len(baseline_rows)),
            "correct": int(np.sum(baseline_prediction == baseline["labels"])),
            "accuracy": baseline_accuracy,
            "same_scope_comparison": same_scope,
        },
        "attribution": attribution,
        "best_checkpoint_paths": best_paths,
        "submission_gate": {
            "criterion": "A18 best Session OOF must exceed P89 safe on identical 2470-row scope",
            "passed": exceeds_baseline,
            "discussion_authorized": exceeds_baseline,
            "submission_generated": False,
        },
    }
    np.savez_compressed(
        output / "oof_predictions.npz",
        sample_ids=data.sample_ids,
        users=data.users,
        labels=data.labels,
        fold_ids=fold_ids,
        best_raw_probability=best_raw_probability.astype(np.float32),
        best_session_probability=best_session_probability.astype(np.float32),
        final_raw_probability=final_raw_probability.astype(np.float32),
        final_session_probability=final_session_probability.astype(np.float32),
        best_zero_skeleton_probability=zero_probability.astype(np.float32),
        best_shuffle_skeleton_probability=shuffle_probability.astype(np.float32),
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["status"] = "formal_complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["submission_gate_passed"] = exceeds_baseline
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "best_raw_top1": systems["best_raw_oof"]["metrics"]["top1"],
                "best_session_top1": systems["best_session_oof"]["metrics"]["top1"],
                "final_raw_top1": systems["final_raw_oof"]["metrics"]["top1"],
                "final_session_top1": systems["final_session_oof"]["metrics"]["top1"],
                "p89_safe_same_scope": baseline_accuracy,
                "a18_best_session_same_scope": same_scope["a18_best_session"],
                "submission_gate_passed": exceeds_baseline,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
