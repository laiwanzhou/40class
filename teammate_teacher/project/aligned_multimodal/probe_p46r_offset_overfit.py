from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from p46_event_data import P46EventDataset, collate_p46_events
from p46r_event_bottleneck_model import (
    P46R_OFFSET_FRACTIONS,
    P46REventModel,
    circular_shift_visual_batch,
    parameter_count,
)
from train_p46_step10 import atomic_json, move_batch, seed_everything, write_csv


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EVENT_RUN = PROJECT_DIR / "runs" / "p46_event_inputs_full"
DEFAULT_CONTEXT_RUN = PROJECT_DIR / "runs" / "p30_shared_dir_roi_features_full"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p46r_offset_overfit_gate"
OFFSET_COMPONENT_PREFIXES = (
    "motion.",
    "visual.",
    "motion_sync_project.",
    "visual_sync_project.",
    "offset_head.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit a fixed real P46-R batch on all five visual/motion offsets. "
            "This is a mechanism learnability gate, not a classification experiment."
        )
    )
    parser.add_argument("--event-run", type=Path, default=DEFAULT_EVENT_RUN)
    parser.add_argument("--context-run", type=Path, default=DEFAULT_CONTEXT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--accuracy-threshold", type=float, default=0.90)
    parser.add_argument("--loss-threshold", type=float, default=0.35)
    parser.add_argument("--minimum-per-offset-accuracy", type=float, default=0.75)
    return parser.parse_args()


def selected_indices(total: int, samples: int) -> list[int]:
    if samples < len(P46R_OFFSET_FRACTIONS):
        raise ValueError("samples must be at least the number of offset classes")
    if samples > total:
        raise ValueError(f"requested {samples} samples from a dataset of {total}")
    # Even spacing avoids taking only the first class/subject from the sorted manifest.
    indices = np.linspace(0, total - 1, num=samples, dtype=np.int64).tolist()
    if len(set(indices)) != samples:
        raise RuntimeError("fixed-batch selection produced duplicate indices")
    return [int(value) for value in indices]


def offset_parameters(model: P46REventModel) -> list[torch.nn.Parameter]:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(OFFSET_COMPONENT_PREFIXES) and parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("no P46-R offset parameters selected")
    return parameters


def gradient_report(model: P46REventModel) -> dict[str, Any]:
    groups = {
        "motion": "motion.",
        "visual": "visual.",
        "motion_sync": "motion_sync_project.",
        "visual_sync": "visual_sync_project.",
        "offset_head": "offset_head.",
    }
    report: dict[str, Any] = {}
    for group, prefix in groups.items():
        gradients = [
            parameter.grad.detach().float()
            for name, parameter in model.named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        finite = bool(gradients) and all(torch.isfinite(value).all() for value in gradients)
        squared_norm = sum(float(value.square().sum()) for value in gradients)
        report[group] = {
            "parameters_with_gradient": len(gradients),
            "finite": finite,
            "l2_norm": math.sqrt(squared_norm),
        }
    return report


@torch.inference_mode()
def evaluate_all_offsets(
    model: P46REventModel,
    batch: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    labels_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    correlations: list[torch.Tensor] = []
    batch_size = len(batch["detail_index"])
    for offset_index in range(len(P46R_OFFSET_FRACTIONS)):
        labels = torch.full(
            (batch_size,), offset_index, device=batch["frame_mask"].device, dtype=torch.long
        )
        shifted, _ = circular_shift_visual_batch(batch, labels)
        output = model(shifted)
        labels_all.append(labels.cpu())
        logits_all.append(output["offset_logits"].float().cpu())
        correlations.append(output["offset_correlation"].float().cpu())

    labels = torch.cat(labels_all)
    logits = torch.cat(logits_all)
    predictions = logits.argmax(1)
    per_offset = {
        str(index): float((predictions[labels == index] == index).float().mean())
        for index in range(len(P46R_OFFSET_FRACTIONS))
    }
    confusion = torch.zeros(
        len(P46R_OFFSET_FRACTIONS), len(P46R_OFFSET_FRACTIONS), dtype=torch.long
    )
    for label, prediction in zip(labels.tolist(), predictions.tolist()):
        confusion[label, prediction] += 1
    stacked_correlation = torch.stack(correlations, dim=1)
    return {
        "loss": float(F.cross_entropy(logits, labels)),
        "accuracy": float((predictions == labels).float().mean()),
        "per_offset_accuracy": per_offset,
        "confusion": confusion.tolist(),
        # A zero value means that the five shifted inputs are indistinguishable to
        # the offset head before its classifier and the task cannot be learned.
        "mean_correlation_std_across_offsets": float(
            stacked_correlation.std(dim=1, unbiased=False).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.eval_every < 1:
        raise ValueError("steps and eval-every must be positive")
    if args.torch_threads < 1 or args.interop_threads < 1:
        raise ValueError("thread counts must be positive")
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.interop_threads)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = P46EventDataset(
        args.event_run,
        args.context_run,
        split="train",
        load_context=False,
    )
    indices = selected_indices(len(dataset), args.samples)
    items = [dataset[index] for index in indices]
    raw_batch = collate_p46_events(items, include_context=False)
    batch = move_batch(raw_batch, device)
    model = P46REventModel(width=args.width, dropout=0.0).to(device)
    parameters = offset_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)

    initial = evaluate_all_offsets(model, batch)
    history: list[dict[str, Any]] = []
    first_gradient: dict[str, Any] | None = None
    label_counts: Counter[int] = Counter()
    started = time.perf_counter()
    consecutive_passes = 0

    for step in range(1, args.steps + 1):
        model.train()
        # Across every five steps, every fixed sample is presented at every offset.
        # This prevents memorising a permanent sample-to-offset assignment.
        labels = (
            torch.arange(args.samples, device=device, dtype=torch.long) + step - 1
        ).remainder(len(P46R_OFFSET_FRACTIONS))
        label_counts.update(int(value) for value in labels.tolist())
        shifted, shifts = circular_shift_visual_batch(batch, labels)
        optimizer.zero_grad(set_to_none=True)
        logits = model(shifted)["offset_logits"]
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite offset loss at step {step}")
        loss.backward()
        if step == 1:
            first_gradient = gradient_report(model)
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate_all_offsets(model, batch)
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "train_accuracy": float(
                    (logits.detach().argmax(1) == labels).float().mean()
                ),
                "gradient_norm": float(gradient_norm),
                "minimum_shift_frames": int(shifts.abs().min()),
                "maximum_shift_frames": int(shifts.abs().max()),
                "eval_loss": evaluation["loss"],
                "eval_accuracy": evaluation["accuracy"],
                **{
                    f"eval_offset_{key}_accuracy": value
                    for key, value in evaluation["per_offset_accuracy"].items()
                },
            }
            history.append(row)
            write_csv(output / "history.csv", history)
            print(
                f"step={step}/{args.steps} loss={evaluation['loss']:.4f} "
                f"accuracy={evaluation['accuracy']:.4f} "
                f"per_offset={evaluation['per_offset_accuracy']}",
                flush=True,
            )
            threshold_pass = (
                evaluation["accuracy"] >= args.accuracy_threshold
                and evaluation["loss"] <= args.loss_threshold
                and min(evaluation["per_offset_accuracy"].values())
                >= args.minimum_per_offset_accuracy
            )
            consecutive_passes = consecutive_passes + 1 if threshold_pass else 0
            if consecutive_passes >= 3:
                break

    final = evaluate_all_offsets(model, batch)
    gradients_ok = bool(first_gradient) and all(
        item["finite"] and item["l2_norm"] > 0.0 for item in first_gradient.values()
    )
    balanced_schedule = max(label_counts.values()) - min(label_counts.values()) <= 1
    gate = {
        "overall_accuracy": final["accuracy"] >= args.accuracy_threshold,
        "cross_entropy": final["loss"] <= args.loss_threshold,
        "every_offset_accuracy": min(final["per_offset_accuracy"].values())
        >= args.minimum_per_offset_accuracy,
        "offset_inputs_are_distinguishable": final["mean_correlation_std_across_offsets"]
        > 0.0,
        "all_offset_gradient_groups_are_finite_and_nonzero": gradients_ok,
        "training_offset_schedule_is_balanced": balanced_schedule,
    }
    summary = {
        "stage": "P46-R_offset_fixed_batch_overfit_gate",
        "passed": all(gate.values()),
        "gate": gate,
        "chance_accuracy": 1.0 / len(P46R_OFFSET_FRACTIONS),
        "initial": initial,
        "final": final,
        "first_step_gradients": first_gradient,
        "steps_completed": history[-1]["step"],
        "elapsed_seconds": time.perf_counter() - started,
        "label_counts": dict(sorted(label_counts.items())),
        "samples": {
            "count": args.samples,
            "indices": indices,
            "source_ids": raw_batch["source_id"],
            "subjects": raw_batch["user_id"],
            "class_ids": raw_batch["label"].tolist(),
            "frame_counts": [int(value.sum()) for value in raw_batch["frame_mask"]],
        },
        "configuration": {
            "event_run": str(args.event_run.resolve()),
            "context_loaded": False,
            "device": str(device),
            "steps_requested": args.steps,
            "eval_every": args.eval_every,
            "learning_rate": args.learning_rate,
            "width": args.width,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "seed": args.seed,
            "model_parameters": parameter_count(model),
            "trained_parameter_count": sum(parameter.numel() for parameter in parameters),
            "offset_fractions": list(P46R_OFFSET_FRACTIONS),
            "thresholds": {
                "accuracy": args.accuracy_threshold,
                "loss": args.loss_threshold,
                "minimum_per_offset_accuracy": args.minimum_per_offset_accuracy,
                "required_consecutive_evaluations": 3,
            },
        },
    }
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
