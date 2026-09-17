from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from audit_p87_sequence_decoder import classification_metrics, decode_sessions
from p88_aligned_repeat_holdout import decode_aligned_repeat
from p88_oof_candidate_ensemble import load_protocol
from p88_train_depth_residual import log_softmax_numpy, rescue_harm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select class-bias regularization by H1 user-LOO and transfer once to H2."
    )
    parser.add_argument("--selection-run", type=Path, required=True)
    parser.add_argument("--selection-users", nargs="+", required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--confirmation-users", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-targets", type=Path, default=Path("runs/p85_fullwindow_knn_teacher_v3/teacher_targets.npz"))
    parser.add_argument("--train-metadata", type=Path, default=Path("data/p85_recording_metadata/train_recording_metadata.csv"))
    parser.add_argument("--repeat-config-summary", type=Path, default=Path("runs/p88_aligned_repeat_h1_v1/summary.json"))
    parser.add_argument("--regularizations", type=float, nargs="+", default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0))
    return parser.parse_args()


def protocol(args: argparse.Namespace, run: Path, users: list[str]):
    return load_protocol(SimpleNamespace(
        base_run=run, holdout_users=users,
        teacher_targets=args.teacher_targets,
        train_metadata=args.train_metadata,
        repeat_config_summary=args.repeat_config_summary,
    ))


def load_logits(run: Path) -> np.ndarray:
    return np.asarray(np.load(run.resolve() / "subject_holdout_logits.npy"), dtype=np.float64)


def fit_bias(logits: np.ndarray, labels: np.ndarray, regularization: float) -> np.ndarray:
    values = torch.from_numpy(np.asarray(logits, dtype=np.float64))
    target = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    bias = torch.zeros(values.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [bias], lr=0.5, max_iter=100, tolerance_grad=1e-10,
        tolerance_change=1e-12, line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        centered = bias - bias.mean()
        loss = F.cross_entropy(values + centered, target)
        loss = loss + 0.5 * float(regularization) * centered.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    result = bias.detach().numpy(); result -= result.mean()
    return result


def evaluate(logits, labels, base_decoded, metadata, indices, sessions, transition, decoder, repeat):
    logp = log_softmax_numpy(logits)
    raw = logp.argmax(axis=1)
    decoded = decode_sessions(logp, sessions, transition, decoder)
    aligned, grouping = decode_aligned_repeat(
        logp, indices, metadata, transition, decoder, repeat
    )
    return {
        "raw": classification_metrics(labels, raw),
        "decoded": classification_metrics(labels, decoded),
        "aligned": classification_metrics(labels, aligned),
        "aligned_rescue_harm_vs_base": rescue_harm(labels, base_decoded, aligned),
        "grouping": grouping,
    }


def main() -> None:
    args = parse_args()
    selection = protocol(args, args.selection_run, args.selection_users)
    (
        _ids1, labels1, _prob1, base1, metadata1, indices1, sessions1,
        transition1, decoder1, repeat1,
    ) = selection
    logits1 = load_logits(args.selection_run)
    users1 = metadata1.users
    candidates = []
    for regularization in args.regularizations:
        calibrated = logits1.copy()
        fold_bias_norms = []
        for user in sorted(set(users1)):
            validation = users1 == user
            fit = ~validation
            bias = fit_bias(logits1[fit], labels1[fit], regularization)
            calibrated[validation] += bias
            fold_bias_norms.append(float(np.linalg.norm(bias)))
        metrics = evaluate(
            calibrated, labels1, base1, metadata1, indices1, sessions1,
            transition1, decoder1, repeat1,
        )
        candidates.append({
            "regularization": float(regularization),
            "user_loo": metrics,
            "mean_fold_bias_l2": float(np.mean(fold_bias_norms)),
        })
    candidates.sort(key=lambda value: (
        value["user_loo"]["aligned"]["correct"],
        value["user_loo"]["aligned"]["balanced_accuracy"],
        value["user_loo"]["decoded"]["correct"],
        value["regularization"],
    ), reverse=True)
    selected_regularization = float(candidates[0]["regularization"])
    final_bias = fit_bias(logits1, labels1, selected_regularization)

    confirmation = protocol(args, args.confirmation_run, args.confirmation_users)
    (
        _ids2, labels2, _prob2, base2, metadata2, indices2, sessions2,
        transition2, decoder2, repeat2,
    ) = confirmation
    logits2 = load_logits(args.confirmation_run)
    confirmation_repeat_base = evaluate(
        logits2, labels2, base2, metadata2, indices2, sessions2,
        transition2, decoder2, repeat2,
    )
    confirmation_metrics = evaluate(
        logits2 + final_bias, labels2, base2, metadata2, indices2, sessions2,
        transition2, decoder2, repeat2,
    )
    reverse_bias = fit_bias(logits2, labels2, selected_regularization)
    reverse_selection_metrics = evaluate(
        logits1 + reverse_bias, labels1, base1, metadata1, indices1, sessions1,
        transition1, decoder1, repeat1,
    )
    combined_bias = fit_bias(
        np.concatenate((logits1, logits2), axis=0),
        np.concatenate((labels1, labels2), axis=0),
        selected_regularization,
    )
    summary = {
        "stage": "P88_class_bias_user_LOO_transfer", "status": "complete",
        "protocol": (
            "Regularization selected only by leave-one-user-out predictions inside H1; "
            "one bias vector then fit on all H1 labels and transferred unchanged to H2."
        ),
        "selected_regularization": selected_regularization,
        "selection_base_decoded": classification_metrics(labels1, base1),
        "selection_best_user_loo": candidates[0],
        "confirmation_base_decoded": classification_metrics(labels2, base2),
        "confirmation_repeat_base": confirmation_repeat_base,
        "confirmation": confirmation_metrics,
        "confirmation_aligned_delta_correct_vs_p88_repeat": int(
            confirmation_metrics["aligned"]["correct"]
            - confirmation_repeat_base["aligned"]["correct"]
        ),
        "reverse_transfer_h2_fit_to_h1": reverse_selection_metrics,
        "reverse_aligned_delta_correct_vs_p88_repeat": int(
            reverse_selection_metrics["aligned"]["correct"]
            - evaluate(
                logits1, labels1, base1, metadata1, indices1, sessions1,
                transition1, decoder1, repeat1,
            )["aligned"]["correct"]
        ),
        "bias_l2": float(np.linalg.norm(final_bias)),
        "bias_max_abs": float(np.max(np.abs(final_bias))),
        "repeat_configuration": asdict(repeat2),
        "decoder_configuration_h1": asdict(decoder1),
        "decoder_configuration_h2": asdict(decoder2),
        "all_selection_candidates": candidates,
    }
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    np.save(output / "class_bias.npy", final_bias.astype(np.float32))
    np.save(output / "class_bias_reverse.npy", reverse_bias.astype(np.float32))
    np.save(output / "class_bias_combined.npy", combined_bias.astype(np.float32))
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
