from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import collect_predictions
from src.models.depth_ir_pose_roi_expert import DepthIRPoseROIExpert
from src.train_unimodal import build_datasets, load_config, loader_for, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--b2-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_person_crop_40class_256_fold0/depth_ir_person_crop_40class_256_14train_4val",
    )
    parser.add_argument(
        "--supcon-dir", type=Path,
        default=PROJECT_ROOT / "outputs/depth_ir_person_crop_cross_user_supcon_fold0/depth_ir_person_crop_cross_user_supcon_14train_4val",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def save_archive(path: Path, output: dict[str, object], users: list[str]) -> None:
    logits = np.asarray(output["logits"], dtype=np.float32)
    probabilities = torch.from_numpy(logits).softmax(1).numpy()
    np.savez_compressed(
        path,
        sample_ids=np.asarray(output["sample_ids"], dtype=str),
        user_ids=np.asarray(users, dtype=str),
        labels=np.asarray(output["labels"], dtype=np.int64),
        predictions=logits.argmax(1),
        logits=logits,
        probabilities=probabilities,
        embeddings=np.asarray(output["embeddings"], dtype=np.float32),
    )


def ensure_b2_archives(b2_dir: Path, device_name: str, num_workers: int) -> None:
    train_path = b2_dir / "train_predictions_best_epoch25.npz"
    val_path = b2_dir / "val_predictions_best_epoch25_with_users.npz"
    if train_path.exists() and val_path.exists():
        return
    config = load_config(
        b2_dir / "config.yaml",
        argparse.Namespace(
            data_root=None, manifest=None, fold=None, output_root=None,
            device=device_name, seed=None, smoke_test=False, max_epochs=None,
            num_workers=num_workers, max_train_batches=None, max_val_batches=None,
            run_id=None,
        ),
    )
    set_seed(int(config["seed"]))
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for B2 extraction but unavailable")
    train_dataset, val_dataset = build_datasets(config)
    train_dataset.training = False
    val_dataset.training = False
    model = DepthIRPoseROIExpert(
        num_classes=int(config["num_classes"]), expected_views=4,
        embedding_dim=int(config["embedding_dim"]), frame_feature_dim=int(config["frame_feature_dim"]),
        dropout=float(config["dropout"]), pretrained=False,
    ).to(device)
    checkpoint = torch.load(b2_dir / "best_accuracy.pt", map_location=device, weights_only=True)
    if int(checkpoint["epoch"]) != 25:
        raise ValueError(f"Expected B2 epoch 25, got {checkpoint['epoch']}")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = nn.CrossEntropyLoss()
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    for name, dataset, path in (
        ("train", train_dataset, train_path), ("val", val_dataset, val_path),
    ):
        loader = loader_for(dataset, {**config, "num_workers": num_workers}, training=False)
        output = collect_predictions(model, loader, criterion, device, amp)
        user_by_sample = {str(row["sample_id"]): str(row["user_id"]) for row in dataset.samples}
        users = [user_by_sample[str(sample_id)] for sample_id in output["sample_ids"]]
        save_archive(path, output, users)
        print(json.dumps({"b2_extraction": name, "samples": len(users), "path": str(path)}), flush=True)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    required = {"sample_ids", "user_ids", "labels", "logits", "embeddings"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"Missing arrays in {path}: {sorted(missing)}")
    logits = result["logits"].astype(np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    result["probabilities"] = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    result["predictions"] = logits.argmax(1)
    return result


def align(reference: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reference_ids = reference["sample_ids"].astype(str)
    current_ids = current["sample_ids"].astype(str)
    if len(np.unique(current_ids)) != len(current_ids) or set(reference_ids) != set(current_ids):
        raise ValueError("Prediction sample sets differ")
    lookup = {sample_id: index for index, sample_id in enumerate(current_ids)}
    order = np.asarray([lookup[sample_id] for sample_id in reference_ids], dtype=np.int64)
    aligned = {key: value[order] if len(value) == len(order) else value for key, value in current.items()}
    if not np.array_equal(reference["labels"], aligned["labels"]):
        raise ValueError("Labels differ after sample alignment")
    return aligned


def metrics(arrays: dict[str, np.ndarray]) -> tuple[dict[str, float | int], pd.DataFrame]:
    labels = arrays["labels"].astype(int)
    predictions = arrays["predictions"].astype(int)
    probabilities = arrays["probabilities"].astype(np.float64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=np.arange(40), zero_division=0,
    )
    correct = predictions == labels
    confidence = probabilities.max(axis=1)
    summary: dict[str, float | int] = {
        "accuracy": float(correct.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "loss": float(log_loss(labels, probabilities, labels=np.arange(40))),
        "top3_accuracy": float(np.mean([label in row for label, row in zip(labels, np.argpartition(probabilities, -3, axis=1)[:, -3:], strict=True)])),
        "top5_accuracy": float(np.mean([label in row for label, row in zip(labels, np.argpartition(probabilities, -5, axis=1)[:, -5:], strict=True)])),
        "predicted_class_count": int(np.unique(predictions).size),
        "zero_f1_class_count": int((f1 == 0).sum()),
        "zero_recall_class_count": int((recall == 0).sum()),
        "high_confidence_error_count": int((~correct & (confidence >= 0.8)).sum()),
        "high_confidence_error_rate": float((~correct & (confidence >= 0.8)).mean()),
    }
    per_class = pd.DataFrame({
        "class_id": np.arange(40), "precision": precision, "recall": recall, "f1": f1,
        "support": support.astype(int), "predicted_count": np.bincount(predictions, minlength=40),
        "correct_count": np.bincount(labels[correct], minlength=40),
    })
    return summary, per_class


def user_probe(embeddings: np.ndarray, users: np.ndarray, seed: int) -> tuple[float, float]:
    _, encoded = np.unique(users.astype(str), return_inverse=True)
    class_counts = np.bincount(encoded)
    folds = min(5, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Not enough samples per user for a probe")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced", random_state=seed),
    )
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, embeddings, encoded, cv=split, scoring="accuracy", n_jobs=1)
    return float(scores.mean()), float(scores.std(ddof=0))


def action_probe(
    train_embeddings: np.ndarray, train_labels: np.ndarray,
    val_embeddings: np.ndarray, val_labels: np.ndarray, seed: int,
) -> tuple[float, float]:
    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=seed),
    )
    model.fit(train_embeddings, train_labels)
    predictions = model.predict(val_embeddings)
    _, _, f1, _ = precision_recall_fscore_support(
        val_labels, predictions, labels=np.arange(40), zero_division=0,
    )
    return float(np.mean(predictions == val_labels)), float(f1.mean())


def same_action_cross_user_distance(arrays: dict[str, np.ndarray]) -> tuple[float, float, int]:
    embeddings = arrays["embeddings"].astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    labels = arrays["labels"].astype(int)
    users = arrays["user_ids"].astype(str)
    distances: list[np.ndarray] = []
    for class_id in range(40):
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < 2:
            continue
        similarity = embeddings[indices] @ embeddings[indices].T
        different_user = users[indices, None] != users[indices][None, :]
        upper = np.triu(np.ones_like(similarity, dtype=bool), k=1)
        distances.append(1.0 - similarity[different_user & upper])
    combined = np.concatenate([values for values in distances if len(values)])
    return float(combined.mean()), float(np.median(combined)), int(len(combined))


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6f}")
    lines = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join("---" for _ in display.columns) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in display.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_b2_archives(args.b2_dir, args.device, args.num_workers)
    supcon_summary_path = args.supcon_dir / "run_summary.json"
    if not supcon_summary_path.exists():
        raise FileNotFoundError("Formal SupCon run has not completed")
    run_summary = json.loads(supcon_summary_path.read_text(encoding="utf-8"))
    if run_summary.get("test_read") is not False or int(run_summary.get("epochs_completed", 0)) != 30:
        raise ValueError("SupCon run integrity check failed")
    sources = {
        "B2": {
            "train": load_archive(args.b2_dir / "train_predictions_best_epoch25.npz"),
            "val": load_archive(args.b2_dir / "val_predictions_best_epoch25_with_users.npz"),
        },
        "CrossUserSupCon": {
            "train": load_archive(args.supcon_dir / "train_predictions_primary.npz"),
            "val": load_archive(args.supcon_dir / "val_predictions_primary.npz"),
        },
    }
    sources["CrossUserSupCon"]["val"] = align(sources["B2"]["val"], sources["CrossUserSupCon"]["val"])
    class_map = pd.read_csv(PROJECT_ROOT / "reports/depth_ir_pose_roi_40class_class_map.csv", encoding="utf-8-sig").sort_values("class_id")
    if len(class_map) != 40:
        raise ValueError("Class map must contain exactly 40 rows")
    seed = 20260715
    summary_rows: list[dict[str, Any]] = []
    per_class: dict[str, pd.DataFrame] = {}
    diagnostic_rows: list[dict[str, Any]] = []
    for system, splits in sources.items():
        train_metrics, _ = metrics(splits["train"])
        val_metrics, per_class[system] = metrics(splits["val"])
        train_user_probe, train_user_probe_std = user_probe(splits["train"]["embeddings"], splits["train"]["user_ids"], seed)
        val_user_probe, val_user_probe_std = user_probe(splits["val"]["embeddings"], splits["val"]["user_ids"], seed)
        action_accuracy, action_macro_f1 = action_probe(
            splits["train"]["embeddings"], splits["train"]["labels"],
            splits["val"]["embeddings"], splits["val"]["labels"], seed,
        )
        train_distance = same_action_cross_user_distance(splits["train"])
        val_distance = same_action_cross_user_distance(splits["val"])
        summary_rows.append({
            "system": system,
            "checkpoint_epoch": 25 if system == "B2" else int(run_summary["primary_epoch"]),
            "train_accuracy": train_metrics["accuracy"], "val_accuracy": val_metrics["accuracy"],
            "accuracy_gap": float(train_metrics["accuracy"]) - float(val_metrics["accuracy"]),
            "train_macro_f1": train_metrics["macro_f1"], "val_macro_f1": val_metrics["macro_f1"],
            "macro_f1_gap": float(train_metrics["macro_f1"]) - float(val_metrics["macro_f1"]),
            "val_loss": val_metrics["loss"], "val_weighted_f1": val_metrics["weighted_f1"],
            "top3_accuracy": val_metrics["top3_accuracy"], "top5_accuracy": val_metrics["top5_accuracy"],
            "predicted_class_count": val_metrics["predicted_class_count"],
            "zero_f1_class_count": val_metrics["zero_f1_class_count"],
            "zero_recall_class_count": val_metrics["zero_recall_class_count"],
            "high_confidence_error_count": val_metrics["high_confidence_error_count"],
            "high_confidence_error_rate": val_metrics["high_confidence_error_rate"],
            "train_user_probe_accuracy": train_user_probe,
            "val_user_probe_accuracy": val_user_probe,
            "action_probe_val_accuracy": action_accuracy, "action_probe_val_macro_f1": action_macro_f1,
        })
        diagnostic_rows.extend((
            {"system": system, "split": "train", "diagnostic": "user_id_probe_accuracy", "value": train_user_probe, "std": train_user_probe_std, "pairs": np.nan},
            {"system": system, "split": "val", "diagnostic": "user_id_probe_accuracy", "value": val_user_probe, "std": val_user_probe_std, "pairs": np.nan},
            {"system": system, "split": "train", "diagnostic": "same_action_cross_user_cosine_distance_mean", "value": train_distance[0], "std": np.nan, "pairs": train_distance[2]},
            {"system": system, "split": "train", "diagnostic": "same_action_cross_user_cosine_distance_median", "value": train_distance[1], "std": np.nan, "pairs": train_distance[2]},
            {"system": system, "split": "val", "diagnostic": "same_action_cross_user_cosine_distance_mean", "value": val_distance[0], "std": np.nan, "pairs": val_distance[2]},
            {"system": system, "split": "val", "diagnostic": "same_action_cross_user_cosine_distance_median", "value": val_distance[1], "std": np.nan, "pairs": val_distance[2]},
            {"system": system, "split": "train_to_val", "diagnostic": "action_probe_accuracy", "value": action_accuracy, "std": np.nan, "pairs": np.nan},
            {"system": system, "split": "train_to_val", "diagnostic": "action_probe_macro_f1", "value": action_macro_f1, "std": np.nan, "pairs": np.nan},
        ))
    summary = pd.DataFrame(summary_rows)
    comparison = class_map[["class_id", "action_name", "train_support", "val_support"]].copy()
    for system in sources:
        prefix = "b2" if system == "B2" else "supcon"
        for column in ("precision", "recall", "f1", "predicted_count", "correct_count"):
            comparison[f"{column}_{prefix}"] = per_class[system][column].to_numpy()
    comparison["delta_f1_supcon_minus_b2"] = comparison["f1_supcon"] - comparison["f1_b2"]
    b2_val = sources["B2"]["val"]
    supcon_val = sources["CrossUserSupCon"]["val"]
    labels = b2_val["labels"].astype(int)
    b2_correct = b2_val["predictions"] == labels
    supcon_correct = supcon_val["predictions"] == labels
    outcomes = pd.DataFrame({
        "sample_id": b2_val["sample_ids"].astype(str), "user_id": b2_val["user_ids"].astype(str),
        "class_id": labels, "action_name": class_map.set_index("class_id").loc[labels, "action_name"].to_numpy(),
        "prediction_b2": b2_val["predictions"], "prediction_supcon": supcon_val["predictions"],
        "correct_b2": b2_correct, "correct_supcon": supcon_correct,
        "rescued": ~b2_correct & supcon_correct, "harmed": b2_correct & ~supcon_correct,
    })
    rescued = outcomes.groupby("class_id")["rescued"].sum().reindex(range(40), fill_value=0)
    harmed = outcomes.groupby("class_id")["harmed"].sum().reindex(range(40), fill_value=0)
    comparison["rescued_samples"] = rescued.to_numpy(dtype=int)
    comparison["harmed_samples"] = harmed.to_numpy(dtype=int)
    comparison["net_rescue"] = comparison["rescued_samples"] - comparison["harmed_samples"]
    reports = PROJECT_ROOT / "reports"
    summary.to_csv(reports / "b2_256_cross_user_supcon_summary.csv", index=False, encoding="utf-8-sig")
    comparison.to_csv(reports / "b2_256_cross_user_supcon_per_class.csv", index=False, encoding="utf-8-sig")
    outcomes.to_csv(reports / "b2_256_cross_user_supcon_sample_outcomes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(diagnostic_rows).to_csv(
        reports / "b2_256_cross_user_supcon_embedding_diagnostics.csv", index=False, encoding="utf-8-sig",
    )
    b2_row = summary.loc[summary.system == "B2"].iloc[0]
    supcon_row = summary.loc[summary.system == "CrossUserSupCon"].iloc[0]
    rescue_count = int(outcomes["rescued"].sum())
    harm_count = int(outcomes["harmed"].sum())
    success = bool(
        supcon_row.val_accuracy + 1e-12 >= b2_row.val_accuracy
        and supcon_row.val_macro_f1 > b2_row.val_macro_f1
        and supcon_row.accuracy_gap < b2_row.accuracy_gap
        and supcon_row.train_user_probe_accuracy < b2_row.train_user_probe_accuracy
    )
    best = comparison.sort_values("delta_f1_supcon_minus_b2", ascending=False).head(10)
    worst = comparison.sort_values("delta_f1_supcon_minus_b2").head(10)
    report = [
        "# B2-256 same-action cross-user supervised contrastive experiment", "",
        "## Protocol", "",
        "- Baseline: B2-256 epoch 25.",
        f"- Contrastive checkpoint: {run_summary['primary_checkpoint']}, epoch {run_summary['primary_epoch']}.",
        "- Loss: CE + 0.1 x cross-user SupCon; temperature 0.1; same-user different-action negative weight 2.0.",
        "- Split: fixed 14 train users / 4 unseen validation users. Competition test read: no.", "",
        "## Overall comparison", "", markdown_table(summary), "",
        f"- Accuracy delta: {supcon_row.val_accuracy - b2_row.val_accuracy:+.6f}.",
        f"- Macro-F1 delta: {supcon_row.val_macro_f1 - b2_row.val_macro_f1:+.6f}.",
        f"- Accuracy generalization-gap delta: {supcon_row.accuracy_gap - b2_row.accuracy_gap:+.6f}.",
        f"- Train-user probe delta: {supcon_row.train_user_probe_accuracy - b2_row.train_user_probe_accuracy:+.6f}.",
        f"- Rescued / harmed / net rescue: {rescue_count} / {harm_count} / {rescue_count - harm_count}.",
        f"- All stated success criteria met: {'yes' if success else 'no'}.", "",
        "## Largest per-class gains", "", markdown_table(best[["action_name", "f1_b2", "f1_supcon", "delta_f1_supcon_minus_b2", "net_rescue"]]), "",
        "## Largest per-class losses", "", markdown_table(worst[["action_name", "f1_b2", "f1_supcon", "delta_f1_supcon_minus_b2", "net_rescue"]]), "",
        "## Interpretation boundary", "",
        "The user-ID probes are diagnostic linear separability tests, not identity classifiers used by training. "
        "The validation-user probe is cross-validated within the four held-out users; the action probe is fitted only on train-user embeddings and evaluated on held-out users.",
    ]
    (reports / "b2_256_cross_user_supcon.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "success": success, "rescued": rescue_count, "harmed": harm_count,
        "net_rescue": rescue_count - harm_count, "test_read": False,
        "reports": [
            "b2_256_cross_user_supcon.md", "b2_256_cross_user_supcon_summary.csv",
            "b2_256_cross_user_supcon_per_class.csv", "b2_256_cross_user_supcon_sample_outcomes.csv",
            "b2_256_cross_user_supcon_embedding_diagnostics.csv",
        ],
    }
    print(f"RESULT_JSON={json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
