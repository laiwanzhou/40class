from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel, fp32_size_mb
from predict_test import build_test_manifest


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="冻结独立 Skeleton/Depth 专家，以受保护的残差 logits 方式融合"
    )
    parser.add_argument(
        "--skeleton-checkpoint",
        type=Path,
        default=PROJECT_DIR / "runs" / "skeleton_only" / "best_accuracy.pt",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=PROJECT_DIR / "runs" / "depth_layer3_spatial" / "best_accuracy.pt",
    )
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--weight", type=float, default=0.25, help="Depth residual 权重")
    parser.add_argument(
        "--extra-weight",
        type=float,
        action="append",
        default=[],
        help="额外落盘的固定 Depth 权重；可重复指定，且不会重复模型推理",
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_DIR / "data" / "manifest.csv")
    parser.add_argument("--test-csv", type=Path, default=REPO_DIR / "Testing" / "test.csv")
    parser.add_argument("--test-root", type=Path, default=REPO_DIR / "Testing" / "data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-logits",
        type=Path,
        default=None,
        help="可选：保存逐样本 Skeleton/Depth logits，供无泄漏校准实验复用",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_model(checkpoint: dict, device: torch.device) -> AlignedMultimodalModel:
    config = checkpoint["config"]
    model = AlignedMultimodalModel(
        list(config["modalities"]),
        dropout=float(config["dropout"]),
        use_aux_heads=bool(config.get("use_aux_heads", False)),
        use_modality_masks=bool(config.get("use_modality_masks", False)),
        use_cross_attention=bool(config.get("use_cross_attention", False)),
        cross_attention_grid=tuple(config.get("cross_attention_grid", [2, 3])),
        cross_attention_heads=int(config.get("cross_attention_heads", 4)),
        depth_input_channels=int(config.get("depth_input_channels", 3)),
        use_layer3_spatial=bool(config.get("use_layer3_spatial", False)),
        visual_stem_fusion=str(config.get("visual_stem_fusion", "concat")),
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
        use_ir_motion=bool(config.get("use_ir_motion", False)),
        use_ir_local=bool(config.get("use_ir_local", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def write_val_predictions(
    path: Path, sample_ids: list[str], labels: np.ndarray, predictions: np.ndarray
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "prediction"])
        writer.writerows(zip(sample_ids, labels.tolist(), predictions.tolist()))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.weight <= 1.0:
        raise ValueError("--weight 必须位于 [0, 1]")
    if any(not 0.0 <= weight <= 1.0 for weight in args.extra_weight):
        raise ValueError("--extra-weight 必须位于 [0, 1]")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    skeleton_checkpoint_path = args.skeleton_checkpoint.resolve()
    depth_checkpoint_path = args.depth_checkpoint.resolve()
    skeleton_checkpoint = torch.load(
        skeleton_checkpoint_path, map_location="cpu", weights_only=False
    )
    depth_checkpoint = torch.load(depth_checkpoint_path, map_location="cpu", weights_only=False)
    skeleton_config = skeleton_checkpoint["config"]
    depth_config = depth_checkpoint["config"]
    if list(skeleton_config["modalities"]) != ["skeleton"]:
        raise ValueError("Skeleton checkpoint 必须是 skeleton-only")
    if list(depth_config["modalities"]) != ["depth"]:
        raise ValueError("Depth checkpoint 必须是 depth-only")
    for key in ("num_frames", "image_height", "image_width"):
        if int(skeleton_config[key]) != int(depth_config[key]):
            raise ValueError(f"两个专家的 {key} 不一致")

    official_paths: list[str] | None = None
    if args.split == "test":
        manifest_path = output.parent / "test_manifest.csv"
        official_paths = build_test_manifest(
            args.test_csv.resolve(), args.test_root.resolve(), manifest_path
        )
        cache_dir = None
    else:
        manifest_path = args.manifest.resolve()
        cache_dir = depth_config.get("cache_dir")

    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split=args.split,
        modalities=["depth", "skeleton"],
        num_frames=int(depth_config["num_frames"]),
        image_height=int(depth_config["image_height"]),
        image_width=int(depth_config["image_width"]),
        augment=False,
        cache_dir=cache_dir,
        skeleton_strategy=skeleton_config.get("skeleton_strategy", "first"),
        depth_representation=depth_config.get("depth_representation", "jet_rgb"),
        visual_normalization=depth_config.get("visual_normalization", "legacy"),
        skeleton_representation=skeleton_config.get("skeleton_representation", "frame_joint"),
        skeleton_raw_cache_dir=skeleton_config.get("skeleton_raw_cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(depth_config["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(depth_config.get("use_amp", True) and device.type == "cuda")
    skeleton_model = build_model(skeleton_checkpoint, device)
    depth_model = build_model(depth_checkpoint, device)

    sample_ids: list[str] = []
    labels_all: list[int] = []
    skeleton_logits_all: list[torch.Tensor] = []
    depth_logits_all: list[torch.Tensor] = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                skeleton_logits = skeleton_model(
                    {"skeleton": batch["skeleton"].to(device, non_blocking=True)}
                )
                depth_logits = depth_model(
                    {"depth": batch["depth"].to(device, non_blocking=True)}
                )
            sample_ids.extend(batch["sample_id"])
            labels_all.extend(batch["label"].tolist())
            skeleton_logits_all.append(skeleton_logits.float().cpu())
            depth_logits_all.append(depth_logits.float().cpu())

    skeleton_logits = torch.cat(skeleton_logits_all)
    depth_logits = torch.cat(depth_logits_all)
    # 等价于 Skeleton logits + w * (Depth logits - Skeleton logits)。
    fused_logits = skeleton_logits + args.weight * (depth_logits - skeleton_logits)
    probabilities = torch.softmax(fused_logits, dim=1)
    predictions = probabilities.argmax(1).numpy()
    labels = np.asarray(labels_all, dtype=np.int64)
    top_probabilities, top_classes = probabilities.topk(3, dim=1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)

    if args.save_logits is not None:
        logits_path = args.save_logits.resolve()
        logits_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            logits_path,
            sample_ids=np.asarray(sample_ids),
            labels=labels,
            skeleton_logits=skeleton_logits.numpy(),
            depth_logits=depth_logits.numpy(),
        )

    if args.split == "val":
        write_val_predictions(output, sample_ids, labels, predictions)
        skeleton_predictions = skeleton_logits.argmax(1).numpy()
        depth_predictions = depth_logits.argmax(1).numpy()
        prediction_files = {
            "primary": str(output),
            "skeleton": str(output.with_name(output.stem + "_skeleton.csv")),
            "depth": str(output.with_name(output.stem + "_depth.csv")),
        }
        write_val_predictions(
            Path(prediction_files["skeleton"]), sample_ids, labels, skeleton_predictions
        )
        write_val_predictions(Path(prediction_files["depth"]), sample_ids, labels, depth_predictions)
        extra_weights = {}
        for weight in sorted(set(float(value) for value in args.extra_weight)):
            extra_predictions = (
                skeleton_logits + weight * (depth_logits - skeleton_logits)
            ).argmax(1).numpy()
            extra_path = output.with_name(output.stem + f"_w{round(weight * 100):03d}.csv")
            write_val_predictions(extra_path, sample_ids, labels, extra_predictions)
            extra_weights[f"{weight:.6g}"] = {
                "path": str(extra_path),
                **metric_dict(labels, extra_predictions),
            }
        overall = metric_dict(labels, predictions)
        per_user = {}
        users_array = np.asarray([sample.user_id for sample in dataset.samples])
        for user in sorted(set(users_array.tolist())):
            indices = users_array == user
            per_user[user] = {"samples": int(indices.sum()), **metric_dict(labels[indices], predictions[indices])}
        weight_scan = []
        for weight in np.linspace(0.0, 0.5, 51):
            scan_predictions = (
                skeleton_logits + float(weight) * (depth_logits - skeleton_logits)
            ).argmax(1).numpy()
            weight_scan.append({"depth_weight": round(float(weight), 2), **metric_dict(labels, scan_predictions)})
        summary = {
            "overall": overall,
            "per_user": per_user,
            "weight_scan": weight_scan,
            "expert_metrics": {
                "skeleton": metric_dict(labels, skeleton_predictions),
                "depth": metric_dict(labels, depth_predictions),
            },
            "extra_weights": extra_weights,
            "prediction_files": prediction_files,
        }
    else:
        assert official_paths is not None
        if len(official_paths) != len(predictions):
            raise RuntimeError(f"测试条数不一致：{len(predictions)} != {len(official_paths)}")
        with output.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "prediction"])
            writer.writerows(zip(official_paths, predictions.tolist()))
        detailed_path = output.with_name(output.stem + "_detailed.csv")
        with detailed_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "prediction", "confidence", "entropy", "top3"])
            for sample_id, prediction, confidence, sample_entropy, classes, probs in zip(
                sample_ids,
                predictions,
                top_probabilities[:, 0],
                entropy,
                top_classes,
                top_probabilities,
            ):
                top3 = "|".join(
                    f"{int(class_id)}:{float(probability):.6f}"
                    for class_id, probability in zip(classes, probs)
                )
                writer.writerow(
                    [sample_id, int(prediction), float(confidence), float(sample_entropy), top3]
                )
        counts = Counter(predictions.tolist())
        summary = {
            "num_samples": len(predictions),
            "mean_confidence": float(top_probabilities[:, 0].mean()),
            "mean_entropy": float(entropy.mean()),
            "max_entropy": math.log(40),
            "predicted_class_count": len(counts),
            "class_counts": {str(class_id): counts[class_id] for class_id in range(40)},
            "accuracy": None,
            "accuracy_note": "测试标签未公开，真实 Accuracy 需由 Kaggle 评分。",
        }

    summary.update(
        {
            "method": "Skeleton logits + w * (Depth logits - Skeleton logits)",
            "depth_weight": float(args.weight),
            "skeleton_checkpoint": str(skeleton_checkpoint_path),
            "depth_checkpoint": str(depth_checkpoint_path),
            "combined_fp32_parameter_size_mb": fp32_size_mb(skeleton_model)
            + fp32_size_mb(depth_model),
            "device": str(device),
            "seconds": round(time.time() - started, 2),
            "saved_logits": str(args.save_logits.resolve()) if args.save_logits else None,
        }
    )
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"预测：{output}")


if __name__ == "__main__":
    main()
