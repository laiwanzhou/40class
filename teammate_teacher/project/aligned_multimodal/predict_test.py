from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset, frame_map
from aligned_model import AlignedMultimodalModel


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DEFAULT_TEST_CSV = REPO_DIR / "Testing" / "test.csv"
DEFAULT_TEST_ROOT = REPO_DIR / "Testing" / "data"
DEFAULT_CHECKPOINT = PROJECT_DIR / "runs" / "depth_ir_skeleton" / "best_accuracy.pt"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "depth_ir_skeleton" / "test_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对官方 405 条匿名测试片段运行三模态推理")
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--test-root", type=Path, default=DEFAULT_TEST_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def build_test_manifest(test_csv: Path, test_root: Path, output: Path) -> list[str]:
    with test_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        official_rows = list(csv.DictReader(handle))
    rows: list[dict[str, str | int]] = []
    official_paths: list[str] = []
    for row in official_rows:
        official_path = row["path"]
        trial_dir = (test_root / official_path).resolve()
        sample_id = trial_dir.name
        paths = {
            "depth": trial_dir / "Depth_Color",
            "ir": trial_dir / "IR",
            "skeleton": trial_dir / "Skeleton",
        }
        if not all(path.is_dir() for path in paths.values()):
            missing = [name for name, path in paths.items() if not path.is_dir()]
            raise FileNotFoundError(f"{sample_id} 缺少模态目录：{missing}")
        frame_sets = {name: set(frame_map(path, name)) for name, path in paths.items()}
        common = set.intersection(*frame_sets.values())
        if not common:
            raise RuntimeError(f"{sample_id} 的 Depth/IR/Skeleton 没有共同帧")
        rows.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "class_id": -1,
                "class_name": "",
                "user_id": "",
                "trial_id": sample_id,
                "depth_dir": str(paths["depth"]),
                "ir_dir": str(paths["ir"]),
                "skeleton_dir": str(paths["skeleton"]),
                "num_aligned_frames": len(common),
            }
        )
        official_paths.append(official_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return official_paths


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    manifest_path = PROJECT_DIR / "data" / "test_manifest.csv"
    official_paths = build_test_manifest(args.test_csv.resolve(), args.test_root.resolve(), manifest_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split="test",
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        depth_representation=config.get("depth_representation", "jet_rgb"),
        visual_normalization=config.get("visual_normalization", "legacy"),
        skeleton_representation=config.get("skeleton_representation", "frame_joint"),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(args.num_workers) > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlignedMultimodalModel(
        modalities,
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
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")

    records: list[dict[str, object]] = []
    started = time.time()
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                modality: batch[modality].to(device, non_blocking=True)
                for modality in modalities
            }
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                probabilities = torch.softmax(model(inputs), dim=1)
            top_probabilities, top_classes = probabilities.topk(3, dim=1)
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
            for sample_id, classes, probs, sample_entropy in zip(
                batch["sample_id"], top_classes.cpu(), top_probabilities.cpu(), entropy.cpu()
            ):
                records.append(
                    {
                        "sample_id": sample_id,
                        "prediction": int(classes[0]),
                        "confidence": float(probs[0]),
                        "entropy": float(sample_entropy),
                        "top3": "|".join(f"{int(cls)}:{float(prob):.6f}" for cls, prob in zip(classes, probs)),
                    }
                )

    if len(records) != len(official_paths):
        raise RuntimeError(f"推理条数不一致：{len(records)} != {len(official_paths)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "prediction"])
        writer.writerows((path, record["prediction"]) for path, record in zip(official_paths, records))
    detailed_path = output.with_name(output.stem + "_detailed.csv")
    with detailed_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    counts = Counter(int(record["prediction"]) for record in records)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "num_samples": len(records),
        "seconds": round(time.time() - started, 2),
        "mean_confidence": sum(float(record["confidence"]) for record in records) / len(records),
        "mean_entropy": sum(float(record["entropy"]) for record in records) / len(records),
        "max_entropy": math.log(40),
        "predicted_class_count": len(counts),
        "known_missing_ir_trials": (
            ["SM_test_0012", "SM_test_0014", "SM_test_0154", "SM_test_0194"] if "ir" in modalities else []
        ),
        "missing_ir_fallback": (
            "不可读的零填充 IR 作为归一化黑场（-1）输入；其他模态保留。"
            if "ir" in modalities
            else None
        ),
        "class_counts": {str(i): counts[i] for i in range(40)},
        "accuracy": None,
        "accuracy_note": "官方测试标签未公开，本地只能生成预测，真实 Accuracy 需由 Kaggle 评分。",
    }
    summary_path = output.with_name(output.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"提交格式预测：{output}")
    print(f"带置信度预测：{detailed_path}")


if __name__ == "__main__":
    main()
