from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from aligned_model import AlignedMultimodalModel


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D+S 条件置换重要性诊断")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def metric_vector(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            accuracy_score(labels, predictions),
            balanced_accuracy_score(labels, predictions),
            f1_score(labels, predictions, average="macro", zero_division=0),
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve() if args.output else checkpoint_path.with_name("modality_shuffle.json")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    if modalities != ["depth", "skeleton"]:
        raise ValueError(f"当前诊断只支持 modalities=['depth', 'skeleton']，实际为 {modalities}")

    dataset = AlignedMultimodalDataset(
        manifest_path=args.manifest.resolve(),
        split="val",
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        cache_dir=config.get("cache_dir"),
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
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
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
        skeleton_input_dim=int(config.get("skeleton_input_dim", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    visual_features: list[torch.Tensor] = []
    skeleton_features: list[torch.Tensor] = []
    spatial_features: list[torch.Tensor] = []
    skeleton_sequences: list[torch.Tensor] = []
    labels_all: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                if model.use_cross_attention:
                    visual_output = model.visual(
                        {"depth": batch["depth"].to(device)}, return_spatial=True
                    )
                    assert isinstance(visual_output, tuple)
                    visual, spatial = visual_output
                    skeleton_sequence = model.skeleton.encode_sequence(
                        batch["skeleton"].to(device)
                    )
                    spatial_features.append(spatial.cpu())
                    skeleton_sequences.append(skeleton_sequence.cpu())
                else:
                    visual = model.visual_project(
                        model.visual({"depth": batch["depth"].to(device)})
                    )
                    skeleton = model.skeleton_project(
                        model.skeleton(batch["skeleton"].to(device))
                    )
            visual_features.append(visual.float().cpu())
            if not model.use_cross_attention:
                skeleton_features.append(skeleton.float().cpu())
            labels_all.extend(batch["label"].tolist())
    visual = torch.cat(visual_features).to(device)
    skeleton = torch.cat(skeleton_features).to(device) if skeleton_features else None
    spatial = torch.cat(spatial_features).to(device) if spatial_features else None
    skeleton_sequence = (
        torch.cat(skeleton_sequences).to(device) if skeleton_sequences else None
    )
    labels = np.asarray(labels_all, dtype=np.int64)
    users = np.asarray([sample.user_id for sample in dataset.samples])
    indices = np.arange(len(dataset))

    def fused_logits(
        depth_indices: torch.Tensor | None = None,
        skeleton_indices: torch.Tensor | None = None,
        depth_mean: bool = False,
        skeleton_mean: bool = False,
    ) -> torch.Tensor:
        visual_input = visual
        skeleton_input = skeleton
        if depth_indices is not None:
            visual_input = visual_input[depth_indices]
        elif depth_mean:
            visual_input = visual_input.mean(0, keepdim=True).expand_as(visual_input)

        if model.use_cross_attention:
            assert spatial is not None and skeleton_sequence is not None
            spatial_input = spatial
            sequence_input = skeleton_sequence
            if depth_indices is not None:
                spatial_input = spatial_input[depth_indices]
            elif depth_mean:
                spatial_input = spatial_input.mean(0, keepdim=True).expand_as(spatial_input)
            if skeleton_indices is not None:
                sequence_input = sequence_input[skeleton_indices]
            elif skeleton_mean:
                sequence_input = sequence_input.mean(0, keepdim=True).expand_as(sequence_input)

            batch_size, time_steps = sequence_input.shape[:2]
            pooled_spatial = torch.nn.functional.adaptive_avg_pool2d(
                spatial_input.reshape(
                    batch_size * time_steps, 512, *spatial_input.shape[-2:]
                ),
                model.cross_attention_grid,
            )
            depth_tokens = pooled_spatial.flatten(2).transpose(1, 2)
            assert model.depth_token_project is not None and model.cross_attention is not None
            depth_tokens = model.depth_token_project(depth_tokens)
            query = sequence_input.reshape(batch_size * time_steps, 1, 256)
            attended, _ = model.cross_attention(
                query, depth_tokens, depth_tokens, need_weights=False
            )
            assert model.cross_attention_norm is not None
            assert model.cross_attention_scale is not None
            aligned = query + model.cross_attention_scale * model.cross_attention_norm(attended)
            skeleton_input = model.skeleton.pool(
                aligned.reshape(batch_size, time_steps, 256)
            )
            visual_input = model.visual_project(visual_input)
            skeleton_input = model.skeleton_project(skeleton_input)
        else:
            assert skeleton_input is not None
            if skeleton_indices is not None:
                skeleton_input = skeleton_input[skeleton_indices]
            elif skeleton_mean:
                skeleton_input = skeleton_input.mean(0, keepdim=True).expand_as(skeleton_input)

        if model.use_modality_masks:
            present = torch.ones(len(visual_input), 1, device=device, dtype=visual_input.dtype)
            gate_input = torch.cat([visual_input, skeleton_input, present, present], dim=1)
        else:
            gate_input = torch.cat([visual_input, skeleton_input], dim=1)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            gate = torch.sigmoid(model.gate(gate_input))
            fused = gate * visual_input + (1.0 - gate) * skeleton_input
            return model.classifier(torch.cat([visual_input, skeleton_input, fused], dim=1)).float()

    with torch.inference_mode():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            baseline_logits = fused_logits()
    baseline_predictions = baseline_logits.argmax(1).cpu().numpy()
    baseline_metrics = metric_vector(labels, baseline_predictions)

    def permutation(seed: int, mode: str) -> torch.Tensor:
        rng = np.random.default_rng(seed)
        donors: list[int] = []
        for index in indices:
            if mode == "same_user_different_label":
                candidates = indices[(users == users[index]) & (labels != labels[index])]
            elif mode == "same_label":
                candidates = indices[(labels == labels[index]) & (indices != index)]
            else:
                raise ValueError(mode)
            if len(candidates) == 0:
                candidates = indices[indices != index]
            donors.append(int(rng.choice(candidates)))
        return torch.as_tensor(donors, device=device)

    def repeated_test(modality: str, mode: str) -> dict[str, object]:
        values: list[np.ndarray] = []
        prediction_changes: list[float] = []
        for repeat in range(int(args.repeats)):
            donor = permutation(int(args.seed) + repeat, mode)
            with torch.inference_mode():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    logits = fused_logits(
                        depth_indices=donor if modality == "depth" else None,
                        skeleton_indices=donor if modality == "skeleton" else None,
                    )
            predictions = logits.argmax(1).cpu().numpy()
            values.append(metric_vector(labels, predictions))
            prediction_changes.append(float(np.mean(predictions != baseline_predictions)))
        array = np.stack(values)
        return {
            "mean": dict(zip(("accuracy", "balanced_accuracy", "macro_f1"), array.mean(0).tolist())),
            "std": dict(zip(("accuracy", "balanced_accuracy", "macro_f1"), array.std(0).tolist())),
            "prediction_change_rate": float(np.mean(prediction_changes)),
        }

    def mean_replacement(modality: str) -> dict[str, float]:
        with torch.inference_mode():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = fused_logits(
                    depth_mean=modality == "depth",
                    skeleton_mean=modality == "skeleton",
                )
        predictions = logits.argmax(1).cpu().numpy()
        values = metric_vector(labels, predictions)
        return {
            "accuracy": float(values[0]),
            "balanced_accuracy": float(values[1]),
            "macro_f1": float(values[2]),
            "prediction_change_rate": float(np.mean(predictions != baseline_predictions)),
        }

    summary = {
        "checkpoint": str(checkpoint_path),
        "manifest": str(args.manifest.resolve()),
        "device": str(device),
        "samples": len(dataset),
        "method": (
            "交换编码器输出；Cross-Attention 模型会同时交换 Depth 全局特征与逐帧空间特征，"
            "再重新计算 Skeleton→Depth 注意力和融合"
            if model.use_cross_attention
            else "独立编码器输出特征置换；对后融合 D+S 架构等价于交换对应原始模态后再编码"
        ),
        "baseline": dict(
            zip(("accuracy", "balanced_accuracy", "macro_f1"), baseline_metrics.tolist())
        ),
        "depth_wrong_class_same_user": repeated_test("depth", "same_user_different_label"),
        "depth_same_class": repeated_test("depth", "same_label"),
        "skeleton_wrong_class_same_user": repeated_test(
            "skeleton", "same_user_different_label"
        ),
        "depth_mean_replacement": mean_replacement("depth"),
        "skeleton_mean_replacement": mean_replacement("skeleton"),
    }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
