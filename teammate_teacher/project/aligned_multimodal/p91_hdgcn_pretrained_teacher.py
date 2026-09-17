"""Official NTU60 HD-GCN six-stream transfer, screened on P90 fold 0."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from p90_motionbert_teacher import interpolate_pose
from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomaev2_distilled_teacher import class_sample_weights
from train_p46_videomae_head import l2_normalize, make_model


HERE = Path(__file__).resolve().parent
HDGCN_ROOT = REPO_ROOT.parent / "external_data/HD-GCN"
PRETRAINED = HDGCN_ROOT / "pretrained"
RAW_CACHE = HERE / "cache/skeleton_raw"
DEFAULT_OUTPUT = REPO_ROOT / "runs/p91_hdgcn_ntu60_xsub_fold0_v1"
STREAMS = (
    ("joint_com1", "ntu60_xsub_joint_com1.pt", 1, False),
    ("joint_com2", "ntu60_xsub_joint_com2.pt", 2, False),
    ("joint_com21", "ntu60_xsub_joint_com21.pt", 21, False),
    ("bone_com1", "ntu60_xsub_bone_com1.pt", 1, True),
    ("bone_com2", "ntu60_xsub_bone_com2.pt", 2, True),
    ("bone_com21", "ntu60_xsub_bone_com21.pt", 21, True),
)
H36M_TO_NTU = np.asarray(
    [0, 7, 9, 10, 11, 12, 13, 13, 14, 15, 16, 16, 4, 5, 6, 6, 1, 2, 3, 3, 8, 13, 13, 16, 16],
    dtype=np.int64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reuse-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def prepare_ntu_cache(path: Path) -> dict[str, np.ndarray]:
    protocol = load_protocol()
    metadata = json.loads((RAW_CACHE / "metadata.json").read_text(encoding="utf-8"))
    if metadata["sample_ids"] != protocol.sample_ids.tolist():
        raise ValueError("raw Skeleton cache order differs from P90 protocol")
    raw = np.load(RAW_CACHE / "skeleton_raw_float32.npy", mmap_mode="r")
    pose = np.zeros((len(protocol.labels), 3, 64, 25, 2), dtype=np.float32)
    started = time.time()
    for row, (start, length) in enumerate(metadata["offsets"]):
        xyz, confidence = interpolate_pose(np.asarray(raw[start : start + length]), 64)
        mapped = xyz[:, H36M_TO_NTU].copy()
        mapped -= mapped[0, 1][None, None, :]
        mapped[confidence[:, H36M_TO_NTU] <= 0] = 0
        pose[row, :, :, :, 0] = mapped.transpose(2, 0, 1)
        if (row + 1) % 500 == 0:
            print(f"  NTU map={row + 1}/{len(protocol.labels)}", flush=True)
    payload = {
        "sample_ids": protocol.sample_ids,
        "labels": protocol.labels,
        "pose": pose,
        "mapping": H36M_TO_NTU,
    }
    np.savez_compressed(path, **payload)
    (path.parent / "pose_cache_summary.json").write_text(
        json.dumps(
            {
                "source": str(RAW_CACHE / "skeleton_raw_float32.npy"),
                "target_shape": list(pose.shape),
                "normalization": "subtract first-frame mapped NTU spine-mid joint",
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def bone_input(pose: torch.Tensor) -> torch.Tensor:
    if str(HDGCN_ROOT) not in sys.path:
        sys.path.insert(0, str(HDGCN_ROOT))
    from feeders.bone_pairs import ntu_pairs

    output = torch.zeros_like(pose)
    for child, parent in ntu_pairs:
        output[:, :, :, child - 1] = pose[:, :, :, child - 1] - pose[:, :, :, parent - 1]
    return output


def build_model(com: int, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    if str(HDGCN_ROOT) not in sys.path:
        sys.path.insert(0, str(HDGCN_ROOT))
    from model.HDGCN import Model

    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        graph="graph.ntu_rgb_d_hierarchy.Graph",
        graph_args={"labeling_mode": "spatial", "CoM": com},
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def forward_tokens(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    n, c, t, v, m = x.shape
    value = x.permute(0, 4, 3, 1, 2).contiguous().reshape(n, m * v * c, t)
    value = model.data_bn(value)
    value = value.reshape(n, m, v, c, t).permute(0, 1, 3, 4, 2).reshape(n * m, c, t, v)
    for layer in (model.l1, model.l2, model.l3, model.l4, model.l5, model.l6, model.l7, model.l8, model.l9, model.l10):
        value = layer(value)
    channels, frames = value.shape[1], value.shape[2]
    value = value.reshape(n, m, channels, frames, v).mean(dim=4).mean(dim=1)
    return value.transpose(1, 2).contiguous()


@torch.inference_mode()
def extract_features(
    pose_data: dict[str, np.ndarray], cache: Path, batch_size: int
) -> dict[str, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose = torch.from_numpy(np.asarray(pose_data["pose"], dtype=np.float32))
    loader = DataLoader(TensorDataset(pose), batch_size=batch_size, shuffle=False, num_workers=0)
    stream_tokens = []
    summary: dict[str, Any] = {}
    for name, filename, com, use_bone in STREAMS:
        checkpoint = PRETRAINED / filename
        model = build_model(com, checkpoint, device)
        parts = []
        started = time.time()
        for batch_number, (batch,) in enumerate(loader):
            batch = batch.to(device, non_blocking=True)
            if use_bone:
                batch = bone_input(batch)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                tokens = forward_tokens(model, batch)
            parts.append(tokens.float().cpu().numpy().astype(np.float16))
            if (batch_number + 1) % 25 == 0:
                print(
                    f"  {name}={min((batch_number + 1) * batch_size, len(pose))}/{len(pose)}",
                    flush=True,
                )
        values = np.concatenate(parts)
        stream_tokens.append(values)
        summary[name] = {
            "checkpoint": str(checkpoint),
            "tokens": list(values.shape),
            "elapsed_seconds": time.time() - started,
        }
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None
    tokens = np.stack(stream_tokens, axis=1)
    payload = {
        "sample_ids": pose_data["sample_ids"],
        "labels": pose_data["labels"],
        "stream_names": np.asarray([row[0] for row in STREAMS]),
        "tokens": tokens,
        "features": tokens.mean(axis=2),
    }
    np.savez_compressed(cache, **payload)
    (cache.parent / "feature_cache_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def evaluate(data: dict[str, np.ndarray], output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    if not np.array_equal(data["sample_ids"].astype(str), protocol.sample_ids.astype(str)):
        raise ValueError("HD-GCN cache order mismatch")
    features = l2_normalize(np.asarray(data["features"], dtype=np.float32))
    names = data["stream_names"].astype(str).tolist()
    matrices = {name: features[:, index] for index, name in enumerate(names)}
    matrices.update(
        {
            "joint_mean": l2_normalize(features[:, :3].mean(axis=1)),
            "bone_mean": l2_normalize(features[:, 3:].mean(axis=1)),
            "six_mean": l2_normalize(features.mean(axis=1)),
            "six_concat": features.reshape(len(features), -1),
        }
    )
    train = protocol.train_indices(0)
    val = protocol.val_indices(0)
    labels = protocol.labels
    with np.load(
        REPO_ROOT / "runs/p90_crossuser_visual_router_v1/gate_predictions.npz",
        allow_pickle=False,
    ) as router:
        router_lookup = {
            str(key): int(value)
            for key, value in zip(router["sample_ids"], router["router_prediction"])
        }
    base = np.asarray([router_lookup[str(value)] for value in protocol.sample_ids[val]])
    fusion = load_npz(REPO_ROOT / "runs/p91_hierarchical_multimodal_h3_v3/predictions.npz")
    fusion_lookup = {
        str(key): int(value)
        for key, value in zip(fusion["sample_ids"], fusion["blended_prediction"])
    }
    fusion_prediction = np.asarray(
        [fusion_lookup[str(value)] for value in protocol.sample_ids[val]], dtype=np.int64
    )
    results = {}
    logits_out = {}
    union = fusion_prediction == labels[val]
    for name, values in matrices.items():
        model = make_model(1000.0 if values.shape[1] <= 256 else 3000.0)
        model.fit(
            values[train],
            labels[train],
            ridge__sample_weight=class_sample_weights(labels[train], 0.75),
        )
        logits = np.asarray(model.decision_function(values[val]), dtype=np.float32)
        prediction = logits.argmax(axis=1)
        correct = prediction == labels[val]
        results[name] = {
            "dimensions": int(values.shape[1]),
            "metrics": classification_metrics(logits, labels[val]),
            "rescue_over_p90": int(np.sum((base != labels[val]) & correct)),
            "rescue_over_p91_fusion": int(
                np.sum((fusion_prediction != labels[val]) & correct)
            ),
            "union_with_p91_accuracy": float(np.mean(union | correct)),
        }
        union |= correct
        logits_out[name] = logits
        print(f"  {name}: {results[name]}", flush=True)
    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=protocol.sample_ids[val],
        labels=labels[val],
        p90_prediction=base,
        p91_fusion_prediction=fusion_prediction,
        **{f"{name}_logits": values for name, values in logits_out.items()},
    )
    return {
        "protocol": "official NTU60 xsub frozen weights; P90 fold0 fixed Ridge heads",
        "baselines": {
            "p90": float(np.mean(base == labels[val])),
            "p91_fusion": float(np.mean(fusion_prediction == labels[val])),
        },
        "candidates": results,
        "all_union_with_p91_accuracy": float(union.mean()),
    }


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose_cache = output / "ntu25_pose.npz"
    feature_cache = output / "complete_features.npz"
    pose_data = load_npz(pose_cache) if args.reuse_cache and pose_cache.exists() else prepare_ntu_cache(pose_cache)
    features = (
        load_npz(feature_cache)
        if args.reuse_cache and feature_cache.exists()
        else extract_features(pose_data, feature_cache, args.batch_size)
    )
    report = evaluate(features, output)
    (output / "fold0_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
