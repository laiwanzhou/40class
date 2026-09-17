"""P92 V-JEPA 2 SSV2 twelve-view visual teacher, fold-0 first.

Views:
  * early/late x scene/person/workspace (existing P90 dynamic person ROIs)
  * full/motion-peak x left-hand/right-hand/interaction (P91 hand ROIs)

Extraction is resumable at trial granularity.  Candidate, regularization and
blend choices are made on H2 after training on H1+embargo.  The exact choices
are then refit on all fold-0 source users and audited on H3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from scipy.special import softmax
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoVideoProcessor,
    VJEPA2ForVideoClassification,
)

from audit_yolo11_pose_skeleton import frame_map
from build_p30_shared_dir_roi_feature_cache import read_ir
from build_p46_videomae_cache import safe_relative, square_crop
from build_p46_videomae_multiclip_cache import WINDOW_BOUNDS
from p90_teacher_common import REPO_ROOT, classification_metrics, load_protocol
from p90_videomae_lora_teacher import P29_RUN, prepare_clips, read_aligned_rows
from p90_videomaev2_distilled_teacher import class_sample_weights
from p91_hierarchical_multimodal_teacher import audit, blend_prediction
from p91_unrestricted_fusion_teacher import build_cohorts
from p91_videomaev2_hand_teacher import (
    centers,
    fallback_box,
    interaction_box,
    prepare_hand_clips,
)
from train_p46_videomae_head import l2_normalize, make_model, row_standardize


MODEL_REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
DEFAULT_OUTPUT = REPO_ROOT / "runs/p92_vjepa2_vitl_ssv2_12view_fold0_v1"
P91_CHAMPION = REPO_ROOT / "runs/p91_hierarchical_multimodal_h3_v3"
VMAE = REPO_ROOT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz"
IV2 = REPO_ROOT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_REPO)
    parser.add_argument("--clip-batch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def align(ids: np.ndarray, values: np.ndarray, target_ids: np.ndarray) -> np.ndarray:
    lookup = {str(sample_id): row for row, sample_id in enumerate(ids)}
    return np.asarray([values[lookup[str(sample_id)]] for sample_id in target_ids])


def bounded_indices(low: int, high: int, count: int) -> np.ndarray:
    if high <= low:
        return np.full(count, max(low, 0), dtype=np.int64)
    return np.rint(np.linspace(low, high - 1, count)).astype(np.int64)


def prepare_long_global_clips(
    row: dict[str, str], frames_per_clip: int
) -> list[list[np.ndarray]]:
    cache = P29_RUN / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(cache, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        names = np.asarray(data["region_names"]).astype(str).tolist()
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
    person = names.index("full_body")
    workspace = names.index("hand_workspace")
    paths = frame_map(Path(row["ir_dir"]), "ir")
    clips: list[list[np.ndarray]] = []
    end = len(frame_ids) - 1
    for low, high in WINDOW_BOUNDS:
        chosen = np.rint(
            np.linspace(low * end, high * end, frames_per_clip)
        ).astype(np.int64)
        scene_video: list[np.ndarray] = []
        person_video: list[np.ndarray] = []
        workspace_video: list[np.ndarray] = []
        for frame in chosen:
            image = read_ir(paths[frame_ids[frame]])
            person_box = (
                boxes[frame, person]
                if valid[frame, person]
                else np.full(4, np.nan, dtype=np.float32)
            )
            workspace_box = (
                boxes[frame, workspace] if valid[frame, workspace] else person_box
            )
            scene_video.append(image)
            person_video.append(square_crop(image, person_box, scale=1.15))
            workspace_video.append(square_crop(image, workspace_box, scale=1.40))
        clips.extend((scene_video, person_video, workspace_video))
    return clips


def prepare_long_hand_clips(
    row: dict[str, str], frames_per_clip: int
) -> list[list[np.ndarray]]:
    cache = P29_RUN / "trial_roi_cache" / safe_relative(row["source_id"]).with_suffix(".npz")
    with np.load(cache, allow_pickle=False) as data:
        frame_ids = np.asarray(data["frame_ids"]).astype(str)
        names = np.asarray(data["region_names"]).astype(str).tolist()
        boxes = np.asarray(data["roi_boxes_xyxy"], dtype=np.float32)
        valid = np.asarray(data["roi_valid"], dtype=bool)
    left = names.index("left_hand")
    right = names.index("right_hand")
    workspace = names.index("hand_workspace")
    person = names.index("full_body")
    left_center = centers(boxes[:, left], valid[:, left])
    right_center = centers(boxes[:, right], valid[:, right])
    velocity = np.zeros(len(boxes), dtype=np.float32)
    if len(boxes) > 1:
        velocity[1:] = np.linalg.norm(np.diff(left_center, axis=0), axis=1)
        velocity[1:] += np.linalg.norm(np.diff(right_center, axis=0), axis=1)
    if len(velocity) >= 5:
        velocity = np.convolve(
            velocity, np.ones(5, dtype=np.float32) / 5, mode="same"
        )
    center = int(np.argmax(velocity))
    span = min(len(boxes), max(frames_per_clip, int(round(len(boxes) * 0.45))))
    motion_low = min(max(center - span // 2, 0), max(len(boxes) - span, 0))
    windows = [
        bounded_indices(0, len(frame_ids), frames_per_clip),
        bounded_indices(motion_low, motion_low + span, frames_per_clip),
    ]
    paths = frame_map(Path(row["ir_dir"]), "ir")
    clips: list[list[np.ndarray]] = []
    for chosen in windows:
        left_video: list[np.ndarray] = []
        right_video: list[np.ndarray] = []
        interaction_video: list[np.ndarray] = []
        for frame in chosen:
            image = read_ir(paths[frame_ids[frame]])
            left_box = fallback_box(boxes, valid, frame, left, workspace, person)
            right_box = fallback_box(boxes, valid, frame, right, workspace, person)
            both_box = interaction_box(
                boxes, valid, frame, left, right, workspace, person
            )
            left_video.append(square_crop(image, left_box, scale=1.85))
            right_video.append(square_crop(image, right_box, scale=1.85))
            interaction_video.append(square_crop(image, both_box, scale=1.55))
        clips.extend((left_video, right_video, interaction_video))
    return clips


class TwelveViewDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        indices: np.ndarray,
        frames_per_clip: int = 16,
    ) -> None:
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.frames_per_clip = int(frames_per_clip)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index = int(self.indices[index])
        row = self.rows[row_index]
        if self.frames_per_clip == 16:
            global_views = prepare_clips(row, "ir")
            hand_views = prepare_hand_clips(row)
        else:
            global_views = prepare_long_global_clips(row, self.frames_per_clip)
            hand_views = prepare_long_hand_clips(row, self.frames_per_clip)
        if len(global_views) != 6 or len(hand_views) != 6:
            raise ValueError(
                f"expected 6+6 views, got {len(global_views)}+{len(hand_views)}"
            )
        return {
            "row": row_index,
            "sample_id": row["sample_id"],
            "videos": [*global_views, *hand_views],
        }


def one_trial(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("trial loader must use batch_size=1")
    return batch[0]


def model_ready_video(
    video: list[np.ndarray], frames_per_clip: int, crop_size: int
) -> np.ndarray:
    """Match V-JEPA resize+center-crop before the processor batches frames."""
    frames = []
    for frame in video:
        image = Image.fromarray(np.asarray(frame))
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        resize_short = int(round(crop_size * 292.0 / 256.0))
        scale = resize_short / max(min(width, height), 1)
        resized = image.resize(
            (
                max(crop_size, int(round(width * scale))),
                max(crop_size, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        left = max((resized.width - crop_size) // 2, 0)
        top = max((resized.height - crop_size) // 2, 0)
        frames.append(
            np.asarray(
                resized.crop((left, top, left + crop_size, top + crop_size))
            )
        )
    value = np.stack(frames)
    if value.shape != (frames_per_clip, crop_size, crop_size, 3):
        raise ValueError(f"unexpected prepared video shape: {value.shape}")
    return value


def open_cache(
    output: Path, samples: int, hidden: int, labels: int
) -> tuple[np.memmap, np.memmap | None, np.memmap]:
    specs = [
        ("features.npy", np.float16, (samples, 12, hidden)),
        ("done.npy", np.bool_, (samples,)),
    ]
    if labels > 0:
        specs.insert(1, ("ssv2_logits.npy", np.float16, (samples, 12, labels)))
    arrays: list[np.memmap] = []
    for filename, dtype, shape in specs:
        path = output / filename
        if path.exists():
            value = np.lib.format.open_memmap(path, mode="r+")
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(
                    f"cache contract mismatch for {path}: {value.shape}/{value.dtype} "
                    f"!= {shape}/{dtype}"
                )
        else:
            value = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
            value[...] = False if dtype == np.bool_ else 0
            value.flush()
        arrays.append(value)
    if labels > 0:
        return arrays[0], arrays[1], arrays[2]
    return arrays[0], None, arrays[1]


def write_progress(
    output: Path,
    model_repo: str,
    model_path: Path,
    done: np.ndarray,
    started: float,
    peak_gib: float,
    complete: bool,
    frames_per_clip: int,
    crop_size: int,
    hidden_size: int,
    classifier_labels: int,
) -> None:
    (output / "cache_summary.json").write_text(
        json.dumps(
            {
                "model_repo": model_repo,
                "snapshot": str(model_path),
                "views": (
                    "early/late x scene/person/workspace + "
                    "full/motion-peak x left/right/interaction"
                ),
                "completed_samples": int(done.sum()),
                "total_samples": int(len(done)),
                "complete": complete,
                "frames_per_clip": frames_per_clip,
                "crop_size": crop_size,
                "hidden_size": hidden_size,
                "classifier_labels": classifier_labels,
                "elapsed_seconds_this_run": time.time() - started,
                "peak_cuda_gib": peak_gib,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@torch.inference_mode()
def extract(args: argparse.Namespace, output: Path) -> bool:
    protocol = load_protocol()
    rows = read_aligned_rows()
    if [row["sample_id"] for row in rows] != protocol.sample_ids.tolist():
        raise ValueError("row order differs from P90 protocol")
    snapshot = Path(
        snapshot_download(
            args.model,
            allow_patterns=("*.json", "*.txt", "*.safetensors"),
            local_files_only=True,
        )
    )
    processor = AutoVideoProcessor.from_pretrained(snapshot, local_files_only=True)
    model_config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    architectures = set(getattr(model_config, "architectures", ()) or ())
    has_classifier = "VJEPA2ForVideoClassification" in architectures
    if has_classifier:
        model = VJEPA2ForVideoClassification.from_pretrained(
            snapshot, local_files_only=True, dtype=dtype
        ).to(device)
    else:
        model = AutoModel.from_pretrained(
            snapshot, local_files_only=True, dtype=dtype
        ).to(device)
    model.eval()
    hidden = int(model.config.hidden_size)
    label_count = int(model.config.num_labels) if has_classifier else 0
    frames_per_clip = int(model.config.frames_per_clip)
    crop_size = int(model.config.crop_size)
    if frames_per_clip not in {16, 32, 64}:
        raise ValueError(f"unexpected frame contract: {frames_per_clip}")
    features, action_logits, done = open_cache(
        output, len(protocol.labels), hidden, label_count
    )
    pending = np.flatnonzero(~np.asarray(done, dtype=bool))
    if args.max_samples:
        pending = pending[: args.max_samples]
    if not len(pending):
        return bool(np.asarray(done).all())
    loader = DataLoader(
        TwelveViewDataset(rows, pending, frames_per_clip),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=one_trial,
        pin_memory=False,
        persistent_workers=args.num_workers > 0,
    )
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    completed_this_run = 0
    for item in loader:
        row = int(item["row"])
        feature_parts = []
        logit_parts = []
        videos = item["videos"]
        for low in range(0, len(videos), args.clip_batch):
            prepared = [
                model_ready_video(video, frames_per_clip, crop_size)
                for video in videos[low : low + args.clip_batch]
            ]
            encoded = processor(
                prepared,
                return_tensors="pt",
                do_resize=False,
                do_center_crop=False,
            )
            pixels = encoded.pixel_values_videos.to(
                device=device, dtype=dtype, non_blocking=True
            )
            with torch.autocast(
                device_type=device.type, dtype=dtype, enabled=device.type == "cuda"
            ):
                if has_classifier:
                    backbone = model.vjepa2(
                        pixel_values_videos=pixels, skip_predictor=True
                    )
                    pooled = model.pooler(backbone.last_hidden_state)
                    logits = model.classifier(pooled)
                else:
                    backbone = model(
                        pixel_values_videos=pixels, skip_predictor=True
                    )
                    pooled = backbone.last_hidden_state.mean(dim=1)
                    logits = None
            feature_parts.append(pooled.float().cpu().numpy())
            if logits is not None:
                logit_parts.append(logits.float().cpu().numpy())
        feature_value = np.concatenate(feature_parts)
        if feature_value.shape != (12, hidden):
            raise ValueError(f"unexpected V-JEPA feature output {feature_value.shape}")
        features[row] = feature_value.astype(np.float16)
        if action_logits is not None:
            logit_value = np.concatenate(logit_parts)
            if logit_value.shape != (12, label_count):
                raise ValueError(f"unexpected V-JEPA classifier output {logit_value.shape}")
            action_logits[row] = logit_value.astype(np.float16)
        done[row] = True
        completed_this_run += 1
        if completed_this_run % args.flush_every == 0:
            features.flush()
            if action_logits is not None:
                action_logits.flush()
            done.flush()
            peak = (
                float(torch.cuda.max_memory_allocated() / 2**30)
                if device.type == "cuda"
                else 0.0
            )
            write_progress(
                output,
                args.model,
                snapshot,
                done,
                started,
                peak,
                False,
                frames_per_clip,
                crop_size,
                hidden,
                label_count,
            )
            print(
                f"vjepa2 extracted={int(np.asarray(done).sum())}/{len(done)} "
                f"this_run={completed_this_run} elapsed_min={(time.time()-started)/60:.1f} "
                f"peak_gib={peak:.2f}",
                flush=True,
            )
    features.flush()
    if action_logits is not None:
        action_logits.flush()
    done.flush()
    complete = bool(np.asarray(done).all())
    peak = (
        float(torch.cuda.max_memory_allocated() / 2**30)
        if device.type == "cuda"
        else 0.0
    )
    write_progress(
        output,
        args.model,
        snapshot,
        done,
        started,
        peak,
        complete,
        frames_per_clip,
        crop_size,
        hidden,
        label_count,
    )
    print(
        f"extraction run finished complete={complete} "
        f"done={int(np.asarray(done).sum())}/{len(done)}",
        flush=True,
    )
    return complete


def feature_candidates(output: Path) -> dict[str, np.ndarray]:
    features = np.asarray(np.load(output / "features.npy", mmap_mode="r"), dtype=np.float32)
    features = l2_normalize(features)
    global_mean = l2_normalize(features[:, :6].mean(axis=1))
    hand_mean = l2_normalize(features[:, 6:].mean(axis=1))
    all_mean = l2_normalize(features.mean(axis=1))
    vmae = load_npz(VMAE)
    iv2 = load_npz(IV2)
    protocol = load_protocol()
    if not np.array_equal(vmae["sample_ids"].astype(str), protocol.sample_ids.astype(str)):
        raise ValueError("VMAE ID order mismatch")
    if not np.array_equal(iv2["sample_ids"].astype(str), protocol.sample_ids.astype(str)):
        raise ValueError("IV2 ID order mismatch")
    old_ir = np.concatenate(
        (
            l2_normalize(vmae["features"].astype(np.float32)).reshape(len(features), -1),
            l2_normalize(iv2["features"].astype(np.float32)).reshape(len(features), -1),
        ),
        axis=1,
    )
    candidates = {
        "vjepa_global6": features[:, :6].reshape(len(features), -1),
        "vjepa_hand6": features[:, 6:].reshape(len(features), -1),
        "vjepa_all12": features.reshape(len(features), -1),
        "vjepa_mean3": np.concatenate((global_mean, hand_mean, all_mean), axis=1),
        "old_ir_plus_vjepa_mean3": np.concatenate(
            (old_ir, global_mean, hand_mean, all_mean), axis=1
        ),
    }
    action_path = output / "ssv2_logits.npy"
    if action_path.exists():
        action = np.asarray(np.load(action_path, mmap_mode="r"), dtype=np.float32)
        action = row_standardize(action)
        candidates["vjepa_all12_ssv2"] = np.concatenate(
            (features.reshape(len(features), -1), action.reshape(len(features), -1)),
            axis=1,
        )
    return candidates


def h2_h3_champion(
    h2_ids: np.ndarray, h3_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inner = load_npz(P91_CHAMPION / "inner_predictions.npz")
    target = load_npz(P91_CHAMPION / "predictions.npz")
    h2_logits = align(inner["sample_ids"].astype(str), inner["direct_logits"], h2_ids)
    h2_teacher = align(
        inner["sample_ids"].astype(str), inner["teacher_prediction"], h2_ids
    )
    weight = float(np.asarray(inner["selected_constant_weight"]).reshape(-1)[0])
    h2_prediction = blend_prediction(h2_logits, h2_teacher, weight)
    h3_prediction = align(
        target["sample_ids"].astype(str), target["blended_prediction"], h3_ids
    )
    return h2_prediction, h3_prediction


def conservative_blend(logits: np.ndarray, base: np.ndarray, weight: float) -> np.ndarray:
    probability = softmax(logits, axis=1)
    anchor = np.full((len(base), 40), 0.04 / 39.0, dtype=np.float64)
    anchor[np.arange(len(base)), base] = 0.96
    score = weight * np.log(np.clip(probability, 1e-9, 1.0))
    score += (1.0 - weight) * np.log(np.clip(anchor, 1e-9, 1.0))
    return score.argmax(axis=1)


def user_metrics(
    labels: np.ndarray, base: np.ndarray, prediction: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    values = {}
    nets = []
    for user in np.unique(users):
        selected = users == user
        values[str(user)] = float(np.mean(prediction[selected] == labels[selected]))
        nets.append(
            int(np.sum(prediction[selected] == labels[selected]))
            - int(np.sum(base[selected] == labels[selected]))
        )
    return {
        "per_user": values,
        "worst_user_accuracy": float(min(values.values())),
        "negative_users": int(np.sum(np.asarray(nets) < 0)),
        "worst_user_net": int(min(nets, default=0)),
    }


def evaluate(output: Path) -> dict[str, Any]:
    protocol = load_protocol()
    candidates = feature_candidates(output)
    cohorts = build_cohorts()
    index = {str(sample_id): row for row, sample_id in enumerate(protocol.sample_ids)}
    h1 = np.asarray([index[str(value)] for value in cohorts["H1_selection"].sample_ids])
    h2 = np.asarray([index[str(value)] for value in cohorts["H2_confirmation"].sample_ids])
    embargo = np.asarray(
        [index[str(value)] for value in cohorts["E0_p87_sequence_source"].sample_ids]
    )
    h3 = protocol.val_indices(0)
    inner_train = np.concatenate((h1, embargo))
    final_train = protocol.train_indices(0)
    labels = protocol.labels
    h2_base, h3_base = h2_h3_champion(protocol.sample_ids[h2], protocol.sample_ids[h3])
    recipes = {
        "vjepa_global6": 3000.0,
        "vjepa_hand6": 3000.0,
        "vjepa_all12": 5000.0,
        "vjepa_mean3": 1500.0,
        "old_ir_plus_vjepa_mean3": 5000.0,
    }
    if "vjepa_all12_ssv2" in candidates:
        recipes["vjepa_all12_ssv2"] = 5000.0
    inner_results: dict[str, Any] = {}
    selected_rows = []
    for name, values in candidates.items():
        model = make_model(recipes[name])
        model.fit(
            values[inner_train], labels[inner_train],
            ridge__sample_weight=class_sample_weights(labels[inner_train], 0.75),
        )
        logits = np.asarray(model.decision_function(values[h2]), dtype=np.float32)
        grid = []
        for weight in np.linspace(0.0, 1.0, 41):
            prediction = conservative_blend(logits, h2_base, float(weight))
            row = {
                "weight": float(weight),
                **audit(labels[h2], h2_base, prediction),
                **user_metrics(labels[h2], h2_base, prediction, protocol.users[h2]),
            }
            grid.append(row)
        selected = max(
            grid,
            key=lambda row: (
                row["correct"], -row["harm"], -row["negative_users"],
                row["worst_user_net"], -row["weight"],
            ),
        )
        direct = logits.argmax(1)
        inner_results[name] = {
            "dimensions": int(values.shape[1]),
            "direct": classification_metrics(logits, labels[h2]),
            "direct_vs_champion": audit(labels[h2], h2_base, direct),
            "selected_blend": selected,
            "top_blends": sorted(grid, key=lambda row: row["correct"], reverse=True)[:5],
        }
        selected_rows.append({"candidate": name, **selected})
        print(
            f"H2 {name}: direct={inner_results[name]['direct']['accuracy']:.6f} "
            f"blend={selected['accuracy']:.6f} net={selected['net']:+d}", flush=True
        )
    chosen = max(
        selected_rows,
        key=lambda row: (
            row["correct"], -row["harm"], -row["negative_users"],
            row["worst_user_net"], -row["weight"],
        ),
    )

    final_results: dict[str, Any] = {}
    logits_to_save = {}
    union = h3_base == labels[h3]
    for name, values in candidates.items():
        model = make_model(recipes[name])
        model.fit(
            values[final_train], labels[final_train],
            ridge__sample_weight=class_sample_weights(labels[final_train], 0.75),
        )
        logits = np.asarray(model.decision_function(values[h3]), dtype=np.float32)
        direct = logits.argmax(1)
        weight = float(inner_results[name]["selected_blend"]["weight"])
        blended = conservative_blend(logits, h3_base, weight)
        final_results[name] = {
            "direct": classification_metrics(logits, labels[h3]),
            "direct_vs_champion": audit(labels[h3], h3_base, direct),
            "source_selected_blend": {
                **audit(labels[h3], h3_base, blended),
                **user_metrics(labels[h3], h3_base, blended, protocol.users[h3]),
                "weight": weight,
            },
            "union_oracle_with_champion": float(
                np.mean((h3_base == labels[h3]) | (direct == labels[h3]))
            ),
        }
        union |= direct == labels[h3]
        logits_to_save[f"{name}_logits"] = logits
        print(
            f"H3 {name}: direct={final_results[name]['direct']['accuracy']:.6f} "
            f"blend={final_results[name]['source_selected_blend']['accuracy']:.6f} "
            f"net={final_results[name]['source_selected_blend']['net']:+d}", flush=True
        )
    selected_name = str(chosen["candidate"])
    cache_summary = json.loads((output / "cache_summary.json").read_text(encoding="utf-8"))
    report = {
        "protocol": (
            "H1+embargo trains fixed Ridge; H2 selects candidate/blend; fold0 source refit; "
            "one frozen H3 audit. V-JEPA backbone remains frozen."
        ),
        "model": {
            "repo": cache_summary["model_repo"],
            "hidden_size": int(cache_summary.get("hidden_size", 1024)),
            "frames_per_clip": int(cache_summary.get("frames_per_clip", 16)),
            "crop_size": int(cache_summary.get("crop_size", 256)),
            "pretraining_head": (
                f"video classifier with {cache_summary.get('classifier_labels', 174)} labels"
                if int(cache_summary.get("classifier_labels", 174)) > 0
                else "self-supervised V-JEPA 2 representation; mean token pooling"
            ),
            "views": 12,
        },
        "baselines": {
            "h2_p91_champion": float(np.mean(h2_base == labels[h2])),
            "h3_p91_champion": float(np.mean(h3_base == labels[h3])),
        },
        "inner": inner_results,
        "selected_on_h2": chosen,
        "final": final_results,
        "selected_h3": final_results[selected_name]["source_selected_blend"],
        "all_candidate_union_oracle": float(union.mean()),
    }
    np.savez_compressed(
        output / "fold0_logits.npz",
        sample_ids=protocol.sample_ids[h3], labels=labels[h3],
        champion_prediction=h3_base, selected_candidate=np.asarray(selected_name),
        **logits_to_save,
    )
    (output / "fold0_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    complete = extract(args, output)
    if not complete or args.extract_only:
        print(json.dumps({"complete": complete, "evaluation_skipped": True}), flush=True)
        return
    report = evaluate(output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
