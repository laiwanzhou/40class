"""P91 unrestricted cross-modal fusion teacher on frozen subject-disjoint features.

This is the first, size-unconstrained stage of P91.  It deliberately does not
patch the P87 student.  Instead, it learns a new dense fusion representation
from the strongest available visual, skeleton and IMU teachers, while keeping
the P89 safe distribution as one input expert rather than as the architecture.

Protocol
--------
* H1 + H2 are the labelled source users.
* Model/epoch/blend choices are made by three inner user-disjoint folds.
* H3/fold0 is touched once for the outer result.
* Every learned feature and logit consumed here is OOF for its own classifier.
* The large frozen teacher features are allowed to exceed the final 100 MB
  budget.  Compression is a later stage and is not performed by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from p90_crossuser_visual_router import load_splits
from p90_teacher_fusion_audit import align
from p90_visual_teacher_safe_fusion_audit import load_visual_candidates


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_unrestricted_fusion_h3_v1"
EPSILON = 1e-7


@dataclass
class Cohort:
    name: str
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    safe_prediction: np.ndarray
    expert_names: list[str]
    expert_probability: np.ndarray
    vmae_tokens: np.ndarray
    iv2_tokens: np.ndarray
    motionbert_tokens: np.ndarray
    skeleton_statistics: np.ndarray
    imu_statistics: np.ndarray
    relation_statistics: np.ndarray


@dataclass
class Prepared:
    name: str
    sample_ids: np.ndarray
    labels: np.ndarray
    users: np.ndarray
    safe_prediction: np.ndarray
    expert_probability: np.ndarray
    vmae_tokens: np.ndarray
    iv2_tokens: np.ndarray
    motionbert_tokens: np.ndarray
    skeleton_statistics: np.ndarray
    imu_statistics: np.ndarray
    relation_statistics: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-3)
    parser.add_argument("--statistics-dim", type=int, default=64)
    parser.add_argument("--seeds", default="17,43,71")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--include-embargo-source",
        action="store_true",
        help="Add user1/user2/user21 with strictly OOF P87 sequence anchors to the H1+H2 source.",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def normalise_probability(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64)
    probability = np.clip(probability, EPSILON, None)
    return (probability / probability.sum(axis=1, keepdims=True)).astype(np.float32)


def safe_decoded_probability(prediction: np.ndarray, confidence: float = 0.94) -> np.ndarray:
    output = np.full((len(prediction), 40), (1.0 - confidence) / 39.0, dtype=np.float32)
    output[np.arange(len(prediction)), prediction.astype(np.int64)] = confidence
    return output


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def aligned_array(
    source: dict[str, np.ndarray], target_ids: np.ndarray, key: str
) -> np.ndarray:
    return align(source["sample_ids"].astype(str), source[key], target_ids.astype(str))


def build_cohorts() -> dict[str, Cohort]:
    splits = load_splits()
    vmae = load_npz(PROJECT / "runs/p90_videomaev2_distilled_teacher_v1/complete_features.npz")
    iv2 = load_npz(PROJECT / "runs/p90_internvideo2_l_k400_teacher_v1/complete_features.npz")
    motionbert_features = load_npz(
        PROJECT / "runs/p90_motionbert_teacher_v1/features_pretrain_front_t81.npz"
    )
    motionbert_logits = load_npz(
        PROJECT / "runs/p90_motionbert_teacher_v1/motionbert_pretrain_front_linear_oof.npz"
    )
    skeleton = load_npz(
        HERE / "runs/p89_skeleton_invariant_expert_v1/oof_logits.npz"
    )
    imu = load_npz(
        PROJECT / "runs/p90_imu_teacher_blend_v1/imu_p90_sensorwise_plus_deep_crossfit_oof.npz"
    )
    statistics = load_npz(
        HERE / "runs/p89_crossmodal_statistics_v1/crossmodal_statistics.npz"
    )

    output: dict[str, Cohort] = {}
    for name, split in splits.items():
        ids = split.sample_ids.astype(str)
        visual_names = sorted(split.visual_probability)
        expert_names = [
            "p89_safe_decoded",
            "p89_safe_raw",
            *visual_names,
            "skeleton_invariant",
            "motionbert_front",
            "imu_sensorwise_deep",
        ]
        expert_probability = [
            safe_decoded_probability(split.safe_prediction),
            normalise_probability(split.safe_probability),
            *[normalise_probability(split.visual_probability[key]) for key in visual_names],
            normalise_probability(
                softmax(aligned_array(skeleton, ids, "skeleton_logits"), axis=1)
            ),
            normalise_probability(aligned_array(motionbert_logits, ids, "probabilities")),
            normalise_probability(aligned_array(imu, ids, "probabilities")),
        ]
        cross = aligned_array(statistics, ids, "features").astype(np.float32)
        # See p89_crossmodal_statistics_v1/summary.json for the fixed block layout.
        skeleton_statistics = np.concatenate((cross[:, :2629], cross[:, 5729:5740]), axis=1)
        imu_statistics = np.concatenate((cross[:, 2629:5729], cross[:, 5740:5795]), axis=1)
        relation_statistics = cross[:, 5795:6195]
        mb = aligned_array(motionbert_features, ids, "features").astype(np.float32)
        if mb.shape[1] % 768:
            raise ValueError(f"unexpected MotionBERT feature shape: {mb.shape}")
        output[name] = Cohort(
            name=name,
            sample_ids=ids,
            labels=split.labels.astype(np.int64),
            users=split.users.astype(str),
            safe_prediction=split.safe_prediction.astype(np.int64),
            expert_names=expert_names,
            expert_probability=np.stack(expert_probability, axis=1).astype(np.float32),
            vmae_tokens=aligned_array(vmae, ids, "features").reshape(len(ids), -1, 768).astype(np.float32),
            iv2_tokens=aligned_array(iv2, ids, "features").reshape(len(ids), -1, 768).astype(np.float32),
            motionbert_tokens=mb.reshape(len(ids), -1, 768),
            skeleton_statistics=skeleton_statistics,
            imu_statistics=imu_statistics,
            relation_statistics=relation_statistics,
        )
    names = {tuple(value.expert_names) for value in output.values()}
    if len(names) != 1:
        raise ValueError("expert order differs across cohorts")

    # The three P87 embargo users were intentionally excluded from P89 H1/H2/H3,
    # but all frozen modality teachers have strict OOF features for them.  They
    # are useful source-only data for P91.  Their anchor is the P87 sequence OOF
    # prediction (80.18% on these users), never an in-fold refit.
    used = set()
    for split in output.values():
        used.update(split.sample_ids.tolist())
    all_ids = vmae["sample_ids"].astype(str)
    embargo_mask = np.asarray([sample_id not in used for sample_id in all_ids], dtype=bool)
    embargo_ids = all_ids[embargo_mask]
    if len(embargo_ids):
        sequence = load_npz(HERE / "runs/p87_sequence_decoder_v1/oof_predictions.npz")
        structured = load_npz(
            HERE / "runs/p87s_holdout1_structured_targets_v1/structured_targets.npz"
        )
        reference_ids, all_visual = load_visual_candidates()
        safe_prediction = aligned_array(sequence, embargo_ids, "sequence_predictions").astype(
            np.int64
        )
        safe_probability = normalise_probability(
            aligned_array(structured, embargo_ids, "emission_probability")
        )
        visual_names = sorted(all_visual)
        expert_names = [
            "p89_safe_decoded",
            "p89_safe_raw",
            *visual_names,
            "skeleton_invariant",
            "motionbert_front",
            "imu_sensorwise_deep",
        ]
        expert_probability = [
            safe_decoded_probability(safe_prediction),
            safe_probability,
            *[
                normalise_probability(align(reference_ids, all_visual[key], embargo_ids))
                for key in visual_names
            ],
            normalise_probability(
                softmax(aligned_array(skeleton, embargo_ids, "skeleton_logits"), axis=1)
            ),
            normalise_probability(
                aligned_array(motionbert_logits, embargo_ids, "probabilities")
            ),
            normalise_probability(aligned_array(imu, embargo_ids, "probabilities")),
        ]
        cross = aligned_array(statistics, embargo_ids, "features").astype(np.float32)
        mb = aligned_array(motionbert_features, embargo_ids, "features").astype(np.float32)
        output["E0_p87_sequence_source"] = Cohort(
            name="E0_p87_sequence_source",
            sample_ids=embargo_ids,
            labels=aligned_array(vmae, embargo_ids, "labels").astype(np.int64),
            users=aligned_array(vmae, embargo_ids, "users").astype(str),
            safe_prediction=safe_prediction,
            expert_names=expert_names,
            expert_probability=np.stack(expert_probability, axis=1).astype(np.float32),
            vmae_tokens=aligned_array(vmae, embargo_ids, "features")
            .reshape(len(embargo_ids), -1, 768)
            .astype(np.float32),
            iv2_tokens=aligned_array(iv2, embargo_ids, "features")
            .reshape(len(embargo_ids), -1, 768)
            .astype(np.float32),
            motionbert_tokens=mb.reshape(len(embargo_ids), -1, 768),
            skeleton_statistics=np.concatenate(
                (cross[:, :2629], cross[:, 5729:5740]), axis=1
            ),
            imu_statistics=np.concatenate(
                (cross[:, 2629:5729], cross[:, 5740:5795]), axis=1
            ),
            relation_statistics=cross[:, 5795:6195],
        )
    return output


def concatenate(name: str, cohorts: list[Cohort]) -> Cohort:
    expert_names = cohorts[0].expert_names
    return Cohort(
        name=name,
        sample_ids=np.concatenate([item.sample_ids for item in cohorts]),
        labels=np.concatenate([item.labels for item in cohorts]),
        users=np.concatenate([item.users for item in cohorts]),
        safe_prediction=np.concatenate([item.safe_prediction for item in cohorts]),
        expert_names=expert_names,
        expert_probability=np.concatenate([item.expert_probability for item in cohorts]),
        vmae_tokens=np.concatenate([item.vmae_tokens for item in cohorts]),
        iv2_tokens=np.concatenate([item.iv2_tokens for item in cohorts]),
        motionbert_tokens=np.concatenate([item.motionbert_tokens for item in cohorts]),
        skeleton_statistics=np.concatenate([item.skeleton_statistics for item in cohorts]),
        imu_statistics=np.concatenate([item.imu_statistics for item in cohorts]),
        relation_statistics=np.concatenate([item.relation_statistics for item in cohorts]),
    )


class Preprocessor:
    def __init__(self, statistics_dim: int, seed: int) -> None:
        self.statistics_dim = statistics_dim
        self.seed = seed
        self.token_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.pcas: dict[str, PCA] = {}

    def fit(self, source: Cohort) -> "Preprocessor":
        for name in ("vmae_tokens", "iv2_tokens", "motionbert_tokens"):
            values = getattr(source, name).reshape(-1, 768).astype(np.float64)
            mean = values.mean(axis=0).astype(np.float32)
            std = values.std(axis=0).astype(np.float32)
            self.token_stats[name] = (mean, np.maximum(std, 1e-4))
        for name in ("skeleton_statistics", "imu_statistics", "relation_statistics"):
            values = getattr(source, name).astype(np.float64)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values)
            dimensions = min(self.statistics_dim, scaled.shape[0] - 1, scaled.shape[1])
            pca = PCA(
                n_components=dimensions,
                whiten=True,
                svd_solver="randomized",
                random_state=self.seed,
            )
            pca.fit(scaled)
            self.scalers[name] = scaler
            self.pcas[name] = pca
        return self

    def transform(self, source: Cohort) -> Prepared:
        tokens: dict[str, np.ndarray] = {}
        for name, (mean, std) in self.token_stats.items():
            tokens[name] = ((getattr(source, name) - mean) / std).astype(np.float32)
        statistics: dict[str, np.ndarray] = {}
        for name in self.scalers:
            scaled = self.scalers[name].transform(getattr(source, name).astype(np.float64))
            statistics[name] = self.pcas[name].transform(scaled).astype(np.float32)
        return Prepared(
            name=source.name,
            sample_ids=source.sample_ids,
            labels=source.labels,
            users=source.users,
            safe_prediction=source.safe_prediction,
            expert_probability=source.expert_probability,
            vmae_tokens=tokens["vmae_tokens"],
            iv2_tokens=tokens["iv2_tokens"],
            motionbert_tokens=tokens["motionbert_tokens"],
            skeleton_statistics=statistics["skeleton_statistics"],
            imu_statistics=statistics["imu_statistics"],
            relation_statistics=statistics["relation_statistics"],
        )

    def summary(self) -> dict[str, Any]:
        return {
            name: {
                "input_dim": int(self.pcas[name].n_features_in_),
                "output_dim": int(self.pcas[name].n_components_),
                "explained_variance": float(self.pcas[name].explained_variance_ratio_.sum()),
            }
            for name in self.pcas
        }


class PreparedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, data: Prepared, indices: np.ndarray, sample_weight: np.ndarray) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        self.sample_weight = np.asarray(sample_weight, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = int(self.indices[index])
        return {
            "vmae": torch.from_numpy(self.data.vmae_tokens[row]),
            "iv2": torch.from_numpy(self.data.iv2_tokens[row]),
            "motionbert": torch.from_numpy(self.data.motionbert_tokens[row]),
            "skeleton": torch.from_numpy(self.data.skeleton_statistics[row]),
            "imu": torch.from_numpy(self.data.imu_statistics[row]),
            "relation": torch.from_numpy(self.data.relation_statistics[row]),
            "experts": torch.from_numpy(self.data.expert_probability[row]),
            "label": torch.tensor(self.data.labels[row], dtype=torch.long),
            "safe": torch.tensor(self.data.safe_prediction[row], dtype=torch.long),
            "weight": torch.tensor(self.sample_weight[row], dtype=torch.float32),
            "row": torch.tensor(row, dtype=torch.long),
        }


class P91FusionTeacher(nn.Module):
    def __init__(
        self,
        expert_count: int,
        statistics_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.expert_count = expert_count
        self.vmae_project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, model_dim))
        self.iv2_project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, model_dim))
        self.motionbert_project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, model_dim))
        self.skeleton_project = nn.Sequential(
            nn.LayerNorm(statistics_dim), nn.Linear(statistics_dim, model_dim), nn.GELU()
        )
        self.imu_project = nn.Sequential(
            nn.LayerNorm(statistics_dim), nn.Linear(statistics_dim, model_dim), nn.GELU()
        )
        self.relation_project = nn.Sequential(
            nn.LayerNorm(statistics_dim), nn.Linear(statistics_dim, model_dim), nn.GELU()
        )
        self.probability_project = nn.Sequential(nn.LayerNorm(40), nn.Linear(40, model_dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.type_embedding = nn.Parameter(torch.randn(1, 7 + expert_count, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.gate = nn.Linear(model_dim, expert_count)
        self.residual = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, 40),
        )
        self.visual_aux = nn.Linear(model_dim, 40)
        self.sensor_aux = nn.Linear(model_dim, 40)
        gate_prior = torch.zeros(expert_count)
        gate_prior[0] = 2.2  # decoded P89 is a strong initial anchor, not a fixed path
        if expert_count > 1:
            gate_prior[1] = 0.8
        self.register_buffer("gate_prior", gate_prior)
        self.residual_strength = nn.Parameter(torch.tensor(-0.5))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        vmae = self.vmae_project(batch["vmae"])
        iv2 = self.iv2_project(batch["iv2"])
        motionbert = self.motionbert_project(batch["motionbert"])
        skeleton = self.skeleton_project(batch["skeleton"]).unsqueeze(1)
        imu = self.imu_project(batch["imu"]).unsqueeze(1)
        relation = self.relation_project(batch["relation"]).unsqueeze(1)
        probability = self.probability_project(torch.log(batch["experts"].clamp_min(EPSILON)))

        # Modality dropout is applied to complete sensor tokens, not individual
        # coordinates, so the fusion teacher learns graceful fallback behavior.
        if self.training:
            for value, probability_drop in ((skeleton, 0.10), (imu, 0.10), (relation, 0.08)):
                keep = (torch.rand(len(value), 1, 1, device=value.device) >= probability_drop).to(value.dtype)
                value.mul_(keep)

        visual_summary = 0.5 * (vmae.mean(dim=1) + iv2.mean(dim=1))
        sensor_summary = (motionbert.mean(dim=1) + skeleton[:, 0] + imu[:, 0]) / 3.0
        # Collapse each frozen temporal stream only after its own projection;
        # the six/twelve temporal tokens remain visible to the transformer.
        cls = self.cls.expand(len(vmae), -1, -1)
        tokens = torch.cat((cls, vmae, iv2, motionbert, skeleton, imu, relation, probability), dim=1)
        # Type embeddings are repeated within each temporal stream.
        type_ids = [0] + [1] * vmae.shape[1] + [2] * iv2.shape[1] + [3] * motionbert.shape[1]
        type_ids += [4, 5, 6] + list(range(7, 7 + self.expert_count))
        types = self.type_embedding[:, type_ids]
        encoded = self.encoder(tokens + types)
        representation = self.norm(encoded[:, 0])
        gate = torch.softmax(self.gate(representation) + self.gate_prior, dim=1)
        expert_log_probability = torch.log(batch["experts"].clamp_min(EPSILON))
        base_logits = torch.einsum("be,bec->bc", gate, expert_log_probability)
        strength = 2.5 * torch.sigmoid(self.residual_strength)
        logits = base_logits + strength * torch.tanh(self.residual(representation))
        return {
            "logits": logits,
            "gate": gate,
            "visual_logits": self.visual_aux(visual_summary),
            "sensor_logits": self.sensor_aux(sensor_summary),
            "residual_strength": strength,
        }


def sample_weights(data: Prepared, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    class_count = np.bincount(data.labels[indices], minlength=40).astype(np.float64)
    class_weight = 1.0 / np.sqrt(np.maximum(class_count, 1.0))
    class_weight /= class_weight[data.labels[indices]].mean()
    user_values, user_count = np.unique(data.users[indices], return_counts=True)
    user_weight = {user: 1.0 / math.sqrt(count) for user, count in zip(user_values, user_count)}
    output = np.ones(len(data.labels), dtype=np.float32)
    output[indices] = np.asarray(
        [class_weight[data.labels[row]] * user_weight[data.users[row]] for row in indices],
        dtype=np.float32,
    )
    output[indices] /= output[indices].mean()
    output[indices] *= np.where(
        data.safe_prediction[indices] == data.labels[indices], 1.0, 1.35
    ).astype(np.float32)
    return output


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def infer(
    model: P91FusionTeacher,
    data: Prepared,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(data.labels), dtype=np.float32)
    loader = DataLoader(
        PreparedDataset(data, indices, weights),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    logits: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch)
        logits.append(output["logits"].float().cpu().numpy())
        gates.append(output["gate"].float().cpu().numpy())
        rows.append(batch["row"].cpu().numpy())
    return np.concatenate(logits), np.concatenate(gates), np.concatenate(rows)


def train_model(
    data: Prepared,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    expert_count: int,
    args: argparse.Namespace,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[P91FusionTeacher, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    set_seed(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = P91FusionTeacher(
        expert_count=expert_count,
        statistics_dim=data.skeleton_statistics.shape[1],
        model_dim=args.model_dim,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = fixed_epochs if fixed_epochs is not None else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    weights = sample_weights(data, train_indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PreparedDataset(data, train_indices, weights),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_accuracy = -1.0
    best_epoch = 0
    best_logits: np.ndarray | None = None
    best_gates: np.ndarray | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                main = nn.functional.cross_entropy(
                    output["logits"], batch["label"], reduction="none", label_smoothing=0.02
                )
                main = (main * batch["weight"]).mean()
                visual = nn.functional.cross_entropy(output["visual_logits"], batch["label"])
                sensor = nn.functional.cross_entropy(output["sensor_logits"], batch["label"])
                gate_penalty = -(output["gate"].clamp_min(EPSILON).log() * output["gate"]).sum(1).mean()
                loss = main + 0.10 * visual + 0.08 * sensor - 0.006 * gate_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record: dict[str, float] = {"epoch": float(epoch), "loss": float(np.mean(losses))}
        if validation_indices is not None:
            validation_logits, validation_gates, _ = infer(
                model, data, validation_indices, device, args.batch_size
            )
            accuracy = float(
                np.mean(validation_logits.argmax(1) == data.labels[validation_indices])
            )
            record["validation_accuracy"] = accuracy
            if accuracy > best_accuracy + 1e-12:
                best_accuracy = accuracy
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                best_logits = validation_logits
                best_gates = validation_gates
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("inner validation did not save a model")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
        best_accuracy = float("nan")
    audit = {
        "seed": seed,
        "epochs_requested": epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    return model, audit, best_logits, best_gates


def inner_user_folds(users: np.ndarray) -> list[np.ndarray]:
    unique = sorted(np.unique(users).tolist())
    # Interleave users instead of slicing adjacent names; user sample counts and
    # collection dates are uneven.
    return [np.asarray(unique[offset::3], dtype=str) for offset in range(3)]


def blend_probability(
    neural_logits: np.ndarray, safe_prediction: np.ndarray, neural_weight: float
) -> np.ndarray:
    neural = softmax(neural_logits, axis=1)
    safe = safe_decoded_probability(safe_prediction)
    logp = neural_weight * np.log(np.clip(neural, EPSILON, 1.0))
    logp += (1.0 - neural_weight) * np.log(np.clip(safe, EPSILON, 1.0))
    return softmax(logp, axis=1).astype(np.float32)


def select_classwise_blend(
    neural_logits: np.ndarray,
    labels: np.ndarray,
    safe_prediction: np.ndarray,
    users: np.ndarray,
    global_weight: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select conservative per-safe-class weights from inner OOF predictions.

    A class override is accepted only with at least two net rescues overall and
    non-negative transfer in all but at most one source user.  Otherwise it
    inherits the globally selected blend.  This uses predictions from held
    inner users, never their in-fold logits.
    """

    weights = np.full(40, global_weight, dtype=np.float32)
    audit: list[dict[str, Any]] = []
    grid = np.linspace(0.0, 1.0, 41)
    for class_id in range(40):
        selected = safe_prediction == class_id
        support = int(selected.sum())
        base_correct = int(np.sum(safe_prediction[selected] == labels[selected]))
        best: dict[str, Any] | None = None
        if support >= 20:
            for weight in grid:
                probability = blend_probability(
                    neural_logits[selected], safe_prediction[selected], float(weight)
                )
                prediction = probability.argmax(1)
                correct = int(np.sum(prediction == labels[selected]))
                user_nets: list[int] = []
                for user in np.unique(users[selected]):
                    mask = users[selected] == user
                    user_nets.append(
                        int(np.sum(prediction[mask] == labels[selected][mask]))
                        - int(np.sum(safe_prediction[selected][mask] == labels[selected][mask]))
                    )
                candidate = {
                    "class_id": class_id,
                    "support": support,
                    "weight": float(weight),
                    "correct": correct,
                    "net": correct - base_correct,
                    "negative_users": int(np.sum(np.asarray(user_nets) < 0)),
                    "user_nets": user_nets,
                }
                if best is None or (candidate["correct"], -candidate["negative_users"], -weight) > (
                    best["correct"],
                    -best["negative_users"],
                    -best["weight"],
                ):
                    best = candidate
        accepted = bool(
            best is not None and best["net"] >= 2 and best["negative_users"] <= 1
        )
        if accepted:
            weights[class_id] = float(best["weight"])
        audit.append(
            {
                "class_id": class_id,
                "support": support,
                "accepted": accepted,
                "selected_weight": float(weights[class_id]),
                "candidate": best,
            }
        )
    return weights, audit


def classwise_blend_probability(
    neural_logits: np.ndarray, safe_prediction: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    output = np.empty((len(neural_logits), 40), dtype=np.float32)
    for weight in np.unique(weights):
        selected = weights[safe_prediction] == weight
        output[selected] = blend_probability(
            neural_logits[selected], safe_prediction[selected], float(weight)
        )
    return output


def rescue_harm(
    labels: np.ndarray, safe: np.ndarray, candidate: np.ndarray
) -> dict[str, int]:
    safe_correct = safe == labels
    candidate_correct = candidate == labels
    return {
        "rescue": int(np.sum(~safe_correct & candidate_correct)),
        "harm": int(np.sum(safe_correct & ~candidate_correct)),
        "net": int(np.sum(candidate_correct) - np.sum(safe_correct)),
        "changed": int(np.sum(candidate != safe)),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.seeds = "17"
        args.statistics_dim = min(args.statistics_dim, 16)
        args.model_dim = min(args.model_dim, 64)
        args.layers = 1
        args.heads = 4
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cohorts = build_cohorts()
    source_parts = [cohorts["H1_selection"], cohorts["H2_confirmation"]]
    if args.include_embargo_source:
        source_parts.append(cohorts["E0_p87_sequence_source"])
    source = concatenate("P91_source", source_parts)
    target = cohorts["H3_independent_fold0"]
    preprocessor = Preprocessor(args.statistics_dim, seed=seeds[0]).fit(source)
    source_prepared = preprocessor.transform(source)
    target_prepared = preprocessor.transform(target)

    folds = inner_user_folds(source.users)
    inner_logits = np.zeros((len(source.labels), 40), dtype=np.float32)
    inner_gates = np.zeros(
        (len(source.labels), source.expert_probability.shape[1]), dtype=np.float32
    )
    inner_audit: list[dict[str, Any]] = []
    for fold_index, validation_users in enumerate(folds):
        validation_indices = np.flatnonzero(np.isin(source.users, validation_users))
        train_indices = np.flatnonzero(~np.isin(source.users, validation_users))
        model, audit, logits, gates = train_model(
            source_prepared,
            train_indices,
            validation_indices,
            source.expert_probability.shape[1],
            args,
            seed=seeds[fold_index % len(seeds)],
        )
        assert logits is not None and gates is not None
        inner_logits[validation_indices] = logits
        inner_gates[validation_indices] = gates
        audit["fold"] = fold_index
        audit["train_users"] = sorted(np.unique(source.users[train_indices]).tolist())
        audit["validation_users"] = sorted(validation_users.tolist())
        audit["safe_metrics"] = metrics(
            source.labels[validation_indices], source.safe_prediction[validation_indices]
        )
        audit["neural_metrics"] = metrics(
            source.labels[validation_indices], logits.argmax(1)
        )
        audit["rescue_harm"] = rescue_harm(
            source.labels[validation_indices],
            source.safe_prediction[validation_indices],
            logits.argmax(1),
        )
        inner_audit.append(audit)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    blend_grid: list[dict[str, Any]] = []
    for weight in np.linspace(0.0, 1.0, 41):
        probability = blend_probability(inner_logits, source.safe_prediction, float(weight))
        prediction = probability.argmax(1)
        blend_grid.append(
            {
                "neural_weight": float(weight),
                **metrics(source.labels, prediction),
                **rescue_harm(source.labels, source.safe_prediction, prediction),
            }
        )
    selected_blend = max(
        blend_grid,
        key=lambda row: (row["accuracy"], row["balanced_accuracy"], -row["neural_weight"]),
    )
    classwise_weights, classwise_audit = select_classwise_blend(
        inner_logits,
        source.labels,
        source.safe_prediction,
        source.users,
        float(selected_blend["neural_weight"]),
    )
    inner_classwise_probability = classwise_blend_probability(
        inner_logits, source.safe_prediction, classwise_weights
    )
    inner_classwise_prediction = inner_classwise_probability.argmax(1)
    best_epochs = [int(row["best_epoch"]) for row in inner_audit]
    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    if args.smoke:
        final_epochs = 1

    target_logits: list[np.ndarray] = []
    target_gates: list[np.ndarray] = []
    final_models: list[dict[str, Any]] = []
    all_source = np.arange(len(source.labels), dtype=np.int64)
    for seed in seeds:
        model, audit, _, _ = train_model(
            source_prepared,
            all_source,
            None,
            source.expert_probability.shape[1],
            args,
            seed=seed,
            fixed_epochs=final_epochs,
        )
        logits, gates, rows = infer(
            model,
            target_prepared,
            np.arange(len(target.labels), dtype=np.int64),
            torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"),
            args.batch_size,
        )
        if not np.array_equal(rows, np.arange(len(target.labels))):
            raise RuntimeError("target inference order changed")
        target_logits.append(logits)
        target_gates.append(gates)
        checkpoint = output / f"fusion_seed{seed}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "expert_names": source.expert_names,
                "model_dim": args.model_dim,
                "layers": args.layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "statistics_dim": int(source_prepared.skeleton_statistics.shape[1]),
                "epochs": final_epochs,
            },
            checkpoint,
        )
        audit["checkpoint"] = str(checkpoint)
        final_models.append(audit)

    ensemble_logits = np.mean(target_logits, axis=0)
    ensemble_gates = np.mean(target_gates, axis=0)
    raw_prediction = ensemble_logits.argmax(1)
    blended_probability = blend_probability(
        ensemble_logits, target.safe_prediction, selected_blend["neural_weight"]
    )
    blended_prediction = blended_probability.argmax(1)
    classwise_probability = classwise_blend_probability(
        ensemble_logits, target.safe_prediction, classwise_weights
    )
    classwise_prediction = classwise_probability.argmax(1)
    safe_metrics = metrics(target.labels, target.safe_prediction)
    raw_metrics = metrics(target.labels, raw_prediction)
    blended_metrics = metrics(target.labels, blended_prediction)
    classwise_metrics = metrics(target.labels, classwise_prediction)

    np.savez_compressed(
        output / "predictions.npz",
        sample_ids=target.sample_ids,
        labels=target.labels,
        users=target.users,
        safe_prediction=target.safe_prediction,
        neural_logits=ensemble_logits.astype(np.float32),
        neural_prediction=raw_prediction.astype(np.int64),
        blended_probability=blended_probability,
        blended_prediction=blended_prediction.astype(np.int64),
        classwise_probability=classwise_probability,
        classwise_prediction=classwise_prediction.astype(np.int64),
        classwise_weights=classwise_weights,
        expert_names=np.asarray(source.expert_names),
        expert_probability=target.expert_probability,
        gate=ensemble_gates.astype(np.float32),
    )
    summary = {
        "stage": "P91_unrestricted_crossmodal_fusion_teacher_H3_v1",
        "protocol": {
            "source_cohorts": [item.name for item in source_parts],
            "outer_target": "H3_independent_fold0",
            "inner_selection": "three user-disjoint folds on source users",
            "size_constraint": "disabled for teacher ceiling; compress only after gain",
        },
        "samples": {
            "source": len(source.labels),
            "target": len(target.labels),
            "source_users": sorted(np.unique(source.users).tolist()),
            "target_users": sorted(np.unique(target.users).tolist()),
        },
        "expert_names": source.expert_names,
        "preprocessor": preprocessor.summary(),
        "inner": {
            "folds": inner_audit,
            "oof_neural": metrics(source.labels, inner_logits.argmax(1)),
            "oof_rescue_harm": rescue_harm(
                source.labels, source.safe_prediction, inner_logits.argmax(1)
            ),
            "blend_grid": blend_grid,
            "selected_blend": selected_blend,
            "classwise_blend": {
                "weights": classwise_weights.tolist(),
                "selection": classwise_audit,
                "oof_metrics": metrics(source.labels, inner_classwise_prediction),
                "oof_rescue_harm": rescue_harm(
                    source.labels, source.safe_prediction, inner_classwise_prediction
                ),
            },
            "selected_final_epochs": final_epochs,
        },
        "outer_H3": {
            "safe": safe_metrics,
            "neural": raw_metrics,
            "neural_rescue_harm": rescue_harm(
                target.labels, target.safe_prediction, raw_prediction
            ),
            "blended": blended_metrics,
            "blended_rescue_harm": rescue_harm(
                target.labels, target.safe_prediction, blended_prediction
            ),
            "classwise": classwise_metrics,
            "classwise_rescue_harm": rescue_harm(
                target.labels, target.safe_prediction, classwise_prediction
            ),
        },
        "models": final_models,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["inner"]["selected_blend"], ensure_ascii=False))
    print(json.dumps(summary["outer_H3"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
