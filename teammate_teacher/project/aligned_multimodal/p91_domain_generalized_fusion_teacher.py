"""Large cross-user domain-generalized P91 fusion teacher (fold-0 screen).

This is a structural replacement experiment, not another threshold router.  It
uses the complete token model, a learned mixture of all dense expert
probabilities, GroupDRO over source users, cross-user supervised contrastive
alignment and adversarial removal of user identity.  Model/epoch/blend choices
are made on H2 and the frozen choice is audited on H3 once.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from torch.utils.data import DataLoader, Sampler

from p91_hierarchical_multimodal_teacher import (
    FAMILIES,
    FusionData,
    FusionDataset,
    HierarchicalTeacher,
    Preprocessor,
    audit,
    build_data,
    move,
    set_seed,
)
from p91_subject_domain_normalization_probe import champion_predictions


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_OUTPUT = PROJECT / "runs/p91_domain_generalized_fusion_h3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-3)
    parser.add_argument("--statistics-dim", type=int, default=96)
    parser.add_argument("--groupdro-eta", type=float, default=0.005)
    parser.add_argument("--contrastive-weight", type=float, default=0.12)
    parser.add_argument("--domain-weight", type=float, default=0.08)
    parser.add_argument("--domain-grl", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


class DomainDataset(FusionDataset):
    def __init__(
        self,
        data: FusionData,
        indices: np.ndarray,
        user_lookup: dict[str, int],
    ) -> None:
        super().__init__(data, indices, np.ones(len(data.labels), dtype=np.float32))
        self.user_lookup = user_lookup

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(index)
        row = int(self.indices[index])
        item["user_code"] = torch.tensor(
            self.user_lookup[str(self.data.users[row])], dtype=torch.long
        )
        return item


class CrossUserClassBatchSampler(Sampler[list[int]]):
    """Every selected class contributes examples from two different users."""

    def __init__(
        self,
        data: FusionData,
        indices: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = max(4, batch_size - batch_size % 2)
        self.seed = seed
        self.epoch = 0
        self.grouped: dict[int, dict[str, np.ndarray]] = {}
        for class_id in range(40):
            by_user = {}
            for user in np.unique(data.users[self.indices]):
                positions = np.flatnonzero(
                    (data.labels[self.indices] == class_id)
                    & (data.users[self.indices] == user)
                )
                if len(positions):
                    by_user[str(user)] = positions
            if len(by_user) >= 2:
                self.grouped[class_id] = by_user
        self.classes = np.asarray(sorted(self.grouped), dtype=np.int64)
        if len(self.classes) < 2:
            raise ValueError("not enough cross-user classes for balanced batches")
        self.batches = max(1, math.ceil(len(indices) / self.batch_size))

    def __len__(self) -> int:
        return self.batches

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        self.epoch += 1
        classes_per_batch = self.batch_size // 2
        for _ in range(self.batches):
            chosen = rng.choice(
                self.classes,
                size=classes_per_batch,
                replace=classes_per_batch > len(self.classes),
            )
            batch = []
            for class_id in chosen:
                by_user = self.grouped[int(class_id)]
                users = rng.choice(np.asarray(sorted(by_user), dtype=object), 2, replace=False)
                for user in users:
                    batch.append(int(rng.choice(by_user[str(user)])))
            rng.shuffle(batch)
            yield batch


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, strength: float) -> torch.Tensor:
        ctx.strength = strength
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.strength * gradient, None


class DomainGeneralizedTeacher(nn.Module):
    def __init__(
        self,
        statistics_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        dropout: float,
        expert_count: int,
        user_count: int,
        domain_grl: float,
    ) -> None:
        super().__init__()
        self.backbone = HierarchicalTeacher(
            statistics_dim, model_dim, layers, heads, dropout
        )
        self.expert_count = expert_count + 1  # P90 cross-user teacher plus nine experts.
        self.domain_grl = domain_grl
        self.gate = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim, self.expert_count),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        with torch.no_grad():
            self.gate[-1].bias[0] = 3.0
        self.residual = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 2, 40),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.residual_strength = nn.Parameter(torch.tensor(-0.85))
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.GELU(), nn.Linear(model_dim, 128)
        )
        self.user_head = nn.Sequential(
            nn.LayerNorm(model_dim), nn.Linear(model_dim, model_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(model_dim, user_count)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.backbone(batch)
        representation = output["representation"]
        p90 = torch.full(
            (len(representation), 40),
            0.04 / 39.0,
            dtype=representation.dtype,
            device=representation.device,
        )
        p90.scatter_(1, batch["teacher"].unsqueeze(1), 0.96)
        experts = torch.cat((p90.unsqueeze(1), batch["expert_probability"]), dim=1)
        experts = experts.clamp_min(1e-7)
        experts = experts / experts.sum(dim=2, keepdim=True)
        gate = torch.softmax(self.gate(representation), dim=1)
        mixture = (gate.unsqueeze(2) * experts).sum(dim=1).clamp_min(1e-7)
        strength = 0.75 * torch.sigmoid(self.residual_strength)
        logits = torch.log(mixture) + strength * torch.tanh(
            self.residual(representation) / 3.0
        )
        reversed_representation = GradientReverse.apply(representation, self.domain_grl)
        output.update(
            {
                "logits": logits,
                "gate": gate,
                "projection": F.normalize(self.projection(representation), dim=1),
                "user_logits": self.user_head(reversed_representation),
                "residual_strength": strength,
            }
        )
        return output


def supervised_cross_user_contrastive(
    projection: torch.Tensor,
    labels: torch.Tensor,
    users: torch.Tensor,
    temperature: float = 0.12,
) -> torch.Tensor:
    similarity = projection @ projection.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & users[:, None].ne(users[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not bool(valid.any()):
        return similarity.sum() * 0.0
    log_probability = similarity - torch.logsumexp(
        similarity.masked_fill(eye, -torch.inf), dim=1, keepdim=True
    )
    loss = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1) / positives.sum(dim=1).clamp_min(1))
    return loss[valid].mean()


def groupdro_loss(
    per_sample: torch.Tensor,
    users: torch.Tensor,
    group_weights: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    present = users.unique()
    group_losses = torch.stack([per_sample[users == group].mean() for group in present])
    with torch.no_grad():
        group_weights[present] *= torch.exp(eta * group_losses.detach().float())
        group_weights /= group_weights.sum().clamp_min(1e-8)
    weights = group_weights[present]
    weights = weights / weights.sum().clamp_min(1e-8)
    return (weights * group_losses).sum()


@torch.no_grad()
def infer(
    model: DomainGeneralizedTeacher,
    data: FusionData,
    indices: np.ndarray,
    user_lookup: dict[str, int],
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        DomainDataset(data, indices, user_lookup), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda"
    )
    model.eval()
    logits, gates, rows = [], [], []
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            output = model(batch)
        logits.append(output["logits"].float().cpu().numpy())
        gates.append(output["gate"].float().cpu().numpy())
        rows.append(batch["row"].cpu().numpy())
    if not np.array_equal(np.concatenate(rows), indices):
        raise ValueError("inference order changed")
    return np.concatenate(logits), np.concatenate(gates)


def per_user_metrics(
    labels: np.ndarray, prediction: np.ndarray, users: np.ndarray
) -> dict[str, Any]:
    values = {
        str(user): float(np.mean(prediction[users == user] == labels[users == user]))
        for user in np.unique(users)
    }
    return {"per_user": values, "worst_user_accuracy": float(min(values.values()))}


def train_model(
    args: argparse.Namespace,
    data: FusionData,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    user_lookup: dict[str, int],
    fixed_epochs: int | None = None,
) -> tuple[DomainGeneralizedTeacher, dict[str, Any], np.ndarray | None, np.ndarray | None]:
    set_seed(args.seed + (10000 if fixed_epochs is not None else 0))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = DomainGeneralizedTeacher(
        data.statistics["skeleton"].shape[1], args.model_dim, args.layers,
        args.heads, args.dropout, data.expert_probability.shape[1],
        len(user_lookup), args.domain_grl,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    epochs = fixed_epochs or args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = DomainDataset(data, train_indices, user_lookup)
    sampler = CrossUserClassBatchSampler(data, train_indices, args.batch_size, args.seed)
    loader = DataLoader(
        dataset, batch_sampler=sampler, num_workers=0, pin_memory=device.type == "cuda"
    )
    group_weights = torch.ones(len(user_lookup), device=device) / len(user_lookup)
    best_state = None
    best_logits = None
    best_gates = None
    best_epoch = 0
    best_key = (-1, -1.0)
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in loader:
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                per_sample = F.cross_entropy(
                    output["logits"], batch["label"], reduction="none", label_smoothing=0.02
                )
                per_sample = per_sample * torch.where(
                    batch["teacher"] != batch["label"], 1.45, 1.0
                )
                main = groupdro_loss(
                    per_sample, batch["user_code"], group_weights, args.groupdro_eta
                )
                family = F.cross_entropy(output["family_logits"], batch["family"])
                visual = F.cross_entropy(output["global_visual_logits"], batch["label"])
                local = F.cross_entropy(output["local_visual_logits"], batch["label"])
                sensor = F.cross_entropy(output["sensor_logits"], batch["label"])
                contrastive = supervised_cross_user_contrastive(
                    output["projection"], batch["label"], batch["user_code"]
                )
                domain = F.cross_entropy(output["user_logits"], batch["user_code"])
                gate_entropy = -(output["gate"] * output["gate"].clamp_min(1e-7).log()).sum(1).mean()
                loss = (
                    main + 0.14 * family + 0.06 * visual + 0.10 * local + 0.07 * sensor
                    + args.contrastive_weight * contrastive + args.domain_weight * domain
                    - 0.012 * gate_entropy
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        scheduler.step()
        record: dict[str, Any] = {
            "epoch": epoch,
            "loss": float(np.mean(epoch_losses)),
            "residual_strength": float(output["residual_strength"].detach().cpu()),
            "group_weights": group_weights.detach().cpu().tolist(),
        }
        if validation_indices is not None:
            logits, gates = infer(
                model, data, validation_indices, user_lookup, device, args.batch_size
            )
            prediction = logits.argmax(1)
            correct = int(np.sum(prediction == data.labels[validation_indices]))
            user_metrics = per_user_metrics(
                data.labels[validation_indices], prediction, data.users[validation_indices]
            )
            key = (correct, user_metrics["worst_user_accuracy"])
            record.update(
                validation_accuracy=float(correct / len(validation_indices)),
                validation_correct=correct,
                worst_user_accuracy=user_metrics["worst_user_accuracy"],
            )
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_logits = logits
                best_gates = gates
                best_state = copy.deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
                stale = 0
            else:
                stale += 1
            if epoch == 1 or epoch % 5 == 0 or key >= best_key:
                print(
                    f"epoch={epoch} loss={record['loss']:.4f} "
                    f"val={correct}/{len(validation_indices)} "
                    f"worst={user_metrics['worst_user_accuracy']:.4f}", flush=True
                )
            if stale >= args.patience:
                history.append(record)
                break
        history.append(record)
    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("no validation state")
        model.load_state_dict(best_state)
    else:
        best_epoch = epochs
    return model, {
        "best_epoch": best_epoch,
        "best_key": list(best_key),
        "epochs_ran": len(history),
        "parameter_count": int(sum(value.numel() for value in model.parameters())),
        "history": history,
    }, best_logits, best_gates


def hard_blend(logits: np.ndarray, base: np.ndarray, weight: float) -> np.ndarray:
    probability = softmax(logits, axis=1)
    anchor = np.full((len(base), 40), 0.04 / 39.0, dtype=np.float64)
    anchor[np.arange(len(base)), base] = 0.96
    score = weight * np.log(np.clip(probability, 1e-8, 1.0))
    score += (1.0 - weight) * np.log(np.clip(anchor, 1e-8, 1.0))
    return score.argmax(1)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.model_dim = min(args.model_dim, 128)
        args.layers = min(args.layers, 2)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = build_data()
    h1 = raw.boundaries["H1_selection"]
    h2 = raw.boundaries["H2_confirmation"]
    embargo = raw.boundaries["E0_p87_sequence_source"]
    h3 = raw.boundaries["H3_independent_fold0"]
    inner_train = np.concatenate((h1, embargo))
    final_train = np.concatenate((h1, h2, embargo))
    user_lookup = {str(user): index for index, user in enumerate(sorted(np.unique(raw.users)))}
    h2_champion, h3_champion = champion_predictions(raw, h2, h3)

    print("preparing inner source domains", flush=True)
    inner_pre = Preprocessor(args.statistics_dim, args.seed).fit(raw, inner_train)
    inner = inner_pre.transform(raw)
    _, inner_audit, h2_logits, h2_gates = train_model(
        args, inner, inner_train, h2, user_lookup
    )
    if h2_logits is None or h2_gates is None:
        raise RuntimeError("missing H2 predictions")
    blend_grid = []
    for weight in np.linspace(0.0, 1.0, 41):
        prediction = hard_blend(h2_logits, h2_champion, float(weight))
        row = {"weight": float(weight), **audit(inner.labels[h2], h2_champion, prediction)}
        row.update(per_user_metrics(inner.labels[h2], prediction, inner.users[h2]))
        blend_grid.append(row)
    selected_blend = max(
        blend_grid,
        key=lambda row: (row["correct"], -row["harm"], row["worst_user_accuracy"], -row["weight"]),
    )
    fixed_epochs = max(1, int(inner_audit["best_epoch"]))
    print(
        f"source selection epoch={fixed_epochs} blend={selected_blend['weight']} "
        f"correct={selected_blend['correct']}", flush=True
    )

    final_pre = Preprocessor(args.statistics_dim, args.seed + 1000).fit(raw, final_train)
    final = final_pre.transform(raw)
    model, final_audit, _, _ = train_model(
        args, final, final_train, None, user_lookup, fixed_epochs=fixed_epochs
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    h3_logits, h3_gates = infer(model, final, h3, user_lookup, device, args.batch_size)
    direct = h3_logits.argmax(1)
    selected = hard_blend(h3_logits, h3_champion, float(selected_blend["weight"]))
    report = {
        "protocol": (
            "H1+embargo train/H2 epoch and blend selection; H1+H2+embargo refit; "
            "one frozen H3 audit; GroupDRO + cross-user SupCon + user adversary."
        ),
        "model": {
            "parameter_count": final_audit["parameter_count"],
            "expert_count": int(final.expert_probability.shape[1] + 1),
            "model_dim": args.model_dim,
            "layers": args.layers,
        },
        "inner": {
            "training": inner_audit,
            "direct_vs_p90": audit(inner.labels[h2], inner.teacher_prediction[h2], h2_logits.argmax(1)),
            "direct_vs_champion": audit(inner.labels[h2], h2_champion, h2_logits.argmax(1)),
            "selected_blend": selected_blend,
            "top_blends": sorted(blend_grid, key=lambda row: row["correct"], reverse=True)[:10],
        },
        "final": {
            "training": final_audit,
            "direct_vs_p90": audit(final.labels[h3], final.teacher_prediction[h3], direct),
            "direct_vs_champion": audit(final.labels[h3], h3_champion, direct),
            "source_selected_vs_champion": audit(final.labels[h3], h3_champion, selected),
            "direct_users": per_user_metrics(final.labels[h3], direct, final.users[h3]),
            "selected_users": per_user_metrics(final.labels[h3], selected, final.users[h3]),
            "mean_gate": h3_gates.mean(axis=0).tolist(),
        },
    }
    torch.save(
        {"state_dict": model.state_dict(), "args": vars(args), "audit": final_audit},
        output / "teacher.pt",
    )
    np.savez_compressed(
        output / "predictions.npz", sample_ids=final.sample_ids[h3], labels=final.labels[h3],
        p90_prediction=final.teacher_prediction[h3], champion_prediction=h3_champion,
        logits=h3_logits.astype(np.float32), gates=h3_gates.astype(np.float32),
        direct_prediction=direct, selected_prediction=selected,
    )
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    main()
