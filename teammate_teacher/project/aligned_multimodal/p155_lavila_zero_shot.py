"""Fixed-prompt LaViLa zero-shot audit over cached projected video features."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audit_p87_sequence_decoder import classification_metrics
from p155_lavila_teacher import CHECKPOINT, EXTERNAL, load_checkpoint


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
RUN = HERE / "runs/p155_lavila_timesformer_teacher_v1"
MAPPING = PROJECT / "class_mapping.csv"
OUTPUT = RUN / "zero_shot_predictions.npz"
PROMPTS = ("{}", "a person is {}", "someone is {}")


def class_phrases() -> list[str]:
    with MAPPING.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    phrases = []
    for row in rows:
        name = row["action_name"].split("_", 1)[-1]
        phrases.append(name.replace("_", " ").lower())
    if len(phrases) != 40:
        raise RuntimeError(f"class mapping changed: {len(phrases)}")
    return phrases


def l2(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def main() -> None:
    sys.path.insert(0, str(EXTERNAL))
    from lavila.models.openai_clip import tokenize
    from lavila.models.openai_model import Transformer

    class TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            mask = torch.empty(77, 77)
            mask.fill_(float("-inf"))
            mask.triu_(1)
            self.transformer = Transformer(
                width=512, layers=12, heads=8, attn_mask=mask
            )
            self.token_embedding = nn.Embedding(49408, 512)
            self.positional_embedding = nn.Parameter(torch.empty(77, 512))
            self.ln_final = nn.LayerNorm(512)
            self.text_projection = nn.Parameter(torch.empty(512, 256))

        def encode_text(self, text: torch.Tensor) -> torch.Tensor:
            values = self.token_embedding(text) + self.positional_embedding
            values = values.permute(1, 0, 2)
            values = self.transformer(values)
            values = self.ln_final(values.permute(1, 0, 2))
            return values[
                torch.arange(values.shape[0]), text.argmax(dim=-1)
            ] @ self.text_projection

    _, state = load_checkpoint(CHECKPOINT)
    model = TextEncoder()
    text_state = {
        key: value
        for key, value in state.items()
        if key.startswith("transformer.")
        or key.startswith("token_embedding.")
        or key.startswith("ln_final.")
        or key in ("positional_embedding", "text_projection")
    }
    missing, unexpected = model.load_state_dict(text_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"text mismatch missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phrases = class_phrases()
    embeddings = []
    with torch.inference_mode():
        for phrase in phrases:
            texts = tokenize([template.format(phrase) for template in PROMPTS])
            values = model.encode_text(texts).float().cpu().numpy()
            embeddings.append(l2(values).mean(axis=0))
    text_features = l2(np.stack(embeddings).astype(np.float32))

    source = np.load(RUN / "oof_predictions.npz")
    projected = np.asarray(
        np.load(RUN / "projected_features.npy", mmap_mode="r"), dtype=np.float32
    )
    variants = {
        "workspace": l2(projected[:, 2]),
        "all_views": l2(l2(projected.reshape(len(projected), 3, 256)).mean(axis=1)),
    }
    payload = {
        "sample_ids": source["sample_ids"],
        "labels": source["labels"],
        "fold_ids": source["fold_ids"],
        "text_features": text_features,
    }
    report = {
        "stage": "P155_LaViLa_fixed_prompt_zero_shot",
        "protocol": {
            "prompts": list(PROMPTS),
            "prompt_selected_on_labels": False,
            "user_id_used_as_feature": False,
            "test_rows_loaded": 0,
            "test_labels_loaded": 0,
            "submission_generated": False,
        },
        "variants": {},
    }
    labels = source["labels"].astype(np.int64)
    folds = source["fold_ids"].astype(np.int64)
    for name, video_features in variants.items():
        logits = video_features @ text_features.T
        probability = np.exp(100.0 * (logits - logits.max(axis=1, keepdims=True)))
        probability /= probability.sum(axis=1, keepdims=True)
        prediction = logits.argmax(axis=1)
        report["variants"][name] = {
            "metrics": classification_metrics(labels, prediction),
            "folds": [
                {"fold": fold, **classification_metrics(labels[folds == fold], prediction[folds == fold])}
                for fold in range(3)
            ],
        }
        payload[f"{name}_logits"] = logits.astype(np.float32)
        payload[f"{name}_probability"] = probability.astype(np.float32)
    np.savez_compressed(OUTPUT, **payload)
    (RUN / "zero_shot_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
