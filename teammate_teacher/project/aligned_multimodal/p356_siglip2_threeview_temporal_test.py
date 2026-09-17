"""All-Train/Test counterpart of the frozen P344 temporal head."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from p90_teacher_common import load_protocol, softmax
from p142_vjepa_token_transformer_oof import train_fold
from p344_siglip2_threeview_temporal_oof import SEEDS, cfg


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
OUT = RUNS / "p356_siglip2_threeview_temporal_test_v1"
TRAIN = RUNS / "p340_siglip2_threeview_state_cache_v1/features.npy"
TEST = RUNS / "p355_siglip2_threeview_test_cache_v1/features.npy"
TEST_IDS = RUNS / "p355_siglip2_threeview_test_cache_v1/sample_ids.npy"


def tokens(values: np.ndarray) -> np.ndarray:
    states = []
    for view in range(3):
        early = values[:, view, 0].astype(np.float32).mean(1)
        late = values[:, view, 1].astype(np.float32).mean(1)
        states.extend((early, late, late - early, np.abs(late - early)))
    return np.concatenate((values.reshape(len(values), 24, 768), np.stack(states, 1).astype(np.float16)), axis=1)


def main() -> None:
    print("P356 refits the frozen P344 temporal head on all Train and infers Test.", flush=True)
    protocol = load_protocol()
    train = tokens(np.asarray(np.load(TRAIN, mmap_mode="r"), np.float16))
    test = tokens(np.asarray(np.load(TEST, mmap_mode="r"), np.float16))
    combined = np.concatenate((train, test), axis=0)
    train_idx = np.arange(len(train), dtype=np.int64)
    test_idx = np.arange(len(train), len(combined), dtype=np.int64)
    domain = np.concatenate((protocol.fold_id, np.zeros(len(test), dtype=protocol.fold_id.dtype)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    members = [train_fold(combined, protocol.labels, train_idx, test_idx, domain, seed, cfg(), device, None, None) for seed in SEEDS]
    logits = np.mean(members, axis=0)
    probability = softmax(logits)
    ids = np.load(TEST_IDS).astype(str)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "test_predictions.npz", sample_ids=ids, logits=logits.astype(np.float32), probability=probability.astype(np.float32))
    report = {
        "stage": "P356_P344_threeview_temporal_Test",
        "status": "complete",
        "train_rows": len(train),
        "test_rows": len(test),
        "seeds": list(SEEDS),
        "epochs": cfg().epochs,
        "test_labels_read": False,
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
