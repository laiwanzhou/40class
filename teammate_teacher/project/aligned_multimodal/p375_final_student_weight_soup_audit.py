"""Unlabeled weight-space path audit around the P315 compact champion."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from p87s_test_data import P87STestCachedSequenceMotionDataset, collate_p87s_test
from predict_p87s_test_student import load_student
from train_p86_mobind_fusion_proxy import model_forward


HERE = Path(__file__).resolve().parent
OUT = HERE / "runs/p375_final_student_weight_soup_audit_v1"
CHAMPION = HERE / "runs/p310_student_test_adapt_e40_v1/unified_student.pt"
ENDPOINTS = {
    "p246": HERE / "runs/p246_student_test_adapt_e40_v1/unified_student.pt",
    "p280": HERE / "runs/p280_student_test_adapt_e40_v1/unified_student.pt",
    "p359": HERE / "runs/p359_student_test_adapt_e40_v1/unified_student.pt",
}
SEQUENCE = HERE / "runs/p87s_test_mc3_sequence_v1"
MOTION = HERE / "runs/p87s_test_motion_window_t16_v1"
PIXELS = HERE / "runs/p87s_test_pixel_cache_t16_r160_v1"
P309 = HERE / "runs/p309_union_repeat_group_test_v1/predictions.npz"
P310 = HERE / "runs/p310_union_repeat_precedence_teacher_v1/student_test_targets.npz"
CHAMPION_WEIGHTS = (0.50, 0.65, 0.80, 0.90, 0.95, 1.00)


def infer(model, loader, device):
    logits = []
    ids = []
    model.eval().to(device)
    with torch.inference_mode():
        for batch in loader:
            ids.extend(map(str, batch["sample_id"]))
            batch = {
                key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits.append(model_forward(model, batch)["logits"].float().cpu().numpy())
    return np.asarray(ids), np.concatenate(logits)


def main():
    print(
        "P375 audits single-checkpoint weight interpolation from weaker scored endpoints "
        "toward the P315 champion; Test labels are unavailable and unused.",
        flush=True,
    )
    champion = torch.load(CHAMPION, map_location="cpu", weights_only=False)
    model, _, _ = load_student(CHAMPION)
    dataset = P87STestCachedSequenceMotionDataset(SEQUENCE, MOTION, PIXELS, temporal_augment=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=collate_p87s_test)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = np.load(P310)
    group = np.load(P309)
    report = {
        "stage": "P375_final_Student_weight_soup_audit",
        "status": "complete",
        "protocol": {
            "champion": str(CHAMPION),
            "champion_kaggle": 0.91542,
            "endpoint_scores": {"p246": 0.91044, "p280": 0.91044, "p359": 0.91044},
            "champion_weights": list(CHAMPION_WEIGHTS),
            "single_checkpoint_at_inference": True,
            "test_labels_read": False,
            "selection_from_test_labels": False,
        },
        "paths": {},
    }
    champion_state = champion["model_state"]
    reference_ids = target["sample_ids"].astype(str)
    reference_prediction = target["emission_prediction"].astype(int)
    if not np.array_equal(reference_ids, group["sample_ids"].astype(str)):
        raise RuntimeError("P309/P310 order mismatch")
    group_probability = group["probability"].astype(float)

    for endpoint_name, endpoint_path in ENDPOINTS.items():
        endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
        if endpoint["model_state"].keys() != champion_state.keys():
            raise RuntimeError(f"state key mismatch: {endpoint_name}")
        path_rows = []
        for champion_weight in CHAMPION_WEIGHTS:
            merged = {}
            for key, champion_value in champion_state.items():
                endpoint_value = endpoint["model_state"][key]
                if champion_value.is_floating_point():
                    merged[key] = torch.lerp(endpoint_value, champion_value, float(champion_weight))
                else:
                    if not torch.equal(champion_value, endpoint_value):
                        raise RuntimeError(f"nonfloating mismatch: {endpoint_name}/{key}")
                    merged[key] = champion_value
            model.load_state_dict(merged, strict=True)
            ids, logits = infer(model, loader, device)
            if not np.array_equal(ids.astype(str), reference_ids):
                raise RuntimeError(f"Test order mismatch: {endpoint_name}/{champion_weight}")
            prediction = logits.argmax(axis=1).astype(int)
            changed = np.flatnonzero(prediction != reference_prediction)
            rows = np.arange(len(prediction))
            support = group_probability[rows, prediction] - group_probability[rows, reference_prediction]
            path_rows.append(
                {
                    "champion_weight": champion_weight,
                    "changes_vs_p315": int(len(changed)),
                    "changed_rows_zero_based": changed.tolist(),
                    "changed_pairs": [f"{reference_prediction[index]}->{prediction[index]}" for index in changed],
                    "p309_support_gaps": [float(support[index]) for index in changed],
                    "supported_positive_changes": int(np.sum(support[changed] > 0.0)),
                    "supported_gap_ge_030": int(np.sum(support[changed] >= 0.30)),
                    "mean_confidence": float(torch.softmax(torch.from_numpy(logits), dim=1).max(dim=1).values.mean()),
                }
            )
            print(json.dumps({"endpoint": endpoint_name, **path_rows[-1]}), flush=True)
        report["paths"][endpoint_name] = path_rows

    model.to("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "notes.txt").write_text(
        "Run 1: unlabeled single-model interpolation paths around P315.\n"
        "No candidate is promoted without independent OOF or mechanism evidence.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["status"], "paths": {name: [row["changes_vs_p315"] for row in rows] for name, rows in report["paths"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
