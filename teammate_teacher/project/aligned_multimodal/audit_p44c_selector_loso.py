from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
RUN = PROJECT_DIR / "runs" / "p44c_group_experts_fold0"
GROUP = np.asarray((6, 7, 8, 9, 10, 11, 14, 37), dtype=np.int64)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def features(
    base_probability: np.ndarray,
    detail_probability: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    base_order = np.sort(base_probability, axis=1)
    detail_order = np.sort(detail_probability, axis=1)
    base_index = base_probability.argmax(1)
    detail_index = detail_probability.argmax(1)
    return np.column_stack(
        (
            base_probability,
            detail_probability,
            detail_probability - base_probability,
            base_probability.max(1),
            base_order[:, -1] - base_order[:, -2],
            -(base_probability * np.log(base_probability + 1e-12)).sum(1),
            detail_probability.max(1),
            detail_order[:, -1] - detail_order[:, -2],
            -(detail_probability * np.log(detail_probability + 1e-12)).sum(1),
            quality,
            np.eye(8)[base_index],
            np.eye(8)[detail_index],
            (base_index == detail_index).astype(np.float32),
        )
    )


def main() -> None:
    detail = np.load(RUN / "final_detail.npz", allow_pickle=False)
    residual = np.load(RUN / "final_base_detail_residual.npz", allow_pickle=False)
    labels = detail["labels"]
    users = detail["users"].astype(str)
    base_probability = softmax(residual["base_group_logits"].astype(np.float64))
    detail_probability = softmax(residual["detail_logits"].astype(np.float64))
    base_prediction = GROUP[base_probability.argmax(1)]
    detail_prediction = GROUP[detail_probability.argmax(1)]
    source = features(
        base_probability, detail_probability, detail["roi_quality"].astype(np.float64)
    )
    decisive = ((detail_prediction == labels) & (base_prediction != labels)) | (
        (base_prediction == labels) & (detail_prediction != labels)
    )
    target = ((detail_prediction == labels) & (base_prediction != labels)).astype(np.int64)
    disagreement = detail_prediction != base_prediction
    factories = {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000),
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=3,
            min_samples_leaf=3,
            max_features=0.6,
            class_weight="balanced",
            random_state=44032,
            n_jobs=-1,
        ),
    }
    report: dict[str, object] = {
        "protocol": "P44-C selector subject-LOO diagnostic",
        "status": "exploratory_not_a_formal_fold0_score",
        "reason": "Group-A was identified from the same fold0 validation subjects",
        "decisive_samples": int(decisive.sum()),
        "detail_better": int(target[decisive].sum()),
        "base_better": int((1 - target[decisive]).sum()),
        "models": {},
    }
    arrays: dict[str, np.ndarray] = {
        "sample_ids": detail["sample_ids"],
        "users": users,
        "labels": labels,
        "base_predictions": base_prediction,
        "detail_predictions": detail_prediction,
    }
    for name, factory in factories.items():
        prediction = base_prediction.copy()
        detail_score = np.zeros(len(labels), dtype=np.float64)
        for held_user in sorted(set(users)):
            train = (users != held_user) & decisive
            test = (users == held_user) & disagreement
            selector = factory()
            selector.fit(source[train], target[train])
            detail_score[test] = selector.predict_proba(source[test])[:, 1]
            prediction[test] = np.where(
                detail_score[test] >= 0.5,
                detail_prediction[test],
                base_prediction[test],
            )
        report["models"][name] = {
            "accuracy": float(accuracy_score(labels, prediction)),
            "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
            "switches": int((prediction != base_prediction).sum()),
            "rescues": int(((prediction == labels) & (base_prediction != labels)).sum()),
            "new_errors": int(((prediction != labels) & (base_prediction == labels)).sum()),
        }
        arrays[f"{name}_predictions"] = prediction
        arrays[f"{name}_detail_score"] = detail_score.astype(np.float32)
    (RUN / "selector_loso_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(RUN / "selector_loso_predictions.npz", **arrays)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
