from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    PROJECT_DIR / "runs" / "p3_sd_imu_rf_full18" / "imu_random_forest.joblib"
)
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p11_model_size_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Losslessly recompress the trained IMU Random Forest"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    load_started = time.time()
    model = joblib.load(source)
    source_load_seconds = time.time() - load_started
    feature_count = int(model.n_features_in_)
    rng = np.random.default_rng(20260724)
    probe = rng.normal(size=(32, feature_count)).astype(np.float32)
    reference = model.predict_proba(probe)
    candidates = [
        ("zlib3", 3),
        ("gzip3", ("gzip", 3)),
        ("lzma3", ("lzma", 3)),
        ("lzma6", ("lzma", 6)),
    ]
    results = []
    for name, compression in candidates:
        path = output_dir / f"imu_random_forest_{name}.joblib"
        dump_started = time.time()
        joblib.dump(model, path, compress=compression)
        dump_seconds = time.time() - dump_started
        reload_started = time.time()
        reloaded = joblib.load(path)
        reload_seconds = time.time() - reload_started
        candidate = reloaded.predict_proba(probe)
        results.append(
            {
                "name": name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "size_mib": path.stat().st_size / 1024**2,
                "compression_ratio_vs_source": (
                    path.stat().st_size / source.stat().st_size
                ),
                "dump_seconds": round(dump_seconds, 3),
                "load_seconds": round(reload_seconds, 3),
                "probe_max_abs_probability_difference": float(
                    np.max(np.abs(reference - candidate))
                ),
                "sha256": sha256(path),
            }
        )
    summary = {
        "protocol": (
            "Serialization-only recompression. The fitted estimator is not retrained "
            "or pruned. A deterministic 32-row probe must reproduce predict_proba "
            "exactly after reload."
        ),
        "source": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "size_mib": source.stat().st_size / 1024**2,
            "load_seconds": round(source_load_seconds, 3),
            "feature_count": feature_count,
            "sha256": sha256(source),
        },
        "candidates": results,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
