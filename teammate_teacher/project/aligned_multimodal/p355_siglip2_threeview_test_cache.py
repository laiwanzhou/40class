"""Exact Test counterpart of the P340 three-view SigLIP2 state cache."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from p335_siglip2_workspace_state_cache import H, POSITIONS, model


PIX = H / "runs/p87s_test_pixel_cache_t16_r160_v1"
WORKSPACE = H / "runs/p348_siglip2_workspace_test_cache_v1"
OUT = H / "runs/p355_siglip2_threeview_test_cache_v1"
SHAPE = (405, 3, 2, 4, 768)


def main() -> None:
    print("P355 extracts scene/person Test states and reuses the verified workspace cache.", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    images = np.load(PIX / "images.npy", mmap_mode="r")
    feat = np.lib.format.open_memmap(OUT / "features.npy", mode="w+", dtype=np.float16, shape=SHAPE)
    feat[:, 2] = np.load(WORKSPACE / "features.npy", mmap_mode="r")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model(device)
    start = time.time()
    for view in (0, 1):
        for lo in range(0, len(images), 8):
            idx = np.arange(lo, min(lo + 8, len(images)))
            raw = np.asarray(images[idx][:, :, POSITIONS, view], np.float32)
            values = torch.from_numpy(raw).reshape(-1, 1, 160, 160).to(device) / 255.0
            values = values.repeat(1, 3, 1, 1)
            values = torch.nn.functional.interpolate(values, size=(256, 256), mode="bicubic", align_corners=False)
            values = (values - 0.5) / 0.5
            with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = encoder(values)
            feat[idx, view] = encoded.reshape(len(idx), 2, 4, 768).half().cpu().numpy()
    feat.flush()
    ids = np.load(WORKSPACE / "sample_ids.npy").astype(str)
    np.save(OUT / "sample_ids.npy", ids)
    report = {
        "stage": "P355_SigLIP2_threeview_Test_cache",
        "status": "complete",
        "rows": len(ids),
        "shape": list(feat.shape),
        "workspace_reused_from_p348": True,
        "test_labels_read": False,
        "elapsed_seconds": time.time() - start,
    }
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
