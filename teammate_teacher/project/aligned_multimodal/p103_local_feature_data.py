"""Leakage-safe loaders for P103-B3 explicit local visual representations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p100a_global_teacher_data import H3_USERS


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_VMAE = HERE / "runs/p103_b3_videomaev2_local_dev_v1/complete_features.npz"
DEFAULT_VJEPA = PROJECT_ROOT / "runs/p96_vjepa2_vitl_ssv2_dense24_fold0_v1"
MASTER_MANIFEST = HERE / "data/manifest.csv"
VJEPA_VIEW_INDICES = np.asarray([2, 5, 8, 11, *range(12, 24)], dtype=np.int64)
VJEPA_VIEW_NAMES = (
    "workspace_full",
    "workspace_early",
    "workspace_middle",
    "workspace_late",
    "hand_full_left",
    "hand_full_right",
    "hand_full_interaction",
    "hand_early_left",
    "hand_early_right",
    "hand_early_interaction",
    "hand_late_left",
    "hand_late_right",
    "hand_late_interaction",
    "hand_motion_peak_left",
    "hand_motion_peak_right",
    "hand_motion_peak_interaction",
)
VMAE_VIEW_NAMES = (
    "hand_full_left",
    "hand_full_right",
    "hand_full_interaction",
    "hand_motion_peak_left",
    "hand_motion_peak_right",
    "hand_motion_peak_interaction",
)


@dataclass(frozen=True)
class P103LocalData:
    sample_ids: np.ndarray
    users: np.ndarray
    vmae_features: np.ndarray
    vmae_actions: np.ndarray
    vjepa_features: np.ndarray
    vjepa_actions: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "rows": int(len(self.sample_ids)),
            "subjects": sorted(set(self.users.astype(str).tolist())),
            "h3_rows_selected": int(np.sum(np.isin(self.users, list(H3_USERS)))),
            "h3_users_loaded": sorted(set(self.users.tolist()) & set(H3_USERS)),
            "videomaev2": {
                "features": list(self.vmae_features.shape),
                "actions": list(self.vmae_actions.shape),
                "views": list(VMAE_VIEW_NAMES),
            },
            "vjepa2": {
                "features": list(self.vjepa_features.shape),
                "actions": list(self.vjepa_actions.shape),
                "views": list(VJEPA_VIEW_NAMES),
            },
        }


def _manifest_order() -> tuple[np.ndarray, np.ndarray]:
    # Explicit columns prevent class_id/class_name from entering the B3 process.
    frame = pd.read_csv(MASTER_MANIFEST, usecols=["sample_id", "user_id"])
    return (
        frame["sample_id"].astype(str).to_numpy(dtype=str),
        frame["user_id"].astype(str).to_numpy(dtype=str),
    )


def _allowlist_indices(
    source_ids: np.ndarray, requested_ids: np.ndarray, source_name: str
) -> np.ndarray:
    lookup = {str(sample_id): index for index, sample_id in enumerate(source_ids)}
    if len(lookup) != len(source_ids):
        raise RuntimeError(f"duplicate sample ID in {source_name}")
    missing = [str(sample_id) for sample_id in requested_ids if str(sample_id) not in lookup]
    if missing:
        raise RuntimeError(f"{source_name} misses {len(missing)} requested dev rows")
    return np.asarray([lookup[str(sample_id)] for sample_id in requested_ids], dtype=np.int64)


def load_p103_local_data(
    sample_ids: np.ndarray,
    users: np.ndarray,
    vmae_path: Path = DEFAULT_VMAE,
    vjepa_root: Path = DEFAULT_VJEPA,
) -> P103LocalData:
    requested_ids = np.asarray(sample_ids).astype(str)
    requested_users = np.asarray(users).astype(str)
    if len(requested_ids) != 1941 or len(set(requested_ids.tolist())) != 1941:
        raise RuntimeError("P103-B3 requires the frozen 1941-row dev allowlist")
    if set(requested_users.tolist()) & set(H3_USERS):
        raise RuntimeError("H3 user reached P103-B3 allowlist")

    # Resolve and validate the dev allowlist before reading any feature tensor.
    with np.load(vmae_path.resolve(), allow_pickle=False) as cache:
        vmae_ids = cache["sample_ids"].astype(str)
        vmae_users = cache["users"].astype(str)
        vmae_order = _allowlist_indices(vmae_ids, requested_ids, "B3 VideoMAEv2")
        if not np.array_equal(vmae_users[vmae_order], requested_users):
            raise RuntimeError("B3 VideoMAEv2 user alignment changed")
        vmae_features = np.asarray(cache["features"][vmae_order], dtype=np.float16)
        vmae_actions = np.asarray(cache["action_logits"][vmae_order], dtype=np.float16)

    manifest_ids, manifest_users = _manifest_order()
    vjepa_order = _allowlist_indices(manifest_ids, requested_ids, "B3 V-JEPA2")
    if not np.array_equal(manifest_users[vjepa_order], requested_users):
        raise RuntimeError("B3 V-JEPA2 user alignment changed")
    cache_summary = json.loads(
        (vjepa_root.resolve() / "cache_summary.json").read_text(encoding="utf-8")
    )
    if not cache_summary.get("label_free_extraction") or not cache_summary.get("complete"):
        raise RuntimeError("B3 V-JEPA2 cache is not a complete label-free extraction")
    done = np.load(vjepa_root.resolve() / "done.npy", mmap_mode="r")
    if not np.asarray(done[vjepa_order], dtype=bool).all():
        raise RuntimeError("B3 V-JEPA2 requested dev rows are incomplete")
    feature_memmap = np.load(vjepa_root.resolve() / "features.npy", mmap_mode="r")
    action_memmap = np.load(vjepa_root.resolve() / "ssv2_logits.npy", mmap_mode="r")
    vjepa_features = np.asarray(
        feature_memmap[vjepa_order[:, None], VJEPA_VIEW_INDICES[None, :]], dtype=np.float16
    )
    vjepa_actions = np.asarray(
        action_memmap[vjepa_order[:, None], VJEPA_VIEW_INDICES[None, :]], dtype=np.float16
    )

    data = P103LocalData(
        sample_ids=requested_ids,
        users=requested_users,
        vmae_features=vmae_features,
        vmae_actions=vmae_actions,
        vjepa_features=vjepa_features,
        vjepa_actions=vjepa_actions,
    )
    expected = {
        "vmae_features": (1941, 6, 768),
        "vmae_actions": (1941, 6, 710),
        "vjepa_features": (1941, 16, 1024),
        "vjepa_actions": (1941, 16, 174),
    }
    for name, shape in expected.items():
        value = getattr(data, name)
        if value.shape != shape or not np.isfinite(value).all():
            raise RuntimeError(f"invalid B3 local tensor {name}: {value.shape}")
    if data.summary()["h3_rows_selected"] != 0:
        raise RuntimeError("H3 row entered B3 local tensors")
    return data
