"""Pre-encoder evidence contracts for the isolated Visual90 experiment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ClipSelection:
    indices: np.ndarray
    bounds: np.ndarray
    valid: np.ndarray
    source_breaks: np.ndarray
    unique_fraction: np.ndarray


def select_continuous_clips(
    timestamps_ms: np.ndarray,
    *,
    frame_ids: np.ndarray | None = None,
    verified_counter_step: int | None = None,
) -> ClipSelection:
    """Select the longest single source segment inside each of four bins.

    NaN timestamps are accepted only when a verified source counter is supplied.
    No interpolation or padding may borrow from another segment or bin.
    """
    times = np.asarray(timestamps_ms, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError('timestamps must be a vector')
    n = len(times)
    time_evidence = bool(np.isfinite(times).all()) and n > 0
    counter_evidence = verified_counter_step is not None
    if counter_evidence:
        if verified_counter_step < 1 or frame_ids is None:
            raise ValueError('verified counter requires positive step and IDs')
        ids = np.asarray(frame_ids)
        if ids.shape != times.shape or not np.isfinite(ids).all():
            raise ValueError('invalid verified counter IDs')
    if n and not (time_evidence or counter_evidence):
        raise ValueError('continuity_unverified: no trusted timestamp or source counter')
    breaks = np.zeros(max(0, n - 1), dtype=bool)
    if time_evidence and n > 1:
        delta = np.diff(times)
        positive = delta[delta > 0]
        breaks |= delta <= 0
        if len(positive):
            breaks |= delta > 3.0 * float(np.median(positive))
    if counter_evidence:
        breaks |= np.diff(ids) != verified_counter_step
    source_breaks = np.unique(np.r_[0, np.flatnonzero(breaks) + 1, n]).astype(np.int64)
    indices = np.full((4, 16), -1, dtype=np.int64)
    bounds = np.full((4, 2), -1, dtype=np.int64)
    valid = np.zeros(4, dtype=bool)
    unique = np.zeros(4, dtype=np.float32)
    for clip in range(4):
        lo, hi = clip * n // 4, (clip + 1) * n // 4
        if lo == hi:
            continue
        cuts = np.r_[lo, source_breaks[(source_breaks > lo) & (source_breaks < hi)], hi]
        candidates = [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:])]
        start, stop = min(candidates, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
        chosen = np.floor(np.linspace(start, stop - 1, 16) + 0.5).astype(np.int64)
        indices[clip] = chosen
        bounds[clip] = [start, stop]
        valid[clip] = True
        unique[clip] = len(np.unique(chosen)) / 16
    return ClipSelection(indices, bounds, valid, source_breaks, unique)


@dataclass(frozen=True)
class ROITrack:
    boxes: np.ndarray
    eligible: bool
    interpolated: np.ndarray
    reason: str


def prepare_roi_track(boxes: np.ndarray, width: int, height: int) -> ROITrack:
    """Fill short interior gaps then smooth using only this selected segment."""
    raw = np.asarray(boxes, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 4 or width < 2 or height < 2:
        raise ValueError('invalid ROI track shape or image dimensions')
    count = len(raw)
    finite = np.isfinite(raw).all(axis=1)
    valid = finite & (raw[:, 2] > raw[:, 0]) & (raw[:, 3] > raw[:, 1])
    valid &= (raw[:, :2] >= 0).all(axis=1)
    valid &= (raw[:, 2] <= width) & (raw[:, 3] <= height)
    interpolated = np.zeros(count, dtype=bool)

    def reject(reason: str) -> ROITrack:
        return ROITrack(np.zeros((count, 4), dtype=np.float32), False, interpolated, reason)

    if not count or not valid.any():
        return reject('no_valid_roi')
    if not valid[0] or not valid[-1]:
        return reject('unbounded_roi_gap')
    filled = raw.copy()
    i = 0
    while i < count:
        if valid[i]:
            i += 1
            continue
        stop = i
        while stop < count and not valid[stop]:
            stop += 1
        if stop - i > 3:
            return reject('long_roi_gap')
        for j in range(i, stop):
            alpha = (j - i + 1) / (stop - i + 1)
            filled[j] = (1 - alpha) * filled[i - 1] + alpha * raw[stop]
        interpolated[i:stop] = True
        i = stop
    centers = (filled[:, :2] + filled[:, 2:]) / 2
    smooth = np.stack([np.median(centers[max(0, i - 2):i + 3], axis=0) for i in range(count)])
    half = np.maximum(smooth - filled[:, :2], filled[:, 2:] - smooth).max(axis=0)
    size = np.minimum(2 * half, [width, height])
    low = np.clip(smooth - size / 2, 0, np.array([width, height]) - size)
    result = np.concatenate((low, low + size), axis=1).astype(np.float32)
    if (result[:, :2] > filled[:, :2] + 1e-5).any() or (result[:, 2:] < filled[:, 2:] - 1e-5).any():
        return reject('smoothed_roi_cannot_cover_evidence')
    return ROITrack(result, True, interpolated, '')


def require_geometry(report: dict[str, Any]) -> None:
    """Fail closed: a boolean or timestamp pairing alone is not geometry evidence."""
    required = ('reviewer', 'source_sha256', 'pose_sha256', 'configurations', 'observations')
    if report.get('status') != 'verified' or any(not report.get(k) for k in required):
        raise ValueError('geometry_unverified: local ROI correspondence evidence required')
    if any(item.get('verdict') != 'pass' for item in report['observations']):
        raise ValueError('geometry_unverified: correspondence observation not passed')
