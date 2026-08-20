from __future__ import annotations

from collections.abc import Sequence
import math


DEFAULT_WINDOWS = ((0.0, 0.5), (0.25, 0.75), (0.5, 1.0))


def normalized_window_indices(
    frame_count: int,
    *,
    windows: tuple[tuple[float, float], ...] = DEFAULT_WINDOWS,
    frames_per_window: int = 16,
) -> tuple[tuple[int, ...], ...]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if frames_per_window < 1:
        raise ValueError("frames_per_window must be positive")
    if not windows:
        raise ValueError("at least one normalized window is required")

    scale = frame_count - 1
    output: list[tuple[int, ...]] = []
    for start, end in windows:
        if not (0.0 <= start <= end <= 1.0):
            raise ValueError("normalized windows must satisfy 0 <= start <= end <= 1")
        if frames_per_window == 1:
            positions = ((start + end) / 2.0,)
        else:
            positions = tuple(
                start + (end - start) * index / (frames_per_window - 1)
                for index in range(frames_per_window)
            )
        output.append(
            tuple(
                min(frame_count - 1, max(0, math.floor(position * scale + 0.5)))
                for position in positions
            )
        )
    return tuple(output)


def normalized_probe_indices(frame_count: int, probe_count: int = 8) -> tuple[int, ...]:
    indices = normalized_window_indices(
        frame_count,
        windows=((0.0, 1.0),),
        frames_per_window=probe_count,
    )[0]
    return tuple(dict.fromkeys(indices))


def uniqueness_mask(indices: Sequence[int]) -> tuple[bool, ...]:
    seen: set[int] = set()
    mask: list[bool] = []
    for index in indices:
        is_unique = int(index) not in seen
        mask.append(is_unique)
        seen.add(int(index))
    return tuple(mask)
