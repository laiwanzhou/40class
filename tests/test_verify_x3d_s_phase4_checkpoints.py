from pathlib import Path

import numpy as np
import pytest

from scripts.verify_x3d_s_phase4_checkpoints import aligned_max_delta, parse_entry


def test_parse_entry_preserves_windows_path() -> None:
    seed, fold, path = parse_entry("20260715:2:C:/runs/fold_2")
    assert (seed, fold) == (20260715, 2)
    assert path == Path("C:/runs/fold_2").resolve()


def test_aligned_max_delta_requires_exact_discrete_fields() -> None:
    assert aligned_max_delta(np.asarray([1.0]), np.asarray([1.0 + 1e-7])) == pytest.approx(1e-7)
    with pytest.raises(ValueError, match="Non-floating"):
        aligned_max_delta(np.asarray(["a"]), np.asarray(["b"]))
