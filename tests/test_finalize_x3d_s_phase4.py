import numpy as np

from scripts.finalize_x3d_s_phase4 import aligned


def test_alignment_reorders_trial_fields_and_preserves_scalar_metadata() -> None:
    reference = {"sample_ids": np.asarray(["b", "a"])}
    other = {
        "sample_ids": np.asarray(["a", "b"]),
        "labels": np.asarray([1, 2]),
        "class_map_hash": np.asarray("hash"),
    }
    result = aligned(reference, other)
    assert result["labels"].tolist() == [2, 1]
    assert result["class_map_hash"].item() == "hash"
