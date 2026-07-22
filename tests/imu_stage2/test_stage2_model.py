from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml


MODEL_INVARIANCE_ATOL = 1e-6


def _batch(batch_size: int = 2, time_steps: int = 5) -> dict[str, object]:
    generator = torch.Generator().manual_seed(123)
    values = torch.randn((batch_size, time_steps, 5, 16), generator=generator)
    valid_mask = torch.ones((batch_size, time_steps, 5), dtype=torch.bool)
    valid_mask[:, 2, 1] = False
    values[:, 2, 1] = 0
    valid_mask[:, :, 4] = False
    values[:, :, 4] = 0
    sequence_mask = torch.ones((batch_size, time_steps), dtype=torch.bool)
    return {
        "values": values,
        "valid_mask": valid_mask,
        "sequence_mask": sequence_mask,
        "sensor_mask": torch.ones((batch_size, 5), dtype=torch.bool),
        "usable_sensor_mask": valid_mask.any(dim=1),
        "timestamps_ms": torch.arange(time_steps).repeat(batch_size, 1) * 100,
        "lengths": torch.full((batch_size,), time_steps, dtype=torch.int64),
        "sample_id": [f"s{index}" for index in range(batch_size)],
        "imu_modality_mask": torch.ones(batch_size, dtype=torch.bool),
    }


def _model():
    from scripts.build_imu_training_index import ClassOrderContract
    from src.models.imu_stage2_tcn import IMUStage2Classifier

    classes = tuple(
        {"class_id": index * 10, "class_name": f"c{index}", "label_index": index}
        for index in range(7)
    )
    class_order = ClassOrderContract(classes, "a" * 64, len(classes))
    torch.manual_seed(7)
    model = IMUStage2Classifier(
        num_classes=class_order.num_classes,
        embedding_dim=32,
        channels=(16, 24),
        dropout=0.2,
    )
    return model


def _assert_logits_equal(left: torch.Tensor, right: torch.Tensor) -> None:
    torch.testing.assert_close(left, right, rtol=0, atol=MODEL_INVARIANCE_ATOL)


def test_model_emits_finite_logits_with_derived_seven_class_shape() -> None:
    model = _model().eval()

    with torch.inference_mode():
        result = model(_batch())

    assert result["embedding"].shape == (2, 32)
    assert result["logits"].shape == (2, 7)
    assert torch.isfinite(result["logits"]).all()


def test_logits_are_invariant_to_invalid_values_unusable_sensor_and_right_padding() -> None:
    model = _model().eval()
    base = _batch()
    changed = copy.deepcopy(base)
    invalid = ~changed["valid_mask"]
    changed["values"][invalid] = torch.randn_like(changed["values"][invalid]) * 1e6
    padded = copy.deepcopy(changed)
    padded["values"] = torch.cat(
        [padded["values"], torch.randn((2, 3, 5, 16)) * 1e6], dim=1
    )
    padded["valid_mask"] = torch.cat(
        [padded["valid_mask"], torch.zeros((2, 3, 5), dtype=torch.bool)], dim=1
    )
    padded["sequence_mask"] = torch.cat(
        [padded["sequence_mask"], torch.zeros((2, 3), dtype=torch.bool)], dim=1
    )
    padded["timestamps_ms"] = torch.cat(
        [padded["timestamps_ms"], torch.full((2, 3), -1, dtype=torch.int64)], dim=1
    )

    with torch.inference_mode():
        base_logits = model(base)["logits"]
        changed_logits = model(changed)["logits"]
        padded_logits = model(padded)["logits"]

    _assert_logits_equal(base_logits, changed_logits)
    _assert_logits_equal(base_logits, padded_logits)


def test_unavailable_modality_placeholder_is_content_invariant() -> None:
    model = _model().eval()
    first = _batch(batch_size=1)
    first["imu_modality_mask"][:] = False
    second = copy.deepcopy(first)
    second["values"] = torch.randn_like(second["values"]) * 1e8
    second["valid_mask"][:] = True
    second["usable_sensor_mask"][:] = True

    with torch.inference_mode():
        first_result = model(first)
        second_result = model(second)

    _assert_logits_equal(first_result["embedding"], second_result["embedding"])
    _assert_logits_equal(first_result["logits"], second_result["logits"])


def test_batch_partition_does_not_change_logits() -> None:
    model = _model().eval()
    batch = _batch(batch_size=2)
    single = {
        key: (
            value[:1]
            if isinstance(value, torch.Tensor)
            else value[:1]
            if key == "sample_id"
            else value
        )
        for key, value in batch.items()
    }

    with torch.inference_mode():
        alone = model(single)["logits"][0]
        together = model(batch)["logits"][0]

    _assert_logits_equal(alone, together)


def test_eval_mode_is_repeatable_and_argmax_ties_choose_lowest_index() -> None:
    from src.models.imu_stage2_tcn import predict_label_indices

    model = _model().eval()
    batch = _batch()
    with torch.inference_mode():
        first = model(batch)["logits"]
        second = model(batch)["logits"]
    _assert_logits_equal(first, second)
    logits = torch.tensor([[1.0, 3.0, 3.0, 2.0]])
    assert predict_label_indices(logits).tolist() == [1]


def test_checkpoint_metadata_requires_all_six_hashes_and_derived_class_count() -> None:
    from src.models.imu_stage2_tcn import build_checkpoint_metadata

    bindings = {
        "stage2_contract_sha256": "a" * 64,
        "training_index_sha256": "b" * 64,
        "normalization_contract_sha256": "c" * 64,
        "normalization_file_sha256": "d" * 64,
        "class_order_sha256": "e" * 64,
        "submission_contract_sha256": "f" * 64,
    }
    metadata = build_checkpoint_metadata(num_classes=7, **bindings)
    assert metadata["num_classes"] == 7
    assert all(metadata[key] == value for key, value in bindings.items())

    for missing in bindings:
        invalid = dict(bindings)
        invalid[missing] = ""
        with pytest.raises(ValueError, match=missing):
            build_checkpoint_metadata(num_classes=7, **invalid)
    with pytest.raises(ValueError, match="num_classes"):
        build_checkpoint_metadata(num_classes=0, **bindings)


def test_training_modality_dropout_produces_null_embedding_gradient() -> None:
    from src.models.imu_stage2_tcn import build_imu_stage2_model

    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )
    config.update(
        embedding_dim=32,
        tcn_channels=[16, 24],
        dropout=0.0,
        imu_modality_dropout=1.0,
    )
    model = build_imu_stage2_model(config, num_classes=7).train()

    result = model(_batch())
    torch.nn.functional.cross_entropy(
        result["logits"],
        torch.tensor([0, 1], dtype=torch.int64),
    ).backward()

    gradient = model.null_embedding.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient).item() > 0


def test_v1_model_config_enables_controlled_modality_dropout() -> None:
    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )

    assert 0.0 < float(config["imu_modality_dropout"]) < 1.0


def test_production_factory_consumes_all_v1_model_fields() -> None:
    from src.models import build_imu_stage2_model

    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )
    config.update(
        embedding_dim=37,
        tcn_channels=[19, 23],
        dropout=0.17,
        imu_modality_dropout=0.37,
    )

    model = build_imu_stage2_model(config, num_classes=7)

    assert model.num_classes == 7
    assert model.embedding_dim == 37
    assert model.channels == (19, 23)
    assert model.dropout_probability == 0.17
    assert model.modality_dropout == 0.37


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("embedding_dim", None, "embedding_dim"),
        ("embedding_dim", True, "embedding_dim"),
        ("embedding_dim", 0, "embedding_dim"),
        ("tcn_channels", None, "tcn_channels"),
        ("tcn_channels", [], "tcn_channels"),
        ("tcn_channels", [16, True], "tcn_channels"),
        ("tcn_channels", [16, 0], "tcn_channels"),
        ("dropout", None, "dropout"),
        ("dropout", True, "dropout"),
        ("dropout", -0.1, "dropout"),
        ("dropout", 1.1, "dropout"),
        ("imu_modality_dropout", None, "imu_modality_dropout"),
        ("imu_modality_dropout", True, "imu_modality_dropout"),
        ("imu_modality_dropout", -0.1, "imu_modality_dropout"),
        ("imu_modality_dropout", 1.1, "imu_modality_dropout"),
    ],
)
def test_production_factory_rejects_invalid_model_fields(
    field: str,
    value: object,
    message: str,
) -> None:
    from src.models import build_imu_stage2_model

    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )
    if value is None:
        config.pop(field)
    else:
        config[field] = value

    with pytest.raises((KeyError, TypeError, ValueError), match=message):
        build_imu_stage2_model(config, num_classes=7)


@pytest.mark.parametrize("num_classes", [0, -1, True])
def test_production_factory_rejects_invalid_derived_class_count(num_classes: object) -> None:
    from src.models import build_imu_stage2_model

    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="num_classes"):
        build_imu_stage2_model(config, num_classes=num_classes)  # type: ignore[arg-type]


def test_eval_output_is_independent_of_modality_dropout_probability() -> None:
    from src.models import build_imu_stage2_model

    config = yaml.safe_load(
        Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8")
    )
    config.update(embedding_dim=32, tcn_channels=[16, 24], dropout=0.0)
    torch.manual_seed(77)
    disabled = build_imu_stage2_model(
        {**config, "imu_modality_dropout": 0.0}, num_classes=7
    ).eval()
    torch.manual_seed(77)
    forced = build_imu_stage2_model(
        {**config, "imu_modality_dropout": 1.0}, num_classes=7
    ).eval()

    with torch.inference_mode():
        disabled_logits = disabled(_batch())["logits"]
        forced_logits = forced(_batch())["logits"]
        repeated_logits = forced(_batch())["logits"]

    _assert_logits_equal(disabled_logits, forced_logits)
    _assert_logits_equal(forced_logits, repeated_logits)
