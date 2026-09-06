from __future__ import annotations

import pytest

from scripts.audit_visual90_inputs import choose_smoke_rows, inspect_frame_sequence, training_disposition


def test_smoke_selection_never_accepts_validation_rows():
    with pytest.raises(ValueError, match='training'):
        choose_smoke_rows([{'partition': 'validation', 'sample_id': 'x', 'ir_frames': 10}])


def test_smoke_selection_covers_length_quartiles_without_class_labels():
    rows = [{'partition': 'train', 'sample_id': str(i), 'ir_frames': i + 1} for i in range(8)]
    assert len(choose_smoke_rows(rows)) == 8


def test_legacy_filename_does_not_silently_gain_timestamp_evidence(tmp_path):
    path = tmp_path / 'IR_00000010.png'
    path.touch()
    result = inspect_frame_sequence([path], 'IR')
    assert result['continuity_status'] == 'continuity_unverified'
    assert result['unparsed_frames'] == 1
    assert result['missing_timestamp_frames'] == 1


def test_real_timestamp_keys_generate_continuous_selection(tmp_path):
    paths = [tmp_path / f'IR_2025-06-10_11-31-11.{i:03d}_{i:08d}.png' for i in range(8)]
    result = inspect_frame_sequence(paths, 'IR')
    assert result['continuity_status'] == 'verified_timestamps'
    assert result['source_segment_count'] == 1


def test_missing_timestamp_train_trial_is_excluded_before_training():
    row = {'partition': 'train', 'ir_frames': 2, 'sequences': {
        'ir': {'missing_timestamp_frames': 2, 'continuity_status': 'continuity_unverified'},
        'depth_color': {'missing_timestamp_frames': 0, 'continuity_status': 'verified_timestamps'},
    }}
    assert training_disposition(row) == 'excluded_missing_timestamps'


def test_validation_is_never_silently_excluded_by_training_permission():
    row = {'partition': 'validation', 'ir_frames': 2, 'sequences': {
        'ir': {'missing_timestamp_frames': 2, 'continuity_status': 'continuity_unverified'},
    }}
    assert training_disposition(row) == 'validation_only'


def test_other_continuity_failures_are_not_authorized_exclusions():
    row = {'partition': 'train', 'ir_frames': 2, 'sequences': {
        'ir': {'missing_timestamp_frames': 0, 'continuity_status': 'continuity_unverified'},
    }}
    assert training_disposition(row) == 'blocked_other_continuity'


def test_missing_visual_is_a_separate_data_issue_not_a_timestamp_exclusion():
    assert training_disposition({'partition':'train','ir_frames':0,'sequences':{}}) == 'unsupported_visual_data_issue'
