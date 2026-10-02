from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest


def test_timestamp_restoration_preserves_ir_acquisition_keys(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map
    ir=tmp_path/'IR';sk=tmp_path/'Skeleton/predictions';ir.mkdir();sk.mkdir(parents=True)
    for n in (1,2):
        key=f'20250717_120000_{n:08}'
        (ir/f'IR_{key}.png').write_bytes(b'image')
        (sk/f'Color_{key}.json').write_text('[]')
    maps=compatible_frame_map(sk.parent,'skeleton',ir)
    assert set(maps)=={'20250717_120000_00000001','20250717_120000_00000002'}


def test_counter_only_skeleton_does_not_invent_time(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map
    sk=tmp_path/'predictions';sk.mkdir();(sk/'Color_00000001.json').write_text('[]')
    with pytest.raises(ValueError,match='timeline'):compatible_frame_map(tmp_path,'skeleton')


def test_ambiguous_ir_counters_rejected(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map
    ir=tmp_path/'ir';sk=tmp_path/'sk/predictions';ir.mkdir();sk.mkdir(parents=True)
    for timestamp in ('20250717_120000','20250718_120000'):
        (ir/f'IR_{timestamp}_00000001.png').write_bytes(b'')
    (sk/'Color_00000001.json').write_text('[]')
    with pytest.raises(ValueError,match='Ambiguous'):compatible_frame_map(sk.parent,'skeleton',ir)


def test_missing_pose_is_complete_unavailable_label_free_and_partial(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,load_stage_inputs
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.pose_roi_adapter import build_pose_roi
    config,_=no_vote_fixture;p=load_protocol(config)
    source=tmp_path/'source.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(40):w.writerow({'sample_id':f'c{c}-{user}','class_id':c,'user_id':user})
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    registry=ArtifactRegistry(p)
    weight=registry.register(stage='pose_initializer',kind='public_weights',phase='public',
        files=[p.weights['yolo']],config={'role':'yolo'})
    inputs=load_stage_inputs(p,'train12','select')
    refs=build_pose_roi(inputs,weight,p.run_root/'pose_smoke',protocol=p,max_trials=1)
    for stage,ref in refs.items():
        record=registry.verify(ref)
        assert not record.complete and record.kind=='raw_cache'
        caches=list((p.run_root/'pose_smoke'/stage).rglob('*.npz'))
        assert len(caches)==1
        with np.load(caches[0],allow_pickle=False) as data:
            assert 'class_id' not in data.files and 'label' not in data.files
            assert not bool(data['available']) and bool(data['completed'])
            assert data['acquisition_ids'].shape==data['frame_ids'].shape
    again=build_pose_roi(inputs,weight,p.run_root/'pose_smoke',protocol=p,max_trials=1)
    assert refs==again
    cache=next((p.run_root/'pose_smoke/p28').rglob('*.npz'))
    with cache.open('wb') as f:np.savez(f,class_id=np.array([0]))
    with pytest.raises(ValueError,match='hash'):
        build_pose_roi(inputs,weight,p.run_root/'pose_smoke',protocol=p,max_trials=1)


def test_interrupted_trial_cache_hash_is_checked_without_global_summary(no_vote_fixture,tmp_path):
    # The preceding missing-input test provides a real completed pair; execute
    # that setup without its final mutation, then remove the aggregate summary.
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,load_stage_inputs
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.pose_roi_adapter import build_pose_roi
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(40):w.writerow(dict(sample_id=f'{user}{c}',class_id=c,user_id=user))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    r=ArtifactRegistry(p);weight=r.register(stage='pose_initializer',kind='public_weights',phase='public',files=[p.weights['yolo']],config={})
    inputs=load_stage_inputs(p,'train12','raw');out=p.run_root/'interrupted'
    build_pose_roi(inputs,weight,out,protocol=p,max_trials=1)
    (out/'summary.json').unlink()
    cache=next((out/'p28').rglob('*.npz'))
    with cache.open('wb') as f:np.savez(f,frame_ids=np.array([]),available=False,completed=True)
    with pytest.raises(ValueError,match='hash'):
        build_pose_roi(inputs,weight,out,protocol=p,max_trials=1)


def test_raw_membership_snapshot_detects_new_frames(tmp_path):
    from src.experiments.pose_roi_adapter import snapshot_raw_files
    ir=tmp_path/'ir';ir.mkdir()
    row={'ir_path':str(ir),'depth_path':'','skeleton_path':''}
    before=snapshot_raw_files(row)
    (ir/'IR_20250717_120000_00000001.png').write_bytes(b'new')
    assert snapshot_raw_files(row)!=before


def test_numeric_ir_matches_timestamped_skeleton_without_losing_acquisition_time(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map, acquisition_ids
    ir=tmp_path/'ir';sk=tmp_path/'sk/predictions';ir.mkdir();sk.mkdir(parents=True)
    stamp='2025-06-11_16-53-26.216_00000063'
    (ir/'IR_00000063.png').write_bytes(b'')
    (sk/f'Color_{stamp}.json').write_text('[]')
    maps={'ir':compatible_frame_map(ir,'ir'),'skeleton':compatible_frame_map(sk.parent,'skeleton',ir)}
    assert set(maps['skeleton'])=={'00000063'}
    assert acquisition_ids(maps,['00000063']).tolist()==[stamp]


def test_duplicate_skeleton_counters_cannot_silently_overwrite_frames(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map
    ir=tmp_path/'ir';sk=tmp_path/'sk/predictions';ir.mkdir();sk.mkdir(parents=True)
    (ir/'IR_00000063.png').write_bytes(b'')
    for date in ('2025-06-11','2025-06-12'):
        (sk/f'Color_{date}_16-53-26.216_00000063.json').write_text('[]')
    with pytest.raises(ValueError,match='Ambiguous'):
        compatible_frame_map(sk.parent,'skeleton',ir)


def test_numeric_ir_and_skeleton_align_but_do_not_invent_absolute_time(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map,acquisition_ids
    ir=tmp_path/'ir';sk=tmp_path/'sk/predictions';ir.mkdir();sk.mkdir(parents=True)
    (ir/'IR_00000063.png').write_bytes(b'')
    (sk/'Color_00000063.json').write_text('[]')
    maps={'ir':compatible_frame_map(ir,'ir'),'skeleton':compatible_frame_map(sk.parent,'skeleton',ir)}
    assert set(maps['skeleton'])=={'00000063'}
    assert acquisition_ids(maps,['00000063']).tolist()==['']


def test_equal_numeric_skeleton_alias_uses_unique_real_timestamp(tmp_path):
    from src.experiments.pose_roi_adapter import compatible_frame_map
    ir=tmp_path/'ir';sk=tmp_path/'sk/predictions';ir.mkdir();sk.mkdir(parents=True)
    (ir/'IR_00000063.png').write_bytes(b'')
    stamped=sk/'Color_2025-06-11_16-53-26.216_00000063.json'
    stamped.write_text('[]');(sk/'Color_00000063.json').write_text('[]')
    assert compatible_frame_map(sk.parent,'skeleton',ir)=={'00000063':stamped}
    (sk/'Color_00000063.json').write_text('[1]')
    with pytest.raises(ValueError,match='Ambiguous'):
        compatible_frame_map(sk.parent,'skeleton',ir)
