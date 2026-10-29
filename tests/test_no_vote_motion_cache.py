"""Task6: source physical operators, exact pixel grid and no fitted transform."""
from dataclasses import replace
import importlib
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.artifact_record import ArtifactRegistry, VerificationContext
from src.experiments.no_vote_types import write_json
from test_no_vote_source_ops import v3_protocol


def module():
    assert importlib.util.find_spec('src.experiments.motion_cache'), 'Task6 motion_cache is missing'
    return importlib.import_module('src.experiments.motion_cache')


def trial_values():
    n=22;times=np.arange(n,dtype=np.float64)*.1
    xyz=np.zeros((n,17,4),np.float32);xyz[:,:,3]=.9
    xyz[:,0,:3]=[0,0,2];xyz[:,8,:3]=[0,.6,2];xyz[:,10,:3]=[0,1,2]
    xyz[:,11,:3]=[-.3,.7,2];xyz[:,14,:3]=[.3,.7,2]
    xyz[:,13,0]=np.linspace(-.8,-.2,n);xyz[:,16,0]=.4
    imu=np.zeros((120,10),np.float32);imu[:,0]=np.sin(np.arange(120)*.1);imu[:,2]=9.8;imu[:,6]=1
    return xyz,times,{'WTC':(times[0]+np.linspace(-.1,2.2,120),imu)}


def test_no_added_global_standardization(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);c=VerificationContext(p);ops=m.source_motion_ops(p,c)
    x={k:np.array([[[12.,-300.]]],np.float32) for k in ops['MOTION_FIELDS'] if k not in ops['BOOLEAN_MOTION_FIELDS']}
    x.update({k:np.array([[[True,False]]]) for k in ops['BOOLEAN_MOTION_FIELDS']})
    original={k:v.copy() for k,v in x.items()}
    transformed=m.apply_source_transform(x)
    assert set(transformed)==set(ops['MOTION_FIELDS'])
    for k in x:np.testing.assert_array_equal(transformed[k],original[k]);np.testing.assert_array_equal(x[k],original[k])
    assert 'labels' not in __import__('inspect').signature(m.apply_source_transform).parameters
    assert 'fit' not in __import__('inspect').signature(m.apply_source_transform).parameters


def test_source_window_alignment(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_motion_ops(p,VerificationContext(p))
    raw,times,imu=trial_values();trial=m.prepare_source_arrays(raw,np.ones(len(raw),np.uint8),times,imu,ops)
    chosen=np.stack([np.rint(np.linspace(0,14,16)),np.rint(np.linspace(6,21,16))]).astype(np.int64)
    actual=m.window_arrays(trial,chosen,ops)
    rotated=ops['rotate_skeleton_cache'](trial['skeleton_features'],trial['skeleton_feature_mask'],trial['skeleton_joint_mask'],trial['skeleton_relations'],trial['skeleton_relation_mask'],trial['frame_time_seconds'])
    np.testing.assert_array_equal(actual['skeleton_features'],rotated['features'][chosen].astype(np.float16))
    expected=ops['aggregate_imu'](trial['frame_time_seconds'],chosen,trial['imu_values'],trial['imu_time_seconds'],trial['imu_device_offsets'],4)
    mapping={'imu_sequences':'sequences','imu_sequence_mask':'sequence_mask','imu_bin_statistics':'statistics','imu_bin_mask':'bin_mask','imu_global_statistics':'global_statistics','imu_global_mask':'global_mask'}
    for k,v in mapping.items():np.testing.assert_array_equal(actual[k],expected[v].astype(actual[k].dtype))
    assert actual['imu_global_statistics'].dtype==np.float32
    assert not actual['imu_bin_mask'][:,:,1:].any()


def test_source_p31_masks_and_point_accounting(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_motion_ops(p,VerificationContext(p))
    raw,times,imu=trial_values();raw[4,13]=np.nan
    got=m.prepare_source_arrays(raw,np.ones(len(raw),np.uint8),times,imu,ops)
    reference=ops['build_skeleton_features'](raw,times)
    for a,b in [('skeleton_features','features'),('skeleton_feature_mask','feature_mask'),('skeleton_joint_mask','joint_mask'),('skeleton_relations','relations'),('skeleton_relation_mask','relation_mask')]:np.testing.assert_array_equal(got[a],reference[b])
    assert got['imu_device_offsets'].tolist()==[0,120,120,120,120,120]
    assert int(got['imu_interval_counts'].sum())==120
    assert not got['skeleton_joint_mask'][4,13]


def test_missing_inputs_keep_source_masks(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_motion_ops(p,VerificationContext(p))
    raw=np.full((3,17,4),np.nan,np.float32)
    trial=m.prepare_source_arrays(raw,np.zeros(3,np.uint8),np.arange(3)*.1,{},ops)
    actual=m.window_arrays(trial,np.zeros((2,16),np.int64),ops)
    for k in ops['BOOLEAN_MOTION_FIELDS']:assert not actual[k].any()
    assert not actual['imu_global_mask'].any() and not actual['imu_global_statistics'].any()
    with pytest.raises(ValueError,match='frame|indices|grid'):m.window_arrays(trial,np.full((2,16),3),ops)


def test_dataset_temporal_axes_match_source(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_motion_ops(p,VerificationContext(p))
    raw,times,imu=trial_values();trial=m.prepare_source_arrays(raw,np.ones(len(raw),np.uint8),times,imu,ops)
    arrays=m.window_arrays(trial,np.tile(np.arange(16),(2,1)),ops);order=np.arange(16)[::-1]
    got=m.motion_item(arrays,temporal_index=order)
    source=ops['source_dataset'].__new__(ops['source_dataset'])
    source.motion={k:v[None] for k,v in arrays.items()};source.indices=np.array([0]);source.temporal_augment=True
    source.temporal_indices=lambda steps:order
    source.rows=[{'sample_id':'synthetic','user_id':'synthetic','class_id':'0'}]
    for attr in ['teacher_logits','teacher_features','imu_teacher_logits','imu_event_features']:setattr(source,attr,None)
    source_item=source[0]
    for key in ops['MOTION_FIELDS']:
        expected=arrays[key][:,order] if key in ops['TEMPORAL_MOTION_FIELDS'] else arrays[key]
        dtype=bool if key in ops['BOOLEAN_MOTION_FIELDS'] else np.float32
        np.testing.assert_array_equal(got[key],expected.astype(dtype))
        np.testing.assert_array_equal(got[key],source_item[key].numpy())


def test_transform_descriptor_has_no_fit_or_labels(no_vote_fixture,monkeypatch):
    m=module();p=v3_protocol(no_vote_fixture);c=VerificationContext(p);registry=ArtifactRegistry(p,context=c)
    path=p.run_root/'motion_stub.json';write_json(path,{'completed':True})
    motion=registry.register(stage='motion_windows',kind='raw_cache',phase='raw',files=[path],config={})
    original=Path.open
    def guarded(path,*a,**k):
        assert 'labels' not in path.name and path.name!='final_labels.csv'
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    transform=m.describe_motion_transform(motion,p,context=c)
    record=registry.verify(transform,'source_transform','public');body=json.loads(next(Path(x) for x in record.files).read_text(encoding='utf-8'))
    assert body['cross_sample_fit'] is False and body['imu_instance_normalization'] is False
    assert body['external_transform']=='identity' and not record.fit_users
    assert not any(x in body for x in ['mean','std','labels'])


def test_rejects_added_statistical_normalization(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture)
    bad=replace(p,recipe={**p.recipe,'motion':{'imu_instance_normalization':True}})
    with pytest.raises(ValueError,match='normalization|identity'):m.source_motion_ops(bad,VerificationContext(bad))


@pytest.fixture
def prepared_motion(no_vote_fixture):
    from src.experiments.no_vote_types import Partition
    from src.experiments.no_vote_manifest import PUBLIC_COLUMNS,csv_bytes,row_index,load_stage_inputs
    m=module();p=v3_protocol(no_vote_fixture)
    parts={k:Partition(k,p.partitions[k].users,2 if k=='refit14' else 1) for k in p.partitions}
    p=replace(p,partitions=parts);c=VerificationContext(p);registry=ArtifactRegistry(p,context=c)
    all_rows=[]
    for user in ['fit1','dev1','target1']:
        row={k:'' for k in PUBLIC_COLUMNS};row.update(sample_id=hashlib.sha256(user.encode()).hexdigest(),user_id=user)
        row.update({k:'0' for k in PUBLIC_COLUMNS if k.endswith('_available')})
        all_rows.append(row)
    refs={}
    for name,part in p.partitions.items():
        rows=[r for r in all_rows if r['user_id'] in part.users];path=p.public_manifests[name]
        path.write_bytes(csv_bytes(PUBLIC_COLUMNS,rows));index=row_index(rows)
        ref=registry.register(stage='manifest_'+name,kind='public_manifest',phase='raw',files=[path],rows=index)
        refs[name]={'record_path':str(ref.record_path),'sha256':ref.sha256}
    receipt=p.run_root/'protocol/prepared_inputs.json';write_json(receipt,{'partitions':refs,'supervised_label_hashes':{}})
    write_json(p.run_root/'run_state.json',{'protocol_sha256':p.identity(),'state':'prepared','prepared_inputs_sha256':c.hash_file(receipt)})
    inputs=load_stage_inputs(p,'refit14','raw',context=c);index=row_index([r for r in all_rows if r['user_id']!='target1'])
    paths=[]
    for sid in index.sample_ids:
        path=p.run_root/'pose'/(sid+'.npz');path.parent.mkdir(exist_ok=True)
        np.savez(path,sample_id=sid,completed=True,frame_ids=np.array([],dtype=str),acquisition_ids=np.array([],dtype=str),
            skeleton_h36m_xyz_conf_raw=np.empty((0,17,4),np.float32),skeleton_person_count=np.empty(0,np.uint8))
        paths.append(path)
    pose=registry.register(stage='p28',kind='raw_cache',phase='raw',files=paths,rows=index,parents=[inputs.parents['manifest']])
    root=p.run_root/'pixel_fixture';root.mkdir();n=len(index.sample_ids)
    values={'images':np.zeros((n,2,16,3,160,160),np.uint8),'view_valid':np.zeros((n,2,16,3),bool),
        'view_quality':np.zeros((n,2,16,3),np.float32),'source_frame_indices':np.zeros((n,2,16),np.int64),
        'source_time_seconds':np.full((n,2,16),np.nan,np.float64),'completed':np.ones(n,bool)}
    for k,v in values.items():np.save(root/(k+'.npy'),v)
    rows=root/'rows.csv';rows.write_bytes(p.public_manifests['refit14'].read_bytes())
    pixels=registry.register(stage='pixels',kind='raw_cache',phase='raw',files=list(root.iterdir()),rows=index,parents=[pose])
    return m,p,c,registry,inputs,pose,pixels


def test_complete_cache_pipeline_is_label_free_and_idempotent(prepared_motion,monkeypatch):
    m,p,c,registry,inputs,pose,pixels=prepared_motion
    original=Path.open
    def guard(path,*a,**k):
        assert 'labels' not in path.name
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guard)
    root=p.run_root/'motion_complete'
    source=m.build_motion_source(inputs,pose,root/'source',protocol=p,context=c)
    window=m.build_motion_windows(inputs,source,pixels,root/'windows',protocol=p,context=c)
    transform=m.describe_motion_transform(window,p,context=c)
    record,arrays=m.load_motion(window,protocol=p,context=c)
    assert record.complete and record.rows.sample_ids==tuple(hashlib.sha256(user.encode()).hexdigest() for user in ['fit1','dev1'])
    assert m.build_motion_source(inputs,pose,root/'source',protocol=p,context=c)==source
    assert m.build_motion_windows(inputs,source,pixels,root/'windows',protocol=p,context=c)==window
    assert m.describe_motion_transform(window,p,context=c)==transform
    assert not arrays['skeleton_joint_mask'].any() and not arrays['imu_bin_mask'].any()
    assert not arrays['skeleton_features'].flags.writeable
    assert arrays['source_time_seconds'].dtype==np.float64
    assert registry.verify(transform).kind=='descriptor'
    # A later consumer reads headers/mmap, not another full motion payload SHA.
    bounded=VerificationContext(p);real_hash=bounded.hash_file
    def no_array_hash(path,*a,**k):
        assert Path(path).suffix!='.npy', 'motion reader must not hash a whole array'
        return real_hash(path,*a,**k)
    monkeypatch.setattr(bounded,'hash_file',no_array_hash)
    again,_=m.load_motion(window,protocol=p,context=bounded)
    assert again.rows==record.rows


def test_window_reader_rejects_changed_payload_stamp(prepared_motion):
    m,p,c,registry,inputs,pose,pixels=prepared_motion;root=p.run_root/'motion_tamper'
    source=m.build_motion_source(inputs,pose,root/'source',protocol=p,context=c)
    ref=m.build_motion_windows(inputs,source,pixels,root/'windows',protocol=p,context=c)
    path=root/'windows/imu_sequences.npy'
    with path.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='stamp drift'):m.load_motion(ref,protocol=p,context=VerificationContext(p))


def test_source_window_absolute_time_alignment(no_vote_fixture):
    m=module();assert hasattr(m,'validate_pixel_timeline'), 'Task6 must verify actual pixel timestamp grid'
    times=1.75e9+np.arange(22)*.1;chosen=np.tile(np.arange(16),(2,1))
    trial={'frame_absolute_time_seconds':times}
    m.validate_pixel_timeline(trial,chosen,times[chosen])
    with pytest.raises(ValueError,match='timestamp'):m.validate_pixel_timeline(trial,chosen,times[chosen]+.1)
    m.validate_pixel_timeline({'frame_absolute_time_seconds':np.empty(0)},np.zeros((2,16),np.int64),np.full((2,16),np.nan))
