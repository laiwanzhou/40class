"""Task3: bounded source geometry parity and exact v3 imported-cache consumption."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.experiments import pose_roi_adapter as adapter
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_manifest import load_stage_inputs
from src.experiments.no_vote_types import RowIndex
from test_no_vote_cache_import import prepared,raw_chain
from test_no_vote_source_ops import v3_protocol


@pytest.fixture
def source_ops(no_vote_fixture):
    return adapter._load_ops(v3_protocol(no_vote_fixture))


def test_pose_source_loading_is_bounded(no_vote_fixture,monkeypatch):
    original=adapter.verify_teammate_source
    def bounded(*args,**kwargs):
        assert kwargs.get('verify_contents') is False, 'Task3 must not hash the entire source snapshot'
        return original(*args,**kwargs)
    monkeypatch.setattr(adapter,'verify_teammate_source',bounded)
    adapter._load_ops(v3_protocol(no_vote_fixture))


def test_imported_pose_roi_sample_is_consumable(no_vote_fixture,tmp_path,monkeypatch):
    from src.experiments.no_vote_cache_import import prepare_from_existing,import_public_cache
    old,new=prepared(no_vote_fixture,tmp_path);prepare_from_existing(old,new)
    r,rows,old_pose,_,_=raw_chain(old);pose_files=[];roi_files=[]
    for sid in rows.sample_ids:
        pose=adapter._empty_pose([])
        pose.update(acquisition_ids=np.array([],str),sample_id=np.asarray(sid),available=np.bool_(False),completed=np.bool_(True))
        roi=dict(frame_ids=pose['frame_ids'],acquisition_ids=pose['acquisition_ids'],region_names=np.array(adapter.REGIONS),
            roi_boxes_xyxy=np.empty((0,7,4),np.float32),roi_valid=np.empty((0,7),bool),
            roi_quality=np.empty((0,7),np.float32),roi_source=np.empty((0,7),np.uint8),
            sample_id=np.asarray(sid),available=np.bool_(False),completed=np.bool_(True))
        for stage,values,files in [('p28',pose,pose_files),('p29',roi,roi_files)]:
            path=old.run_root/'geometry'/stage/(sid+'.npz');adapter._atomic_npz(path,values);files.append(path)
    config={'algorithm':'teammate_adaptive_ir_pose_roi','settings':vars(adapter._settings('cpu'))}
    source_pose=r.register(stage='p28',kind='raw_cache',phase='raw',files=pose_files,rows=rows,
        parents=r.read(old_pose).parents,config=config)
    source_roi=r.register(stage='p29',kind='raw_cache',phase='raw',files=roi_files,rows=rows,parents=[source_pose],config=config)
    context=VerificationContext(new);imported=import_public_cache(source_roi,new,rows,context,source_protocol=old)
    registry=ArtifactRegistry(new,context=context);pose_ref=registry.read(imported).parents[0]
    inputs=load_stage_inputs(new,'train12','raw',context=context)
    assert hasattr(adapter,'load_pose_roi_sample'), 'Task3 imported-cache consumer missing'
    def forbidden(*a,**k):raise AssertionError('cached geometry rescanned raw files or invoked detector')
    monkeypatch.setattr(adapter,'snapshot_raw_files',forbidden)
    monkeypatch.setattr(adapter,'_process_trial',forbidden)
    pose,roi=adapter.load_pose_roi_sample(inputs,pose_ref,imported,rows.sample_ids[0],protocol=new,context=context)
    assert pose['frame_ids'].shape==(0,) and roi['roi_boxes_xyxy'].shape==(0,7,4)
    assert not pose['frame_ids'].flags.writeable
    with pytest.raises(ValueError,match='sample|ID'):
        adapter.load_pose_roi_sample(inputs,pose_ref,imported,'not-in-this-partition',protocol=new,context=context)
    with pytest.raises(ValueError,match='stage|parent|P28'):
        adapter.load_pose_roi_sample(inputs,imported,imported,rows.sample_ids[0],protocol=new,context=context)


def test_geometry_settings_and_missing_roi_source_parity(source_ops,monkeypatch):
    import sys
    base,adaptive,roi=source_ops
    monkeypatch.setattr(sys,'argv',['fixture'])
    settings=adapter._settings('cpu')
    for parsed in [adaptive.parse_args(),roi.parse_args()]:
        for name in vars(settings).keys() & vars(parsed).keys():
            if name!='device':assert getattr(settings,name)==getattr(parsed,name),name
    empty=adapter._empty_pose([])
    actual=roi.build_trial_rois(empty,settings)
    assert tuple(actual['region_names'])==adapter.REGIONS
    assert actual['roi_boxes_xyxy'].shape==(0,7,4)


def test_two_pass_geometry_and_roi_fixture(source_ops,tmp_path,monkeypatch):
    base,adaptive,roi=source_ops;args=adapter._settings('cpu')
    box=np.array([100,40,400,440,.9],np.float32)
    joints=np.zeros((17,3),np.float32);joints[:,:2]=[200,160];joints[:,2]=.9
    joints[5,:2]=[180,120];joints[7,:2]=[150,200];joints[9,:2]=[130,280]
    joints[6,:2]=[280,120];joints[8,:2]=[310,200];joints[10,:2]=[330,280]
    def candidate(present=True):return dict(boxes=box[None].copy() if present else np.empty((0,5),np.float32),
        keypoints=joints[None].copy() if present else np.empty((0,17,3),np.float32),shape=(480,640))
    primary=[candidate(),candidate(False),candidate(False),candidate()]
    fallback=[candidate(),candidate(False)]
    keys=[f'{i:08}' for i in range(4)];calls=[]
    def predict(model,paths,**kwargs):
        calls.append((list(paths),kwargs));return primary if kwargs['imgsz']==640 else fallback
    monkeypatch.setattr(adaptive,'predict_candidates',predict)
    maps={'ir':{k:tmp_path/(k+'.png') for k in keys},'depth':{},'skeleton':{}}
    pose,actual=adapter._process_trial(maps,object(),source_ops,args)
    assert len(calls)==2 and [x[1]['imgsz'] for x in calls]==[640,1280]
    assert calls[0][1]['conf']==.10 and calls[1][1]['conf']==.03
    assert calls[0][1]['batch']==32 and calls[1][1]['batch']==16
    assert pose['fallback_selected'].tolist()==[False,True,False,False]
    assert pose['box_interpolated'].tolist()==[False,False,True,False]
    assert np.isnan(pose['ir_keypoints_xy_conf'][2]).all()
    assert not pose['depth_frame_valid'].any() and np.isnan(pose['depth_boxes_from_ir_xyxy_conf']).all()
    assert pose['acquisition_ids'].tolist()==['']*4
    expected=roi.build_trial_rois(pose,adapter._settings('cpu'))
    for name in expected:
        if expected[name].dtype.kind in 'fci':np.testing.assert_allclose(actual[name],expected[name],rtol=0,atol=0,equal_nan=True)
        else:np.testing.assert_array_equal(actual[name],expected[name])


def test_person_workspace_crop_and_missing_fallback(no_vote_fixture,tmp_path):
    import cv2
    from src.experiments.teammate_source import verify_teammate_source,load_teammate_symbol
    from src.experiments.visual_teacher import _prepare_clips
    p=v3_protocol(no_vote_fixture)
    report=verify_teammate_source(p.source_root,p.source_root.parent/'source_manifest.json',verify_contents=False)
    crop=load_teammate_symbol(report,'build_p46_videomae_cache','square_crop')
    read_ir=load_teammate_symbol(report,'build_p30_shared_dir_roi_feature_cache','read_ir')
    folder=tmp_path/'ir';folder.mkdir();image=np.tile(np.arange(640,dtype=np.uint16)%256,(480,1)).astype(np.uint8)
    for i in range(3):assert cv2.imwrite(str(folder/f'IR_{i:08}.png'),image)
    ids=np.array([f'{i:08}' for i in range(3)])
    boxes=np.full((3,7,4),np.nan,np.float32);valid=np.zeros((3,7),bool)
    boxes[0,0]=[100,60,300,400];boxes[0,5]=[120,120,200,220];valid[0,[0,5]]=True
    boxes[1,0]=boxes[0,0];valid[1,0]=True
    path=tmp_path/'roi.npz';np.savez(path,sample_id='fixture',completed=True,frame_ids=ids,
        acquisition_ids=np.array(['']*3),region_names=adapter.REGIONS,roi_boxes_xyxy=boxes,roi_valid=valid)
    clips,metadata=_prepare_clips({'sample_id':'fixture','ir_path':str(folder)},path,crop,read_ir)
    assert len(clips)==6 and metadata['frame_indices'].shape==(2,16)
    original=read_ir(folder/'IR_00000000.png')
    for w in range(2):
        for t,index in enumerate(metadata['frame_indices'][w]):
            expected_person=crop(original,boxes[index,0],1.15) if valid[index,0] else original
            wb=boxes[index,5] if valid[index,5] else boxes[index,0]
            expected_workspace=crop(original,wb,1.40)
            np.testing.assert_array_equal(clips[w*3][t],original)
            np.testing.assert_array_equal(clips[w*3+1][t],expected_person)
            np.testing.assert_array_equal(clips[w*3+2][t],expected_workspace)
    assert metadata['acquisition_ids'].tolist()==[['']*16]*2
