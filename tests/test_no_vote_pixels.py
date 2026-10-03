"""Pixel contracts keep IR views and source-time ownership explicit."""
import numpy as np
import pytest


def pixel_arrays(n=2):
    return dict(images=np.zeros((n,2,16,3,160,160),np.uint8),
        view_valid=np.ones((n,2,16,3),bool),view_quality=np.ones((n,2,16,3),np.float32),
        source_frame_indices=np.tile(np.arange(16),(n,2,1)),
        source_time_seconds=np.tile(np.arange(16,dtype=float),(n,2,1)),
        completed=np.ones(n,bool))


def test_pixels_contract_and_label_fields():
    from src.experiments.pixel_cache import validate_pixels
    arrays=pixel_arrays();validate_pixels(arrays,2)
    arrays['labels']=np.zeros(2)
    with pytest.raises(ValueError,match='schema|label'):validate_pixels(arrays,2)


def test_pixel_view_axis_and_completion_are_not_rgb_or_availability():
    from src.experiments.pixel_cache import validate_pixels
    arrays=pixel_arrays();arrays['view_valid'][1]=False;arrays['view_quality'][1]=0
    arrays['source_time_seconds'][1]=np.nan
    validate_pixels(arrays,2)
    arrays['images']=arrays['images'].astype(np.float32)
    with pytest.raises(ValueError,match='dtype|uint8'):validate_pixels(arrays,2)


def test_inference_dataset_does_not_require_teacher_or_labels():
    from src.experiments.no_vote_datasets import NoVotePixelDataset
    from src.experiments.no_vote_types import RowIndex
    index=RowIndex(('first','second'),('user1','user2'),tuple(range(40)))
    dataset=NoVotePixelDataset(pixel_arrays(),index)
    sample=dataset[0]
    assert sample['sample_id']=='first' and sample['images'].shape==(2,16,3,160,160)
    assert not {'label','teacher_logits','teacher_features','teacher_valid'}&set(sample)


def test_recorded_timestamps_are_restored_without_inventing_counter_time():
    from src.experiments.pixel_cache import acquisition_seconds
    known=acquisition_seconds('2025-05-08_13-57-38.716_00000238')
    later=acquisition_seconds('2025-05-08_13-57-38.816_00000239')
    assert later-known==pytest.approx(.1,abs=1e-6)
    assert np.isnan(acquisition_seconds('00000238'))


def test_build_missing_ir_pixels_keeps_completed_and_unavailable(no_vote_fixture,tmp_path,monkeypatch):
    import csv
    from pathlib import Path
    from src.experiments.pixel_cache import build_pixels,load_pixels
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows,row_index,load_stage_inputs
    from src.experiments.artifact_record import ArtifactRegistry
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);writer.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(40):writer.writerow(dict(sample_id=f'{user}-{c}',class_id=c,user_id=user))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    rows=read_public_rows(p.public_manifests['refit14']);files=[]
    for row in rows:
        path=p.run_root/(row['sample_id']+'.npz');np.savez(path,completed=True);files.append(path)
    registry=ArtifactRegistry(p);roi=registry.register(stage='p29',kind='raw_cache',phase='raw',
        files=files,rows=row_index(rows))
    ref=build_pixels(load_stage_inputs(p,'refit14','raw'),roi,p.run_root/'pixels',protocol=p,max_trials=1)
    record,arrays=load_pixels(ref,protocol=p,complete=False)
    assert not record.complete and len(record.rows.sample_ids)==1
    assert arrays['completed'].all() and not arrays['view_valid'].any()
    assert np.isnan(arrays['source_time_seconds']).all()
    # Writer digests agree with actual newly written data; this tiny fixture
    # is not a request to rescan a formal multi-GiB cache.
    from src.experiments.teammate_source import sha256_file
    assert all(sha256_file(Path(file))==digest for file,digest in record.files.items())
    (p.run_root/'pixels/artifact.json').unlink()
    original=np.ascontiguousarray
    def no_image_rehash(value,*args,**kwargs):
        if np.shape(value)==(2,16,3,160,160):raise AssertionError('completed pixel row reread for SHA')
        return original(value,*args,**kwargs)
    monkeypatch.setattr(np,'ascontiguousarray',no_image_rehash)
    restored=build_pixels(load_stage_inputs(p,'refit14','raw'),roi,p.run_root/'pixels',protocol=p,max_trials=1)
    assert restored==ref


def test_training_subset_joins_full_teacher_by_ids_without_changing_artifact_index(tmp_path):
    from src.experiments.no_vote_datasets import NoVotePixelDataset
    from src.experiments.no_vote_types import RowIndex,Prediction,TeacherTargets,ArtifactRef
    full=RowIndex(('first','second'),('user1','user2'),tuple(range(40)))
    subset=RowIndex(('second',),('user2',),tuple(range(40)))
    logits=np.stack([np.ones(40),np.full(40,2.)])
    target=TeacherTargets(Prediction(full,logits,np.full((2,40),1/40),np.ones(2,bool),
        ArtifactRef(tmp_path/'unused.json','0'*64)),np.zeros((2,2,3,1024),np.float32))
    dataset=NoVotePixelDataset(pixel_arrays(),full,rows=subset,teacher=target,labels={'second':3})
    assert np.all(dataset[0]['teacher_logits'].numpy()==2)
    assert target.prediction.index is full
