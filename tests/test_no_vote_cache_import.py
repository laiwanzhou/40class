"""Task2: immutable public-only bridges and bounded shared verification."""
from dataclasses import replace
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import RowIndex


def api():
    import src.experiments.artifact_record as ar
    assert hasattr(ar, 'VerificationContext'), 'Task2 shared context is missing'
    module = importlib.import_module('src.experiments.no_vote_cache_import')
    return ar, module


def pair(fixture):
    p=load_protocol(fixture[0])
    root=p.run_root.parent/'v3'
    root.mkdir()
    (root/'protocol').mkdir()
    (root/'protocol/weights_manifest.json').write_bytes((p.run_root/'protocol/weights_manifest.json').read_bytes())
    new=replace(p,run_id='v3',run_root=root,
        public_manifests={k:root/'protocol'/v.name for k,v in p.public_manifests.items()},
        supervised_labels={k:root/'protocol'/v.name for k,v in p.supervised_labels.items()},
        recipe={**p.recipe,'source_contract_version':3})
    return p,new


def prepared(fixture,tmp_path):
    from src.experiments.no_vote_manifest import prepare_inputs
    p,new=pair(fixture)
    source=tmp_path/'source.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id']);w.writeheader()
        for user in ['fit1','dev1','target1']:
            for label in range(40):w.writerow(dict(sample_id=f'{user}-{label}',class_id=label,user_id=user))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    return p,new


def raw_chain(p):
    from src.experiments.no_vote_manifest import read_public_rows,row_index
    r=ArtifactRegistry(p)
    receipt=json.loads((p.run_root/'protocol/prepared_inputs.json').read_text())
    from src.experiments.no_vote_types import ArtifactRef
    entry=receipt['partitions']['train12'];manifest=ArtifactRef(Path(entry['record_path']),entry['sha256'])
    rows=row_index(read_public_rows(p.public_manifests['train12']))
    weight=p.weights['yolo']
    init=r.register(stage='pose_initializer',kind='public_weights',phase='public',files=[weight],config={'role':'yolo'})
    paths=[]
    for i,sid in enumerate(rows.sample_ids):
        path=p.run_root/'p28'/f'{sid}.npz';path.parent.mkdir(exist_ok=True)
        np.savez(path,sample_id=sid,acquisition_ids=np.array(['2026-07-01_13-02-03.125_000001'] if i==0 else [],dtype=str),
            frame_ids=np.array(['frame1']),completed=np.bool_(True))
        paths.append(path)
    path=paths[0]
    a=r.register(stage='p28',kind='raw_cache',phase='raw',files=paths,parents=[manifest,init],rows=rows)
    b=r.register(stage='p29',kind='raw_cache',phase='raw',files=paths,parents=[a],rows=rows)
    return r,rows,a,b,path


def test_prepare_v3_reads_only_public_and_three_allowed_labels(no_vote_fixture,tmp_path,monkeypatch):
    _,m=api();old,new=prepared(no_vote_fixture,tmp_path)
    forbidden=tmp_path/'private'
    original=Path.open
    def guarded(path,*a,**k):
        assert not path.resolve().is_relative_to(forbidden)
        assert path.name!='source.csv', 'must not reopen canonical labels'
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    m.prepare_from_existing(old,new)
    assert not (new.run_root/'protocol/final_labels.csv').exists()
    assert all(new.supervised_labels[k].exists() for k in new.supervised_labels)
    assert all(old.public_manifests[k].read_bytes()==new.public_manifests[k].read_bytes() for k in old.partitions)
    from src.experiments.no_vote_manifest import load_stage_inputs
    assert load_stage_inputs(new,'final4','predict').labels is None


def test_import_hardlink_bridge_consumable(no_vote_fixture,tmp_path,monkeypatch):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,a,b,path=raw_chain(old);context=ar.VerificationContext(new)
    original=ar.sha256_file
    def no_raw(p):
        assert Path(p)!=path, 'bridge must not hash old raw cache payload'
        return original(p)
    monkeypatch.setattr(ar,'sha256_file',no_raw)
    imported=m.import_public_cache(b,new,rows,context,source_protocol=old)
    registry=ar.ArtifactRegistry(new,context=context)
    record=registry.verify(imported,'p29','raw',rows)
    linked=Path(next(iter(record.files)))
    assert os.path.samefile(path,linked)
    assert registry.verify(record.parents[0]).stage=='p28'
    assert all(parent.record_path.is_relative_to(new.run_root) for parent in record.parents)
    assert m.import_public_cache(b,new,rows,context,source_protocol=old)==imported
    assert record.config['import_provenance']['source_ref_sha256']==b.sha256
    assert r.verify(b).protocol_sha256==old.identity()


def test_imported_payload_cannot_be_opened_for_writing(no_vote_fixture,tmp_path):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    _,rows,_,ref,_=raw_chain(old)
    dest=m.import_public_cache(ref,new,rows,ar.VerificationContext(new),source_protocol=old)
    path=Path(next(iter(ar.ArtifactRegistry(new).read(dest).files)))
    with pytest.raises(ValueError,match='immutable'):
        m.assert_writable_output(path,new)


@pytest.mark.parametrize('stage,kind',[('A1','supervised_model'),('A2','raw_cache'),('visual_sequence','raw_cache'),('targets','raw_cache')])
def test_reuse_raw_only_rejects_learned(no_vote_fixture,tmp_path,stage,kind):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,a,_,path=raw_chain(old)
    kwargs={'fit_users':old.partitions['train12'].users} if kind=='supervised_model' else {}
    ref=r.register(stage=stage,kind=kind,phase='select' if kwargs else 'raw',files=[path],rows=rows,**kwargs)
    with pytest.raises(ValueError,match='import|learned|public'):
        m.import_public_cache(ref,new,rows,ar.VerificationContext(new),source_protocol=old)


def test_import_rejects_old_learned_parents(no_vote_fixture,tmp_path):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,_,_,path=raw_chain(old)
    model=r.register(stage='A1',kind='supervised_model',phase='select',files=[path],fit_users=old.partitions['train12'].users)
    raw=r.register(stage='pixels',kind='raw_cache',phase='raw',files=[path],parents=[model],rows=rows)
    with pytest.raises(ValueError,match='import|learned|public'):
        m.import_public_cache(raw,new,rows,ar.VerificationContext(new),source_protocol=old)


@pytest.mark.parametrize('bad',['ids','users','windows','views'])
def test_import_exact_ids_axes(no_vote_fixture,tmp_path,bad):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,_,_,_=raw_chain(old)
    root=old.run_root/'pixels';root.mkdir()
    p=root/'images.npy'
    np.save(p,np.zeros((40,1 if bad=='windows' else 2,16,2 if bad=='views' else 3,160,160),np.uint8))
    ref=r.register(stage='pixels',kind='raw_cache',phase='raw',files=[p],rows=rows,
        config={'frames':16,'resolution':160,'views':['scene','person','workspace']})
    if bad=='ids':rows=RowIndex(tuple(reversed(rows.sample_ids)),rows.user_ids,rows.class_ids)
    if bad=='users':rows=RowIndex(rows.sample_ids,('dev1',)*len(rows.sample_ids),rows.class_ids)
    with pytest.raises(ValueError,match='IDs|axis|schema|shape|user|files'):
        m.import_public_cache(ref,new,rows,ar.VerificationContext(new),source_protocol=old)


def test_shared_ancestors_read_once_and_register_consume_no_rehash(no_vote_fixture,tmp_path,monkeypatch):
    ar,_=api();old,new=pair(no_vote_fixture);c=ar.VerificationContext(new)
    r=ar.ArtifactRegistry(new,context=c)
    parent=new.run_root/'parent.json';parent.write_text('{}')
    ref=r.register(stage='p28',kind='raw_cache',phase='raw',files=[parent])
    original=ar.sha256_file;reads=[]
    def counted(path):reads.append(Path(path));return original(path)
    monkeypatch.setattr(ar,'sha256_file',counted)
    for i in range(3):
        path=new.run_root/f'child{i}.json';path.write_text('{}')
        child=r.register(stage='p29',kind='raw_cache',phase='raw',files=[path],parents=[ref])
        r.verify(child);r.verify_file(r.read(child),path)
    assert reads.count(ref.record_path)==0
    assert all(reads.count(new.run_root/f'child{i}.json')==1 for i in range(3))
    parent.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='drift|hash'):r.verify_file(r.read(ref),parent)


def test_six_heads_load_once(no_vote_fixture,monkeypatch):
    ar,_=api();p=load_protocol(no_vote_fixture[0]);c=ar.VerificationContext(p)
    path=p.run_root/'features.npz';np.savez(path,features=np.zeros((3,2,3,1024),np.float32))
    original=np.load;calls=[]
    def counted(*a,**k):calls.append(a[0]);return original(*a,**k)
    monkeypatch.setattr(np,'load',counted)
    objects=[c.load_arrays(path) for _ in range(6)]
    assert len(calls)==1 and all(x is objects[0] for x in objects)
    assert not objects[0]['features'].flags.writeable


def test_imported_visual_loader_six_heads_no_full_hash(no_vote_fixture,tmp_path,monkeypatch):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,_,roi,_=raw_chain(old)
    path=old.run_root/'features.npz'
    np.savez(path,sample_ids=rows.sample_ids,class_ids=np.arange(40),features=np.zeros((40,2,3,1024),np.float32),
        kinetics_logits=np.zeros((40,2,3,400),np.float32),valid=np.ones(40,bool))
    features=r.register(stage='visual_features',kind='raw_cache',phase='raw',files=[path],parents=[roi],rows=rows,
        config={'backbone_frozen':True,'partition':'train12'})
    context=ar.VerificationContext(new)
    dest=m.import_public_cache(features,new,rows,context,source_protocol=old)
    original=ar.sha256_file
    def guard(p):
        assert Path(p).name!='features.npz','must not hash whole frozen feature table'
        return original(p)
    monkeypatch.setattr(ar,'sha256_file',guard)
    from src.experiments.visual_teacher import _load_features
    arrays=[_load_features(dest,new,context=context)[1] for _ in range(6)]
    assert all(x is arrays[0] for x in arrays)


def test_recording_sidecar_label_free(no_vote_fixture,tmp_path):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    _,rows,_,ref,_=raw_chain(old);context=ar.VerificationContext(new)
    dest=m.import_public_cache(ref,new,rows,context,source_protocol=old)
    from src.experiments.no_vote_manifest import load_stage_inputs,build_recording_metadata
    artifact=build_recording_metadata(load_stage_inputs(new,'train12','raw'),dest,protocol=new,context=context)
    record=ar.ArtifactRegistry(new,context=context).verify(artifact)
    with Path(next(iter(record.files))).open() as f:
        reader=csv.DictReader(f);names=reader.fieldnames;data=list(reader)
    assert names==['sample_id','user_id','partition','recording_date','start_seconds','timestamp_available','time_source']
    assert len(data)==40 and data[0]['recording_date']=='2026-07-01'
    assert float(data[0]['start_seconds'])==13*3600+2*60+3.125
    assert all(x['recording_date']=='' and x['timestamp_available']=='0' for x in data[1:])
    assert record.fit_users==()


def test_v3_preparation_rejects_canonical_label_source(no_vote_fixture,tmp_path):
    api();_,new=prepared(no_vote_fixture,tmp_path)
    from src.experiments.no_vote_manifest import prepare_inputs
    with pytest.raises(ValueError,match='existing public'):
        prepare_inputs(tmp_path/'source.csv',new,new.run_root/'protocol',tmp_path/'private2')


def test_public_expert_modalities_whitelist(no_vote_fixture,tmp_path):
    api();old,new=prepared(no_vote_fixture,tmp_path)
    from src.experiments import no_vote_manifest as nm
    assert hasattr(nm,'read_expert_public_rows'), 'Task2 expert modality whitelist missing'
    rows=nm.read_public_rows(old.public_manifests['train12']);index=nm.row_index(rows)
    path=tmp_path/'thermal.csv'
    table=[dict(sample_id=x['sample_id'],user_id=x['user_id'],thermal_path='',thermal_available='0') for x in rows]
    path.write_bytes(nm.csv_bytes(nm.EXPERT_PUBLIC_COLUMNS,table))
    assert len(nm.read_expert_public_rows(path,index))==40
    path.write_bytes(nm.csv_bytes((*nm.EXPERT_PUBLIC_COLUMNS,'trial_id'),[{**x,'trial_id':'1'} for x in table]))
    with pytest.raises(ValueError,match='whitelist'):nm.read_expert_public_rows(path,index)


def test_pixels_positive_and_readonly_mmap(no_vote_fixture,tmp_path,monkeypatch):
    ar,m=api();old,new=prepared(no_vote_fixture,tmp_path);m.prepare_from_existing(old,new)
    r,rows,_,roi,_=raw_chain(old)
    from src.experiments.no_vote_manifest import csv_bytes,PUBLIC_COLUMNS,read_public_rows
    root=old.run_root/'pixels';root.mkdir()
    specs={'images':((40,2,16,3,160,160),np.uint8),'view_valid':((40,2,16,3),bool),
        'view_quality':((40,2,16,3),np.float32),'source_frame_indices':((40,2,16),np.int64),
        'source_time_seconds':((40,2,16),np.float64),'completed':((40,),bool)}
    files=[]
    for name,(shape,dtype) in specs.items():
        p=root/(name+'.npy');a=np.lib.format.open_memmap(p,mode='w+',dtype=dtype,shape=shape)
        a[:]=True if name=='completed' else 0;a.flush();del a;files.append(p)
    path=root/'rows.csv';path.write_bytes(csv_bytes(PUBLIC_COLUMNS,read_public_rows(old.public_manifests['train12'])));files.append(path)
    ref=r.register(stage='pixels',kind='raw_cache',phase='raw',files=files,parents=[roi],rows=rows,
        config={'partition':'train12','frames':16,'resolution':160,'views':['scene','person','workspace']})
    context=ar.VerificationContext(new);dest=m.import_public_cache(ref,new,rows,context,source_protocol=old)
    original=ar.sha256_file
    def guard(p):
        assert Path(p).suffix!='.npy','must not hash pixel arrays after bridge'
        return original(p)
    monkeypatch.setattr(ar,'sha256_file',guard)
    from src.experiments.pixel_cache import load_pixels
    _,arrays=load_pixels(dest,protocol=new,context=context)
    assert arrays['images'].shape==specs['images'][0] and arrays['images'].mode=='r'
    with pytest.raises(ValueError):arrays['images'][0]=1
