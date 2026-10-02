from __future__ import annotations

import csv
from pathlib import Path

import pytest


def make_manifest(path: Path):
    columns = ['sample_id','class_id','action_name','user_id','trial_id','ir_path','depth_color_path','skeleton_path','imu_path']
    with path.open('w',newline='',encoding='utf-8') as f:
        out=csv.DictWriter(f,fieldnames=columns);out.writeheader()
        for user in ('fit1','dev1','target1'):
            for label in range(40):
                out.writerow(dict(sample_id=f'train__c{label:02}__{user}__one',class_id=label,
                    action_name=f'action{label}',user_id=user,trial_id='one',ir_path='',depth_color_path='',skeleton_path='',imu_path=''))


def test_final_whitelist_and_private_mapping(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs, PUBLIC_COLUMNS, load_stage_inputs
    config,_=no_vote_fixture; p=load_protocol(config)
    source=tmp_path/'canonical.csv';make_manifest(source)
    private=tmp_path/'private'
    paths=prepare_inputs(source,p,p.run_root/'protocol',private)
    for partition in p.partitions:
        with paths[partition].open(encoding='utf-8') as f:
            reader=csv.DictReader(f); assert reader.fieldnames==list(PUBLIC_COLUMNS)
            rows=list(reader)
        assert len(rows)==p.partitions[partition].expected_rows
        assert all(len(r['sample_id'])==64 and '__c' not in r['sample_id'] for r in rows)
    assert (private/'final_labels.csv').is_file() and (private/'id_mapping.csv').is_file()
    final=load_stage_inputs(p,'final4','predict')
    assert final.labels is None
    assert len(list(csv.DictReader((private/'final_labels.csv').open())))==40
    with pytest.raises(ValueError):load_stage_inputs(p,'final4','select')


def test_generation_never_reads_private_labels(no_vote_fixture,tmp_path,monkeypatch):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,load_stage_inputs,read_public_rows
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv';make_manifest(source)
    private=tmp_path/'private';prepare_inputs(source,p,p.run_root/'protocol',private)
    first=read_public_rows(load_stage_inputs(p,'final4','predict').public_manifest)
    (private/'final_labels.csv').write_text('sample_id,class_id\nwrong,99\n')
    original=Path.open
    def guarded(path,*args,**kwargs):
        assert path.resolve()!=source.resolve() and not path.resolve().is_relative_to(private)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    second=read_public_rows(load_stage_inputs(p,'final4','predict').public_manifest)
    assert first==second


def test_missing_inputs_preserved_and_unknown_labels_rejected(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv';make_manifest(source)
    paths=prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    assert all(r['ir_available']=='0' for r in read_public_rows(paths['final4']))
    text=paths['final4'].read_text().replace('sample_id,user_id','sample_id,action_name,user_id')
    paths['final4'].write_text(text)
    with pytest.raises(ValueError):read_public_rows(paths['final4'])


def test_prepare_is_idempotent_and_refuses_drift(no_vote_fixture,tmp_path):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs
    p=load_protocol(no_vote_fixture[0]);source=tmp_path/'canonical.csv';make_manifest(source)
    private=tmp_path/'private';paths=prepare_inputs(source,p,p.run_root/'protocol',private)
    original={k:v.read_bytes() for k,v in paths.items()}
    prepare_inputs(source,p,p.run_root/'protocol',private)
    assert original=={k:v.read_bytes() for k,v in paths.items()}
    source.write_text(source.read_text().replace('target1__one','target1__two'))
    with pytest.raises(ValueError,match='drift'):prepare_inputs(source,p,p.run_root/'protocol',private)


def test_imu_available_does_not_count_repository_text_files(tmp_path):
    from src.experiments.no_vote_manifest import _available
    (tmp_path/'requirements-x3d.txt').write_text('torch')
    (tmp_path/'.git').write_text('gitdir: elsewhere')
    assert not _available(tmp_path,'imu')
    (tmp_path/'up(LA+RA+C).csv').write_text('sensor,timestamp\n')
    assert _available(tmp_path,'imu')
