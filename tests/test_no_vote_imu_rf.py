"""Task7 RF source semantics, consumer profiles and fixed-phase ownership."""
import importlib
import csv
from pathlib import Path
import numpy as np
import pytest

from src.experiments.artifact_record import VerificationContext
from test_no_vote_source_ops import v3_protocol


def module():
    assert importlib.util.find_spec('src.experiments.imu_rf_teacher'), 'Task7 RF teacher is missing'
    return importlib.import_module('src.experiments.imu_rf_teacher')


def test_source_rf_recipe(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);model=m.make_rf(p)
    got=model.get_params()
    for key,value in dict(n_estimators=400,max_depth=18,min_samples_leaf=2,max_features='sqrt',
            class_weight='balanced_subsample',n_jobs=-1,random_state=20260723).items():assert got[key]==value


def test_source_device_dropout(no_vote_fixture):
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_rf_ops(p,VerificationContext(p))
    features=np.arange(4*240,dtype=np.float32).reshape(4,240)
    masks=np.array([[1,.5,1,1,0,0,0,0,0,0],[0,0,0,0,1,.25,0,0,0,0],[0]*10,[1,1]*5],np.float32)
    labels=np.array([3,5,8,9]);x,y,selected=m.dropout_training_data(features,masks,labels,ops)
    devices=ops['random_present_devices'](masks,np.random.default_rng(20260723))
    f,mask=ops['drop_devices'](features,masks,devices)
    np.testing.assert_array_equal(selected,devices)
    np.testing.assert_array_equal(x,np.concatenate([np.c_[features,masks],np.c_[f,mask]]))
    np.testing.assert_array_equal(y,[3,5,8,9,3,5,8,9]);assert devices[2]==-1


def test_rf_logits_export_profiles_and_class_order():
    m=module();assert hasattr(m,'aligned_rf_logits'), 'consumer-aware RF export is missing'
    probabilities=np.array([[0.,.25,.75],[.6,.4,0.]])
    classes=np.array([0,5,39])
    fit=m.aligned_rf_logits(probabilities,classes,'fit_teacher_target')
    terminal=m.aligned_rf_logits(probabilities,classes,'terminal_export')
    np.testing.assert_array_equal(fit[:,classes],np.log(np.clip(probabilities,1e-12,1)))
    np.testing.assert_array_equal(terminal[:,classes],fit[:,classes])
    assert np.all(fit[:,1]==-1e6) and np.all(terminal[:,1]==np.log(1e-12))
    with pytest.raises(ValueError,match='profile'):m.aligned_rf_logits(probabilities,classes,'auto')
    with pytest.raises(ValueError,match='classes'):m.aligned_rf_logits(probabilities,np.array([0,0,39]),'terminal_export')


def test_source_32bin_statistics_and_duplicate_timestamp_semantics(no_vote_fixture,tmp_path):
    m=module();assert hasattr(m,'statistics_for_row'), 'source32 statistic adapter is missing'
    p=v3_protocol(no_vote_fixture);ops=m.source_rf_ops(p,VerificationContext(p))
    folder=tmp_path/'imu';folder.mkdir()
    with (folder/'test.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow([str(i) for i in range(21)])
        for timestamp,ax in [('2025-06-11 12:00:00',0),('2025-06-11 12:00:00',2),('2025-06-11 12:00:01',5)]:
            w.writerow([timestamp,'WTC',ax,0,9.8,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
        # A known-device row with only18 fields is valid for P31 but not this source32 loader.
        w.writerow(['2025-06-11 12:00:02','WTLA',1,0,9.8,0,0,0,0,0,0,0,0,0,1,0,0,0])
    row={'imu_path':str(folder),'imu_available':'1'}
    feature,mask,meta=m.statistics_for_row(row,ops)
    loaded=ops['load_trial'](folder);values,tm,dm,duration=ops['resample_trial'](loaded,32)
    expected,em=ops['feature_vector'](values,tm,dm)
    np.testing.assert_array_equal(feature,expected);np.testing.assert_array_equal(mask,em)
    assert len(loaded['WTC'][0])==2 and 'WTLA' not in loaded and loaded['WTC'][1][0,0]==1
    assert meta['usable'] and meta['device_count']==1 and meta['duration_seconds']==duration
    empty,em,meta=m.statistics_for_row({'imu_path':'','imu_available':'0'},ops)
    assert not meta['usable'] and not empty.any() and not em.any()


@pytest.fixture
def rf_population(no_vote_fixture):
    from dataclasses import replace
    import hashlib,json
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_types import Partition,write_json
    from src.experiments.no_vote_manifest import PUBLIC_COLUMNS,csv_bytes,row_index
    p=v3_protocol(no_vote_fixture)
    sizes={'train12':8,'development2':2,'refit14':10,'final4':3}
    p=replace(p,partitions={k:Partition(k,p.partitions[k].users,n) for k,n in sizes.items()})
    rows=[];truth={}
    for user,n in [('fit1',8),('dev1',2),('target1',3)]:
        for i in range(n):
            sid=hashlib.sha256((user+str(i)).encode()).hexdigest()
            row={k:'' for k in PUBLIC_COLUMNS};row.update(sample_id=sid,user_id=user)
            row.update({k:'0' for k in PUBLIC_COLUMNS if k.endswith('_available')})
            if i!=n-1:
                folder=p.run_root/'raw_imu'/sid;folder.mkdir(parents=True)
                with (folder/'device.csv').open('w',newline='') as f:
                    writer=csv.writer(f);writer.writerow([str(x) for x in range(21)])
                    for t in range(5):writer.writerow([f'2025-06-11 12:00:0{t}','WTC',i+t,0,9.8,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
                row['imu_path']=str(folder);row['imu_available']='1'
            rows.append(row);truth[sid]=[0,5,39][i%3]
    registry=ArtifactRegistry(p);partrefs={};labelhashes={}
    for name,part in p.partitions.items():
        population=[r for r in rows if r['user_id'] in part.users];path=p.public_manifests[name]
        path.write_bytes(csv_bytes(PUBLIC_COLUMNS,population));index=row_index(population)
        ref=registry.register(stage='manifest_'+name,kind='public_manifest',phase='raw',files=[path],rows=index)
        partrefs[name]={'record_path':str(ref.record_path),'sha256':ref.sha256}
        if name!='final4':
            labels=p.supervised_labels[name];labels.write_bytes(csv_bytes(('sample_id','class_id'),[{'sample_id':r['sample_id'],'class_id':truth[r['sample_id']]} for r in population]));labelhashes[name]=hashlib.sha256(labels.read_bytes()).hexdigest()
    receipt=p.run_root/'protocol/prepared_inputs.json';write_json(receipt,{'partitions':partrefs,'supervised_label_hashes':labelhashes})
    write_json(p.run_root/'run_state.json',{'state':'prepared','protocol_sha256':p.identity(),'prepared_inputs_sha256':hashlib.sha256(receipt.read_bytes()).hexdigest()})
    # A synthetic Task6 public completion is sufficient; no motion/teacher payload is imported.
    write_json(p.run_root/'protocol/task6_complete.json',{'complete':True,'protocol_sha256':p.identity(),'training_executed':False,'task7_started':False})
    return p


def test_fixed_phase_rf_pipeline_and_consumer_targets(rf_population,monkeypatch):
    m=module();assert hasattr(m,'run_rf_pipeline'), 'Task7 fixed-phase pipeline is missing'
    import json
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_types import ArtifactRef
    p=rf_population;original=Path.open
    def guard(path,*a,**kw):
        assert path.name!='final_labels.csv'
        assert not (path.suffix.lower() in {'.npz','.npy','.csv','.joblib','.pt','.pth'} and 'oof' in path.name.lower())
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    result=m.run_rf_pipeline(p);assert set(result['outputs'])==set(p.partitions)
    registry=ArtifactRegistry(p)
    select=registry.verify(result['models']['select'],'source_expert/imu_rf','select')
    refit=registry.verify(result['models']['refit'],'source_expert/imu_rf','refit')
    assert select.fit_users==p.partitions['train12'].users and refit.fit_users==p.partitions['refit14'].users
    assert not select.select_users and not refit.select_users
    for phase,expected in [('select',7),('refit',8)]:
        record=registry.read(result['models'][phase]);assert record.config['valid_fit_rows']==expected
        assert record.config['augmented_fit_rows']==2*expected
    for partition,profiles in result['outputs'].items():
        assert set(profiles)=={'fit_teacher_target','terminal_export'}
        for profile,target in profiles.items():
            prediction=target.prediction;record=registry.verify(prediction.artifact,'source_expert/imu_rf','predict',prediction.index)
            assert record.config['consumer_profile']==profile and target.features is None
            assert len(prediction.index.sample_ids)==p.partitions[partition].expected_rows
            assert not prediction.valid[-1]
    # Completed invocation consumes existing bound outputs, never refits or writes new identities.
    second=m.run_rf_pipeline(p)
    assert result['models']==second['models']
    def no_labels(path,*a,**kw):
        assert not path.name.endswith('_labels.csv') and path.name!='final_labels.csv'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',no_labels)
    # Raw prediction has no label/teacher-probability payload dependency.
    target=m.predict_imu_teacher(result['models']['refit'],result['statistics']['final4'],'final4',
        protocol=p,context=VerificationContext(p),profile='terminal_export')
    assert target.prediction.artifact==result['outputs']['final4']['terminal_export'].prediction.artifact


def test_rf_scope_does_not_unlock_other_supervised_stages(rf_population):
    m=module();assert hasattr(m,'prepare_rf_capability'), 'scoped RF readiness is missing'
    from src.experiments.artifact_record import ArtifactRegistry
    from src.experiments.no_vote_types import write_json
    p=rf_population;result=m.run_rf_pipeline(p);registry=ArtifactRegistry(p)
    assert p.recipe.get('source_ops_bindings',{}).get('formal_generation_ready',False) is False
    payload=p.run_root/'fake_a5.json';write_json(payload,{'fake':True})
    cap=next(x for x in registry.read(result['models']['select']).parents if registry.read(x).stage=='imu_rf_readiness')
    with pytest.raises(ValueError,match='not ready'):
        registry.register(stage='A5',kind='supervised_model',phase='select',files=[payload],parents=[cap],fit_users=p.partitions['train12'].users)


def test_select_training_never_opens_development_or_final_labels(rf_population,monkeypatch):
    from src.experiments.no_vote_manifest import load_stage_inputs
    from src.experiments.artifact_record import ArtifactRegistry
    p=rf_population;m=module();context=VerificationContext(p)
    stats={pop:m.build_rf_statistics(load_stage_inputs(p,pop,'raw',context=context),protocol=p,context=context) for pop in ['refit14','final4']}
    cap=m.prepare_rf_capability(p,stats,context);original=Path.open
    def guard(path,*a,**kw):
        assert path.resolve()!=p.supervised_labels['development2'].resolve()
        assert path.resolve()!=p.supervised_labels['refit14'].resolve()
        assert path.name!='final_labels.csv'
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'open',guard)
    fit=load_stage_inputs(p,'train12','select',context=context)
    ref=m.train_imu_rf('select',fit,stats['refit14'],cap,protocol=p,context=context)
    assert ArtifactRegistry(p).verify(ref).fit_users==p.partitions['train12'].users
    with pytest.raises(ValueError,match='population'):
        m.train_imu_rf('refit',fit,stats['refit14'],cap,protocol=p,context=context)
    with pytest.raises(ValueError,match='population'):
        m.predict_imu_teacher(ref,stats['final4'],'final4',protocol=p,context=context)


def test_rf_consumer_sentinel_and_rng_matches_source_fit(no_vote_fixture):
    from sklearn.ensemble import RandomForestClassifier
    m=module();p=v3_protocol(no_vote_fixture);ops=m.source_rf_ops(p,VerificationContext(p))
    rng=np.random.default_rng(19);features=rng.normal(size=(12,240)).astype(np.float32);masks=np.tile([1.,1.],(12,5)).astype(np.float32)
    labels=np.repeat([0,5,39],4);data,y,chosen=m.dropout_training_data(features,masks,labels,ops)
    rf=m.make_rf(p).fit(data,y)
    dropped,dm=ops['drop_devices'](features,masks,ops['random_present_devices'](masks,np.random.default_rng(20260723)))
    source=RandomForestClassifier(n_estimators=400,max_depth=18,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=20260723)
    source.fit(np.concatenate([np.c_[features,masks],np.c_[dropped,dm]]),np.r_[labels,labels])
    query=np.c_[features,masks];np.testing.assert_allclose(rf.predict_proba(query),source.predict_proba(query),rtol=0,atol=1e-14)
    for profile in ['fit_teacher_target','terminal_export']:
        scores=m.aligned_rf_logits(rf.predict_proba(query),rf.classes_,profile)
        np.testing.assert_allclose(scores[:,source.classes_],np.log(np.clip(source.predict_proba(query),1e-12,1.)),rtol=0,atol=1e-14)
