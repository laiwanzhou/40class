"""Fixed-split teacher contracts, using synthetic populations and no GPU."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.mark.parametrize('n',[1,388,609])
def test_arbitrary_label_free_population(n):
    from src.experiments.visual_teacher import validate_feature_arrays
    from src.experiments.no_vote_types import RowIndex
    ids=tuple(f'{i:064x}' for i in range(n));index=RowIndex(ids,('target',)*n,tuple(range(40)))
    arrays=dict(sample_ids=np.asarray(ids),class_ids=np.arange(40),
        features=np.zeros((n,2,3,1024),np.float16),kinetics_logits=np.zeros((n,2,3,400),np.float16),
        valid=np.ones(n,bool))
    validate_feature_arrays(arrays,index)
    arrays['sample_ids']=arrays['sample_ids'][::-1]
    if n>1:
        with pytest.raises(ValueError,match='IDs'):validate_feature_arrays(arrays,index)
    arrays['labels']=np.zeros(n)
    with pytest.raises(ValueError,match='schema'):validate_feature_arrays(arrays,index)


def test_grid_and_tie_break_are_frozen():
    from src.experiments.visual_teacher import candidate_grid,choose_candidate
    recipe={'families':['early','late','window_mean','early_late','temporal_delta','kinetics'],
        'class_weight_powers':[0.,.5,.75],'alphas':[300,1000,3000,10000]}
    assert len(candidate_grid(recipe))==72
    base=dict(accuracy=.5,macro_f1=.4,worst_user_accuracy=.3,dimensions=3072)
    rows=[dict(base,candidate_id='z',alpha=300),dict(base,candidate_id='a',alpha=1000),
          dict(base,candidate_id='b',alpha=1000)]
    assert choose_candidate(rows)['candidate_id']=='a'


def test_feature_math_keeps_source_window_and_delta_definition():
    from src.experiments.visual_teacher import feature_sets,window_indices
    x=np.zeros((1,2,3,1024),np.float32);x[0,0,:,0]=2;x[0,1,:,1]=3
    result=feature_sets(x,np.ones((1,2,3,400)))
    assert result['early'].shape==(1,3072) and result['early_late'].shape==(1,6144)
    expected=np.tile([-1,1],(3,1))
    assert np.array_equal(result['temporal_delta'].reshape(1,6,1024)[0,3:,:2],expected)
    assert np.all(result['kinetics']==0)
    assert np.array_equal(window_indices(11,0,.7),np.rint(np.linspace(0,7,16)).astype(int))


@pytest.fixture
def teacher_population(no_vote_fixture,tmp_path,request):
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.no_vote_manifest import prepare_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    config,payload=no_vote_fixture
    params=getattr(request,'param',40)
    nfit=params['nfit'] if isinstance(params,dict) else params
    raw_ir=tmp_path/'raw_ir';raw_ir.mkdir()
    if isinstance(params,dict) and params.get('ir_trial'):(raw_ir/'IR_00000001.png').write_bytes(b'original')
    payload['partitions']['train12']['expected_rows']=nfit
    payload['partitions']['refit14']['expected_rows']=nfit+40
    payload['recipe']['visual_teacher']={'families':['early'], 'class_weight_powers':[0.],
        'alphas':[300,1000]}
    config.write_text(yaml.safe_dump(payload),encoding='utf-8');p=load_protocol(config)
    source=tmp_path/'canonical.csv'
    with source.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['sample_id','class_id','user_id','ir_path']);w.writeheader()
        for user in ('fit1','dev1','target1'):
            for c in range(nfit if user=='fit1' else 40):
                ir=str(raw_ir) if user=='fit1' and c==0 and isinstance(params,dict) and params.get('ir_trial') else ''
                w.writerow(dict(sample_id=f'{user}-{c}',class_id=c%40,user_id=user,ir_path=ir))
    prepare_inputs(source,p,p.run_root/'protocol',tmp_path/'private')
    registry=ArtifactRegistry(p);refs={}
    for part in ('refit14','final4'):
        rows=read_public_rows(p.public_manifests[part]);idx=row_index(rows);n=len(rows)
        # Independent deterministic synthetic features; no production label read.
        features=np.zeros((n,2,3,1024),np.float16)
        labels={r['sample_id']:int(r['class_id']) for r in csv.DictReader(p.supervised_labels['refit14'].open())}
        for i,sid in enumerate(idx.sample_ids):features[i,:,:,labels.get(sid,i%40)]=1
        valid=np.ones(n,bool)
        if part=='final4':valid[-1]=False
        path=p.run_root/(part+'_features.npz')
        np.savez(path,sample_ids=np.asarray(idx.sample_ids),class_ids=np.arange(40),features=features,
            kinetics_logits=np.zeros((n,2,3,400),np.float16),valid=valid)
        refs[part]=registry.register(stage='visual_features',kind='raw_cache',phase='raw',files=[path],rows=idx,
            config={'partition':part,'feature_signature':'same-public-backbone'})
    return p,refs,tmp_path/'private',source


def test_one_selected_head_refit_separation_and_missing_prior(teacher_population,monkeypatch):
    from src.experiments.visual_teacher import select_visual_head,fit_visual_head,predict_visual_teacher,fit_class_prior
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    p,features,private,canonical=teacher_population
    original_open=Path.open
    def guarded_open(path,*args,**kwargs):
        if Path(path).is_relative_to(private) or Path(path)==canonical:
            raise AssertionError('generation accessed private/canonical labels')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded_open)
    tr=load_stage_inputs(p,'train12','select');dev=load_stage_inputs(p,'development2','select')
    selection=select_visual_head(tr,dev,features['refit14'],protocol=p)
    assert selection.budget['candidates']==2
    assert selection.metric['accuracy']==1
    registry=ArtifactRegistry(p)
    assert registry.verify(selection.fit_artifact).fit_users==('fit1',)
    refit=load_stage_inputs(p,'refit14','refit')
    model=fit_visual_head(refit,features['refit14'],selection,protocol=p)
    assert registry.verify(model).fit_users==('dev1','fit1')
    assert selection.fit_artifact not in registry.verify(model).parents
    prior=fit_class_prior(refit,'refit',protocol=p)
    receipt=json.loads((p.run_root/'A1/refit/prior_artifact.json').read_text())
    assert receipt=={'record_path':str(prior.record_path),'sha256':prior.sha256}
    final=load_stage_inputs(p,'final4','predict');rows=row_index(read_public_rows(final.public_manifest))
    target=predict_visual_teacher(model,features['final4'],rows,prior,protocol=p)
    assert target.features.shape==(40,2,3,1024)
    assert np.allclose(target.prediction.probabilities[-1],np.full(40,1/40))
    assert not target.prediction.valid[-1]
    with np.load(next(Path(f) for f in registry.verify(target.prediction.artifact).files if f.endswith('.npz'))) as z:
        assert set(z.files)=={'sample_ids','class_ids','logits','probabilities','valid'}
    with pytest.raises(ValueError,match='phase|population|prior'):
        predict_visual_teacher(selection.fit_artifact,features['final4'],rows,prior,protocol=p)
    assert len(list((p.run_root/'A1/select').glob('*.joblib')))==1


def test_partial_feature_ancestor_is_rejected_before_fitting(teacher_population):
    from src.experiments.visual_teacher import select_visual_head
    from src.experiments.no_vote_manifest import load_stage_inputs
    from src.experiments.artifact_record import ArtifactRegistry
    p,refs,_,_=teacher_population;r=ArtifactRegistry(p);record=r.verify(refs['refit14'])
    partial=r.register(stage=record.stage,kind=record.kind,phase=record.phase,files=record.files,
        rows=record.rows,config=record.config,complete=False)
    with pytest.raises(ValueError,match='complete|partial'):
        select_visual_head(load_stage_inputs(p,'train12','select'),load_stage_inputs(p,'development2','select'),partial,protocol=p)


@pytest.mark.parametrize('teacher_population',[80],indirect=True)
def test_invalid_fit_extreme_features_do_not_change_scaler(teacher_population):
    from src.experiments.visual_teacher import select_visual_head,feature_sets
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    import joblib
    p,refs,_,_=teacher_population;registry=ArtifactRegistry(p);old=registry.verify(refs['refit14'])
    tr=load_stage_inputs(p,'train12','select');dev=load_stage_inputs(p,'development2','select')
    idx=row_index(read_public_rows(tr.public_manifest));path=next(Path(f) for f in old.files)
    with np.load(path) as z:a={k:z[k] for k in z.files}
    bad=list(a['sample_ids']).index(idx.sample_ids[0]);a['features'][bad]=60000;a['valid'][bad]=False
    np.savez(path,**a)
    ref=registry.register(stage=old.stage,kind=old.kind,phase=old.phase,files=[path],rows=old.rows,config=old.config)
    selection=select_visual_head(tr,dev,ref,protocol=p)
    model=joblib.load(p.run_root/'A1/select/head.joblib')
    lookup={s:i for i,s in enumerate(a['sample_ids'])};positions=np.array([lookup[s] for s in idx.sample_ids])
    values=feature_sets(a['features'],a['kinetics_logits'])['early'][positions]
    expected=values[a['valid'][positions]].mean(axis=0,dtype=np.float64)
    assert model.named_steps['scale'].n_samples_seen_==79
    assert np.allclose(model.named_steps['scale'].mean_,expected)


@pytest.mark.parametrize('teacher_population',[{'nfit':40,'ir_trial':True}],indirect=True)
def test_extraction_detects_new_raw_frames_before_any_cache_fast_path(teacher_population):
    from src.experiments.visual_teacher import extract_visual_features
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    p,_,_,_=teacher_population;registry=ArtifactRegistry(p);inputs=load_stage_inputs(p,'refit14','raw')
    rows=read_public_rows(inputs.public_manifest);raw=next(Path(r['ir_path']) for r in rows if r['ir_path'])
    roi_file=p.run_root/'dummy_roi.npz';np.savez(roi_file,completed=True)
    roi=registry.register(stage='p29',kind='raw_cache',phase='raw',files=[roi_file],rows=row_index(rows),
        raw_inputs=[raw/'IR_00000001.png'],config={'schema':1})
    weights=registry.register(stage='visual_initializer',kind='public_weights',phase='public',
        files=list(p.weights['videomae'].iterdir()),config={})
    (raw/'IR_00000002.png').write_bytes(b'new')
    with pytest.raises(ValueError,match='raw input membership'):
        extract_visual_features(inputs,roi,weights,p.run_root/'feature_resume',protocol=p)


@pytest.mark.parametrize('teacher_population',[{'nfit':40,'ir_trial':True}],indirect=True)
def test_head_cannot_bypass_changed_roi_input_membership(teacher_population):
    from src.experiments.visual_teacher import select_visual_head
    from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
    from src.experiments.artifact_record import ArtifactRegistry
    p,refs,_,_=teacher_population;r=ArtifactRegistry(p);raw_rows=read_public_rows(p.public_manifests['refit14'])
    folder=next(Path(row['ir_path']) for row in raw_rows if row['ir_path'])
    cache=p.run_root/'dummy_p29.npz';np.savez(cache,completed=True)
    roi=r.register(stage='p29',kind='raw_cache',phase='raw',files=[cache],rows=row_index(raw_rows),
        raw_inputs=[folder/'IR_00000001.png'],config={})
    old=r.verify(refs['refit14']);feature=r.register(stage=old.stage,kind=old.kind,phase=old.phase,
        files=old.files,parents=[roi],rows=old.rows,config=old.config)
    (folder/'IR_00000002.png').write_bytes(b'new frame')
    with pytest.raises(ValueError,match='raw input membership'):
        select_visual_head(load_stage_inputs(p,'train12','select'),load_stage_inputs(p,'development2','select'),feature,protocol=p)
