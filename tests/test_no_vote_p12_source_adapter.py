"""The P12 producer keeps explicit columns, fit ownership and missing-input masks."""
import importlib
from types import SimpleNamespace
import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('src.experiments.p12_source_adapter'),'formal P12 source adapter is missing'
    return importlib.import_module('src.experiments.p12_source_adapter')


def test_p12_eight_column_order_and_all_missing_prior(no_vote_fixture):
    m=module()
    from src.experiments.p12_fusion import p12_fusion_source_ops
    from src.experiments.artifact_record import VerificationContext
    from test_no_vote_source_ops import v3_protocol
    p=v3_protocol(no_vote_fixture);ops=p12_fusion_source_ops(p,VerificationContext(p))
    prior=np.arange(1,41,dtype=np.float64);prior/=prior.sum()
    rng=np.random.default_rng(22);scores={k:rng.normal(size=(4,40)) for k in ['skeleton','depth','imu','thermal']}
    masks={k:np.array([True,True,False,False]) for k in scores}
    masks['thermal']=np.array([True,False,True,False]);counts=np.array([5,2,0,0])
    state={'imu':{'sd_temperature':1.,'imu_temperature':2.,'selected_imu_weight':.3},
        'thermal':{'base_temperature':1.,'thermal_temperature':1.,'selected_thermal_weight':.4},
        'router':{'kind':'constant','value':1.}}
    out=m.compose_p12(scores,masks,counts,prior,state,ops)
    assert out['expert_names']==('skeleton','depth','sd','imu','sd_imu','thermal','thermal_candidate','final')
    assert out['expert_logits'].shape==(4,8,40) and out['expert_valid'].shape==(4,8)
    np.testing.assert_allclose(out['expert_logits'][0,2],.6*scores['skeleton'][0]+.4*scores['depth'][0],atol=0,rtol=0)
    np.testing.assert_array_equal(out['route'],[True,False,True,False])
    np.testing.assert_array_equal(out['probabilities'][3],prior)
    assert out['valid'].tolist()==[True,True,True,False]
    assert out['expert_valid'][2].tolist()==[False,False,False,False,False,True,True,True]
    assert not out['expert_valid'][3].any()


def test_unknown_phase_rejected_before_recipe_or_label_access(no_vote_fixture):
    m=module()
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import VerificationContext
    p=load_protocol(no_vote_fixture[0]);context=VerificationContext(p)
    with pytest.raises(ValueError,match='phase'):m.build_p12_source_chain(p,'development2',context)
    with pytest.raises(ValueError,match='phase'):m.fit_global_mixture(p,'final4',context)


def test_unbound_global_and_p12_records_do_not_unseal_global_gate(no_vote_fixture):
    m=module()
    from src.experiments.no_vote_protocol import load_protocol
    from src.experiments.artifact_record import VerificationContext,ArtifactRegistry
    p=load_protocol(no_vote_fixture[0]);r=ArtifactRegistry(p,context=VerificationContext(p))
    for stage in ['p12_fusion','global_mixture']:
        record=SimpleNamespace(stage=stage,kind='statistics',phase='select',parents=(),config={'revision':'task11_reconstruction_v2'})
        assert m.allows_p12_global_stage(r,record) is False


def test_registered_p12_global_pipeline_and_wrong_parent_refusal(no_vote_fixture):
    m=module()
    from dataclasses import replace
    from pathlib import Path
    import csv
    from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
    from src.experiments.no_vote_types import write_json,canonical_hash
    from src.experiments.p12_reconstruction import load_reconstruction,REVISION
    from src.experiments.imu_rf_teacher import _write_npz
    from src.experiments.motion_cache import _stamp
    from test_no_vote_source_ops import v3_protocol
    from test_artifact_record import prediction_rows
    old=v3_protocol(no_vote_fixture);p=replace(old,recipe={**old.recipe,'source_ops_bindings':{'formal_generation_ready':True}})
    c=VerificationContext(p);r=ArtifactRegistry(p,context=c);index=prediction_rows(r,'train12')
    folder=p.run_root/'protocol'/REVISION;recipe=folder/'fixture_recipe.json'
    write_json(recipe,{'revision':REVISION,'base_protocol_sha256':p.identity(),'historical_exact_reproduction':False})
    write_json(folder/'preparation.json',{'recipe':{'path':str(recipe),'sha256':c.hash_file(recipe)}})
    load_reconstruction(p,c)
    with p.supervised_labels['train12'].open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['sample_id','class_id']);writer.writerows(zip(index.sample_ids,range(40)))
    manifest=r.register(stage='manifest_train12',kind='public_manifest',phase='raw',files=[p.public_manifests['train12']],rows=index)
    receipt=p.run_root/'protocol/prepared_inputs.json';write_json(receipt,{'partitions':{'train12':manifest},
        'supervised_label_hashes':{'train12':c.hash_file(p.supervised_labels['train12'])}})
    write_json(p.run_root/'run_state.json',{'protocol_sha256':p.identity(),'state':'generating','prepared_inputs_sha256':c.hash_file(receipt)})
    prior=np.full(40,1/40,np.float32);prfile=p.run_root/'A1/select/prior.json';write_json(prfile,{'probabilities':prior.tolist()})
    pr=r.register(stage='class_prior',kind='statistics',phase='select',files=[prfile],rows=index,fit_users=['fit1'])
    write_json(prfile.with_name('prior_artifact.json'),pr)
    scores={};masks={};parents=[];visual=[]
    for role in ['skeleton','depth','thermal','imu',*m.FAMILIES]:
        stage=('visual_head/'+role if role in m.FAMILIES else 'source_expert/'+('imu_rf' if role=='imu' else 'p12_'+role))
        model_file=p.run_root/('fixture_'+role+'.bin');model_file.write_bytes(b'fixture model only')
        model=r.register(stage=stage,kind='supervised_model',phase='select',files=[model_file],rows=index,fit_users=['fit1'])
        values=np.eye(40,dtype=np.float32)*3;valid=np.ones(40,bool);path=p.run_root/('fixture_'+role+'.npz')
        from scipy.special import softmax
        _write_npz(path,dict(sample_ids=np.asarray(index.sample_ids),class_ids=np.arange(40),logits=values,probabilities=softmax(values,axis=1),valid=valid))
        prediction=r.register(stage=stage,kind='predictions',phase='predict',files=[path],parents=[model],rows=index,config={'partition':'train12'})
        if role in m.FAMILIES:visual.append((values,valid,prediction))
        else:scores[role]=values;masks[role]=valid;parents.append(prediction)
        if role in m.FAMILIES:
            write_json(p.run_root/'visual_heads'/role/'select/train12_targets.artifact.json',prediction)
    statfile=p.run_root/'statistics.npz';_write_npz(statfile,dict(sample_ids=np.asarray(index.sample_ids),features=np.zeros((40,240),np.float32),
        masks=np.ones((40,10),np.float32),valid=np.ones(40,bool),device_count=np.full(40,5),duration_seconds=np.ones(40)))
    statistic=r.register(stage='imu_statistics32',kind='raw_cache',phase='raw',files=[statfile],rows=index,
        config={'bins':32,'not_p31_motion_statistics':True,'payload_stamps':{'statistics.npz':_stamp(statfile)}})
    bundle=(index,scores,masks,np.full(40,5),parents,statistic)
    model=m.build_p12_source_chain(p,'select',c,components=bundle)
    record=r.verify(model,'p12_fusion','select');assert m.allows_p12_global_stage(r,record)
    output=m.predict_p12(p,model,'train12',c,components=bundle)
    _,a=m._read_prediction(p,output,'p12_fusion','train12',c)
    table=np.concatenate([a['expert_logits'].astype(np.float64),*[x[0][:,None,:] for x in visual]],axis=1)
    global_model=m.fit_global_mixture(p,'select',c,components=(table,np.ones(40,bool),a['probabilities'],[output,*[x[2] for x in visual]]))
    global_record=r.verify(global_model,'global_mixture','select');assert m.allows_p12_global_stage(r,global_record)
    assert np.isclose(sum(global_record.config['weights']),1.) and global_record.config['fit_rows']==40
    assert not m.allows_p12_global_stage(r,replace(global_record,parents=record.parents))
    assert not m.allows_p12_global_stage(r,replace(record,rows=replace(index,user_ids=('dev1',)*40)))
    bad={**scores,'skeleton':scores['skeleton']+1}
    with pytest.raises(ValueError,match='parent payload'):
        m.predict_p12(p,model,'train12',c,components=(index,bad,masks,np.full(40,5),parents,statistic))
    actual=m.predict_global(p,global_model,'train12',c)
    _,disk=m._read_prediction(p,actual.artifact,'global_mixture','train12',c)
    np.testing.assert_array_equal(actual.logits,disk['logits'])
    np.testing.assert_array_equal(actual.probabilities,disk['probabilities'])
    from src.experiments.no_vote_manifest import PUBLIC_COLUMNS,read_public_rows,row_index,csv_bytes
    public=read_public_rows(p.public_manifests['train12'])
    for i in range(40,80):public.append({**public[0],'sample_id':f'{i:064x}','user_id':'dev1'})
    p.public_manifests['refit14'].write_bytes(csv_bytes(PUBLIC_COLUMNS,public))
    full_index=row_index(public);fullpath=p.run_root/'fullwindow_features/refit14/full/features.npz'
    rng=np.random.default_rng(125)
    _write_npz(fullpath,dict(sample_ids=np.asarray(full_index.sample_ids),class_ids=np.arange(40),
        features=rng.normal(size=(80,3,1024)).astype(np.float16),valid=np.ones(80,bool)))
    feature=r.register(stage='fullwindow_features',kind='raw_cache',phase='raw',files=[fullpath],rows=full_index,
        config={'partition':'refit14','windows':'full_0_1','frames':16,'views':['scene','person','workspace'],'backbone_frozen':True})
    write_json(fullpath.with_name('artifact.json'),feature)
    knn=m.apply_fullwindow_knn(p,actual,'train12',c)
    record,arrays=m._read_prediction(p,knn.artifact,'global_mixture','train12',c)
    assert record.config['operation']=='source_fullwindow_knn' and record.config['pool']=='train12'
    assert arrays['neighbor_indices'].max()<40 and not np.any(arrays['neighbor_indices']==np.arange(40)[:,None])
    assert m.apply_fullwindow_knn(p,actual,'train12',c).artifact==knn.artifact
