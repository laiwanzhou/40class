"""Task10 bounded acceptance; no raw/image/feature/sequence payload rescans."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import RowIndex,canonical_bytes,canonical_hash,write_json
from src.experiments.no_vote_manifest import read_public_rows,row_index
from src.experiments.mobind_fusion import (VARIANTS,ref,fusion_inputs,load_fusion_data,initialize_fusion,source_fusion_math,
    fusion_args,dataset_for,fit_shuffle,MOTION_FIELDS)
from src.experiments.simple_fusion import load_prediction,read_labels


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);c=VerificationContext(p);r=ArtifactRegistry(p,context=c)
    root=p.run_root;start=time.perf_counter();ready=json.loads((root/'protocol/task10_artifacts_ready.json').read_text(encoding='utf-8'))
    assert ready['complete'] and ready['protocol_sha256']==p.identity() and ready['task8_revision']=='ir_supervision_v2'
    refs=fusion_inputs(p,c);assert dict(ready['prerequisites'])==refs['proofs']
    summary={};models={};outputs={};total=0;training_seconds=0
    for phase,pop in [('select','train12'),('refit','refit14')]:
        data=load_fusion_data(p,pop,c,refs);index=data['index'];valid=data['observed_valid']
        chosen=RowIndex(tuple(s for s,v in zip(index.sample_ids,valid) if v),tuple(u for u,v in zip(index.user_ids,valid) if v),index.class_ids)
        expected_count=1957 if phase=='select' else 2342;assert len(chosen.sample_ids)==expected_count
        baseline=None;models[phase]={};summary[phase]={}
        for variant,stage in VARIANTS.items():
            reference=ref(ready['models'][phase][variant]);record=r.read(reference);cfg=record.config;models[phase][variant]=record
            assert record.stage==stage and record.phase==phase and record.fit_users==p.partitions[pop].users and not record.select_users
            assert cfg['fit_partition']==pop and cfg['fit_rows']==expected_count and cfg['stage_epochs']=={'A':4,'B':20}
            assert cfg['fit_ids_sha256']==cfg['class_weight_population_ids_sha256']==canonical_hash(chosen)
            assert cfg['variant']==variant and cfg['fresh_same_phase_initializers']
            assert canonical_bytes(cfg['source_recipe'])==canonical_bytes(p.recipe['fusion'])
            assert cfg['anchor_consumer_dtype']==cfg['quality_consumer_dtype']=='float16_then_float32'
            assert cfg['padding_compatibility']=='safe_mask_first_column_or_empty'
            assert record.source_hashes[str(ROOT/'src/experiments/mobind_fusion.py')]==c.hash_file(ROOT/'src/experiments/mobind_fusion.py')
            assert all(refs['phases'][phase][k] in record.parents for k in ['visual','motion'])
            if variant=='full':baseline=cfg
            else:
                assert cfg['initial_head_sha256']==baseline['initial_head_sha256']
                assert cfg['epoch_order_digests']==baseline['epoch_order_digests']
            assert len(cfg['epoch_order_digests']['A'])==4 and len(cfg['epoch_order_digests']['B'])==20
            assert cfg['shuffle_fit_only']==variant.startswith('shuffle')
            if cfg['shuffle_fit_only']:assert cfg['shuffle_donors_sha256']==canonical_hash(fit_shuffle(chosen,seed=p.recipe['fusion']['seed']).tolist())
            hist=next(Path(x) for x in record.files if Path(x).name=='history.json');r.verify_file(record,hist)
            history=json.loads(hist.read_text(encoding='utf-8'))
            assert [(x['stage'],x['epoch']) for x in history]==[('A',i) for i in range(1,5)]+[('B',i) for i in range(1,21)]
            assert all(x['train_samples']==expected_count and np.isfinite(x['train_loss']) for x in history)
            assert all(not any(k.startswith(('val_','development_')) for k in x) for x in history)
            for block in [history[:4],history[4:]]:
                assert abs(block[0]['encoder_learning_rate']-1e-4)<1e-12 and abs(block[0]['fusion_learning_rate']-4e-4)<1e-12
                assert abs(block[-1]['encoder_learning_rate']-1e-5)<1e-12 and abs(block[-1]['fusion_learning_rate']-1e-5)<1e-12
            assert all(x['visual_learning_rate']==0 for x in history[:4])
            assert abs(history[4]['visual_learning_rate']-2e-5)<1e-12 and abs(history[-1]['visual_learning_rate']-1e-5)<1e-12
            seconds=sum(x['seconds'] for x in history);training_seconds+=seconds
            summary[phase][variant]={'fit_rows':expected_count,'epochs':24,'paired_initialization_and_orders':True,'train_seconds':round(seconds,2)}
    kinds=[*VARIANTS.values(),'A7-zero-S','A7-zero-I'];partitions={}
    for pop in p.partitions:
        phase='select' if pop in {'train12','development2'} else 'refit';data=load_fusion_data(p,pop,c,refs);index=data['index']
        partitions[pop]={};outputs[pop]={};y=None if pop=='final4' else read_labels(p.supervised_labels[pop],index)
        for stage in kinds:
            pred=load_prediction(p,ref(ready['outputs'][pop][stage]),stage,pop,c);record=r.read(pred.artifact)
            variant=next((v for v,s in VARIANTS.items() if s==stage),'full');model=models[phase][variant];prior=np.asarray(model.config['prior'])
            assert ref(ready['models'][phase][variant]) in record.parents
            assert record.config['fit_phase']==phase and record.config['task8_revision']=='ir_supervision_v2'
            np.testing.assert_array_equal(pred.valid,data['observed_valid'])
            if (~pred.valid).any():np.testing.assert_allclose(pred.probabilities[~pred.valid],np.broadcast_to(prior,((~pred.valid).sum(),40)),rtol=0,atol=1e-7)
            result={'rows':len(pred.valid),'valid':int(pred.valid.sum()),'prior_rows':int((~pred.valid).sum()),'correct':None}
            if y is not None:result['correct']=int((pred.logits.argmax(1)==y).sum())
            partitions[pop][stage]=result;outputs[pop][stage]=pred
            total+=sum(Path(x).stat().st_size for x in record.files)
        if pop=='final4':assert all(x['rows']==609 and x['valid']==591 and x['prior_rows']==18 for x in partitions[pop].values())
    # Actual current refit model: construct architecture without loading A2/A5
    # initializer parameters; protocol ancestry digest checks still require their
    # files. This is not a claim that physically moving ancestry files works.
    original_open=Path.open;original_load=torch.load;attempts=[];initializer_loads=[]
    def guard(path,*args,**kwargs):
        if path.name.endswith('_labels.csv') or path.name=='final_labels.csv' or (any(x in {'A1','A4'} for x in path.parts) and path.suffix in {'.npz','.npy','.pt','.joblib'}):
            attempts.append(str(path));raise AssertionError('teacher targets or labels entered fusion inference')
        return original_open(path,*args,**kwargs)
    def load_guard(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and any(x in {'A2','A5','weights'} for x in Path(path).parts):
            initializer_loads.append(str(path));raise AssertionError('initializer parameter load entered fusion inference')
        return original_load(path,*args,**kwargs)
    Path.open=guard;torch.load=load_guard
    try:
        model=models['refit']['full'];path=next(Path(x) for x in model.files if Path(x).name=='checkpoint.pt');r.verify_file(model,path)
        network=initialize_fusion(p,'refit',refs,c,device='cpu',load_parents=False)
        state=torch.load(path,map_location='cpu',weights_only=True);network.load_state_dict(state['model_state']);network.eval().cuda()
        assert all(torch.isfinite(x).all() for x in network.state_dict().values())
        ops=source_fusion_math(p,c);data=load_fusion_data(p,'final4',c,refs);ds=dataset_for(data)
        batch=ops['collate']([ds[i] for i in range(64)]);assert 'label' not in batch and 'teacher_logits' not in batch
        active=np.flatnonzero(data['observed_valid'][:64]);batch=ops['to_device'](batch,torch.device('cuda'));take=torch.as_tensor(active,device='cuda')
        batch={k:v.index_select(0,take) if torch.is_tensor(v) else [v[i] for i in active] for k,v in batch.items()}
        differences={}
        for stage,zero in [('A7',None),('A7-zero-S','skeleton_'),('A7-zero-I','imu_')]:
            current={k:torch.zeros_like(v) if zero and k in MOTION_FIELDS and k.startswith(zero) else v for k,v in batch.items()}
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):out=ops['model_forward'](network,current)
            logits=out['logits'].float().cpu().numpy();expected=outputs['final4'][stage].logits[active]
            np.testing.assert_allclose(logits,expected,rtol=0,atol=.01)
            differences[stage]=float(np.max(np.abs(logits-expected),initial=0))
    finally:Path.open=original_open;torch.load=original_load
    assert not attempts and not initializer_loads
    before=json.loads((ROOT/'outputs/task5_audit/P38-task10-before-serialization-history.json').read_text(encoding='utf-8'))
    after=json.loads((root/'A7/select/history.json').read_text(encoding='utf-8'))
    assert [{k:v for k,v in x.items() if k!='seconds'} for x in before]==[{k:v for k,v in x.items() if k!='seconds'} for x in after]
    result={'protocol_sha256':p.identity(),'complete':True,'models':summary,'partitions':partitions,'outputs_checked':24,
        'inference_real_final_public_rows':64,'inference_logit_max_differences':differences,
        'inference_teacher_label_open_attempts':len(attempts),'inference_initializer_parameter_load_attempts':len(initializer_loads),
        'serialization_recovery_history_equal_except_time':True,'training_seconds':round(training_seconds,2),'target_bytes':total,
        'final_labels_read':False,'raw_image_feature_sequence_full_hashes':0,'independent_review':'not_performed',
        'seconds':round(time.perf_counter()-start,2)}
    write_json(ROOT/'outputs/task5_audit/P38-task10-final-acceptance.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
