"""Two real train12 batches against source Dataset, losses, gradients and update."""
import argparse
from pathlib import Path
import sys
import json
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_manifest import load_stage_inputs
from src.experiments.no_vote_types import RowIndex,write_json
from src.experiments.motion_cache import load_motion,MOTION_FIELDS
from src.experiments.visual_teacher import _labels
from src.experiments.mobind_pretrain import (source_motion_math,motion_args,initialize_motion,source_motion_losses,
    NoVoteMotionDataset,prepare_motion_capability,motion_references,load_motion_teachers,motion_validity,motion_training_mask,TASK8_REVISION)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',default='cuda');args0=parser.parse_args();p=load_protocol(args0.config,inference_only=True)
    c=VerificationContext(p);started=time.perf_counter();prepare_motion_capability(p,c)
    ops=source_motion_math(p,c);args=motion_args(p);refs=motion_references(p)
    record,arrays=load_motion(refs['refit14']['windows'],protocol=p,context=c)
    fit=load_stage_inputs(p,'train12','select',context=c);index,y=_labels(fit,'select',p)
    visual,rf=load_motion_teachers(p,'train12',c);lookup={s:i for i,s in enumerate(record.rows.sample_ids)}
    positions=np.array([lookup[s] for s in index.sample_ids]);sk,imu=motion_validity(arrays)
    valid=motion_training_mask(sk[positions],imu[positions],visual.prediction.valid)
    chosen=np.flatnonzero(valid)[:128]
    partial=np.flatnonzero(valid & ~(sk&imu)[positions])
    for i in range(min(2,len(partial))):
        if partial[i] not in chosen:chosen[63+64*i]=partial[i]
    visual_only=np.flatnonzero(valid & ~(sk|imu)[positions])
    if len(visual_only) and visual_only[0] not in chosen:chosen[62]=visual_only[0]
    rows=RowIndex(tuple(index.sample_ids[i] for i in chosen),tuple(index.user_ids[i] for i in chosen),index.class_ids)
    labels={index.sample_ids[i]:int(y[i]) for i in chosen}
    ours=NoVoteMotionDataset(arrays,record.rows,rows=rows,labels=labels,visual_teacher=visual,rf_teacher=rf,temporal_indices=ops['temporal_indices'])
    # Instantiate the original reader without its historical label-bearing CSV loader.
    source=ops['dataset_class'].__new__(ops['dataset_class']);source.indices=np.arange(len(chosen));source.temporal_augment=True
    source.rows=[{'sample_id':s,'user_id':u,'class_id':labels[s]} for s,u in zip(rows.sample_ids,rows.user_ids)]
    source.motion={k:arrays[k][positions[chosen]] for k in MOTION_FIELDS}
    source.teacher_logits=np.asarray(visual.prediction.logits[chosen],np.float32)
    source.teacher_features=np.asarray(visual.features[chosen],np.float32).mean(2)
    source.imu_teacher_logits=np.asarray(rf.prediction.logits[chosen],np.float32).copy()
    source.imu_teacher_valid=rf.prediction.valid[chosen].copy();source.imu_teacher_logits[~source.imu_teacher_valid]=0
    source.imu_event_features=None
    device=torch.device(args0.device);a=initialize_motion(p,device=args0.device,context=c);b=initialize_motion(p,device=args0.device,context=c)
    weights=ops['class_weights'](y[valid],args.class_weight_power,device)
    optimizers=[torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay) for model in [a,b]]
    scalers=[torch.amp.GradScaler('cuda',enabled=device.type=='cuda') for _ in range(2)]
    proofs=[]
    for number in range(2):
        ids=range(number*64,(number+1)*64);np.random.seed(1720+number)
        actual=torch.utils.data.default_collate([ours[i] for i in ids]);np.random.seed(1720+number)
        expected=torch.utils.data.default_collate([source[i] for i in ids])
        for key in [*MOTION_FIELDS,'teacher_logits','teacher_features','imu_teacher_logits','imu_teacher_valid','label']:
            torch.testing.assert_close(actual[key],expected[key],atol=0,rtol=0)
        batches=[ops['to_device'](x,device) for x in [actual,expected]];losses=[];gradients=[]
        for j,(model,optimizer,scaler,batch) in enumerate(zip([a,b],optimizers,scalers,batches)):
            model.train();optimizer.zero_grad(set_to_none=True);torch.manual_seed(1990+number)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                out=model(ops['motion_fields'](batch),mask_ratio=args.mask_ratio)
                values=source_motion_losses(out,batch,weights,args,ops) if j==0 else ops['losses'](out,batch,weights,args)
            assert torch.isfinite(values['loss'])
            scaler.scale(values['loss']).backward();scaler.unscale_(optimizer)
            gradients.append({k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None})
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);scaler.step(optimizer);scaler.update()
            losses.append({k:float(v.detach()) for k,v in values.items()})
        for key in losses[0]:assert losses[0][key]==losses[1][key],key
        gradient_difference=0.;nonfinite=0
        for key,value in gradients[0].items():
            other=gradients[1][key];torch.testing.assert_close(value,other,atol=0,rtol=0)
            finite=torch.isfinite(value);nonfinite+=int((~finite).sum())
            if finite.any():gradient_difference=max(gradient_difference,float((value[finite]-other[finite]).abs().max()))
        parameter_difference=0.
        for va,vb in zip(a.parameters(),b.parameters()):
            torch.testing.assert_close(va,vb,atol=0,rtol=0);parameter_difference=max(parameter_difference,float((va-vb).abs().max()))
        proofs.append({'batch':number+1,'rows':64,'loss':losses[0]['loss'],'all_loss_terms_equal':True,
            'maximum_finite_gradient_difference':gradient_difference,'nonfinite_gradient_elements':nonfinite,
            'maximum_parameter_difference_after_source_update':parameter_difference,'scaler':scalers[0].get_scale(),
            'partial_rows':int((~(actual['skeleton_joint_mask'].flatten(1).any(1)&actual['imu_global_mask'][:,:,0].any(1))).sum())})
    proof={'protocol_sha256':p.identity(),'task8_revision':TASK8_REVISION,'fit_population_rows':int(valid.sum()),
        'restored_visual_only_rows':int(len(visual_only)),'restored_visual_only_rows_in_real_batches':int(np.isin(chosen,visual_only).sum()),
        'train12_rows':128,'source_dataset_teacher_augmentation_equal':True,'batches':proofs,
        'device':str(device),'peak_cuda_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else 0.,
        'final_labels_read':False,'seconds':round(time.perf_counter()-started,2)}
    write_json(p.run_root/'protocol'/('task8_'+TASK8_REVISION+'_real_batch_acceptance.json'),proof);print(json.dumps(proof),flush=True)


if __name__=='__main__':main()
