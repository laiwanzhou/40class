"""Task5 real train12 batch loss/gradient/native-sequence source acceptance."""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from src.experiments.no_vote_datasets import NoVotePixelDataset
from src.experiments.no_vote_types import RowIndex,write_json
from src.experiments.pixel_cache import load_pixels
from src.experiments.visual_student import sequence_and_anchor,to_device,train_epoch
from src.experiments import visual_student_v3 as source


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();started=time.perf_counter();p=load_protocol(args.config);c=VerificationContext(p)
    source.prepare_student_capability(p,c);ops=source.source_student_math(p,c);recipe=p.recipe['visual_student']
    teacher=source.load_teacher(p,'train12',c)
    record,arrays=load_pixels(source.pixel_refs(p)['refit14'],protocol=p,context=c)
    index=teacher.prediction.index;positions=np.flatnonzero(teacher.prediction.valid)[:recipe['batch_size']]
    rows=RowIndex(tuple(index.sample_ids[i] for i in positions),tuple(index.user_ids[i] for i in positions),index.class_ids)
    from src.experiments.visual_teacher import _labels
    fit=load_stage_inputs(p,'train12','select',context=c);_,y=_labels(fit,'select',p);lookup=dict(zip(index.sample_ids,y.tolist()))
    dataset=NoVotePixelDataset(arrays,record.rows,rows=rows,labels={s:lookup[s] for s in rows.sample_ids},
        teacher=teacher,augmentation=ops['dataset_class']._subject_robust_augment)
    network,_=source.initialize_student(p,device=args.device,context=c)
    # Both implementations receive the identical augmented real batch and RNG.
    loader=DataLoader(dataset,batch_size=recipe['batch_size'],shuffle=False,num_workers=0)
    batch=to_device(next(iter(loader)),torch.device(args.device));network.train()
    weights=ops['class_weights'](y[teacher.prediction.valid],recipe['class_weight_power'],torch.device(args.device))
    initial_rng=torch.get_rng_state();initial_cuda=torch.cuda.get_rng_state_all() if args.device!='cpu' else []
    parameters=[x for x in network.parameters() if x.requires_grad]
    def run_loss(reference):
        torch.set_rng_state(initial_rng)
        if initial_cuda:torch.cuda.set_rng_state_all(initial_cuda)
        network.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type,dtype=torch.float16,enabled=args.device!='cpu'):
            output=network(batch['images'],batch['view_valid'],batch['view_quality'],batch['global_time_position'])
            if reference:
                ce=F.cross_entropy(output['logits'],batch['label'],weight=weights,label_smoothing=recipe['label_smoothing'])
                temperature=recipe['distillation_temperature']
                kd=F.kl_div(F.log_softmax(output['logits']/temperature,1),F.softmax(batch['teacher_logits']/temperature,1),reduction='batchmean')*temperature**2
                relation=ops['relation_loss'](output['clip_embeddings'],batch['teacher_features'],output['clip_mask'])
                loss=ce+recipe['distillation_weight']*kd+recipe['relation_weight']*relation
            else:loss=source.source_hybrid_loss(output,batch,weights,recipe,ops['relation_loss'])['loss']
        loss.backward()
        return float(loss.detach()),[x.grad.detach().clone() if x.grad is not None else None for x in parameters]
    actual,gradients=run_loss(False);expected,reference=run_loss(True)
    if actual!=expected:raise ValueError('real batch source loss differs')
    maximum=0.
    for a,b in zip(gradients,reference):
        if (a is None)!=(b is None):raise ValueError('gradient participation differs')
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if not torch.isfinite(a).all():raise ValueError('real gradient nonfinite')
            maximum=max(maximum,float((a-b).abs().max()))
    network.zero_grad(set_to_none=True)
    with torch.autocast(torch.device(args.device).type,dtype=torch.float16,enabled=args.device!='cpu'):
        sequence,anchor=sequence_and_anchor(network,batch)
        with torch.inference_mode():
            direct=network(batch['images'],batch['view_valid'],batch['view_quality'],batch['global_time_position'])['logits'].float().cpu().numpy()
    np.testing.assert_allclose(anchor,direct,rtol=1e-3,atol=1e-3)
    optimizer=ops['optimizer_for'](network,source.SimpleNamespace(**recipe))
    report=train_epoch(network,[batch],optimizer,weights,recipe,device=args.device,
        loss_function=partial(source.source_hybrid_loss,relation_loss=ops['relation_loss']))
    if report['optimizer_steps']!=1 or report['samples']!=recipe['batch_size']:raise ValueError('real accumulation flush mismatch')
    body={'protocol_sha256':p.identity(),'partition':'train12','batch_rows':len(rows.sample_ids),'source_loss':actual,
        'source_reference_loss':expected,'gradient_max_difference':maximum,'native_sequence_shape':list(sequence.shape),
        'anchor_max_difference':float(np.max(np.abs(anchor-direct))),'optimizer_steps':report['optimizer_steps'],
        'fixed_epochs':16,'teacher_head_temperature_separate_from_kd':True,'formal_student_trained':False,
        'final_labels_read':False,'seconds':round(time.perf_counter()-started,2)}
    write_json(p.run_root/'protocol/task5_real_batch_acceptance.json',body)
    print(json.dumps(body),flush=True)


if __name__=='__main__':main()
