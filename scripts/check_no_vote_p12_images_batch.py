"""Bounded real training batch and source-input equivalence, no held labels."""
from pathlib import Path
import json
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_types import write_json
from src.experiments.p12_visual_reconstruction import (load_reconstruction,visual_source_ops,build_image_index,
    _image_dataset,initialize_image_model,image_initializer)
from src.experiments.no_vote_manifest import load_stage_inputs
from src.experiments.visual_teacher import _labels

p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);recipe,_,_=load_reconstruction(p,c);ops=visual_source_ops(p,c);torch.set_num_threads(4)
summary={};started=time.perf_counter()
for role in ['depth','thermal']:
    cache=build_image_index(p,'refit14',role,c);cfg=dict(recipe['source_models'][role]['configuration']);cfg.update(num_workers=0,val_num_workers=0)
    index,y=_labels(load_stage_inputs(p,'train12','select',context=c),'select',p)
    ds,_=_image_dataset(p,cache,'train12',role,cfg,ops,c,augment=True,labels=dict(zip(index.sample_ids,y.tolist())))
    # One real item compared with original source Dataset under identical RNG.
    torch.manual_seed(57);item=ds[0];torch.manual_seed(57)
    expected=(ops['aligned_dataset_class']._getitem_from_cache(ds.reader,ds.samples[0]) if role=='depth'
        else ops['thermal_dataset_class'].__getitem__(ds.reader,0))
    key='depth' if role=='depth' else 'clip';torch.testing.assert_close(item[key],expected[key],atol=0,rtol=0)
    init=None;initializer_exact=None
    if role=='thermal':
        _,init=image_initializer(p,c);actual=initialize_image_model(role,cfg,ops,device='cpu',initializer=init)
        ops['seed'](cfg['seed']);expected_model=ops['thermal_model_class'](num_classes=40,dropout=cfg['dropout'],imagenet_pretrained=True)
        for k,v in actual.state_dict().items():torch.testing.assert_close(v,expected_model.state_dict()[k],atol=0,rtol=0)
        del actual,expected_model;initializer_exact=True
    model=initialize_image_model(role,cfg,ops,device='cuda',initializer=init)
    source=ops['thermal_train'] if role=='thermal' else sys.modules['train']
    loader=source.create_loader(ds,cfg,True);batch=next(iter(loader));before=next(model.parameters()).detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    observed={}
    def observe_step(opt,args,kwargs):
        grads=[v.grad for v in model.parameters() if v.grad is not None]
        observed.update(count=len(grads),finite=bool(grads) and all(torch.isfinite(v).all().item() for v in grads))
    optimizer.register_step_pre_hook(observe_step)
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    args=[model,[batch],torch.nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing']),torch.device('cuda')]
    if role=='depth':args.append(['depth'])
    scaler=torch.amp.GradScaler('cuda');skipped=0
    for attempt in range(12):
        metrics=source.run_epoch(*args,optimizer,scaler,True,cfg['max_grad_norm'])
        if not np.isfinite(metrics['loss']):raise ValueError('non-finite batch loss')
        if observed.get('finite'):break
        skipped+=1
    else:raise AssertionError('source GradScaler did not recover to a finite update')
    assert not torch.equal(before,next(model.parameters()).detach())
    summary[role]={'batch_rows':len(metrics['sample_ids']),'fit_rows':len(ds),'source_item_max_difference':0,
        'source_original_epoch_function':True,'loss':metrics['loss'],'finite_gradient_tensors':observed['count'],
        'first_parameter_updated':True,'initializer_source_model_exact':initializer_exact,'source_initial_amp_skipped_steps':skipped,
        'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'step_seconds':time.perf_counter()-tick}
    print(json.dumps({'event':'p12_image_batch_check','role':role,**summary[role]}),flush=True)
    del model,optimizer,loader,batch,ds;torch.cuda.empty_cache()
summary.update(final_labels_read=False,development_labels_read=False,seconds=time.perf_counter()-started)
write_json(ROOT/'outputs/task5_audit/P41-p12-images-real-batch.json',summary)
