"""Public full-window VideoMAE features. No class labels or nearest-neighbor fit."""
import hashlib
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from .no_vote_types import ArtifactRef,canonical_bytes,canonical_hash,write_json
from .no_vote_cache_import import immutable_bytes
from .pose_roi_adapter import compatible_frame_map
from .visual_teacher_v3 import read_ref
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file


def prepare_fullwindow_clips(row,roi,crop,read_ir):
    ids=roi['frame_ids'].astype(str);acquired=roi['acquisition_ids'].astype(str)
    if not len(ids):return [],np.zeros(16,np.int64)
    actual=compatible_frame_map(Path(row['ir_path']),'ir')
    if tuple(ids)!=tuple(sorted(actual)) or acquired.shape!=ids.shape:raise ValueError('fullwindow frame/acquisition axis mismatch')
    chosen=np.rint(np.linspace(0,len(ids)-1,16)).astype(np.int64)
    names=tuple(roi['region_names'].astype(str));person=names.index('full_body');workspace=names.index('hand_workspace')
    videos=[[],[],[]]
    for i in chosen:
        image=read_ir(actual[ids[i]])
        pb=roi['roi_boxes_xyxy'][i,person] if roi['roi_valid'][i,person] else np.full(4,np.nan)
        wb=roi['roi_boxes_xyxy'][i,workspace] if roi['roi_valid'][i,workspace] else pb
        for video,view in zip(videos,(image,crop(image,pb,1.15),crop(image,wb,1.40))):video.append(view)
    return videos,chosen


def _write_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True);buffer=io.BytesIO();np.savez_compressed(buffer,**arrays);content=buffer.getvalue()
    if path.exists():raise ValueError('immutable fullwindow output already exists')
    temporary=path.with_suffix(path.suffix+'.building')
    with temporary.open('wb') as f:f.write(content)
    temporary.replace(path)
    return hashlib.sha256(content).hexdigest()


def extract_fullwindow(protocol,partition,*,device='cuda',max_trials=0,context=None):
    if max_trials<0:raise ValueError('fullwindow trial limit must be nonnegative')
    started=time.perf_counter();context=context or VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    inputs=load_stage_inputs(protocol,partition,'raw',context=context);public=read_public_rows(inputs.public_manifest)
    selected=public[:max_trials] if max_trials else public;index=row_index(selected)
    receipt=json.loads((protocol.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'))
    candidates=[ArtifactRef(Path(x['record_path']),x['sha256']) for x in receipt['imports'] if x['stage']=='p29']
    matches=[ref for ref in candidates if set(index.sample_ids)<=set(registry.read(ref).rows.sample_ids)]
    if len(matches)!=1:raise ValueError('fullwindow requires one exact-ID shared ROI source')
    roi=matches[0];rr=registry.verify(roi,'p29','raw')
    owners=dict(zip(rr.rows.sample_ids,rr.rows.user_ids))
    if any(owners.get(s)!=u for s,u in zip(index.sample_ids,index.user_ids)):raise ValueError('fullwindow ROI users mismatch')
    # Import bridge kept core public initializer as its own ancestor, never a head.
    feature_entry=next(x for x in receipt['imports'] if x['stage']=='visual_features' and x['partition']=='refit14')
    fr=registry.read(ArtifactRef(Path(feature_entry['record_path']),feature_entry['sha256']))
    weights=next(parent for parent in fr.parents if registry.read(parent).stage=='visual_initializer')
    wr=registry.verify(weights,'visual_initializer','public')
    for file in wr.files:
        target=protocol.weights['videomae']/Path(file).name
        if not os.path.samefile(file,target):raise ValueError('fullwindow public initializer differs from pinned weights')
    root=protocol.run_root/'fullwindow_features'/partition/('acceptance' if max_trials else 'full')
    cfg={'partition':partition,'windows':'full_0_1','frames':16,'views':['scene','person','workspace'],
        'crops':[1.15,1.40],'backbone_frozen':True,'producer_sha256':sha256_file(Path(__file__))}
    identity=canonical_hash({'protocol':protocol.identity(),'roi':roi,'weights':weights,'rows':index,'config':cfg})
    immutable_bytes(root/'identity.json',canonical_bytes({'identity':identity})+b'\n')
    if (root/'artifact.json').exists():
        ref=read_ref(root/'artifact.json');registry.verify(ref,'fullwindow_features','raw',index);return ref
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
    ops_key=(protocol.identity(),'fullwindow_ops')
    if ops_key not in context.public_rows:
        context.public_rows[ops_key]=tuple(load_teammate_symbol(report,*x) for x in [
            ('build_p46_videomae_cache','square_crop'),('build_p30_shared_dir_roi_feature_cache','read_ir'),('build_p46_videomae_cache','encode')])
    crop,read_ir,encode=context.public_rows[ops_key]
    roi_files={Path(p).stem:Path(p) for p in rr.files if Path(p).suffix=='.npz'}
    all_features=[];all_logits=[];all_valid=[];digests={}
    model_key=(protocol.identity(),'fullwindow_model',device)
    for number,row in enumerate(selected,1):
        path=root/'trial_cache'/(row['sample_id']+'.npz');meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            body=json.loads(meta.read_text(encoding='utf-8'))
            if body['identity']!=identity:raise ValueError('fullwindow resume identity drift')
            context.hash_file(path,body['sha256'])
            arrays=context.load_arrays(path);digest=body['sha256']
        else:
            roi_path=registry.verify_file(rr,roi_files[row['sample_id']]);a=context.load_arrays(roi_path)
            if (str(a['sample_id'].item())!=row['sample_id'] or not bool(a['completed'])
                    or bool(a['available'])!=(row['ir_available']=='1')):raise ValueError('fullwindow ROI identity mismatch')
            videos,chosen=prepare_fullwindow_clips(row,a,crop,read_ir)
            f=np.zeros((3,1024),np.float16);k=np.zeros((3,400),np.float16)
            if videos:
                if model_key not in context.public_rows:
                    import torch
                    from transformers import VideoMAEForVideoClassification,VideoMAEImageProcessor
                    from .no_vote_weights import validate_attention_biases
                    local=protocol.weights['videomae'];model=VideoMAEForVideoClassification.from_pretrained(local,local_files_only=True)
                    validation=validate_attention_biases(model,local/'model.safetensors')
                    if model.config.hidden_size!=1024 or model.config.num_labels!=400:raise ValueError('fullwindow public architecture mismatch')
                    model.eval().to(torch.device(device))
                    for parameter in model.parameters():parameter.requires_grad_(False)
                    processor=VideoMAEImageProcessor.from_pretrained(local,local_files_only=True)
                    context.public_rows[model_key]=(model,processor)
                    write_json(root/'initializer_validation.json',validation)
                import torch
                model,processor=context.public_rows[model_key]
                encoded,logits,peak=encode(model,processor,videos,torch.device(device));f=encoded.astype(np.float16);k=logits.astype(np.float16)
            arrays={'sample_id':np.asarray(row['sample_id']),'features':f,'kinetics_logits':k,'valid':np.asarray(bool(videos)),
                'frame_indices':chosen,'acquisition_ids':a['acquisition_ids'][chosen] if len(a['frame_ids']) else np.full(16,'',dtype='U1')}
            digest=_write_npz(path,arrays);write_json(meta,{'identity':identity,'sha256':digest})
        if arrays['features'].shape!=(3,1024) or arrays['kinetics_logits'].shape!=(3,400):raise ValueError('fullwindow view/feature axes mismatch')
        if not np.isfinite(arrays['features']).all() or not np.isfinite(arrays['kinetics_logits']).all():raise ValueError('nonfinite fullwindow features')
        if str(arrays['sample_id'].item())!=row['sample_id']:raise ValueError('fullwindow cached ID drift')
        all_features.append(arrays['features']);all_logits.append(arrays['kinetics_logits']);all_valid.append(bool(arrays['valid']))
        if number%25==0 or number==len(selected):
            print(json.dumps({'event':'fullwindow_progress','partition':partition,'processed':number,'total':len(selected),
                'seconds':round(time.perf_counter()-started,2)}),flush=True)
    path=root/'features.npz';arrays={'sample_ids':np.asarray(index.sample_ids),'class_ids':np.arange(40),
        'features':np.stack(all_features),'kinetics_logits':np.stack(all_logits),'valid':np.asarray(all_valid,bool)}
    digests[str(path)]=_write_npz(path,arrays)
    ref=registry.register(stage='fullwindow_features',kind='raw_cache',phase='raw',files=[path],file_digests=digests,
        parents=[roi,weights,inputs.parents['manifest']],rows=index,config=cfg,source_files=[Path(__file__)],complete=len(selected)==len(public))
    write_json(root/'artifact.json',ref)
    return ref
