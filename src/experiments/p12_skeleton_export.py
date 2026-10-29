"""Versioned FP16 storage export; retain original trained FP32 models and targets."""
from copy import copy
import json
from pathlib import Path

import torch

from .artifact_record import ArtifactRegistry,VerificationContext
from .no_vote_types import canonical_bytes,write_json
from .no_vote_cache_import import immutable_bytes
from .p12_reconstruction import REVISION,ref,load_reconstruction,predict_skeleton


def fp16_state(state):
    result=copy(state)
    for key,value in state.items():result[key]=value.detach().half() if value.is_floating_point() else value.detach().clone()
    return result


def export_skeleton_component(protocol,*,device='cuda'):
    from .visual_student_v3 import _save_torch
    context=VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context);_,rr,_=load_reconstruction(protocol,context)
    component_path=protocol.run_root/'protocol'/REVISION/'skeleton_component.json'
    master=json.loads(component_path.read_text(encoding='utf-8'));models={}
    for phase in ['select','refit']:
        original=ref(master['models'][phase]);record=registry.verify(original,'source_expert/p12_skeleton',phase)
        root=protocol.run_root/REVISION/'p12_skeleton_fp16'/phase;alias=root/'artifact.json'
        if alias.exists():models[phase]=ref(json.loads(alias.read_text(encoding='utf-8')));registry.verify(models[phase]);continue
        path=next(Path(x) for x in record.files if Path(x).name=='checkpoint.pt');registry.verify_file(record,path)
        state=torch.load(path,map_location='cpu',weights_only=True);state['model_state_dict']=fp16_state(state['model_state_dict'])
        state['storage_dtype']='float16';state['execution_dtype']='float32_with_source_amp'
        root.mkdir(parents=True,exist_ok=True);_save_torch(root/'checkpoint.pt',state)
        cfg={**record.config,'storage_export':'float16_roundtrip','new_training_executed':False,
            'master_model':original,'master_checkpoint_preserved':True}
        write_json(root/'model.json',cfg)
        models[phase]=registry.register(stage='source_expert/p12_skeleton',kind='supervised_model',phase=phase,
            files=[root/'checkpoint.pt',root/'model.json'],parents=[rr,original],rows=record.rows,fit_users=record.fit_users,
            config=cfg,source_files=[Path(__file__)])
        write_json(alias,models[phase])
    outputs={}
    for partition in protocol.partitions:
        phase='select' if partition in {'train12','development2'} else 'refit';cache=ref(master['caches']['final4' if partition=='final4' else 'refit14'])
        prediction=predict_skeleton(protocol,models[phase],cache,partition,context,device=device,output_name='p12_skeleton_fp16')
        outputs[partition]=prediction.artifact
        print(json.dumps({'event':'p12_skeleton_fp16_export','partition':partition,'rows':len(prediction.valid),'valid':int(prediction.valid.sum())}),flush=True)
    body={'revision':REVISION,'storage_export':'float16_roundtrip','base_protocol_sha256':protocol.identity(),
        'skeleton_component_complete':True,'master_component':{'path':str(component_path),'sha256':context.hash_file(component_path)},
        'models':models,'outputs':outputs,'caches':master['caches'],'recipe':rr,'new_training_executed':False,
        'final_labels_read':False,'task11_complete':False,'a8_generated':False}
    path=protocol.run_root/'protocol'/REVISION/'skeleton_fp16_component.json';immutable_bytes(path,canonical_bytes(body)+b'\n')
    write_json(protocol.run_root/'protocol'/REVISION/'p12_skeleton_current.json',{'component_path':str(path),'sha256':context.hash_file(path),
        'revision':REVISION,'storage_export':'float16_roundtrip','master_component_preserved':True})
    return body
