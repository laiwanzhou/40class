"""CPU real-input interface proof, not a formal P85/KNN pseudo model."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_types import RowIndex,write_json
from src.experiments.p12_reconstruction import ref,load_reconstruction
from src.experiments.mobind_fusion import fusion_inputs,load_fusion_data,dataset_for,source_fusion_math
from src.experiments.mobind_pretrain import motion_validity
from src.experiments.source_base_adaptation import source_adaptation_ops,initialize_source_base,adaptation_args,train_source_with_resume

p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);registry=ArtifactRegistry(p,context=c);torch.set_num_threads(2);started=time.perf_counter()
recipe,_,_=load_reconstruction(p,c);ops=source_adaptation_ops(p,c);refs=fusion_inputs(p,c);data=load_fusion_data(p,'train12',c,refs)
base=ref(json.loads((p.run_root/'A7/select/artifact.json').read_text(encoding='utf-8')))
target=p.run_root/'A1/select/train12_targets.npz'
# Existing own-fit A1 distributions are interface probes only. This does not
# register/train a formal source-base model or substitute for the required P85 chain.
with np.load(target,allow_pickle=False) as z:
    assert tuple(z['sample_ids'].tolist())==data['index'].sample_ids;prob=z['probabilities'];valid=z['valid']
positions=np.flatnonzero(valid)[:4].tolist();sk,imu=motion_validity(data['motion']);lookup={s:i for i,s in enumerate(data['motion_index'].sample_ids)}
mi=np.array([lookup[s] for s in data['index'].sample_ids]);ir_only=np.flatnonzero(valid&~sk[mi]&~imu[mi])
if len(ir_only) and int(ir_only[0]) not in positions:positions[-1]=int(ir_only[0])
chosen=RowIndex(tuple(data['index'].sample_ids[i] for i in positions),tuple(data['index'].user_ids[i] for i in positions),data['index'].class_ids)
dataset=dataset_for(data,rows=chosen,temporal_indices=source_fusion_math(p,c)['temporal_indices'])
pseudo=ops['LabelFreePseudoDataset'](dataset,dict(zip(chosen.sample_ids,prob[positions])),{s:1. for s in chosen.sample_ids})
assert all('label' not in pseudo[i] for i in range(len(pseudo)))
args=adaptation_args(recipe['intermediate_source_base']['recipe']);args.epochs=1
network=initialize_source_base(p,'select',base,refs,c,seed=args.seed,device='cpu');original=deepcopy(network);resumed=deepcopy(network)
ops['seed_all'](args.seed);expected=ops['train_label_free'](original,ops['make_loader'](pseudo,64,0,shuffle=True),args,torch.device('cpu'))
ops['seed_all'](args.seed);temporary=ROOT/'outputs/task5_audit/P43-real-source-probe-resume.pt'
if temporary.exists():raise ValueError('previous owned probe resume exists; retain it and choose a new proof path before rerunning')
actual=train_source_with_resume(resumed,ops['make_loader'](pseudo,64,0,shuffle=True),args,ops,temporary,'P43_probe_not_formal',device='cpu')
for k,value in original.state_dict().items():torch.testing.assert_close(value,resumed.state_dict()[k],atol=0,rtol=0)
for key,value in expected[0].items():
    if key!='seconds':assert value==actual[0][key]
assert expected[0]['ground_truth_fields_seen']==0 and actual[0]['train_samples']==len(positions)
summary={'rows':len(positions),'ir_only_included':bool(len(ir_only)),'device':'cpu','epochs':1,'formal_training':False,
    'probe_target_origin':'existing_own_phase_A1_for_interface_only_not_P85_replacement','source_loss_update_state_max_difference':0.,
    'ground_truth_batch_fields_seen':0,'final_labels_read':False,'formal_artifact_registered':False,'seconds':time.perf_counter()-started}
write_json(ROOT/'outputs/task5_audit/P43-real-source-batch.json',summary)
temporary.unlink()  # Only this script's newly created, verified probe checkpoint.
print(json.dumps(summary),flush=True)
