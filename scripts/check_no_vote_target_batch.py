"""A train12-only real-batch source comparison; never a registered final model."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import time
import tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_types import RowIndex,write_json
from src.experiments.mobind_fusion import fusion_inputs,load_fusion_data,dataset_for,source_fusion_math
from src.experiments.source_base_adaptation import source_adaptation_ops,initialize_source_base
from src.experiments.target_adaptation import RejectLabelsDataset,target_args,train_fixed_budget
from src.experiments.teammate_source import verify_teammate_source,load_teammate_symbol
from src.experiments.p12_reconstruction import ref
from src.experiments.mobind_pretrain import motion_validity

p=load_protocol(ROOT/'configs/experiments/teammate_single_teacher_fixed_split_v3.yaml',inference_only=True)
c=VerificationContext(p);torch.set_num_threads(2);started=time.perf_counter()
ops=source_adaptation_ops(p,c);refs=fusion_inputs(p,c);data=load_fusion_data(p,'train12',c,refs)
positions=np.flatnonzero(data['observed_valid'])[:4].tolist()
sk,imu=motion_validity(data['motion']);mi={s:i for i,s in enumerate(data['motion_index'].sample_ids)}
ir_only=[i for i,s in enumerate(data['index'].sample_ids) if data['observed_valid'][i] and not sk[mi[s]] and not imu[mi[s]]]
if ir_only:positions[-1]=ir_only[0]
chosen=RowIndex(tuple(data['index'].sample_ids[i] for i in positions),tuple(data['index'].user_ids[i] for i in positions),data['index'].class_ids)
base=ref(json.loads((p.run_root/'A7/select/artifact.json').read_text(encoding='utf-8')))
dataset=RejectLabelsDataset(dataset_for(data,rows=chosen,temporal_indices=source_fusion_math(p,c)['temporal_indices']))
report=verify_teammate_source(p.source_root,p.source_root.parent/'source_manifest.json',expected_sha256=p.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
source_class=load_teammate_symbol(report,'p87s_test_data','P87STestCachedSequenceMotionDataset')
source=object.__new__(source_class);source.indices=np.arange(len(positions));source.rows=[{'sample_id':s} for s in chosen.sample_ids];source.temporal_augment=True
source.backbone_sequence=np.asarray(data['sequence'][positions])
pix={s:i for i,s in enumerate(data['pixel_index'].sample_ids)};pp=[pix[s] for s in chosen.sample_ids]
source.view_valid=data['pixels']['view_valid'][pp];source.view_quality=data['pixels']['view_quality'][pp];source.source_frame_indices=data['pixels']['source_frame_indices'][pp]
source.motion={k:np.asarray(v[[mi[s] for s in chosen.sample_ids]]) for k,v in data['motion'].items()}
maximum=0.;quality_residual=0.
for i in range(len(positions)):
    np.random.seed(200+i);expected=source[i]
    np.random.seed(200+i);actual=dataset[i]
    for k,v in expected.items():
        if torch.is_tensor(v):
            if k=='view_quality':
                quality_residual=max(quality_residual,float((v-actual[k]).abs().max()))
                # Existing Task5/10 precision exception, explicitly retained.
                # Compare every other value and the consumed precision exactly.
                torch.testing.assert_close(v.half().float(),actual[k],atol=0,rtol=0)
            else:
                torch.testing.assert_close(v,actual[k],atol=0,rtol=0)
    assert expected['sample_id']==actual['sample_id'] and 'label' not in actual
target_file=p.run_root/'A1/select/train12_targets.npz'
with np.load(target_file,allow_pickle=False) as z:prob=z['probabilities'][positions]
pseudo=ops['LabelFreePseudoDataset'](dataset,dict(zip(chosen.sample_ids,prob)),{s:1. for s in chosen.sample_ids})
args=target_args(p,40);args.epochs=1  # Interface probe only, never registered as a formal endpoint.
initial=initialize_source_base(p,'select',base,refs,c,seed=args.seed,device='cpu');original=deepcopy(initial);actual=deepcopy(initial)
ops['seed_all'](args.seed);expected=ops['train_label_free'](original,ops['make_loader'](pseudo,64,0,True),args,torch.device('cpu'))
with tempfile.TemporaryDirectory(prefix='P47-target-probe-',dir=ROOT/'outputs/task5_audit') as temp:
    ops['seed_all'](args.seed)
    result=train_fixed_budget(actual,ops['make_loader'](pseudo,64,0,True),args,ops,Path(temp)/'resume.pt','P47-interface-only',expected_rows=4,device='cpu')
for k,v in original.state_dict().items():torch.testing.assert_close(v,actual.state_dict()[k],atol=0,rtol=0)
for k,v in expected[0].items():
    if k!='seconds':assert result[0][k]==v
assert any(not torch.equal(v,initial.state_dict()[k]) for k,v in actual.state_dict().items())
assert all(torch.isfinite(v).all() for v in actual.state_dict().values() if v.is_floating_point())
assert any(x.requires_grad for x in actual.motion_residual.encoder.parameters())
assert not any(x.requires_grad for key,x in actual.visual.named_parameters() if key.startswith(('stem.','layer1.','layer2.','layer3.','layer4.')))
summary=dict(rows=4,ir_only_included=bool(ir_only),source_final_dataset_nonquality_tensor_max_difference=maximum,
    existing_roi_quality_fp16_max_difference=quality_residual,quality_consumed_fp16_difference=0.,
    source_optimizer_update_max_difference=0.,ground_truth_fields_seen=0,formal_artifact_registered=False,
    probe_target_origin='own_train12_A1_interface_only_not_final_A8_substitute',final_labels_read=False,seconds=time.perf_counter()-started)
write_json(ROOT/'outputs/task5_audit/P47-target-real-batch.json',summary);print(json.dumps(summary),flush=True)
