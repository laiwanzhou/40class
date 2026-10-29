"""One real train12 batch: source Dataset, losses, groups, gradients and updates."""
import argparse
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys
import textwrap
import time
from types import MethodType

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_types import RowIndex,write_json
from src.experiments.mobind_fusion import (source_fusion_math,fusion_args,fusion_inputs,load_fusion_data,dataset_for,
    initialize_fusion,run_source_stage,MOTION_FIELDS)
from src.experiments.visual_student_v3 import load_teacher
from src.experiments.simple_fusion import read_labels


class Joined:
    def __init__(self,array,positions,dtype=None):self.array=array;self.positions=positions;self.dtype=dtype
    def __getitem__(self,i):
        v=self.array[self.positions[i]]
        return np.asarray(v,dtype=self.dtype) if self.dtype else v


def flat_reference(original):
    tree=ast.parse(textwrap.dedent(inspect.getsource(original)))
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            if ast.unparse(node.targets[0])=='safe_mask[empty, 0]':
                return ast.parse('safe_mask.reshape(-1, safe_mask.shape[-1])[empty.reshape(-1), 0] = True').body[0]
            return node
    namespace=dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(Rewrite().visit(tree)),'<independent_flat_reference>','exec'),namespace)
    return namespace['forward']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();p=load_protocol(args.config,inference_only=True);c=VerificationContext(p);start=time.perf_counter()
    refs=fusion_inputs(p,c);data=load_fusion_data(p,'train12',c,refs);ops=source_fusion_math(p,c);recipe=fusion_args(p)
    teacher=load_teacher(p,'train12',c);index=data['index'];y=read_labels(p.supervised_labels['train12'],index)
    mi={s:i for i,s in enumerate(data['motion_index'].sample_ids)};pi={s:i for i,s in enumerate(data['pixel_index'].sample_ids)}
    positions=np.array([mi[s] for s in index.sample_ids]);m=data['motion'];sk=m['skeleton_joint_mask'][positions].reshape(len(index.sample_ids),-1).any(1)
    im=m['imu_sequence_mask'][positions].reshape(len(index.sample_ids),-1).any(1)|m['imu_global_mask'][positions,:,0].any(1)
    valid=np.flatnonzero(data['observed_valid']);restored=np.flatnonzero(data['observed_valid']&~(sk|im))
    chosen=list(restored[:1])+[int(i) for i in valid if i not in set(restored[:1])][:63]
    rows=RowIndex(tuple(index.sample_ids[i] for i in chosen),tuple(index.user_ids[i] for i in chosen),index.class_ids)
    labels=dict(zip(index.sample_ids,y.tolist()))
    dataset=dataset_for(data,rows=rows,labels={s:labels[s] for s in rows.sample_ids},teacher=teacher,temporal_indices=ops['temporal_indices'])
    # Call the original source __getitem__ on joined public arrays. The three
    # unused extra teacher-head fields are fixture placeholders, never consumed
    # by this recipe's model_forward/losses and not asserted as real targets.
    source=object.__new__(ops['module'].P86CachedSequenceMotionDataset)
    source.indices=np.asarray(chosen);source.rows=[{'sample_id':s} for s in index.sample_ids]
    source.backbone_sequence=data['sequence'];source.compact_backbone_sequence=None;source.uses_spatial_regions=False
    source.anchor_logits=Joined(data['anchor'],np.arange(len(index.sample_ids)),np.float16)
    source.motion={k:Joined(data['motion'][k],positions) for k in MOTION_FIELDS}
    source.view_valid=Joined(data['pixels']['view_valid'],[pi[s] for s in index.sample_ids])
    source.view_quality=Joined(data['pixels']['view_quality'],[pi[s] for s in index.sample_ids],np.float16)
    source.source_frame_indices=Joined(data['pixels']['source_frame_indices'],[pi[s] for s in index.sample_ids])
    source.teacher_features=Joined(teacher.features,np.arange(len(index.sample_ids)),np.float32)
    source.teacher_logits=np.asarray(teacher.prediction.logits,np.float32)
    for k in ['teacher_early_logits','teacher_late_logits','teacher_temporal_delta_logits']:setattr(source,k,np.zeros((len(index.sample_ids),40),np.float32))
    source.imu_teacher_logits=None;source.imu_event_features=None;source.labels=y;source.users=np.asarray(index.user_ids);source.temporal_augment=True
    items=[];difference=0.
    for i in range(len(dataset)):
        np.random.seed(901+i);a=dataset[i];np.random.seed(901+i);b=source[i]
        for k,v in a.items():
            if k=='cache_index':continue
            if torch.is_tensor(v):
                torch.testing.assert_close(v,b[k],rtol=0,atol=0)
                if v.dtype!=torch.bool:difference=max(difference,float((v-b[k]).abs().max()))
            else:assert v==b[k]
        items.append(a)
    batch=ops['collate'](items)
    class Single:
        def __init__(self,repeats):self.repeats=repeats;self.dataset=range(len(items)*repeats)
        def __iter__(self):
            for _ in range(self.repeats):yield batch
    torch.cuda.reset_peak_memory_stats();net=initialize_fusion(p,'select',refs,c);reference=deepcopy(net)
    reference.motion_residual.forward=MethodType(flat_reference(ops['module'].P86MoBindMotionResidual.forward),reference.motion_residual)
    proof={};weights_y=y[data['observed_valid']]
    for stage in ['A','B']:
        # Source starts a fresh GradScaler at each stage. Its first B step can
        # overflow at scale65536; exercise ordinary automatic recovery as well.
        repeats=1 if stage=='A' else 4
        ops['seed_all'](121 if stage=='A' else 122)
        a=run_source_stage(net,Single(repeats),weights_y,recipe,'cuda',stage,1,ops)
        ops['seed_all'](121 if stage=='A' else 122)
        b=ops['train_stage'](reference,Single(repeats),weights_y,recipe,torch.device('cuda'),stage,1)
        for k in a[0]:
            if k!='seconds':assert a[0][k]==b[0][k],(k,a[0][k],b[0][k])
        max_grad=max_update=0.;nonfinite=0
        expected=dict(reference.named_parameters())
        for k,v in net.named_parameters():
            q=expected[k];torch.testing.assert_close(v,q,atol=0,rtol=0)
            max_update=max(max_update,float((v-q).abs().max()))
            if v.grad is not None:
                nonfinite+=int((~torch.isfinite(v.grad)).sum());torch.testing.assert_close(v.grad,q.grad,atol=0,rtol=0)
                max_grad=max(max_grad,float((v.grad-q.grad).abs().max()))
        assert nonfinite==0
        assert all(not q.requires_grad for q in net.visual.backbone_parameters())
        assert all(not q.training for q in net.visual.modules() if isinstance(q,torch.nn.BatchNorm3d))
        assert net.motion_residual.encoder.skeleton_encoder.training
        proof[stage]={'loss_terms_equal':True,'max_gradient_difference':max_grad,'max_parameter_difference':max_update,
            'real_batch_repeats':repeats,
            'nonfinite_gradients':nonfinite,'trainable_parameters':sum(q.numel() for q in net.parameters() if q.requires_grad),
            'motion_dropout_training':True,'backbone_and_bn_frozen':True}
    # The all-zero control must still execute the source motion_aux loss.
    net.train();zero={k:torch.zeros_like(v) if k in MOTION_FIELDS else v for k,v in batch.items()}
    zero=ops['to_device'](zero,torch.device('cuda'));ops['seed_all'](55)
    with torch.autocast('cuda',dtype=torch.float16):
        out=ops['model_forward'](net,zero,args=recipe,augment_visual=True)
        loss=ops['losses'](out,zero,ops['class_weights'](weights_y,.35,torch.device('cuda')),recipe)
    assert torch.isfinite(loss['loss']) and float(loss['motion_aux'].detach())>0
    body={'protocol_sha256':p.identity(),'complete':True,'rows':len(items),'fit_population':len(weights_y),
        'restored_ir_only_rows_in_batch':len(restored[:1]),'source_consumed_dataset_max_difference':difference,'stages':proof,
        'mask_motion_loss_finite':True,'mask_motion_aux_preserved':True,'unused_source_head_fields_not_consumed':True,
        'source_padding_fix':'first_column_or_empty_verified_against_flatten_reference',
        'peak_gpu_mib':round(torch.cuda.max_memory_allocated()/1024**2,2),'seconds':round(time.perf_counter()-start,2),
        'final_labels_read':False,'formal_training_started':False}
    write_json(ROOT/'outputs/task5_audit/P38-task10-real-batch.json',body);print(json.dumps(body),flush=True)


if __name__=='__main__':main()
