"""Label-free P28/P29 orchestration using verified teammate operators."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from .artifact_record import ArtifactRegistry,verify_public_file
from .no_vote_manifest import read_public_rows,row_index
from .no_vote_protocol import NoVoteProtocol
from .no_vote_types import ArtifactRef,StageInputs,canonical_hash,write_json
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file

REGIONS=('full_body','left_arm','right_arm','left_hand','right_hand','hand_workspace','global_fallback')


def compatible_frame_map(path: Path | None,modality: str,ir_path: Path | None=None):
    if path is None or not Path(path).is_dir():return {}
    path=Path(path);directory=path/'predictions' if modality=='skeleton' else path
    if not directory.is_dir():return {}
    result={}
    for file in sorted(directory.iterdir()):
        if not file.is_file():continue
        stem=file.stem
        if modality=='skeleton':
            if file.suffix.lower()!='.json' or not stem.startswith('Color_'):continue
            key=stem[len('Color_'):]
        else:
            if file.suffix.lower() not in {'.png','.jpg','.jpeg'}:continue
            if modality=='depth':
                if not stem.startswith('Depth_') or not stem.endswith('_Color'):continue
                key=stem[len('Depth_'):-len('_Color')]
            elif modality=='ir':
                if not stem.startswith('IR_'):continue
                key=stem[len('IR_'):]
            else:raise ValueError('unsupported frame modality')
        if key in result:raise ValueError('duplicate acquisition key')
        result[key]=file
    if modality!='skeleton' or not result:return result
    ir=compatible_frame_map(ir_path,'ir') if ir_path else {}
    if ir:
        counters={key.rsplit('_',1)[-1]:key for key in ir}
        if len(counters)!=len(ir):raise ValueError('Ambiguous IR acquisition counters')
        grouped={}
        for key,file in result.items():grouped.setdefault(key.rsplit('_',1)[-1],[]).append((key,file))
        resolved={}
        for counter,items in grouped.items():
            if len(items)>1:
                dated=[(key,file) for key,file in items if not key.isdigit()]
                if len(items)!=2 or len(dated)!=1 or len({sha256_file(file) for _,file in items})!=1:
                    raise ValueError('Ambiguous Skeleton acquisition counters or alias contents')
                key,file=dated[0]
            else:key,file=items[0]
            resolved[key]=file
        result=resolved
        mapped={counters[key.rsplit('_',1)[-1]]:file for key,file in result.items()
                if key.rsplit('_',1)[-1] in counters}
        if mapped:return mapped
    if all(not key.isdigit() for key in result):return result
    raise ValueError('counter-only Skeleton lacks an absolute timeline')


def acquisition_ids(maps,frame_ids):
    """Keep observed timestamps separately from IR file lookup keys."""
    observed=[]
    for key in frame_ids:
        stamp=key
        if key.isdigit():
            file=maps.get('skeleton',{}).get(key)
            stamp=file.stem[len('Color_'):] if file is not None else ''
            if stamp.isdigit():stamp=''
        observed.append(stamp)
    return np.asarray(observed,dtype=str)


def _load_ops(protocol):
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'])
    for module,symbol in [('audit_yolo11_pose_skeleton','select_track'),
                          ('build_adaptive_yolo11_pose_skeleton_cache','predict_candidates'),
                          ('build_multiscale_dir_rois','build_trial_rois')]:
        load_teammate_symbol(report,module,symbol)
    return (sys.modules['audit_yolo11_pose_skeleton'],
            sys.modules['build_adaptive_yolo11_pose_skeleton_cache'],
            sys.modules['build_multiscale_dir_rois'])


def _settings(device):
    return SimpleNamespace(device=device,primary_imgsz=640,primary_conf=.10,primary_batch=32,
        fallback_imgsz=1280,fallback_conf=.03,fallback_batch=16,keypoint_conf=.25,retry_box_conf=.15,
        retry_min_height_ratio=.20,retry_min_upper_joints=4,retry_min_wrists=2,retry_center_jump_ratio=.15,
        retry_area_ratio=2.5,interpolate_box_gap=3,transfer_margin=.20,width=640,height=480,
        arm_margin=.35,hand_forearm_scale=1.60,hand_min_person_height=.12,hand_max_person_height=.35,
        workspace_margin=.15,single_hand_workspace_margin=.30,max_joint_gap=3)


def _atomic_npz(path,arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.building')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)


def snapshot_raw_files(row):
    """Inventory actual input membership before trusting a resumed record."""
    hashes={}
    for modality in ('ir','depth','skeleton'):
        value=row.get(modality+'_path','')
        if not value:continue
        path=Path(value)
        directory=path/'predictions' if modality=='skeleton' else path
        if not directory.is_dir():continue
        for file in sorted(directory.iterdir()):
            if not file.is_file():continue
            valid=file.suffix.lower()=='.json' and file.stem.startswith('Color_') if modality=='skeleton' else (
                file.suffix.lower() in {'.png','.jpg','.jpeg'} and file.stem.startswith('IR_' if modality=='ir' else 'Depth_'))
            if valid:hashes[str(file.resolve())]=sha256_file(file)
    return hashes


def _empty_pose(frame_ids):
    n=len(frame_ids);boxes=np.full((n,5),np.nan,np.float32);joints=np.full((n,17,3),np.nan,np.float32)
    return {'frame_ids':np.asarray(frame_ids,dtype=str),'ir_boxes_xyxy_conf':boxes,
        'ir_keypoints_xy_conf':joints,'ir_person_count':np.zeros(n,np.uint8),
        'ir_person_roi_boxes_xyxy_conf':boxes.copy(),'depth_search_boxes_from_ir':boxes.copy(),
        'depth_boxes_from_ir_xyxy_conf':boxes.copy(),'depth_keypoints_from_ir_xy_conf':joints.copy(),
        'skeleton_h36m_xyz_conf_raw':np.full((n,17,4),np.nan,np.float32),
        'skeleton_h36m_xyz_conf_normalised':np.full((n,17,4),np.nan,np.float32),
        'skeleton_person_count':np.zeros(n,np.uint8),'final_pose_source':np.zeros(n,np.uint8),
        'retry_reason_bits':np.zeros(n,np.uint8),'roi_box_estimated':np.zeros(n,bool)}


def _process_trial(maps,model,ops,args):
    base,adaptive,roi=ops
    ids=sorted(maps['ir']);images=[maps['ir'][key] for key in ids]
    candidates=adaptive.predict_candidates(model,images,imgsz=args.primary_imgsz,conf=args.primary_conf,
                                           batch=args.primary_batch,device=args.device)
    boxes,keypoints,people=base.select_track(candidates)
    args.height,args.width=map(int,candidates[0]['shape'])
    reasons=adaptive.retry_reasons(boxes,keypoints,image_width=args.width,image_height=args.height,
        keypoint_conf=args.keypoint_conf,box_conf=args.retry_box_conf,min_height_ratio=args.retry_min_height_ratio,
        min_upper_joints=args.retry_min_upper_joints,min_wrists=args.retry_min_wrists,
        center_jump_ratio=args.retry_center_jump_ratio,area_ratio=args.retry_area_ratio)
    retry=np.flatnonzero(reasons!=0)
    fallback=adaptive.predict_candidates(model,[images[int(i)] for i in retry],imgsz=args.fallback_imgsz,
        conf=args.fallback_conf,batch=args.fallback_batch,device=args.device)
    final_boxes,final_kp,final_people=boxes.copy(),keypoints.copy(),people.copy()
    source=np.where(np.isfinite(boxes[:,:4]).all(1),1,0).astype(np.uint8)
    selected=np.zeros(len(ids),bool);references=np.full_like(boxes,np.nan)
    for index,candidate in zip(retry,fallback):
        index=int(index);reference=adaptive.nearest_reference_box(boxes,index)
        if reference is not None:references[index]=reference
        chosen=adaptive.select_fallback_candidate(candidate,reference,args.keypoint_conf,
            strict_reference_scale=not np.isfinite(boxes[index,:4]).all())
        if chosen is not None and adaptive.fallback_is_better(boxes[index],keypoints[index],*chosen,args.keypoint_conf):
            final_boxes[index],final_kp[index]=chosen
            final_people[index]=min(len(candidate['boxes']),255);source[index]=2;selected[index]=True
    final_boxes,source,interpolated=adaptive.interpolate_short_box_gaps(final_boxes,source,args.interpolate_box_gap)
    final_kp[interpolated]=np.nan
    person=final_boxes.copy()
    for i in np.flatnonzero(selected):person[i]=adaptive.union_boxes(references[i],final_boxes[i])
    person,estimated=adaptive.fill_short_roi_gaps(person,args.interpolate_box_gap)
    arrays=_empty_pose(ids)
    arrays.update(primary_ir_boxes_xyxy_conf=boxes,primary_ir_keypoints_xy_conf=keypoints,
        primary_ir_person_count=people,ir_boxes_xyxy_conf=final_boxes,ir_keypoints_xy_conf=final_kp,
        ir_person_count=final_people,ir_person_roi_boxes_xyxy_conf=person,final_pose_source=source,
        retry_reason_bits=reasons,roi_box_estimated=estimated,box_interpolated=interpolated,
        fallback_attempted=reasons!=0,fallback_selected=selected,
        depth_search_boxes_from_ir=base.expanded_search_boxes(person,args.transfer_margin,width=args.width,height=args.height))
    # Transfer geometry only at exact observed Depth keys. Missing counterpart
    # rows remain masked; they are never paired by action or nearest sample.
    depth_valid=np.asarray([key in maps['depth'] for key in ids])
    arrays['depth_boxes_from_ir_xyxy_conf'][depth_valid]=final_boxes[depth_valid]
    arrays['depth_keypoints_from_ir_xy_conf'][depth_valid]=final_kp[depth_valid]
    for i,key in enumerate(ids):
        if key in maps['skeleton']:
            raw,count=base.load_raw_skeleton(maps['skeleton'][key])
            arrays['skeleton_h36m_xyz_conf_raw'][i]=raw
            arrays['skeleton_person_count'][i]=min(count,255)
    valid=np.isfinite(arrays['skeleton_h36m_xyz_conf_raw'][:,:,:3]).any(axis=(1,2))
    if valid.any():arrays['skeleton_h36m_xyz_conf_normalised'][valid]=base.normalise_skeleton(arrays['skeleton_h36m_xyz_conf_raw'][valid])
    arrays['ir_keypoints_bbox_local']=base.bbox_local_keypoints(final_kp,final_boxes)
    semantic=base.semantic_pair_features(arrays['ir_keypoints_bbox_local'],arrays['skeleton_h36m_xyz_conf_normalised'])
    arrays['ir_skeleton_semantic_pairs']=semantic;arrays['depth_skeleton_semantic_pairs']=semantic
    arrays['depth_frame_valid']=depth_valid
    arrays['acquisition_ids']=acquisition_ids(maps,ids)
    rois=roi.build_trial_rois(arrays,args)
    rois['acquisition_ids']=arrays['acquisition_ids']
    return arrays,rois


def build_pose_roi(inputs: StageInputs,pose_weights: ArtifactRef,output: Path,*,
                   protocol: NoVoteProtocol,device='cpu',max_trials: int=0,detector=None):
    registry=ArtifactRegistry(protocol)
    weight=registry.verify(pose_weights,'pose_initializer','public')
    if set(weight.files)!={str(protocol.weights['yolo'])}:raise ValueError('pose weight identity mismatch')
    rows=read_public_rows(inputs.public_manifest)
    if len(rows)!=inputs.partition.expected_rows or set(r['user_id'] for r in rows)!=set(inputs.partition.users):
        raise ValueError('pose input population mismatch')
    registry.verify(inputs.parents['manifest'],'manifest_'+inputs.partition.name,'raw',row_index(rows))
    if max_trials<0:raise ValueError('max_trials cannot be negative')
    selected=rows[:max_trials] if max_trials else rows
    output=Path(output).resolve()
    if not output.is_relative_to(protocol.run_root):raise ValueError('pose output must be inside run')
    config={'schema':1,'device':str(device),'max_trials':max_trials,'algorithm':'teammate_adaptive_ir_pose_roi',
            'settings':vars(_settings(device)),'producer_sha256':sha256_file(Path(__file__))}
    identity={'protocol_sha256':protocol.identity(),'config_sha256':canonical_hash(config),
              'manifest_sha256':sha256_file(inputs.public_manifest),'weight_sha256':pose_weights.sha256}
    identity_path=output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:
        raise ValueError('pose resume identity drift')
    summary_path=output/'summary.json'
    if summary_path.is_file():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        restored={k:ArtifactRef(Path(v['record_path']),v['sha256']) for k,v in summary['artifacts'].items()}
        for stage,ref in restored.items():
            registry.verify(ref,stage,'raw',row_index(selected))
        for row in selected:
            meta=json.loads((output/'p28/trial_summary'/f"{row['sample_id']}.json").read_text())
            if meta['raw_input_hashes']!=snapshot_raw_files(row):raise ValueError('raw input membership drift')
        return restored
    write_json(identity_path,identity)
    ops=None;refs={};raw_inputs=[];files={'p28':[],'p29':[]}
    for row in selected:
        sid=row['sample_id'];paths={m:Path(row[m+'_path']) if row[m+'_path'] else None for m in ('ir','depth','skeleton')}
        maps={m:compatible_frame_map(paths[m],m) for m in ('ir','depth')};reason=''
        try:maps['skeleton']=compatible_frame_map(paths['skeleton'],'skeleton',paths['ir'])
        except ValueError as error:
            if 'Ambiguous' in str(error):raise
            maps['skeleton']={};reason=str(error)
        hashes=snapshot_raw_files(row);trial_raw=[Path(p) for p in hashes];raw_inputs+=trial_raw
        caches={stage:output/stage/'trial_cache'/f'{sid}.npz' for stage in files}
        summaries={stage:output/stage/'trial_summary'/f'{sid}.json' for stage in files}
        if all(p.is_file() for p in (*caches.values(),*summaries.values())):
            for stage in files:
                summary=json.loads(summaries[stage].read_text())
                if summary['raw_input_hashes']!=hashes:raise ValueError('raw input drift')
                if summary['producer_identity']!=identity:raise ValueError('interrupted pose producer identity drift')
                if summary['cache_sha256']!=sha256_file(caches[stage]):raise ValueError('interrupted pose cache hash drift')
                verify_public_file(caches[stage]);verify_public_file(summaries[stage])
        else:
            ids=sorted(maps['ir']);available=bool(ids)
            if available:
                if ops is None:ops=_load_ops(protocol)
                if detector is None:
                    from ultralytics import YOLO
                    detector=YOLO(str(protocol.weights['yolo']),verbose=False)
                pose,rois=_process_trial(maps,detector,ops,_settings(device))
            else:
                pose=_empty_pose([])
                rois={'frame_ids':np.asarray([],dtype=str),'region_names':np.asarray(REGIONS),
                    'roi_boxes_xyxy':np.full((0,7,4),np.nan,np.float32),'roi_valid':np.zeros((0,7),bool),
                    'roi_quality':np.zeros((0,7),np.float32),'roi_source':np.zeros((0,7),np.uint8)}
                reason=reason or 'missing_ir'
            for stage,arrays in (('p28',pose),('p29',rois)):
                arrays.setdefault('acquisition_ids',np.full(len(arrays['frame_ids']),'',dtype='U1'))
                arrays.update(sample_id=np.asarray(sid),available=np.asarray(available),completed=np.asarray(True))
                _atomic_npz(caches[stage],arrays)
                write_json(summaries[stage],{'sample_id':sid,'user_id':row['user_id'],'frames':len(ids),
                    'available':available,'completed':True,'reason':reason,'raw_input_hashes':hashes,
                    'producer_identity':identity,'cache_sha256':sha256_file(caches[stage]),
                    'skeleton_aligned_frames':len(set(ids)&set(maps['skeleton'])),
                    'roi_fallback_policy':'full_frame_if_roi_invalid'})
        for stage in files:files[stage]+=[caches[stage],summaries[stage]]
        print(json.dumps({'processed':len(files['p28'])//2,'selected':len(selected),'partition':inputs.partition.name},ensure_ascii=False),flush=True)
    for stage in files:
        parents=(inputs.parents['manifest'],pose_weights) if stage=='p28' else (refs['p28'],)
        refs[stage]=registry.register(stage=stage,kind='raw_cache',phase='raw',files=files[stage],parents=parents,
            rows=row_index(selected),config=config,raw_inputs=raw_inputs,source_files=[Path(__file__)],
            complete=len(selected)==len(rows))
    write_json(output/'summary.json',{'processed_rows':len(selected),'expected_rows':len(rows),
        'complete':len(selected)==len(rows),'partial':len(selected)!=len(rows),'labels_read':False,
        'artifacts':{k:{'record_path':str(v.record_path),'sha256':v.sha256} for k,v in refs.items()}})
    return refs
