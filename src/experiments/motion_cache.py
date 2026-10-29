"""Label-free source P31/P86 motion caches; no cross-sample fitted scaling."""
from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from .artifact_record import ArtifactRegistry, VerificationContext
from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import load_stage_inputs, read_public_rows, row_index, csv_bytes, PUBLIC_COLUMNS
from .no_vote_types import ArtifactRef, canonical_bytes, canonical_hash, write_json
from .teammate_source import verify_teammate_source, load_teammate_symbol

MOTION_FIELDS = ('skeleton_features','skeleton_feature_mask','skeleton_joint_mask',
    'skeleton_relations','skeleton_relation_mask','skeleton_frame_quality','imu_sequences',
    'imu_sequence_mask','imu_bin_statistics','imu_bin_mask','imu_global_statistics','imu_global_mask')
TEMPORAL_MOTION_FIELDS = MOTION_FIELDS[:10]
BOOLEAN_MOTION_FIELDS = frozenset({'skeleton_feature_mask','skeleton_joint_mask',
    'skeleton_relation_mask','imu_sequence_mask','imu_bin_mask'})
SOURCE_FILES = ('p31_skeleton_imu_preprocessing.py','p46_event_preprocessing.py',
    'build_p86_motion_window_cache.py','p86_cached_motion_data.py','p86_mobind_lite_data.py')


def source_motion_ops(protocol, context):
    recipe=protocol.recipe.get('motion',{})
    if (protocol.recipe.get('source_contract_version')!=3 or recipe.get('imu_instance_normalization',False)
            or recipe.get('external_transform','identity_source_descriptor')!='identity_source_descriptor'):
        raise ValueError('motion requires source identity transform without added normalization')
    required={'frames':16,'joints':17,'skeleton_channels':13,'imu_points_per_bin':4,
        'imu_token_channels':16,'imu_bin_stat_channels':52,'imu_global_stat_channels':48}
    if any(recipe.get(k,v)!=v for k,v in required.items()):raise ValueError('source motion shape/config drift')
    key=(protocol.identity(),'task6_source_motion_ops')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        # Two imports verify their consumed dependency origins and bytes; never run legacy main.
        aggregate=load_teammate_symbol(report,'build_p86_motion_window_cache','aggregate_imu')
        dataset=load_teammate_symbol(report,'p86_mobind_lite_data','P86MoBindMotionDataset')
        p31=sys.modules['p31_skeleton_imu_preprocessing'];event=sys.modules['p46_event_preprocessing']
        cached=sys.modules['p86_cached_motion_data'];window=sys.modules['build_p86_motion_window_cache']
        if (tuple(cached.MOTION_FIELDS)!=MOTION_FIELDS or tuple(cached.TEMPORAL_MOTION_FIELDS)!=TEMPORAL_MOTION_FIELDS
                or set(cached.BOOLEAN_MOTION_FIELDS)!=BOOLEAN_MOTION_FIELDS):raise ValueError('source motion field order drift')
        ops={name:getattr(p31,name) for name in ['build_skeleton_features','frame_times_from_ids',
            'load_full_imu_trial','assign_points_to_frame_intervals','IMU_DEVICE_NAMES','IMU_CHANNEL_NAMES']}
        ops.update(aggregate_imu=aggregate,rotate_skeleton_cache=event.rotate_skeleton_cache,
            MOTION_FIELDS=MOTION_FIELDS,BOOLEAN_MOTION_FIELDS=BOOLEAN_MOTION_FIELDS,
            TEMPORAL_MOTION_FIELDS=TEMPORAL_MOTION_FIELDS,temporal_indices=dataset.temporal_indices,
            bin_assignments=window.bin_assignments,source_dataset=dataset)
        context.public_rows[key]=ops
    return context.public_rows[key]


def prepare_source_arrays(raw, people, absolute_times, by_device, ops):
    """P31 build_trial mathematics, without its manifest labels or legacy paths."""
    raw=np.asarray(raw,np.float32);absolute_times=np.asarray(absolute_times,np.float64)
    if raw.shape!=(len(absolute_times),17,4) or np.shape(people)!=(len(raw),):raise ValueError('P31 frame axes mismatch')
    if not len(raw):
        return dict(frame_time_seconds=np.empty(0,np.float32),skeleton_features=np.empty((0,17,13),np.float32),
            skeleton_feature_mask=np.empty((0,17,13),bool),skeleton_joint_mask=np.empty((0,17),bool),
            skeleton_relations=np.empty((0,18),np.float32),skeleton_relation_mask=np.empty((0,18),bool),
            skeleton_frame_quality=np.empty(0,np.float32),skeleton_person_count=np.empty(0,np.uint8),
            skeleton_clip_scale=np.asarray(1.,np.float32),imu_values=np.empty((0,10),np.float32),
            imu_time_seconds=np.empty(0,np.float32),imu_frame_index=np.empty(0,np.int16),
            imu_device_offsets=np.zeros(6,np.int32),imu_interval_counts=np.zeros((0,5),np.uint16),
            imu_device_mask=np.zeros(5,bool))
    if not np.isfinite(absolute_times).all() or np.any(np.diff(absolute_times)<=0):raise ValueError('P31 nonmonotonic frame times')
    skeleton=ops['build_skeleton_features'](raw,absolute_times)
    times=(absolute_times-absolute_times[0]).astype(np.float32);values=[];point_times=[];frames=[];offsets=[0]
    counts=np.zeros((len(raw),5),np.uint32);mask=np.zeros(5,bool)
    for i,device in enumerate(ops['IMU_DEVICE_NAMES']):
        if device not in by_device:offsets.append(offsets[-1]);continue
        ts,v=by_device[device];v=np.asarray(v,np.float32)
        assigned=ops['assign_points_to_frame_intervals'](ts,absolute_times)
        counts[:,i]=np.bincount(assigned,minlength=len(raw));mask[i]=bool(len(v))
        values.append(v);point_times.append((ts-absolute_times[0]).astype(np.float32));frames.append(assigned.astype(np.int16))
        offsets.append(offsets[-1]+len(v))
    if counts.max(initial=0)>np.iinfo(np.uint16).max:raise OverflowError('IMU interval count overflow')
    out=dict(frame_time_seconds=times,skeleton_features=np.asarray(skeleton['features'],np.float32),
        skeleton_feature_mask=np.asarray(skeleton['feature_mask'],bool),skeleton_joint_mask=np.asarray(skeleton['joint_mask'],bool),
        skeleton_relations=np.asarray(skeleton['relations'],np.float32),skeleton_relation_mask=np.asarray(skeleton['relation_mask'],bool),
        skeleton_frame_quality=np.asarray(skeleton['frame_quality'],np.float32)*(np.asarray(people)>0),
        skeleton_person_count=np.asarray(people,np.uint8),skeleton_clip_scale=np.asarray(skeleton['clip_scale'],np.float32),
        imu_values=np.concatenate(values) if values else np.empty((0,10),np.float32),
        imu_time_seconds=np.concatenate(point_times) if point_times else np.empty(0,np.float32),
        imu_frame_index=np.concatenate(frames) if frames else np.empty(0,np.int16),
        imu_device_offsets=np.asarray(offsets,np.int32),imu_interval_counts=counts.astype(np.uint16),imu_device_mask=mask)
    if int(counts.sum())!=len(out['imu_values']) or offsets[-1]!=len(out['imu_values']):raise ValueError('IMU point accounting mismatch')
    return out


def window_schema(n=None):
    prefix=() if n is None else (n,)
    shapes=((2,16,17,13),(2,16,17,13),(2,16,17),(2,16,18),(2,16,18),(2,16),
        (2,16,5,4,16),(2,16,5,4),(2,16,5,52),(2,16,5),(5,48),(5,2))
    return {k:(np.dtype(np.uint8 if k in BOOLEAN_MOTION_FIELDS else np.float32 if k=='imu_global_statistics' else np.float16),prefix+s)
        for k,s in zip(MOTION_FIELDS,shapes)}


def window_arrays(trial, source_indices, ops):
    chosen=np.asarray(source_indices)
    if chosen.shape!=(2,16) or not np.issubdtype(chosen.dtype,np.integer) or np.any(chosen<0):raise ValueError('source frame indices must be integer [2,16]')
    result={k:np.zeros(shape,dtype) for k,(dtype,shape) in window_schema().items()}
    if not len(trial['frame_time_seconds']):
        if chosen.any() or len(trial['imu_values']):raise ValueError('empty frame grid cannot hide motion input')
        return result
    if chosen.max()>=len(trial['frame_time_seconds']):raise ValueError('source frame indices exceed P31 grid')
    rotated=ops['rotate_skeleton_cache'](trial['skeleton_features'],trial['skeleton_feature_mask'],trial['skeleton_joint_mask'],
        trial['skeleton_relations'],trial['skeleton_relation_mask'],trial['frame_time_seconds'])
    for k,v in [('skeleton_features','features'),('skeleton_feature_mask','feature_mask'),('skeleton_relations','relations'),('skeleton_relation_mask','relation_mask')]:result[k][:]=rotated[v][chosen]
    for k in ['skeleton_joint_mask','skeleton_frame_quality']:result[k][:]=trial[k][chosen]
    imu=ops['aggregate_imu'](trial['frame_time_seconds'],chosen,trial['imu_values'],trial['imu_time_seconds'],trial['imu_device_offsets'],4)
    for k,v in [('imu_sequences','sequences'),('imu_sequence_mask','sequence_mask'),('imu_bin_statistics','statistics'),
            ('imu_bin_mask','bin_mask'),('imu_global_statistics','global_statistics'),('imu_global_mask','global_mask')]:result[k][:]=imu[v]
    for key,value in result.items():
        if not np.isfinite(value).all():raise ValueError('non-finite source motion field: '+key)
    return result


def validate_pixel_timeline(trial, source_indices, pixel_times):
    absolute=np.asarray(trial['frame_absolute_time_seconds'],np.float64);chosen=np.asarray(source_indices,np.int64)
    actual=np.asarray(pixel_times,np.float64)
    if actual.shape!=(2,16):raise ValueError('pixel timestamp shape mismatch')
    if not len(absolute):
        if np.isfinite(actual).any():raise ValueError('pixel timestamp exists without motion frame axis')
        return
    if chosen.max()>=len(absolute):raise ValueError('pixel frame grid exceeds motion timestamp axis')
    known=np.isfinite(actual)
    if known.any() and not np.allclose(actual[known],absolute[chosen][known],atol=1e-6,rtol=0):raise ValueError('pixel/motion timestamp mismatch')


def apply_source_transform(motion):
    """Identity outside source physical operators; no fit/population/label arguments."""
    if set(motion)!=set(MOTION_FIELDS):raise ValueError('motion fields differ from source')
    return dict(motion)


def motion_item(motion, *, temporal_index=None):
    values=apply_source_transform(motion)
    index=np.arange(16) if temporal_index is None else np.asarray(temporal_index)
    if index.shape!=(16,) or not np.issubdtype(index.dtype,np.integer) or np.any((index<0)|(index>=16)):
        raise ValueError('motion temporal index mismatch')
    return {k:np.asarray(v[:,index] if k in TEMPORAL_MOTION_FIELDS else v,
        dtype=bool if k in BOOLEAN_MOTION_FIELDS else np.float32).copy() for k,v in values.items()}


def _ref(path):
    d=json.loads(Path(path).read_text(encoding='utf-8'));return ArtifactRef(Path(d['record_path']),d['sha256'])


def _source_descriptor(protocol, context):
    registry=ArtifactRegistry(protocol,context=context);b=protocol.recipe.get('source_ops_bindings',{})
    if b.get('manifest_path'):path=Path(b['manifest_path'])
    else:
        if protocol.recipe['execution_kind']!='fixture':raise ValueError('motion source_ops binding missing')
        path=protocol.run_root/'protocol/task6_fixture_source.json'
        immutable_bytes(path,canonical_bytes(dict(schema_version=3,training_executed=False,oof_enabled=False,
            source_manifest_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],
            recipe={'motion':'source'},operators=['P31','P86'],consumed_files=list(SOURCE_FILES)))+b'\n')
    return registry.register(stage='source_ops',kind='descriptor',phase='public',files=[path],config={'source_contract_version':3})


def describe_motion_transform(motion, protocol, *, context=None):
    context=context or VerificationContext(protocol);source_motion_ops(protocol,context)
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(motion,'motion_windows','raw')
    body=dict(schema_version=3,scope='task6_motion',cross_sample_fit=False,external_transform='identity',
        imu_instance_normalization=False,operators=['P31_physical_preprocessing','P46_body_rotation','P86_exact_pixel_grid'],
        motion_record=motion,fields=list(MOTION_FIELDS))
    path=protocol.run_root/'protocol'/('motion_transform_'+motion.sha256+'.json')
    immutable_bytes(path,canonical_bytes(body)+b'\n')
    return registry.register(stage='source_transform',kind='descriptor',phase='public',files=[path],
        parents=[_source_descriptor(protocol,context)],config={'source_contract_version':3,'external_transform':'identity'},source_files=[Path(__file__)])


def _write_npz(path, values):
    buffer=BytesIO();np.savez_compressed(buffer,**values);payload=buffer.getvalue()
    immutable_bytes(path,payload);return hashlib.sha256(payload).hexdigest()


def _stamp(path):
    s=Path(path).stat();return [s.st_size,s.st_mtime_ns]


def _imu_inventory(row):
    folder=Path(row['imu_path']) if row.get('imu_path') else None
    return {str(p.resolve()):_stamp(p) for p in sorted(folder.glob('*.csv'))} if folder and folder.is_dir() else {}


def _pose_file_table(record):
    table={Path(x).stem:Path(x) for x in record.files if Path(x).suffix=='.npz'}
    if record.rows is None or set(table)!=set(record.rows.sample_ids):raise ValueError('pose cache exact IDs missing')
    return table


def _trial_from_pose(row, pose_path, registry, pose_record, ops):
    path=registry.verify_file(pose_record,pose_path)
    with np.load(path,allow_pickle=False) as z:
        if str(z['sample_id'].item())!=row['sample_id'] or not bool(z['completed']):raise ValueError('pose sample identity/incomplete')
        ids=z['frame_ids'].astype(str);acquired=z['acquisition_ids'].astype(str)
        raw=z['skeleton_h36m_xyz_conf_raw'].astype(np.float32);people=z['skeleton_person_count'].astype(np.uint8)
    if row['ir_available']=='0':
        if row['skeleton_available']=='1' or row['imu_available']=='1':raise ValueError('motion-only timeline needs explicit source acquisition evidence')
        if len(ids):raise ValueError('missing IR has unexpected pose frames')
    inventory=_imu_inventory(row)
    if row['imu_available']=='1' and not inventory:raise ValueError('declared IMU input missing')
    if row['imu_available']=='0' and inventory:raise ValueError('public IMU availability mismatch')
    by_device,audit=ops['load_full_imu_trial'](Path(row['imu_path']) if row.get('imu_path') else Path('__missing_task6_imu__'))
    if len(ids):
        keys=acquired if len(acquired)==len(ids) and all(acquired) else ids
        absolute=ops['frame_times_from_ids'](keys)
        if all(str(v).isdigit() for v in keys) and by_device:raise ValueError('counter-only camera cannot align absolute IMU times')
        mode='counter_10hz_no_imu' if all(str(v).isdigit() for v in keys) else 'observed_absolute_timestamp'
    else:absolute=np.empty(0,np.float64);mode='unavailable'
    trial=prepare_source_arrays(raw,people,absolute,by_device,ops)
    trial.update(sample_id=np.asarray(row['sample_id']),frame_ids=ids,acquisition_ids=acquired,
        frame_absolute_time_seconds=absolute,
        completed=np.asarray(True),available=np.asarray(bool(len(ids))),time_mode=np.asarray(mode))
    return trial,dict(imu_audit=vars(audit),imu_inventory=inventory,time_mode=mode,
        frames=len(ids),imu_points=len(trial['imu_values']),skeleton_valid_frames=int(trial['skeleton_joint_mask'].any(1).sum()))


def build_motion_source(inputs, pose, output, *, protocol, context=None, max_trials=0):
    context=context or VerificationContext(protocol);ops=source_motion_ops(protocol,context);registry=ArtifactRegistry(protocol,context=context)
    if max_trials<0:raise ValueError('max_trials cannot be negative')
    if inputs!=load_stage_inputs(protocol,inputs.partition.name,'raw',context=context) or inputs.labels is not None:raise ValueError('motion source requires label-free current inputs')
    public=read_public_rows(inputs.public_manifest);selected=public[:max_trials] if max_trials else public;index=row_index(selected)
    pr=registry.verify(pose,'p28','raw');table=_pose_file_table(pr)
    if any(dict(zip(pr.rows.sample_ids,pr.rows.user_ids)).get(x['sample_id'])!=x['user_id'] for x in selected):raise ValueError('pose source users/IDs mismatch')
    root=Path(output).resolve()
    if not root.is_relative_to(protocol.run_root):raise ValueError('motion source output outside run')
    identity={'protocol':protocol.identity(),'manifest':inputs.parents['manifest'],'pose':pose,'max_trials':max_trials,
        'producer':context.hash_file(Path(__file__))}
    immutable_bytes(root/'identity.json',canonical_bytes(identity)+b'\n')
    if (root/'artifact.json').exists():
        ref=_ref(root/'artifact.json');registry.verify(ref,'motion_source','raw',index);return ref
    started=time.perf_counter();digests={};summaries=[]
    for i,row in enumerate(selected):
        sid=row['sample_id'];path=root/'trials'/(sid+'.npz');meta_path=root/'trials'/(sid+'.json')
        current={'pose_stamp':_stamp(table[sid]),'imu_inventory':_imu_inventory(row)}
        if path.exists() or meta_path.exists():
            if not path.exists() or not meta_path.exists():raise ValueError('partial source trial without atomic metadata')
            meta=json.loads(meta_path.read_text(encoding='utf-8'))
            if any(meta.get(k)!=v for k,v in current.items()) or meta['payload_stamp']!=_stamp(path):raise ValueError('source motion resume input drift')
            digest=meta['sha256']
        else:
            values,extra=_trial_from_pose(row,table[sid],registry,pr,ops);digest=_write_npz(path,values)
            context.trust_writer_digest(path,digest);meta={'sample_id':sid,'sha256':digest,'payload_stamp':_stamp(path),**current,**extra}
            write_json(meta_path,meta)
        digests[str(path.resolve())]=digest;summaries.append(meta)
        if (i+1)%50==0 or i+1==len(selected):print(json.dumps({'event':'motion_source_progress','partition':inputs.partition.name,'completed':i+1,'total':len(selected),'seconds':round(time.perf_counter()-started,2)}),flush=True)
    summary=root/'summary.json';write_json(summary,{'partition':inputs.partition.name,'rows':len(selected),
        'frames':sum(x['frames'] for x in summaries),'imu_points':sum(x['imu_points'] for x in summaries),
        'empty_rows':sum(x['frames']==0 for x in summaries),'time_modes':{mode:sum(x['time_mode']==mode for x in summaries) for mode in set(x['time_mode'] for x in summaries)},'seconds':round(time.perf_counter()-started,2)})
    digests[str(summary.resolve())]=context.hash_file(summary)
    ref=registry.register(stage='motion_source',kind='raw_cache',phase='raw',files=list(digests),file_digests=digests,
        parents=[pose,inputs.parents['manifest'],_source_descriptor(protocol,context)],rows=index,
        config={'partition':inputs.partition.name,'max_trials':max_trials,'source_p31':True,'timestamp_restoration':'observed_acquisition_ids',
            'payload_stamps':{Path(x).name:_stamp(x) for x in digests}},complete=not max_trials,source_files=[Path(__file__)])
    write_json(root/'artifact.json',ref);return ref


def _bounded_payload(record, path, context):
    path=Path(path).resolve()
    if str(path) not in record.files or record.config.get('payload_stamps',{}).get(path.name)!=_stamp(path):raise ValueError('motion cache payload stamp drift')
    context.trust_writer_digest(path,record.files[str(path)])
    return path


def build_motion_windows(inputs, source, pixels, output, *, protocol, context=None, max_trials=0):
    from .pixel_cache import load_pixels
    from .visual_student import save_array_with_digest
    context=context or VerificationContext(protocol);ops=source_motion_ops(protocol,context);registry=ArtifactRegistry(protocol,context=context)
    if max_trials<0:raise ValueError('max_trials cannot be negative')
    if inputs!=load_stage_inputs(protocol,inputs.partition.name,'raw',context=context) or inputs.labels is not None:raise ValueError('motion windows require label-free current inputs')
    public=read_public_rows(inputs.public_manifest);selected=public[:max_trials] if max_trials else public;index=row_index(selected)
    sr=registry.verify(source,'motion_source','raw',index);pr,pa=load_pixels(pixels,protocol=protocol,context=context)
    lookup={s:i for i,s in enumerate(pr.rows.sample_ids)}
    if any(s not in lookup or pr.rows.user_ids[lookup[s]]!=u for s,u in zip(index.sample_ids,index.user_ids)):raise ValueError('pixels motion exact IDs/users mismatch')
    root=Path(output).resolve()
    if not root.is_relative_to(protocol.run_root):raise ValueError('motion windows output outside run')
    immutable_bytes(root/'identity.json',canonical_bytes({'source':source,'pixels':pixels,'rows':index,'producer':context.hash_file(Path(__file__))})+b'\n')
    if (root/'artifact.json').exists():
        ref=_ref(root/'artifact.json');registry.verify(ref,'motion_windows','raw',index);return ref
    n=len(index.sample_ids);arrays={k:np.zeros(shape,dtype) for k,(dtype,shape) in window_schema(n).items()}
    arrays.update(completed=np.zeros(n,np.uint8),source_frame_indices=np.zeros((n,2,16),np.int32),source_time_seconds=np.full((n,2,16),np.nan,np.float64))
    table={Path(x).stem:Path(x) for x in sr.files if Path(x).suffix=='.npz'};started=time.perf_counter()
    for i,sid in enumerate(index.sample_ids):
        p=lookup[sid];chosen=np.asarray(pa['source_frame_indices'][p],np.int64)
        path=_bounded_payload(sr,table[sid],context)
        with np.load(path,allow_pickle=False) as archive:trial={k:archive[k] for k in archive.files}
        if str(trial['sample_id'].item())!=sid or not bool(trial['completed']):raise ValueError('P31 source sample mismatch')
        validate_pixel_timeline(trial,chosen,pa['source_time_seconds'][p])
        result=window_arrays(trial,chosen,ops)
        for k in MOTION_FIELDS:arrays[k][i]=result[k]
        arrays['source_frame_indices'][i]=chosen;arrays['completed'][i]=1
        arrays['source_time_seconds'][i]=pa['source_time_seconds'][p]
        if (i+1)%100==0 or i+1==n:print(json.dumps({'event':'motion_windows_progress','partition':inputs.partition.name,'completed':i+1,'total':n,'seconds':round(time.perf_counter()-started,2)}),flush=True)
    digests={}
    for name,array in arrays.items():
        path=root/(name+'.npy');tmp=path.with_suffix('.npy.building');digest=save_array_with_digest(tmp,array);tmp.replace(path);digests[str(path.resolve())]=digest
    rows_path=root/'rows.csv';content=csv_bytes(PUBLIC_COLUMNS,selected);immutable_bytes(rows_path,content);digests[str(rows_path.resolve())]=hashlib.sha256(content).hexdigest()
    summary=root/'summary.json';write_json(summary,{'partition':inputs.partition.name,'rows':n,'completed':int(arrays['completed'].sum()),
        'skeleton_available_rows':int(arrays['skeleton_joint_mask'].reshape(n,-1).any(1).sum()),
        'imu_available_rows':int(arrays['imu_global_mask'][:,:,0].any(1).sum()),'seconds':round(time.perf_counter()-started,2),
        'cross_sample_fit':False,'imu_instance_normalization':False})
    digests[str(summary.resolve())]=context.hash_file(summary)
    ref=registry.register(stage='motion_windows',kind='raw_cache',phase='raw',files=list(digests),file_digests=digests,
        parents=[source,pixels,inputs.parents['manifest']],rows=index,complete=not max_trials,
        config={'partition':inputs.partition.name,'source_p86':True,'source_grid':'exact_pixel_indices','max_trials':max_trials,
            'payload_stamps':{Path(x).name:_stamp(x) for x in digests}},source_files=[Path(__file__)])
    write_json(root/'artifact.json',ref);return ref


def load_motion(ref, *, protocol, context=None):
    context=context or VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    record=registry.verify(ref,'motion_windows','raw');n=len(record.rows.sample_ids);arrays={}
    row_files=[Path(x) for x in record.files if Path(x).name=='rows.csv']
    if len(row_files)!=1 or row_index(read_public_rows(registry.verify_file(record,row_files[0])))!=record.rows:
        raise ValueError('motion row IDs mismatch')
    for name in [*MOTION_FIELDS,'completed','source_frame_indices','source_time_seconds']:
        files=[Path(x) for x in record.files if Path(x).name==name+'.npy']
        if len(files)!=1:raise ValueError('one motion field payload required: '+name)
        arrays[name]=context.load_arrays(_bounded_payload(record,files[0],context))
    for key,(dtype,shape) in window_schema(n).items():
        if arrays[key].shape!=shape or arrays[key].dtype!=dtype:raise ValueError('motion header mismatch: '+key)
    if arrays['completed'].shape!=(n,) or not arrays['completed'].all():raise ValueError('motion cache incomplete')
    if (arrays['source_frame_indices'].shape!=(n,2,16) or arrays['source_frame_indices'].dtype!=np.int32
            or arrays['source_time_seconds'].shape!=(n,2,16) or arrays['source_time_seconds'].dtype!=np.float64):
        raise ValueError('motion source timeline header mismatch')
    return record,arrays


def run_motion_pipeline(protocol, *, partition='all', max_trials=0):
    context=VerificationContext(protocol);registry=ArtifactRegistry(protocol,context=context)
    ops=source_motion_ops(protocol,context)
    if protocol.recipe.get('motion',{}).get('imu_points_per_bin',4)!=4:raise ValueError('source uses four IMU points per bin')
    proof=json.loads((protocol.run_root/'protocol/task2_prepared.json').read_text(encoding='utf-8'));imports=proof['imports']
    requested=['refit14','final4'] if partition=='all' else [partition];outputs={}
    for name in requested:
        cache_part='final4' if name=='final4' else 'refit14'
        px=next(x for x in imports if x['stage']=='pixels' and x['partition']==cache_part)
        pixels=ArtifactRef(Path(px['record_path']),px['sha256'])
        pixel_record=registry.read(pixels)
        rois=[ArtifactRef(Path(x['record_path']),x['sha256']) for x in imports if x['stage']=='p29']
        roi=next(registry.read(r) for r in rois if set(pixel_record.rows.sample_ids)<=set(registry.read(r).rows.sample_ids))
        poses=[r for r in roi.parents if registry.read(r).stage=='p28']
        if len(poses)!=1:raise ValueError('exact P28 source parent missing')
        inputs=load_stage_inputs(protocol,name,'raw',context=context)
        mode='full' if not max_trials else 'acceptance_'+str(max_trials)
        root=protocol.run_root/'motion'/name/mode
        source=build_motion_source(inputs,poses[0],root/'source',protocol=protocol,context=context,max_trials=max_trials)
        windows=build_motion_windows(inputs,source,pixels,root/'windows',protocol=protocol,context=context,max_trials=max_trials)
        transform=describe_motion_transform(windows,protocol,context=context) if not max_trials else None
        outputs[name]={'source':source,'windows':windows,'normalization':transform}
    if partition=='all' and not max_trials:
        body={'protocol_sha256':protocol.identity(),'complete':True,'training_executed':False,'final_labels_read':False,
            'partitions':outputs,'select_subsets':{'train12':'refit14','development2':'refit14'},'external_transform':'identity','imu_instance_normalization':False,'task7_started':False}
        immutable_bytes(protocol.run_root/'protocol/task6_artifacts_ready.json',canonical_bytes(body)+b'\n')
    return outputs
