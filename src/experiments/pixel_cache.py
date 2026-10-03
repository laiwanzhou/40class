"""Label-free IR pixel schema; validation never scans image contents."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .artifact_record import ArtifactRegistry
from .no_vote_manifest import load_stage_inputs,read_public_rows,row_index,csv_bytes,PUBLIC_COLUMNS
from .no_vote_types import ArtifactRef,canonical_hash,write_json
from .pose_roi_adapter import compatible_frame_map
from .teammate_source import verify_teammate_source,load_teammate_symbol,sha256_file
from .visual_teacher import window_indices


PIXEL_KEYS=frozenset({'images','view_valid','view_quality','source_frame_indices',
    'source_time_seconds','completed'})


def validate_pixels(arrays,n):
    if set(arrays)!=PIXEL_KEYS:raise ValueError('pixel schema/label fields mismatch')
    if arrays['images'].shape!=(n,2,16,3,160,160) or arrays['images'].dtype!=np.uint8:
        raise ValueError('pixel shape/dtype must be uint8[N,2,16,3,160,160]')
    if arrays['view_valid'].shape!=(n,2,16,3) or arrays['view_valid'].dtype!=np.bool_:
        raise ValueError('pixel view_valid schema mismatch')
    quality=arrays['view_quality']
    if quality.shape!=(n,2,16,3) or not np.isfinite(quality).all() or np.any((quality<0)|(quality>1)):
        raise ValueError('pixel quality shape/value mismatch')
    indices=arrays['source_frame_indices']
    if indices.shape!=(n,2,16) or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0):
        raise ValueError('pixel source frame indices mismatch')
    times=arrays['source_time_seconds']
    if times.shape!=(n,2,16) or not np.issubdtype(times.dtype,np.floating) or np.isinf(times).any():
        raise ValueError('pixel source times mismatch; unknown timestamps must be NaN')
    done=arrays['completed']
    if done.shape!=(n,) or done.dtype!=np.bool_ or not done.all():
        raise ValueError('pixel cache incomplete')


def acquisition_seconds(key):
    stamp=str(key).rsplit('_',1)[0]
    for pattern in ('%Y-%m-%d_%H-%M-%S.%f','%Y-%m-%d_%H-%M-%S'):
        try:return datetime.strptime(stamp,pattern).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:pass
    try:
        value=float(stamp)
        if 1e9<=value<=4e9:return value
    except ValueError:pass
    return np.nan


def _pixel_ops(protocol):
    report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
        expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
    return (load_teammate_symbol(report,'build_p46_videomae_cache','square_crop'),
            load_teammate_symbol(report,'build_p30_shared_dir_roi_feature_cache','read_ir'))


def _pixel_trial(row,path,ops,registry,roi_record):
    result=dict(images=np.zeros((2,16,3,160,160),np.uint8),view_valid=np.zeros((2,16,3),bool),
        view_quality=np.zeros((2,16,3),np.float32),source_frame_indices=np.zeros((2,16),np.int64),
        source_time_seconds=np.full((2,16),np.nan),completed=np.bool_(True))
    if row['ir_available']!='1':return result
    registry.verify_file(roi_record,path)
    with np.load(path,allow_pickle=False) as z:
        if str(z['sample_id'].item())!=row['sample_id'] or not bool(z['completed']):raise ValueError('ROI trial identity/incomplete')
        ids=z['frame_ids'].astype(str);acquired=z['acquisition_ids'].astype(str)
        names=z['region_names'].astype(str).tolist();boxes=z['roi_boxes_xyxy'];valid=z['roi_valid'];quality=z['roi_quality']
    actual=compatible_frame_map(Path(row['ir_path']),'ir')
    if tuple(ids)!=tuple(sorted(actual)) or not len(ids) or acquired.shape!=ids.shape:
        raise ValueError('pixel ROI/IR acquisition axis mismatch')
    crop,read_ir=ops;person=names.index('full_body');workspace=names.index('hand_workspace')
    chosen=np.stack([window_indices(len(ids),lo,hi) for lo,hi in ((0.,.70),(.30,1.))])
    result['source_frame_indices']=chosen
    times=np.asarray([acquisition_seconds(x) for x in acquired]);result['source_time_seconds']=times[chosen]
    images={i:read_ir(actual[ids[i]])[:,:,0] for i in np.unique(chosen)}
    for w,indices in enumerate(chosen):
        for t,i in enumerate(indices):
            pvalid=bool(valid[i,person]);wvalid=bool(valid[i,workspace])
            pb=boxes[i,person] if pvalid else np.full(4,np.nan)
            wb=boxes[i,workspace] if wvalid else pb
            image=images[i]
            for v,img in enumerate((image,crop(image,pb,1.15),crop(image,wb,1.40))):
                result['images'][w,t,v]=cv2.resize(img,(160,160),interpolation=cv2.INTER_AREA)
            result['view_valid'][w,t]=(True,pvalid,wvalid or pvalid)
            result['view_quality'][w,t]=(1.,quality[i,person] if pvalid else .25,
                quality[i,workspace] if wvalid else (quality[i,person]*.5 if pvalid else .20))
    return result


def build_pixels(inputs,roi,output,*,protocol,max_trials=0):
    started=time.perf_counter();registry=ArtifactRegistry(protocol)
    fresh=load_stage_inputs(protocol,inputs.partition.name,'raw')
    if inputs!=fresh or max_trials<0:raise ValueError('pixel raw input ownership/limit mismatch')
    public=read_public_rows(inputs.public_manifest);selected=public[:max_trials] if max_trials else public
    index=row_index(selected);rr=registry.verify(roi,'p29','raw')
    if rr.kind!='raw_cache' or rr.rows is None:raise ValueError('pixels require an indexed ROI cache')
    lookup=dict(zip(rr.rows.sample_ids,rr.rows.user_ids))
    if any(lookup.get(sid)!=user for sid,user in zip(index.sample_ids,index.user_ids)):
        raise ValueError('pixel ROI sample/user IDs mismatch')
    output=Path(output).resolve()
    if not output.is_relative_to(protocol.run_root):raise ValueError('pixel output outside run')
    output.mkdir(parents=True,exist_ok=True)
    config={'partition':inputs.partition.name,'schema':1,'frames':16,'resolution':160,
        'views':['scene','person','workspace'],'source_time_basis':'recorded wall clock; UTC convention; unknown NaN',
        'producer_sha256':sha256_file(Path(__file__))}
    identity=canonical_hash({'protocol':protocol.identity(),'roi':roi,'rows':index,'config':config})
    receipt=output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text())!={'identity':identity}:raise ValueError('pixel resume identity drift')
    write_json(receipt,{'identity':identity})
    if (output/'artifact.json').exists():
        body=json.loads((output/'artifact.json').read_text());ref=ArtifactRef(Path(body['record_path']),body['sha256'])
        registry.verify(ref,'pixels','raw',index);return ref
    digest_receipt=output/'write-digests.json'
    if digest_receipt.exists():
        body=json.loads(digest_receipt.read_text(encoding='utf-8'))
        if body['identity']!=identity:raise ValueError('pixel writer receipt identity mismatch')
        required={str(output/(name+'.npy')) for name in PIXEL_KEYS}|{str(output/'rows.csv')}
        if set(body['files'])!=required:raise ValueError('pixel writer receipt file set mismatch')
        ref=registry.register(stage='pixels',kind='raw_cache',phase='raw',files=list(body['files']),
            file_digests=body['files'],parents=[roi,inputs.parents['manifest']],rows=index,config=config,
            source_files=[Path(__file__)],complete=rr.complete and len(selected)==len(public))
        write_json(output/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256});return ref
    n=len(selected);specs={'images':(np.uint8,(n,2,16,3,160,160)),
        'view_valid':(np.bool_,(n,2,16,3)),'view_quality':(np.float32,(n,2,16,3)),
        'source_frame_indices':(np.int64,(n,2,16)),'source_time_seconds':(np.float64,(n,2,16)),
        'completed':(np.bool_,(n,))}
    arrays={};digests={};files={}
    for name,(dtype,shape) in specs.items():
        path=output/(name+'.npy');files[name]=path
        array=np.load(path,mmap_mode='r+',allow_pickle=False) if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
        if array.shape!=shape or array.dtype!=dtype:raise ValueError('pixel resume array schema drift')
        arrays[name]=array
        with path.open('rb') as f:digests[name]=hashlib.sha256(f.read(array.offset))
    if arrays['completed'].any():
        raise RuntimeError('partial pixel cache has no final writer receipt; use a new output directory, not a full checksum rescan')
    roi_files={Path(f).stem:Path(f) for f in rr.files if f.endswith('.npz')};ops=None
    print(json.dumps({'event':'pixels_start','rows':n,'partition':inputs.partition.name}),flush=True)
    for i,row in enumerate(selected):
        if not arrays['completed'][i]:
            if row['ir_available']=='1' and ops is None:ops=_pixel_ops(protocol)
            values=_pixel_trial(row,roi_files.get(row['sample_id']),ops,registry,rr)
            for name,array in arrays.items():array[i]=values[name]
        # Only newly produced rows participate in the write-time digest.
        for name,array in arrays.items():digests[name].update(np.ascontiguousarray(array[i]).tobytes())
        if (i+1)%25==0 or i+1==n:
            for array in arrays.values():array.flush()
            print(json.dumps({'event':'pixels_progress','processed':i+1,'total':n,'seconds':round(time.perf_counter()-started,2)}),flush=True)
    validate_pixels(arrays,n)
    content=csv_bytes(PUBLIC_COLUMNS,selected);rows_path=output/'rows.csv';rows_path.write_bytes(content)
    hashes={str(files[name]):digest.hexdigest() for name,digest in digests.items()}
    hashes[str(rows_path)]=hashlib.sha256(content).hexdigest()
    write_json(digest_receipt,{'identity':identity,'files':hashes})
    ref=registry.register(stage='pixels',kind='raw_cache',phase='raw',files=[*files.values(),rows_path],
        file_digests=hashes,parents=[roi,inputs.parents['manifest']],rows=index,config=config,
        source_files=[Path(__file__)],complete=rr.complete and len(selected)==len(public))
    write_json(output/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
    print(json.dumps({'event':'pixels_registered','seconds':round(time.perf_counter()-started,2)}),flush=True)
    return ref


def load_pixels(ref,*,protocol,complete=True):
    registry=ArtifactRegistry(protocol);record=registry.verify(ref,'pixels','raw')
    if record.rows is None or (complete and not record.complete):raise ValueError('pixel cache partial/incomplete')
    by_name={Path(p).name:Path(p) for p in record.files}
    path=by_name['rows.csv'];registry.verify_file(record,path)
    if row_index(read_public_rows(path))!=record.rows:raise ValueError('pixel row IDs drift')
    arrays={}
    for name in PIXEL_KEYS:
        path=by_name[name+'.npy']
        if name!='images':registry.verify_file(record,path)
        arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    validate_pixels(arrays,len(record.rows.sample_ids));return record,arrays

