"""Public Thermal discovery for the authorized approximate reconstruction."""
from pathlib import Path
import hashlib

from .no_vote_cache_import import immutable_bytes
from .no_vote_manifest import EXPERT_PUBLIC_COLUMNS,csv_bytes,read_public_rows,read_expert_public_rows,row_index


def thermal_public_rows(public_rows,train_root):
    root=Path(train_root).resolve();result=[];seen=set()
    for row in public_rows:
        sid=row['sample_id'];user=row['user_id']
        if sid in seen:raise ValueError('duplicate public sample ID')
        seen.add(sid);keys=[]
        for name in ['ir_path','depth_path','skeleton_path','imu_path']:
            if not row.get(name):continue
            path=Path(row[name]).resolve()
            if not path.is_relative_to(root):raise ValueError('public modality path outside explicit train root')
            parts=path.relative_to(root).parts
            if len(parts)<4 or parts[0] not in {'IR','Depth_Color','Skeleton','IMU'}:
                raise ValueError('unknown public trial layout')
            # Folder tokens locate the same raw trial only. No action token is
            # parsed into a label, feature, class index, or predictor input.
            if parts[2]!=user:raise ValueError('public trial user mismatch')
            keys.append(tuple(parts[1:4]))
        if not keys:
            result.append({'sample_id':sid,'user_id':user,'thermal_path':'','thermal_available':'unknown'})
            continue
        if len(set(keys))!=1:raise ValueError('public trial paths disagree')
        trial=(root/'Thermal').joinpath(*keys[0]);available=False
        if trial.is_dir():
            available=any(p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'} for p in trial.iterdir())
        result.append({'sample_id':sid,'user_id':user,'thermal_path':str(trial),'thermal_available':str(int(available))})
    return result


def build_thermal_public_sidecars(protocol,train_root,output):
    output=Path(output).resolve()
    if not output.is_relative_to(protocol.run_root/'protocol'):raise ValueError('Thermal sidecars must stay within run protocol')
    tables={};shared={}
    for partition,path in protocol.public_manifests.items():
        public=read_public_rows(path);observations=thermal_public_rows(public,train_root)
        rows=[{**x,'thermal_available':'0' if x['thermal_available']=='unknown' else x['thermal_available']} for x in observations]
        for row in observations:
            if row['sample_id'] in shared and row!=shared[row['sample_id']]:raise ValueError('Thermal shared-ID mapping drift')
            shared[row['sample_id']]=row
        dest=output/(partition+'.csv');content=csv_bytes(EXPERT_PUBLIC_COLUMNS,rows);immutable_bytes(dest,content)
        observed=output/(partition+'_observations.csv');observed_content=csv_bytes(EXPERT_PUBLIC_COLUMNS,observations)
        immutable_bytes(observed,observed_content)
        read_expert_public_rows(dest,row_index(public))
        tables[partition]={'path':str(dest),'sha256':hashlib.sha256(content).hexdigest(),'rows':len(rows),
            'observations_path':str(observed),'observations_sha256':hashlib.sha256(observed_content).hexdigest(),
            'available':sum(x['thermal_available']=='1' for x in observations),
            'known_absent':sum(x['thermal_available']=='0' for x in observations),
            'unknown':sum(x['thermal_available']=='unknown' for x in observations),
            'unknown_policy':'retain_unknown_observation_no_usable_input_no_label_based_id_reversal',
            'coverage_basis':'image_filename_exists_not_decode_validation'}
    return tables
