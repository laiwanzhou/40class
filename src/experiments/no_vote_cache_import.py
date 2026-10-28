"""Same-volume, read-only import of label-independent caches into a new DAG."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import time

import numpy as np

from .artifact_record import ArtifactRegistry, VerificationContext
from .no_vote_manifest import PUBLIC_COLUMNS, csv_bytes, read_public_rows, row_index
from .no_vote_types import ArtifactRef, RowIndex, canonical_bytes, canonical_hash, write_json
from .teammate_source import sha256_file

RAW_STAGES=frozenset({'p28','p29','visual_features','pixels'})
WEIGHT_STAGES={'pose_initializer':'yolo','visual_initializer':'videomae','mc3_initializer':'mc3'}


def immutable_bytes(path,content):
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=content:raise ValueError('immutable prepared input drift')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(content)


def prepare_from_existing(source_protocol,destination_protocol,*,context=None):
    """Copy four public CSVs and three allowed labels. Never open final truth/raw."""
    src,dst=source_protocol,destination_protocol
    if dst.recipe.get('source_contract_version')!=3 or src.run_root==dst.run_root:
        raise ValueError('preparation needs a distinct v3 run')
    if src.partitions!=dst.partitions:raise ValueError('preparation partition ownership mismatch')
    context=context or VerificationContext(dst);registry=ArtifactRegistry(dst,context=context)
    old=ArtifactRegistry(src,context=VerificationContext(src))
    receipt_path=src.run_root/'protocol/prepared_inputs.json'
    state=json.loads((src.run_root/'run_state.json').read_text(encoding='utf-8'))
    if state['protocol_sha256']!=src.identity() or state['prepared_inputs_sha256']!=sha256_file(receipt_path):
        raise ValueError('source prepared receipt drift')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    public={};labels={};refs={}
    for name,partition in dst.partitions.items():
        entry=receipt['partitions'][name];ref=ArtifactRef(Path(entry['record_path']),entry['sha256'])
        rows=read_public_rows(src.public_manifests[name]);index=row_index(rows)
        old.verify(ref,'manifest_'+name,'raw',index)
        if len(rows)!=partition.expected_rows or set(index.user_ids)!=set(partition.users):
            raise ValueError('source public IDs/users/count mismatch')
        public[name]=(rows,index)
        # Deliberately select names, not a glob or receipt's arbitrary label map.
        if name in {'train12','development2','refit14'}:
            path=src.supervised_labels[name]
            if sha256_file(path)!=receipt['supervised_label_hashes'][name]:raise ValueError('allowed labels drift')
            with path.open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if tuple(reader.fieldnames or ())!=('sample_id','class_id'):raise ValueError('allowed label schema mismatch')
                table=list(reader)
            lookup={x['sample_id']:int(x['class_id']) for x in table}
            if len(lookup)!=len(table) or set(lookup)!=set(index.sample_ids) or any(x not in range(40) for x in lookup.values()):
                raise ValueError('allowed label IDs/values mismatch')
            labels[name]=path.read_bytes()
    tr,dev,refit=(public[k][1] for k in ['train12','development2','refit14'])
    if set(tr.sample_ids)&set(dev.sample_ids) or set(refit.sample_ids)!=set(tr.sample_ids)|set(dev.sample_ids):
        raise ValueError('public partition IDs overlap or refit union mismatch')
    if set(public['final4'][1].sample_ids)&set(refit.sample_ids):raise ValueError('final IDs overlap')
    # Check all destinations before any mutation.
    writes={dst.public_manifests[k]:csv_bytes(PUBLIC_COLUMNS,v[0]) for k,v in public.items()}
    writes.update({dst.supervised_labels[k]:v for k,v in labels.items()})
    for path,content in writes.items():
        if not path.resolve().is_relative_to(dst.run_root):raise ValueError('prepared path outside destination')
        if path.exists() and path.read_bytes()!=content:raise ValueError('prepared input drift')
    for path,content in writes.items():immutable_bytes(path,content)
    for name,(_,index) in public.items():
        ref=registry.register(stage='manifest_'+name,kind='public_manifest',phase='raw',
            files=[dst.public_manifests[name]],rows=index,config={'schema':1},source_files=[Path(__file__)])
        refs[name]={'record_path':str(ref.record_path),'sha256':ref.sha256}
    body={'protocol_sha256':dst.identity(),'partitions':refs,
        'supervised_label_hashes':{k:sha256_file(dst.supervised_labels[k]) for k in labels},
        'provenance':{'source_protocol':src.identity(),'final_labels_read':False,'raw_scan':False}}
    path=dst.run_root/'protocol/prepared_inputs.json';immutable_bytes(path,canonical_bytes(body)+b'\n')
    immutable_bytes(dst.run_root/'run_state.json',canonical_bytes({'protocol_sha256':dst.identity(),
        'state':'prepared','prepared_inputs_sha256':sha256_file(path)})+b'\n')
    return refs


def _source_registry(source,context,source_protocol):
    identity=source_protocol.identity()
    if not source.record_path.is_relative_to(source_protocol.run_root):raise ValueError('source record outside source run')
    if identity not in context.source_contexts:context.source_contexts[identity]=VerificationContext(source_protocol)
    return ArtifactRegistry(source_protocol,context=context.source_contexts[identity])


def _eligible(record):
    if record.fit_users or record.select_users or record.adaptation_users or not record.complete:
        raise ValueError('cannot import learned or incomplete cache')
    if record.kind=='public_manifest' and record.stage.startswith('manifest_') and record.phase=='raw':return
    if record.kind=='public_weights' and record.stage in WEIGHT_STAGES and record.phase=='public':return
    if record.kind=='raw_cache' and record.stage in RAW_STAGES and record.phase=='raw':return
    raise ValueError('not an importable public cache/stage')


def _check_rows(record,rows,destination_protocol):
    if record.rows!=rows:raise ValueError('import exact IDs/users/class columns mismatch')
    if rows is None:return
    # All IDs must correspond to the current public population with exact users.
    lookup={}
    for name in ['refit14','final4']:
        lookup.update({r['sample_id']:r['user_id'] for r in read_public_rows(destination_protocol.public_manifests[name])})
    if any(lookup.get(s)!=u for s,u in zip(rows.sample_ids,rows.user_ids)):
        raise ValueError('import IDs/user not in destination manifests')


def _axes(record,context):
    """Headers, small ID/mask tables and a few samples; no whole pixel scan."""
    files={Path(p).name:Path(p) for p in record.files}
    n=len(record.rows.sample_ids) if record.rows else 0
    if record.stage=='pixels':
        from .pixel_cache import PIXEL_KEYS
        if set(files)!=({k+'.npy' for k in PIXEL_KEYS}|{'rows.csv'}):raise ValueError('pixel schema files mismatch')
        if record.config.get('views') not in (['scene','person','workspace'],('scene','person','workspace')):
            raise ValueError('pixel view axis mismatch')
        if record.config.get('frames')!=16 or record.config.get('resolution')!=160:raise ValueError('pixel shape config mismatch')
        if row_index(read_public_rows(files['rows.csv']))!=record.rows:raise ValueError('pixel IDs/user mismatch')
        shapes={'images':((n,2,16,3,160,160),np.uint8),'view_valid':((n,2,16,3),np.bool_),
            'view_quality':((n,2,16,3),np.float32),'source_frame_indices':((n,2,16),np.int64),
            'source_time_seconds':((n,2,16),np.float64),'completed':((n,),np.bool_)}
        for name,(shape,dtype) in shapes.items():
            a=context.load_arrays(files[name+'.npy'])
            if a.shape!=shape or a.dtype!=dtype:raise ValueError('pixel axis shape/dtype mismatch')
            if name=='completed' and not a.all():raise ValueError('pixel completion mask mismatch')
    elif record.stage=='visual_features':
        if record.config.get('backbone_frozen') is not True:raise ValueError('features not public frozen backbone')
        paths=[p for p in files.values() if p.name=='features.npz' or p.name.endswith('_features.npz')]
        if len(paths)!=1:raise ValueError('features schema files mismatch')
        arrays=context.load_arrays(paths[0])
        from .visual_teacher import validate_feature_arrays
        validate_feature_arrays(arrays,record.rows,bounded=True)
    elif record.stage in {'p28','p29'}:
        if not record.rows:raise ValueError('geometry requires IDs')
        npzs={p.stem:p for p in files.values() if p.suffix=='.npz'}
        if not set(npzs)<=set(record.rows.sample_ids) or not npzs:raise ValueError('geometry cache IDs mismatch')
        if set(npzs)!=set(record.rows.sample_ids):raise ValueError('geometry missing IDs')
    elif record.kind=='public_weights':
        # Public core assets must be the same verified initializer receipt.
        if record.parents:raise ValueError('public initializer cannot have learned parents')


def import_public_cache(source: ArtifactRef,destination_protocol,rows: RowIndex | None,
                        context: VerificationContext,*,source_protocol=None)->ArtifactRef:
    """Re-register only allowed public ancestors; old refs remain provenance.

    source_protocol is explicit on first use; subsequent calls may use the
    stage context's already registered source protocol (never path inference).
    """
    if context.identity!=destination_protocol.identity():raise ValueError('import context protocol mismatch')
    if destination_protocol.recipe.get('source_contract_version')!=3:raise ValueError('import needs v3 protocol')
    if source_protocol is None:
        candidates=[c.protocol for c in context.source_contexts.values() if source.record_path.is_relative_to(c.protocol.run_root)]
        if len(candidates)!=1:raise ValueError('explicit verified source protocol required')
        source_protocol=candidates[0]
    old=_source_registry(source,context,source_protocol);new=ArtifactRegistry(destination_protocol,context=context)
    if source_protocol.recipe['asset_bindings']['source_manifest_sha256']!=destination_protocol.recipe['asset_bindings']['source_manifest_sha256']:
        raise ValueError('import source snapshot recipe mismatch')
    root=old.verify(source);_check_rows(root,rows,destination_protocol)
    planned={};visiting=set()
    def preflight(ref):
        if ref.sha256 in visiting:raise ValueError('import ancestor cycle')
        if ref.sha256 in planned:return
        visiting.add(ref.sha256);record=old.read(ref);_eligible(record)
        if record.protocol_sha256!=source_protocol.identity():raise ValueError('source protocol mismatch')
        _check_rows(record,record.rows,destination_protocol)
        for parent in record.parents:preflight(parent)
        _axes(record,old.context)
        if record.kind=='public_weights':
            role=WEIGHT_STAGES[record.stage]
            receipts=json.loads((destination_protocol.run_root/'protocol/weights_manifest.json').read_text())
            expected={x['sha256'] for x in receipts['weights'][role]['files']}
            if set(record.files.values())!=expected:raise ValueError('import public initializer digest binding mismatch')
        # Stat every linked file to fail before a partial multi-GB bridge.
        for file in record.files:
            p=Path(file)
            if not p.is_relative_to(source_protocol.run_root) or not p.is_file():raise ValueError('missing import file')
            if p.stat().st_dev!=destination_protocol.run_root.stat().st_dev:raise ValueError('same-volume hard link required')
        planned[ref.sha256]=record;visiting.remove(ref.sha256)
    preflight(source)
    def bridge(ref):
        key=(source_protocol.identity(),ref.sha256,str(ref.record_path),context.identity)
        if key in context.import_memo:return context.import_memo[key]
        record=planned[ref.sha256]
        if record.kind=='public_manifest':
            name=record.stage.removeprefix('manifest_')
            if record.rows!=row_index(read_public_rows(destination_protocol.public_manifests[name])):
                raise ValueError('import manifest IDs mismatch')
            receipt=json.loads((destination_protocol.run_root/'protocol/prepared_inputs.json').read_text())
            entry=receipt['partitions'][name];result=ArtifactRef(Path(entry['record_path']),entry['sha256'])
        else:
            parents=[bridge(x) for x in record.parents];files={};provenance={}
            for file,digest in record.files.items():
                path=Path(file);dest=destination_protocol.run_root/'imports'/path.relative_to(source_protocol.run_root)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():
                    if not os.path.samefile(path,dest):raise ValueError('immutable imported path collision')
                else:
                    try:os.link(path,dest)
                    except OSError as e:raise ValueError('hard link unavailable; no copy fallback') from e
                files[str(dest)]=digest
                stat=path.stat()
                provenance[str(dest)]={'path':str(path),'sha256':digest,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            cfg={**record.config,'import_provenance':{'source_protocol':source_protocol.identity(),
                'source_ref_path':str(ref.record_path),'source_ref_sha256':ref.sha256,
                'validation':'inherited_digest_header_ids_masks_only','files':provenance},'immutable':True}
            result=new.register(stage=record.stage,kind=record.kind,phase=record.phase,files=list(files),
                file_digests=files,parents=parents,rows=record.rows,config=cfg,source_files=[Path(__file__)])
        context.import_memo[key]=result
        return result
    return bridge(source)


def verify_imported_file(record,path):
    entry=record.config.get('import_provenance',{}).get('files',{}).get(str(path))
    if (not record.config.get('immutable') or entry is None or entry['sha256']!=record.files[str(path)]
            or path.stat().st_size!=entry['bytes'] or path.stat().st_mtime_ns!=entry['mtime_ns']
            or not os.path.samefile(path,entry['path'])):
        raise ValueError('immutable import provenance/location drift')


def assert_writable_output(path,protocol):
    """All v3 producers must call before resume/open_memmap/r+ or overwrite."""
    path=Path(path).resolve()
    if path.is_relative_to(protocol.run_root/'imports'):
        raise ValueError('immutable imported cache cannot be overwritten or opened r+')
