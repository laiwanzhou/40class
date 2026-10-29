"""Task11 label-free adapters around the original anonymous-date session ops."""
from dataclasses import dataclass
from datetime import date
import csv
from pathlib import Path
import sys

import numpy as np

from .artifact_record import ArtifactRegistry
from .no_vote_manifest import RECORDING_COLUMNS,read_public_rows,row_index
from .teammate_source import verify_teammate_source,load_teammate_symbol


def source_session_ops(protocol,context):
    key=(protocol.identity(),'task11_source_sessions')
    if key not in context.public_rows:
        report=verify_teammate_source(protocol.source_root,protocol.source_root.parent/'source_manifest.json',
            expected_sha256=protocol.recipe['asset_bindings']['source_manifest_sha256'],verify_contents=False)
        build=load_teammate_symbol(report,'audit_p87_sequence_decoder','build_sessions')
        repeat=load_teammate_symbol(report,'p89_global_repeat_decoder','date_session_lists')
        cluster=load_teammate_symbol(report,'p89_global_repeat_decoder','cluster_sessions')
        context.public_rows[key]={'build_sessions':build,'date_session_lists':repeat,'cluster_sessions':cluster,
            'metadata_class':sys.modules['audit_p87_sequence_decoder'].RecordingMetadata}
    return context.public_rows[key]


def validate_recording_rows(index,rows):
    lookup={x['sample_id']:x for x in rows}
    if len(lookup)!=len(rows) or set(lookup)!=set(index.sample_ids):raise ValueError('recording exact IDs/duplicates mismatch')
    ordered=[]
    for sid,user in zip(index.sample_ids,index.user_ids):
        row=lookup[sid]
        if set(row)!=set(RECORDING_COLUMNS) or row['user_id']!=user or row['partition'] not in {'train12','development2','refit14','final4'}:
            raise ValueError('recording whitelist/user/partition mismatch')
        try:value=float(row['start_seconds'])
        except (TypeError,ValueError) as error:raise ValueError('recording time is not numeric') from error
        if row['timestamp_available']=='1':
            date.fromisoformat(row['recording_date'])
            if not np.isfinite(value) or not 0<=value<86400 or row['time_source']!='acquisition_ids':
                raise ValueError('known recording time basis mismatch')
        elif (row['timestamp_available']!='0' or row['recording_date'] or not np.isnan(value)
                or row['time_source']!='unknown'):
            raise ValueError('unknown recording time must remain unknown')
        ordered.append(row)
    return ordered


@dataclass(frozen=True)
class SessionBlocks:
    sessions: tuple
    unknown: np.ndarray
    blocks: tuple


def _local_blocks(index,rows,ops):
    ordered=validate_recording_rows(index,rows);groups={};unknown=[]
    for i,x in enumerate(ordered):
        if x['timestamp_available']=='0':unknown.append(i);continue
        groups.setdefault((x['partition'],x['user_id'],x['recording_date']),[]).append(i)
    result=[]
    for indices in groups.values():
        pos=np.asarray(indices,np.int64)
        meta=ops['metadata_class'](sample_ids=np.asarray([index.sample_ids[i] for i in pos]),
            users=np.asarray([index.user_ids[i] for i in pos]),dates=np.asarray([ordered[i]['recording_date'] for i in pos]),
            starts=np.asarray([float(ordered[i]['start_seconds']) for i in pos]))
        result.append((pos,meta))
    return result,np.asarray(unknown,np.int64)


def session_blocks(index,rows,ops,*,gap_seconds=30.):
    if not np.isfinite(gap_seconds) or gap_seconds<=0:raise ValueError('positive session gap required')
    blocks,unknown=_local_blocks(index,rows,ops);sessions=[]
    for pos,meta in blocks:
        sessions.extend(pos[x] for x in ops['build_sessions'](np.arange(len(pos)),meta,gap_seconds,'anonymous_date'))
    return SessionBlocks(tuple(sessions),unknown,tuple(pos for pos,_ in blocks))


def repeat_session_lists(index,rows,ops,*,short_gap=30.):
    if not np.isfinite(short_gap) or short_gap<=0:raise ValueError('positive repeat gap required')
    blocks,_=_local_blocks(index,rows,ops);groups=[]
    for pos,meta in blocks:
        groups.extend([pos[x] for x in group] for group in ops['date_session_lists'](np.arange(len(pos)),meta,short_gap))
    return groups


def load_recording_rows(protocol,partition,metadata,context):
    registry=ArtifactRegistry(protocol,context=context);record=registry.verify(metadata,'recording_metadata','public')
    if record.config.get('partition')!=partition or len(record.files)!=1:raise ValueError('recording descriptor partition mismatch')
    index=row_index(read_public_rows(protocol.public_manifests[partition]));path=Path(next(iter(record.files)))
    with registry.verify_file(record,path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=RECORDING_COLUMNS:raise ValueError('recording schema mismatch')
        rows=list(reader)
    if any(x['partition']!=partition for x in rows):raise ValueError('recording row partition mismatch')
    return index,validate_recording_rows(index,rows)


def build_session_blocks(index,metadata,*,protocol,context,gap_seconds=30.):
    record=ArtifactRegistry(protocol,context=context).read(metadata);partition=record.config.get('partition')
    expected,rows=load_recording_rows(protocol,partition,metadata,context)
    if expected!=index:raise ValueError('session adapter exact partition IDs required')
    return session_blocks(index,rows,source_session_ops(protocol,context),gap_seconds=gap_seconds)
