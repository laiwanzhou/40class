"""Trusted preparation is separate from label-free generation readers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

from .no_vote_protocol import NoVoteProtocol,PARTITIONS
from .no_vote_types import ArtifactRef,RowIndex,StageInputs,canonical_hash,write_json
from .teammate_source import sha256_file
from .artifact_record import ArtifactRegistry

PUBLIC_COLUMNS=('sample_id','user_id','ir_path','depth_path','skeleton_path','imu_path',
                'ir_available','depth_available','skeleton_available','imu_available')
NAMESPACE='cuhkx-fixed-split-v2:'


def csv_bytes(columns,rows):
    out=io.StringIO(newline='')
    writer=csv.DictWriter(out,fieldnames=columns,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return out.getvalue().encode('utf-8')


def read_public_rows(path: Path) -> list[dict[str,str]]:
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if tuple(reader.fieldnames or ())!=PUBLIC_COLUMNS:raise ValueError('public manifest whitelist mismatch')
        rows=list(reader)
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)) or any(len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in ids):
        raise ValueError('public manifest requires unique opaque IDs')
    for row in rows:
        if None in row or any(row[m+'_available'] not in {'0','1'} for m in ('ir','depth','skeleton','imu')):
            raise ValueError('invalid public row')
    return rows


def row_index(rows) -> RowIndex:
    return RowIndex(tuple(r['sample_id'] for r in rows),tuple(r['user_id'] for r in rows),tuple(range(40)))


def _available(path: Path | None,modality: str) -> bool:
    if path is None or not path.is_dir():return False
    directory=path/'predictions' if modality=='skeleton' else path
    if not directory.is_dir():return False
    suffixes={'.json'} if modality=='skeleton' else {'.png','.jpg','.jpeg'} if modality in {'ir','depth'} else {'.csv'}
    return any(p.is_file() and p.suffix.lower() in suffixes for p in directory.iterdir())


def prepare_inputs(source: Path,protocol: NoVoteProtocol,public_output: Path,
                   private_output: Path,*,data_root: Path | None=None) -> Mapping[str,Path]:
    source,public_output,private_output=Path(source).resolve(),Path(public_output).resolve(),Path(private_output).resolve()
    if public_output!=protocol.run_root/'protocol' or private_output.is_relative_to(protocol.run_root):
        raise ValueError('public output must match protocol; private labels must be outside the run')
    data_root=Path(data_root).resolve() if data_root else source.parent.parent.parent/'datasets/Small-Model-Track/train'
    with source.open(encoding='utf-8-sig',newline='') as f:original=list(csv.DictReader(f))
    expected_users=set(protocol.partitions['refit14'].users)|set(protocol.partitions['final4'].users)
    if {r['user_id'] for r in original}!=expected_users:raise ValueError('canonical source user population mismatch')
    public=[];labels={};mapping=[];old_ids=set()
    for row in original:
        old=row['sample_id']
        if old in old_ids:raise ValueError('duplicate canonical sample ID')
        old_ids.add(old)
        sid=hashlib.sha256((NAMESPACE+old).encode()).hexdigest()
        label=int(row['class_id'])
        if label not in range(40):raise ValueError('source class outside 0..39')
        labels[sid]=label;mapping.append({'sample_id':sid,'original_sample_id':old})
        out={'sample_id':sid,'user_id':row['user_id']}
        for modality,key in (('ir','ir_path'),('depth','depth_color_path'),('skeleton','skeleton_path'),('imu','imu_path')):
            value=row.get(key,'').strip();path=Path(value) if value else None
            if path is not None and not path.is_absolute():path=data_root/path
            out[modality+'_path']=str(path.resolve()) if path else ''
            out[modality+'_available']=str(int(_available(path,modality)))
        public.append(out)
    if len({r['sample_id'] for r in public})!=len(public):raise ValueError('opaque ID collision')
    public.sort(key=lambda r:r['sample_id'])
    outputs={};private={};parts={}
    for name,part in protocol.partitions.items():
        rows=[r for r in public if r['user_id'] in part.users]
        if len(rows)!=part.expected_rows:raise ValueError(f'canonical row count mismatch: {name}')
        parts[name]=rows
        path=protocol.public_manifests[name]
        if path.parent!=public_output:raise ValueError('configured public manifest location mismatch')
        outputs[path]=csv_bytes(PUBLIC_COLUMNS,rows)
        truth=[{'sample_id':r['sample_id'],'class_id':labels[r['sample_id']]} for r in rows]
        if name=='final4':private[private_output/'final_labels.csv']=csv_bytes(('sample_id','class_id'),truth)
        else:
            if name in {'train12','refit14'} and {r['class_id'] for r in truth}!=set(range(40)):
                raise ValueError(f'fit population lacks 40-class support: {name}')
            outputs[protocol.supervised_labels[name]]=csv_bytes(('sample_id','class_id'),truth)
    private[private_output/'id_mapping.csv']=csv_bytes(('sample_id','original_sample_id'),mapping)
    state_path=protocol.run_root/'run_state.json'
    if state_path.exists() and json.loads(state_path.read_text()).get('state')!='prepared':
        raise ValueError('cannot rerun trusted preparation after generation starts')
    for path,content in {**outputs,**private}.items():
        if path.exists() and path.read_bytes()!=content:raise ValueError('prepared input drift')
    for path,content in {**outputs,**private}.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():path.write_bytes(content)
    registry=ArtifactRegistry(protocol);refs={}
    for name,rows in parts.items():
        ref=registry.register(stage='manifest_'+name,kind='public_manifest',phase='raw',
            files=[protocol.public_manifests[name]],rows=row_index(rows),config={'schema':1},source_files=[Path(__file__)])
        refs[name]={'record_path':str(ref.record_path),'sha256':ref.sha256}
    receipt={'protocol_sha256':protocol.identity(),'partitions':refs,
             'supervised_label_hashes':{name:sha256_file(path) for name,path in protocol.supervised_labels.items()}}
    write_json(public_output/'prepared_inputs.json',receipt)
    write_json(state_path,{'protocol_sha256':protocol.identity(),'state':'prepared',
                          'prepared_inputs_sha256':sha256_file(public_output/'prepared_inputs.json')})
    return {name:protocol.public_manifests[name] for name in PARTITIONS}


def load_stage_inputs(protocol: NoVoteProtocol,partition: str,phase: str) -> StageInputs:
    if partition not in PARTITIONS or phase not in {'raw','select','refit','predict','adapt'}:
        raise ValueError('unknown partition/phase')
    if partition=='final4' and phase not in {'raw','predict','adapt'}:
        raise ValueError('final inputs cannot be used for supervised selection/refit')
    if phase=='refit' and partition!='refit14':raise ValueError('refit needs refit14')
    if phase=='select' and partition not in {'train12','development2'}:raise ValueError('select ownership mismatch')
    state=json.loads((protocol.run_root/'run_state.json').read_text(encoding='utf-8'))
    if state['protocol_sha256']!=protocol.identity():raise ValueError('prepared protocol changed')
    if state['state']=='revealed':raise ValueError('generation forbidden after final label reveal')
    receipt_path=protocol.run_root/'protocol/prepared_inputs.json'
    if sha256_file(receipt_path)!=state['prepared_inputs_sha256']:raise ValueError('prepared receipt hash drift')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    ref=ArtifactRef(
        Path(receipt['partitions'][partition]['record_path']),receipt['partitions'][partition]['sha256'])
    rows=read_public_rows(protocol.public_manifests[partition])
    ArtifactRegistry(protocol).verify(ref,'manifest_'+partition,'raw',row_index(rows))
    labels=protocol.supervised_labels.get(partition) if phase in {'select','refit'} else None
    if labels and sha256_file(labels)!=receipt['supervised_label_hashes'][partition]:raise ValueError('supervised labels drift')
    return StageInputs(protocol.partitions[partition],protocol.public_manifests[partition],labels,{'manifest':ref})
