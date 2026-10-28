"""Content-addressed records and experiment-role ancestry verification."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from .no_vote_protocol import NoVoteProtocol
from .no_vote_types import ArtifactRef, RowIndex, canonical_hash, freeze, write_json
from .teammate_source import sha256_file

FORBIDDEN_LABEL_FIELDS = frozenset({'class_id','class_name','action_name','label','labels',
    'correct','confusion','ground_truth','y_true','label_id'})
KINDS = frozenset({'public_weights','raw_cache','public_manifest','supervised_labels',
    'supervised_model','statistics','calibration','transition','predictions','adapted_model'})
LEARNED = frozenset({'supervised_model','statistics','calibration','transition'})
V3_DESCRIPTORS = frozenset({'source_ops', 'teacher_roster', 'source_transform', 'recording_metadata','visual_readiness'})


def _kind_allowed(protocol, kind):
    return kind in KINDS or (kind in {'descriptor','teacher_bank'} and protocol.recipe.get('source_contract_version') == 3)


class VerificationContext:
    """One stage's bounded verification state; never persist trust between stages.

    Cache keys include protocol, ref and resolved path. Stat changes invalidate
    trust. No context may grant cross-protocol registration or writable arrays.
    """
    def __init__(self, protocol):
        self.protocol=protocol
        self.identity=protocol.identity()
        self.validated_records={}
        self.direct_files={}
        self.loaded_arrays={}
        self.import_memo={}
        self.source_contexts={}
        self.public_rows={}

    def stamp(self,path):
        stat=Path(path).stat()
        return stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns

    def hash_file(self,path,expected=None):
        path=Path(path).resolve();key=(self.identity,str(path))
        stamp=self.stamp(path);cached=self.direct_files.get(key)
        if cached:
            if stamp!=cached[0]:raise ValueError('verified file stat drift')
            digest=cached[1]
        else:
            digest=sha256_file(path)
            self.direct_files[key]=(stamp,digest)
        if expected is not None and expected!=digest:raise ValueError('artifact file hash mismatch')
        return digest

    def trust_writer_digest(self,path,digest):
        self.direct_files[(self.identity,str(Path(path).resolve()))]=(self.stamp(path),digest)

    def load_arrays(self,path):
        path=Path(path).resolve();key=(self.identity,str(path));stamp=self.stamp(path)
        if key in self.loaded_arrays:
            saved,arrays=self.loaded_arrays[key]
            if saved!=stamp:raise ValueError('loaded array stat drift')
            return arrays
        if path.suffix=='.npy':
            arrays=np.load(path,mmap_mode='r',allow_pickle=False)
        else:
            with np.load(path,allow_pickle=False) as archive:
                arrays={k:archive[k] for k in archive.files}
            for array in arrays.values():array.flags.writeable=False
        self.loaded_arrays[key]=(stamp,arrays)
        return arrays


def reject_label_fields(value):
    if isinstance(value, Mapping):
        for key,item in value.items():
            if str(key).lower() in FORBIDDEN_LABEL_FIELDS:
                raise ValueError(f'forbidden label field: {key}')
            reject_label_fields(item)
    elif isinstance(value,(list,tuple)):
        for item in value:reject_label_fields(item)


def verify_public_file(path: Path):
    if path.suffix.lower()=='.json':
        reject_label_fields(json.loads(path.read_text(encoding='utf-8')))
    elif path.suffix.lower()=='.csv':
        with path.open(encoding='utf-8-sig',newline='') as f:
            names=csv.DictReader(f).fieldnames or []
        if set(n.lower() for n in names)&FORBIDDEN_LABEL_FIELDS:
            raise ValueError('public CSV contains label fields')
    elif path.suffix.lower()=='.npz':
        with np.load(path,allow_pickle=False) as data:
            if set(n.lower() for n in data.files)&FORBIDDEN_LABEL_FIELDS:
                raise ValueError('public NPZ contains label fields')
            if 'class_ids' in data and not np.array_equal(data['class_ids'],np.arange(40)):
                raise ValueError('class_ids must describe the 40 probability columns')


@dataclass(frozen=True)
class ArtifactRecord:
    stage: str
    kind: str
    phase: str
    protocol_sha256: str
    config: Mapping
    config_sha256: str
    files: Mapping[str,str]
    parents: tuple[ArtifactRef,...]
    fit_users: tuple[str,...]
    select_users: tuple[str,...]
    predict_users: tuple[str,...]
    adaptation_users: tuple[str,...]
    rows: RowIndex | None
    row_set_sha256: str
    source_hashes: Mapping[str,str]
    raw_input_hashes: Mapping[str,str]
    fixture: bool
    complete: bool
    verification_policy: str = 'full-v1'


class ArtifactRegistry:
    def __init__(self,protocol: NoVoteProtocol, *, context: VerificationContext | None=None):
        self.protocol=protocol
        self.root=protocol.run_root.resolve()
        self.identity=protocol.identity()
        if context is not None and context.identity!=self.identity:raise ValueError('verification context protocol mismatch')
        self.context=context

    def read(self,ref: ArtifactRef) -> ArtifactRecord:
        path=ref.record_path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('artifact record hash/location mismatch')
        key=(self.identity,ref.sha256,str(path))
        if self.context:
            self.context.hash_file(path,ref.sha256)
            if key in self.context.validated_records:return self.context.validated_records[key]
        elif sha256_file(path)!=ref.sha256:raise ValueError('artifact record hash/location mismatch')
        d=json.loads(path.read_text(encoding='utf-8'))
        d['parents']=tuple(ArtifactRef(Path(r['record_path']),r['sha256']) for r in d['parents'])
        if d['rows'] is not None:
            d['rows']=RowIndex(tuple(d['rows']['sample_ids']),tuple(d['rows']['user_ids']),tuple(d['rows']['class_ids']))
        for field in ('fit_users','select_users','predict_users','adaptation_users'):d[field]=tuple(d[field])
        record=ArtifactRecord(**d)
        if self.context:self.context.validated_records[key]=record
        return record

    def _roles(self,r: ArtifactRecord):
        p=self.protocol.partitions
        all_users=set(p['refit14'].users)|set(p['final4'].users)
        if any(not set(getattr(r,k))<=all_users for k in ('fit_users','select_users','predict_users','adaptation_users')):
            raise ValueError('unknown artifact users')
        if not _kind_allowed(self.protocol, r.kind):raise ValueError('unapproved artifact kind')
        if r.kind == 'descriptor':
            if (r.stage not in V3_DESCRIPTORS or r.phase != 'public' or r.rows is not None
                    or r.fit_users or r.select_users or r.predict_users or r.adaptation_users
                    or not r.complete or r.config.get('source_contract_version') != 3):
                raise ValueError('invalid v3 descriptor role')
            self._descriptor(r)
        if self.protocol.recipe.get('source_contract_version') == 3:
            from .no_vote_roles import validate_v3_role
            validate_v3_role(self,r)
        if (self.protocol.recipe.get('source_contract_version') == 3
                and r.kind in LEARNED | {'adapted_model'}
                and not self.protocol.recipe.get('source_ops_bindings', {}).get('formal_generation_ready', False)):
            from .visual_teacher_v3 import allows_visual_stage
            if not allows_visual_stage(self,r):
                raise ValueError('source operations/teacher producers are not ready for supervised generation')
        if r.kind in LEARNED:
            pop={'select':'train12','refit':'refit14'}.get(r.phase)
            if pop is None or set(r.fit_users)!=set(p[pop].users):
                raise ValueError('supervised fit population/phase mismatch')
            if r.adaptation_users:raise ValueError('supervised artifact cannot include adaptation users')
        elif r.fit_users:raise ValueError('non-supervised artifact has fit users')
        if r.select_users and (r.phase!='select' or set(r.select_users)!=set(p['development2'].users)):
            raise ValueError('selection ownership mismatch')
        if r.kind=='adapted_model':
            if r.phase!='adapt' or set(r.adaptation_users)!=set(p['final4'].users):
                raise ValueError('adaptation pool mismatch')
        elif r.adaptation_users:raise ValueError('non-adaptation artifact has target-fit users')
        if r.kind=='supervised_labels' and not set(r.predict_users)<=set(p['refit14'].users):
            raise ValueError('final labels cannot enter registry')
        if r.rows and (not set(r.rows.user_ids)<=all_users or set(r.predict_users)!=set(r.rows.user_ids)):
            raise ValueError('row user ownership mismatch')
        if r.kind=='predictions':
            population=r.config.get('partition')
            if population not in p or r.rows is None or not r.complete:
                raise ValueError('predictions require an explicit complete partition/IDs')
            from .no_vote_manifest import read_public_rows,row_index
            expected=row_index(read_public_rows(self.protocol.public_manifests[population]))
            if r.rows!=expected:raise ValueError('prediction IDs/class columns differ from prepared partition')

    def _descriptor(self,r):
        if r.stage=='visual_readiness':
            from .visual_teacher_v3 import validate_visual_capability
            validate_visual_capability(self,r)
            return
        if r.stage in {'recording_metadata','source_transform'}:
            from .no_vote_roles import validate_descriptor
            validate_descriptor(self,r)
            return
        if len(r.files)!=1 or Path(next(iter(r.files))).suffix!='.json':
            raise ValueError('descriptor requires exactly one JSON payload')
        path=Path(next(iter(r.files))).resolve()
        if not path.is_relative_to(self.root/'protocol'):
            raise ValueError('descriptor must be inside protocol')
        body=json.loads(path.read_text(encoding='utf-8'))
        if (body.get('schema_version')!=3 or body.get('training_executed') is not False
                or body.get('oof_enabled') is not False):
            raise ValueError('invalid source descriptor payload')
        bindings=self.protocol.recipe.get('source_ops_bindings',{})
        role='manifest' if r.stage=='source_ops' else 'roster'
        if bindings and (path!=Path(bindings[role+'_path']).resolve()
                or next(iter(r.files.values()))!=bindings[role+'_sha256']):
            raise ValueError('descriptor differs from protocol binding')
        if r.stage=='source_ops':
            expected=self.protocol.recipe['asset_bindings']['source_manifest_sha256']
            if (r.parents or body.get('source_manifest_sha256')!=expected
                    or not body.get('recipe') or not body.get('operators') or not body.get('consumed_files')):
                raise ValueError('source descriptor has invalid content or parent')
        else:
            if len(r.parents)!=1:
                raise ValueError('roster descriptor needs exactly one source parent')
            parent=self.read(r.parents[0])
            if parent.stage!='source_ops' or parent.kind!='descriptor' or len(parent.files)!=1:
                raise ValueError('roster descriptor has wrong source parent')
            if (body.get('source_ops_sha256')!=next(iter(parent.files.values()))
                    or body.get('retained_large_roots')!=['videomae_large']
                    or len(body.get('slots',[]))!=30 or type(body.get('formal_generation_ready')) is not bool):
                raise ValueError('roster descriptor source binding/content invalid')

    def verify(self,ref: ArtifactRef,expected_stage: str | None=None,
               expected_phase: str | None=None,expected_ids: RowIndex | None=None) -> ArtifactRecord:
        visiting=set();checked={}
        def walk(item):
            if item.sha256 in visiting:raise ValueError('artifact ancestor cycle')
            if item.sha256 in checked:return checked[item.sha256]
            visiting.add(item.sha256);r=self.read(item)
            if r.protocol_sha256!=self.identity or r.config_sha256!=canonical_hash(r.config):
                raise ValueError('artifact protocol/config hash mismatch')
            if r.fixture!=(self.protocol.recipe['execution_kind']=='fixture'):
                raise ValueError('fixture/formal artifact mismatch')
            self._roles(r)
            if r.row_set_sha256!=canonical_hash(r.rows):raise ValueError('artifact row-set hash mismatch')
            # Verify content-addressed ancestry metadata without reopening raw
            # payloads. Only the requested small outputs are consumed here;
            # raw/weight tables use verify_file for the selected payload.
            if item == ref and r.kind not in {'raw_cache','public_weights'}:
                for file in r.files:self.verify_file(r,Path(file))
            ancestors=[walk(parent) for parent in r.parents]
            def descendants(parent,seen=None):
                seen={} if seen is None else seen
                key=id(parent)
                if key in seen:return list(seen.values())
                seen[key]=parent
                for child in parent.parents:descendants(checked[child.sha256],seen)
                return list(seen.values())
            unique={}
            for parent in ancestors:descendants(parent,unique)
            lineage=list(unique.values())
            if r.complete and any(not a.complete for a in lineage):
                raise ValueError('complete artifact cannot consume partial ancestors')
            if r.kind in LEARNED:
                if any(a.kind in LEARNED and a.phase!=r.phase for a in lineage):
                    raise ValueError('supervised ancestor phase mismatch')
                if any(a.kind=='adapted_model' for a in lineage):raise ValueError('adapted ancestor in supervised fit')
            if r.kind=='predictions' and r.rows:
                supervised=[a for a in lineage if a.kind in LEARNED]
                required={'A1':{'A1'},'A2':{'A2'},'A5-S':{'A5'},'A5-I':{'A5'},
                    'A6-VS':{'A2','A5'},'A6-VI':{'A2','A5'},'A6-VSI':{'A2','A5'},
                    'A7':{'A7'},'A7-mask':{'A7-mask'},'A7-shuffle-S':{'A7-shuffle-S'},
                    'A7-shuffle-I':{'A7-shuffle-I'},'A7-zero-S':{'A7'},'A7-zero-I':{'A7'},'A8':{'A7'}}
                if self.protocol.recipe.get('source_contract_version') == 3:
                    from .no_vote_roles import verify_prediction_ancestry
                    verify_prediction_ancestry(self,r,lineage)
                needed=required.get(r.stage)
                if needed is not None and not needed<={a.stage for a in supervised}:
                    raise ValueError('prediction missing required supervised ancestor')
                if (needed is None and self.protocol.recipe.get('source_contract_version') != 3
                        and not any(a.kind=='adapted_model' and a.stage==r.stage for a in lineage)):
                    raise ValueError('prediction missing adapted ancestor')
                population=r.config['partition']
                if population in {'train12','development2'}:
                    if any(a.phase=='refit' for a in supervised):raise ValueError('development ancestor is refit')
                if population in {'refit14','final4'}:
                    if any(a.phase=='select' for a in supervised):raise ValueError('final ancestor is select')
            if r.kind=='adapted_model':
                bases=[(parent,a) for parent,a in zip(r.parents,ancestors) if a.stage=='A7' and a.kind=='supervised_model' and a.phase=='refit']
                targets=[a for a in ancestors if a.stage=='A8' and a.kind=='predictions']
                if len(bases)!=1 or len(targets)!=1:raise ValueError('adaptation requires own refit A7 and A8')
                if targets[0].phase!='predict' or targets[0].config.get('partition')!='final4' or not targets[0].complete:
                    raise ValueError('A8 adaptation targets must be complete final4 predictions')
                target_lineage=descendants(targets[0])
                if not any(a==bases[0][1] for a in target_lineage):raise ValueError('A8 does not descend from the same A7')
            visiting.remove(item.sha256);checked[item.sha256]=r;return r
        record=walk(ref)
        if expected_stage is not None and record.stage!=expected_stage:raise ValueError('artifact stage mismatch')
        if expected_phase is not None and record.phase!=expected_phase:raise ValueError('artifact phase mismatch')
        if expected_ids is not None and record.rows!=expected_ids:raise ValueError('artifact IDs/class columns mismatch')
        return record

    def verify_file(self,record: ArtifactRecord,path: Path) -> Path:
        """Check one actually consumed file, never ancestor payloads."""
        path=Path(path).resolve()
        digest=record.files.get(str(path))
        if digest is None or not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('artifact file hash/location mismatch')
        provenance=record.config.get('import_provenance',{})
        if record.kind in {'raw_cache','public_weights'} and provenance:
            from .no_vote_cache_import import verify_imported_file
            verify_imported_file(record,path)
            if self.context:self.context.trust_writer_digest(path,digest)
        elif self.context:self.context.hash_file(path,digest)
        elif sha256_file(path)!=digest:raise ValueError('artifact file hash/location mismatch')
        if record.kind not in {'supervised_labels','supervised_model','adapted_model'}:
            verify_public_file(path)
        return path

    def register(self,*,stage: str,kind: str,phase: str,files,parents=(),fit_users=(),
                 select_users=(),predict_users=(),adaptation_users=(),rows=None,config=None,
                 raw_inputs=(),source_files=(),complete=True,file_digests=None) -> ArtifactRef:
        if not _kind_allowed(self.protocol, kind):raise ValueError('unapproved artifact kind')
        paths=[Path(p).resolve() for p in files]
        if len(set(paths))!=len(paths) or any(not p.is_file() or not p.is_relative_to(self.root) for p in paths):
            raise ValueError('artifact files must be unique and within the run')
        for p in paths:
            if kind not in {'supervised_labels','supervised_model','adapted_model'}:verify_public_file(p)
        config={} if config is None else config
        if file_digests is None:
            digests={str(p):(self.context.hash_file(p) if self.context else sha256_file(p)) for p in paths}
        else:
            # Producers may hash a large cache in the same pass that writes
            # it. This avoids an extra full cache read merely to register it.
            digests={str(Path(p).resolve()):d for p,d in file_digests.items()}
            if set(digests)!={str(p) for p in paths} or any(not isinstance(d,str) or not re.fullmatch('[0-9a-f]{64}',d) for d in digests.values()):
                raise ValueError('writer file digest coverage/format mismatch')
            if self.context:
                for p in paths:self.context.trust_writer_digest(p,digests[str(p)])
        own_sources=tuple(source_files) or (Path(__file__),)
        hash_file=self.context.hash_file if self.context else sha256_file
        record=ArtifactRecord(stage,kind,phase,self.identity,config,canonical_hash(config),
            digests,tuple(parents),tuple(sorted(fit_users)),tuple(sorted(select_users)),
            tuple(sorted(predict_users or (set(rows.user_ids) if rows else ()))),tuple(sorted(adaptation_users)),
            rows,canonical_hash(rows),{str(Path(p).resolve()):hash_file(Path(p)) for p in own_sources},
            {str(Path(p).resolve()):sha256_file(Path(p)) for p in raw_inputs},
            self.protocol.recipe['execution_kind']=='fixture',bool(complete),verification_policy='direct-metadata-v2')
        path=self.root/'artifacts'/(canonical_hash(record)+'.json')
        if not path.exists():write_json(path,record)
        ref=ArtifactRef(path,hash_file(path))
        try:self.verify(ref,stage,phase,rows)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return ref
