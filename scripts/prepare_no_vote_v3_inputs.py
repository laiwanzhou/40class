"""Task2 public-cache bridge; no raw scan, final labels, training or source import."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.artifact_record import ArtifactRegistry,VerificationContext
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_cache_import import prepare_from_existing,import_public_cache,immutable_bytes
from src.experiments.no_vote_manifest import load_stage_inputs,build_recording_metadata
from src.experiments.no_vote_types import ArtifactRef,canonical_bytes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--source-config',required=True,type=Path)
    parser.add_argument('--cache-refs',required=True,type=Path,help='explicit JSON list of source ArtifactRef objects')
    args=parser.parse_args();started=time.perf_counter()
    old=load_protocol(args.source_config);new=load_protocol(args.config);context=VerificationContext(new)
    # Explicit refs only, never scan historical learned artifacts to find caches.
    refs=json.loads(args.cache_refs.read_text(encoding='utf-8'))
    if not isinstance(refs,list):raise ValueError('--cache-refs requires an explicit list')
    sources=[ArtifactRef(Path(x['record_path']),x['sha256']) for x in refs]
    prepare_from_existing(old,new,context=context)
    print(json.dumps({'event':'public_prepared','seconds':round(time.perf_counter()-started,2)}),flush=True)
    registry=ArtifactRegistry(new,context=context);source_registry=ArtifactRegistry(old,context=VerificationContext(old))
    imported=[];acquisition={}
    for i,ref in enumerate(sources):
        record=source_registry.read(ref)
        dest=import_public_cache(ref,new,record.rows,context,source_protocol=old)
        item=registry.read(dest)
        imported.append({'stage':item.stage,'partition':item.config.get('partition'),
            'record_path':str(dest.record_path),'sha256':dest.sha256,'rows':len(item.rows.sample_ids) if item.rows else 0})
        print(json.dumps({'event':'cache_imported','processed':i+1,'total':len(sources),**imported[-1],
            'seconds':round(time.perf_counter()-started,2)}),flush=True)
    # P28 can be a shared ancestor of imported P29/features/pixels.
    for dest in context.import_memo.values():
        record=registry.read(dest)
        if record.stage=='p28' and record.rows is not None:
            for name,part in new.partitions.items():
                if set(part.users)<=set(record.rows.user_ids):acquisition[name]=dest
    metadata={}
    for name,ref in acquisition.items():
        dest=build_recording_metadata(load_stage_inputs(new,name,'raw',context=context),ref,protocol=new,context=context)
        metadata[name]={'record_path':str(dest.record_path),'sha256':dest.sha256}
        print(json.dumps({'event':'recording_metadata_registered','partition':name,
            'seconds':round(time.perf_counter()-started,2)}),flush=True)
    body={'protocol_sha256':new.identity(),'imports':imported,'recording_metadata':metadata,
        'training_executed':False,'final_labels_read':False,'raw_scan':False,
        'formal_generation_ready':bool(new.recipe['source_ops_bindings']['formal_generation_ready'])}
    immutable_bytes(new.run_root/'protocol/task2_prepared.json',canonical_bytes(body)+b'\n')
    print(json.dumps({'event':'task2_prepared','imports':len(imported),'metadata_partitions':list(metadata),
        'seconds':round(time.perf_counter()-started,2),'training_executed':False}),flush=True)


if __name__=='__main__':main()
