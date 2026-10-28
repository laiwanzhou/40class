"""Task1 descriptor-only preparation; never launches a legacy producer."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import read_config,load_protocol
from src.experiments.no_vote_source_ops import compile_source_ops,compile_teacher_roster,export_source_ops,resolve_teacher_roster,_write_immutable
from src.experiments.no_vote_weights import reuse_verified_weights
from src.experiments.no_vote_types import canonical_bytes
from src.experiments.no_vote_source_ops import validate_runtime_recipe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    raw,base=read_config(args.config)
    def path(value):
        p=Path(value);return (p if p.is_absolute() else base/p).resolve()
    if raw['recipe'].get('source_contract_version')!=3:
        raise ValueError('Task1 exporter requires v3 source_contract_version')
    root=path(raw['run_root']);source=path(raw['source']['root'])
    body=compile_source_ops(source)
    roster=compile_teacher_roster(body,tuple(raw['recipe']['retained_large_roots']))
    bindings=raw['recipe']['source_ops_bindings']
    validate_runtime_recipe(raw['recipe'],body)
    for name,data in [('manifest',body),('roster',roster)]:
        if hashlib.sha256(canonical_bytes(data)+b'\n').hexdigest()!=bindings.get(name+'_sha256'):
            raise ValueError('compiled source descriptor differs from the configured binding')
    for name,data in [('manifest',body),('roster',roster)]:
        target=path(bindings[name+'_path'])
        if not target.is_relative_to(root/'protocol'):
            raise ValueError('descriptor output outside the new protocol')
        _write_immutable(target,data)
    specs={name:dict(value,local_path=str(path(value['local_path']))) for name,value in raw['weights'].items()}
    reuse_verified_weights(path(raw['public_weight_receipt_reuse']),root,specs,allow_fixture=raw['execution_kind']=='fixture')
    protocol=load_protocol(args.config)
    source_ref=export_source_ops(source,path(bindings['manifest_path']),protocol=protocol)
    roster_ref=resolve_teacher_roster(source_ref,tuple(raw['recipe']['retained_large_roots']),protocol=protocol,
        output=path(bindings['roster_path']))
    print(json.dumps(dict(source_ops_record=str(source_ref.record_path),source_ops_sha256=source_ref.sha256,
        teacher_roster_record=str(roster_ref.record_path),teacher_roster_sha256=roster_ref.sha256,
        original_expert_slots=len(roster['slots']),formal_generation_ready=roster['formal_generation_ready'],
        blocking_requirements=[x['code'] for x in roster['blocking_requirements']],training_started=False),ensure_ascii=False))


if __name__=='__main__':main()
