"""Bounded orchestration of existing source stages; evaluation stays separate."""
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import builtins
import io
from unittest.mock import patch
from filelock import FileLock
import numpy as np
from .no_vote_types import write_json,canonical_hash

@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: tuple
    commands: tuple
    markers: tuple

def stages():
    rev='protocol/task11_reconstruction_v2/'
    def cmd(script,*args):return (script,*args)
    return (
        Stage('prepared_geometry',(),(),('protocol/task2_prepared.json','protocol/task3_geometry_parity.json')),
        Stage('heads_mechanisms',('prepared_geometry',),(cmd('run_no_vote_visual_teacher.py','--phase','source-pipeline'),),('protocol/task4_heads_complete.json',)),
        Stage('fullwindow',('heads_mechanisms',),(cmd('run_no_vote_visual_teacher.py','--phase','fullwindow','--partition','all'),),('fullwindow_features/refit14/full/artifact.json','fullwindow_features/final4/full/artifact.json')),
        Stage('visual_review',('fullwindow',),(),('protocol/task4_complete.json',)),
        Stage('pixels',('prepared_geometry',),tuple(cmd('build_no_vote_pixels.py','--partition',p) for p in ['refit14','final4']),('pixels/refit14/full/artifact.json','pixels/final4/full/artifact.json')),
        Stage('student',('heads_mechanisms','pixels','visual_review'),(cmd('run_no_vote_visual_student.py','--phase','source-pipeline'),),('protocol/task5_artifacts_ready.json',)),
        Stage('student_review',('student',),(),('protocol/task5_complete.json',)),
        Stage('motion_cache',('pixels',),(cmd('build_no_vote_motion_cache.py'),),('protocol/task6_artifacts_ready.json',)),
        Stage('motion_cache_review',('motion_cache',),(),('protocol/task6_complete.json',)),
        Stage('rf',('motion_cache_review',),(cmd('run_no_vote_imu_rf.py'),),('protocol/task7_artifacts_ready.json',)),
        Stage('rf_review',('rf',),(),('protocol/task7_complete.json',)),
        Stage('motion_teacher',('rf_review','student_review'),(cmd('run_no_vote_mobind_pretrain.py'),),('protocol/task8_ir_supervision_v2_artifacts_ready.json',)),
        Stage('motion_review',('motion_teacher',),(),('protocol/task8_current.json',)),
        Stage('diagnostics',('motion_review',),(cmd('run_no_vote_simple_fusion.py'),),('protocol/task9_artifacts_ready.json',)),
        Stage('fusion',('diagnostics',),(cmd('run_no_vote_mobind_fusion.py'),),('protocol/task10_artifacts_ready.json',)),
        Stage('reconstruction_recipe',('fusion',),(),(rev+'preparation.json',)),
        Stage('p12_skeleton',('reconstruction_recipe',),(cmd('run_no_vote_p12_reconstruction.py'),cmd('export_no_vote_p12_skeleton.py')),(rev+'skeleton_fp16_component.json',)),
        Stage('p12_images',('reconstruction_recipe',),(cmd('run_no_vote_p12_images.py'),),(rev+'depth_fp16_component.json',rev+'thermal_fp16_component.json')),
        Stage('p12_global_knn',('p12_skeleton','p12_images','fullwindow'),(cmd('build_no_vote_p12_global.py'),),(rev+'p12_global_knn_component.json',)),
        Stage('source_base',('p12_global_knn','fusion'),(cmd('fit_no_vote_source_base_decoder.py'),cmd('build_no_vote_source_base_targets.py'),cmd('run_no_vote_source_base_adaptation.py')),(rev+'source_base_adapt12_component.json',)),
        Stage('safe_bank_group_sequence_precedence',('source_base',),(cmd('build_no_vote_source_postprocessing.py'),),(rev+'source_postprocessing_component.json',)),
        Stage('task11_review',('safe_bank_group_sequence_precedence',),(),('protocol/task9_complete.json','protocol/task10_complete.json','protocol/task9_10_independent_review_P39.json','protocol/task11_complete.json')),
        Stage('target_adaptation',('task11_review',),(cmd('run_no_vote_target_adaptation.py'),),('protocol/task12_artifacts_ready.json',)),
        Stage('target_review',('target_adaptation',),(),('protocol/task12_complete.json',)),
        Stage('freeze',('target_review',),(cmd('freeze_no_vote_generation.py'),),('protocol/task13_frozen_generation.json',)),
    )

def _read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def _digest(path):
    path=Path(path)
    if path.stat().st_size>8*1024**2:raise ValueError('orchestrator markers must be small metadata')
    return hashlib.sha256(path.read_bytes()).hexdigest()
def _open(root):
    state=root/'run_state.json'
    if state.exists() and _read(state).get('state') in {'frozen','revealed'}:
        raise ValueError('generation closed after frozen/revealed')

def execute(sequence,root,protocol_sha,runner,*,producer_fingerprints=None):
    root=Path(root).resolve();_open(root)
    workspace=root/'orchestration';workspace.mkdir(parents=True,exist_ok=True)
    done=set()
    with FileLock(workspace/'pipeline.lock',timeout=0):
        for stage in sequence:
            _open(root)
            if not set(stage.dependencies)<=done:raise ValueError('dependency order invalid')
            sources={c[0]:(producer_fingerprints or {}).get(c[0]) for c in stage.commands}
            sources['implementation']=(producer_fingerprints or {}).get('__implementation__')
            receipt=workspace/(stage.name+'.json');identity=canonical_hash((protocol_sha,stage,sources))
            if receipt.exists():
                saved=_read(receipt)
                if saved['identity']!=identity or any(_digest(root/p)!=h for p,h in saved['markers'].items()):
                    raise ValueError('completed stage marker/command identity drift')
                done.add(stage.name);continue
            log=workspace/(stage.name+'.log')
            write_json(workspace/'status.json',dict(status='running',stage=stage.name,protocol_sha256=protocol_sha))
            try:
                # One invocation/context per stage, never per epoch. Each original
                # trainer retains its own context, lock and complete-epoch resume.
                runner(stage,log,dict(protocol_sha256=protocol_sha,stage=stage.name))
                markers={p:_digest(root/p) for p in stage.markers}
                write_json(receipt,dict(identity=identity,markers=markers))
                done.add(stage.name)
            except Exception as error:
                write_json(workspace/'status.json',dict(status='failed',stage=stage.name,error=repr(error)))
                raise
        write_json(workspace/'status.json',dict(status='complete',stages=list(s.name for s in sequence)))
    return workspace/'status.json'

def history_timing(rows):
    seconds=[x.get('seconds',x.get('training',{}).get('seconds')) for x in rows]
    known=bool(rows) and all(isinstance(x,(int,float)) and x>=0 for x in seconds)
    return dict(epochs=len(rows),training_seconds=sum(seconds) if known else None,
                timing_status='measured' if known else 'unknown_missing_epoch_timing')

def preflight(p,config):
    root=p.run_root
    state=_read(root/'run_state.json')
    if state.get('protocol_sha256')!=p.identity():raise ValueError('run state/protocol mismatch')
    repo=Path(__file__).resolve().parents[2]
    missing=[str(repo/'scripts'/c[0]) for s in stages() for c in s.commands if not (repo/'scripts'/c[0]).is_file()]
    if missing:raise ValueError('missing stage producers: '+repr(missing))
    # Known stage names and marker stats only: no raw directories, model hashes
    # or hidden labels. Unknown future time/storage is not fabricated.
    import torch
    gpu=dict(available=torch.cuda.is_available(),name=torch.cuda.get_device_name() if torch.cuda.is_available() else None)
    histories={}
    for name in ['A2/select','A2/refit','A9-40/adapt','A9-12/adapt']:
        path=root/name/'history.json'
        if path.is_file():
            histories[name]=history_timing(_read(path))
    return dict(protocol_sha256=p.identity(),state=state['state'],free_gib=shutil.disk_usage(root).free/1024**3,
        gpu=gpu,config_sha256=_digest(config),measured_training_history=histories,
        generation_allowed=state['state'] not in {'frozen','revealed'},
        required_free_gib=p.recipe.get('minimum_free_gib',20),
        initializer_paths={k:str(v) for k,v in p.weights.items()},
        stages=[dict(name=s.name,dependencies=s.dependencies,commands=s.commands,manual_gate=not bool(s.commands),markers_present={m:(root/m).is_file() for m in s.markers}) for s in stages()],
        missing_manual_prerequisites=[m for s in stages() if not s.commands for m in s.markers if not (root/m).is_file()],
        retained_large_roots=['videomae_large'],candidate_count=16,
        timing_policy='use completed stage histories; no whole-run time or storage guess',
        final_labels_read=False,raw_scan=False)

def smoke(output):
    CANDIDATES=('A1','A2','A5-S','A5-I','A6-VS','A6-VI','A6-VSI','A7','A7-mask','A7-shuffle-S','A7-shuffle-I','A7-zero-S','A7-zero-I','A8','A9-12','A9-40')
    output=Path(output).resolve()
    if output.exists():raise ValueError('smoke requires a fresh isolated directory')
    output.mkdir(parents=True)
    write_json(output/'run_state.json',dict(state='generating',fixture=True))
    order=[]
    def runner(stage,log,context):
        order.append(stage.name)
        for marker in stage.markers:
            write_json(output/marker,dict(fixture=True,stage=stage.name))
        with log.open('a',encoding='utf-8') as f:f.write(stage.name+'\n')
        if stage.name=='freeze':
            for name in CANDIDATES:
                np.savez(output/(name+'.npz'),sample_ids=np.array(['fixture-1','fixture-2']),class_ids=np.arange(40),probabilities=np.full((2,40),.025),valid=np.array([True,False]))
            write_json(output/'run_state.json',dict(state='frozen',fixture=True))
    reads=[];forbidden=[]
    def wrapped(original):
        def guarded(file,*args,**kwargs):
            if isinstance(file,(str,bytes,Path)):
                path=Path(file).resolve();mode=args[0] if args else kwargs.get('mode','r')
                if not path.is_relative_to(output):
                    forbidden.append(str(path));raise ValueError('fixture attempted outside payload access')
                if 'r' in mode:reads.append(str(path))
            return original(file,*args,**kwargs)
        return guarded
    with patch('builtins.open',wrapped(builtins.open)),patch('io.open',wrapped(io.open)):
        execute(stages(),output,'fixture-only-no-model-training',runner)
    result=output/'smoke_summary.json'
    write_json(result,dict(fixture=True,state='frozen',candidates=list(CANDIDATES),stage_order=order,
        truth_reads=len(forbidden),raw_hash_reads=len(forbidden),ancestor_payload_reads=len(forbidden),
        filesystem_read_events=len(reads),epoch_asset_checks='not_applicable_no_training',
        scope='scheduler and synthetic output-contract smoke, not ML numerical equivalence'))
    return result

def run_pipeline(mode,config,*,smoke_output=None):
    from .no_vote_protocol import load_protocol
    p=load_protocol(config,inference_only=True)
    if mode=='preflight':return preflight(p,config)
    if mode=='smoke':
        if smoke_output is None:raise ValueError('explicit isolated smoke output required')
        target=Path(smoke_output).resolve()
        if target==p.run_root or target.is_relative_to(p.run_root):raise ValueError('smoke must be outside formal run')
        return smoke(target)
    _open(p.run_root)
    if mode=='freeze':
        from .no_vote_evaluation import collect_candidates,freeze_generation
        from .artifact_record import ArtifactRegistry
        return freeze_generation(p,ArtifactRegistry(p),collect_candidates(p))
    if mode!='generate':raise ValueError('unknown pipeline mode')
    report=preflight(p,config)
    if report['free_gib']<report['required_free_gib']:raise ValueError('insufficient generation disk space')
    repo=Path(__file__).resolve().parents[2];config=Path(config).resolve()
    def runner(stage,log,context):
        if not stage.commands:
            if not all((p.run_root/x).is_file() for x in stage.markers):
                raise ValueError('manual prepared/geometry or frozen reconstruction recipe prerequisite missing')
            return
        with log.open('ab',buffering=0) as stream:
            for command in stage.commands:
                subprocess.run([sys.executable,'-B',str(repo/'scripts'/command[0]),'--config',str(config),*command[1:]],
                    cwd=repo,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True)
    fingerprints={c[0]:_digest(repo/'scripts'/c[0]) for s in stages() for c in s.commands}
    fingerprints['__implementation__']=canonical_hash({x.name:_digest(x) for x in sorted((repo/'src/experiments').glob('*.py'))})
    return execute(stages(),p.run_root,p.identity(),runner,producer_fingerprints=fingerprints)
