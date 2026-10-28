"""Select/refit one MC3 student or predict without teacher/labels."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from src.experiments.no_vote_types import Selection,ArtifactRef
from src.experiments.visual_student import read_ref,load_teacher,train_visual_student,predict_visual_student,StageVerificationContext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['select','refit','predict','source-pipeline'],required=True)
    parser.add_argument('--partition',choices=['train12','development2','refit14','final4'])
    parser.add_argument('--model-phase',choices=['select','refit'])
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args();started=time.perf_counter();p=load_protocol(args.config,inference_only=args.phase=='predict')
    print(json.dumps({'event':'student_cli_start','phase':args.phase}),flush=True)
    if p.recipe.get('source_contract_version')==3:
        from src.experiments import visual_student_v3 as source
        from src.experiments.artifact_record import VerificationContext
        if args.phase=='source-pipeline':
            source.run_pipeline(p,device=args.device)
            print(json.dumps({'event':'source_student_pipeline_complete'}),flush=True);return
        context=VerificationContext(p)
        if args.phase in ('select','refit'):
            source.prepare_student_capability(p,context)
            part='train12' if args.phase=='select' else 'refit14'
            selection=source._read_selection(p.run_root/'A2/select/selection.json') if args.phase=='refit' else None
            ref,_=source.train_visual_student(args.phase,load_stage_inputs(p,part,args.phase),
                load_stage_inputs(p,'development2','select') if args.phase=='select' else None,
                source.pixel_refs(p)['refit14'],source.load_teacher(p,part,context),selection,
                protocol=p,device=args.device,context=context)
        else:
            if not args.partition or not args.model_phase:parser.error('predict needs --partition and --model-phase')
            rows=row_index(read_public_rows(p.public_manifests[args.partition]))
            pred,_=source.export_population(read_ref(p.run_root/'A2'/args.model_phase/'artifact.json'),
                source.pixel_refs(p)['final4' if args.partition=='final4' else 'refit14'],rows,protocol=p,device=args.device,context=context)
            ref=pred.artifact
        print(json.dumps({'event':'student_cli_complete','record_path':str(ref.record_path),'sha256':ref.sha256}),flush=True);return
    if args.phase in ('select','refit'):
        part='train12' if args.phase=='select' else 'refit14'
        pixels=read_ref(p.run_root/'pixels/refit14/full/artifact.json')
        context=StageVerificationContext(p);teacher=load_teacher(p,part,context);selection=None
        if args.phase=='refit':
            body=json.loads((p.run_root/'A2/select/selection.json').read_text(encoding='utf-8'))
            body['fit_artifact']=ArtifactRef(Path(body['fit_artifact']['record_path']),body['fit_artifact']['sha256'])
            selection=Selection(**body)
        development=load_stage_inputs(p,'development2','select') if args.phase=='select' else None
        ref,_=train_visual_student(args.phase,load_stage_inputs(p,part,args.phase),development,pixels,
            teacher,selection,protocol=p,device=args.device,verification_context=context)
    else:
        if not args.partition or not args.model_phase:parser.error('predict needs --partition and --model-phase')
        pixel_part='final4' if args.partition=='final4' else 'refit14'
        pixels=read_ref(p.run_root/'pixels'/pixel_part/'full/artifact.json')
        model=read_ref(p.run_root/'A2'/args.model_phase/'artifact.json')
        rows=row_index(read_public_rows(load_stage_inputs(p,args.partition,'predict').public_manifest))
        ref=predict_visual_student(model,pixels,rows,protocol=p,device=args.device).artifact
    print(json.dumps({'event':'student_cli_complete','record_path':str(ref.record_path),'sha256':ref.sha256,
        'seconds':round(time.perf_counter()-started,2)}),flush=True)


if __name__=='__main__':main()
