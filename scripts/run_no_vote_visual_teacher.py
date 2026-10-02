"""A1 extraction, fixed-split selection/refit, or label-free prediction."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import load_stage_inputs,read_public_rows,row_index
from src.experiments.no_vote_types import ArtifactRef,Selection,write_json
from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.visual_teacher import extract_visual_features,select_visual_head,fit_visual_head,fit_class_prior,predict_visual_teacher
from src.experiments import no_vote_weights


def read_ref(path):
    body=json.loads(path.read_text(encoding='utf-8'))
    return ArtifactRef(Path(body['record_path']),body['sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['extract','select','refit','predict'],required=True)
    parser.add_argument('--partition',choices=['train12','development2','refit14','final4'])
    parser.add_argument('--features-ref',type=Path)
    parser.add_argument('--roi-summary',type=Path)
    parser.add_argument('--model-phase',choices=['select','refit'])
    parser.add_argument('--max-trials',type=int,default=0)
    parser.add_argument('--clip-batch',type=int,default=6)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke',action='store_true',help='fixture protocols only; real one-trial acceptance uses --max-trials 1')
    args=parser.parse_args();p=load_protocol(args.config);registry=ArtifactRegistry(p)
    if args.smoke and p.recipe['execution_kind']!='fixture':parser.error('--smoke requires a fixture protocol')
    if args.max_trials and args.phase!='extract':parser.error('--max-trials applies only to partial extraction')
    if args.phase=='extract':
        if not args.partition or not args.roi_summary:parser.error('extract requires --partition and --roi-summary')
        if shutil.disk_usage(p.run_root).free<20*1024**3:raise RuntimeError('less than 20 GiB free before extraction')
        body=json.loads(args.roi_summary.read_text());data=body['artifacts']['p29']
        roi=ArtifactRef(Path(data['record_path']),data['sha256'])
        weight=registry.register(stage='visual_initializer',kind='public_weights',phase='public',
            files=[p.weights['videomae']/n for n in ('config.json','preprocessor_config.json','model.safetensors')],
            config={'role':'videomae','asset_identity':p.recipe['asset_bindings']},source_files=[Path(no_vote_weights.__file__)])
        out=p.run_root/'visual_features'/args.partition/('acceptance' if args.max_trials else 'full')
        ref=extract_visual_features(load_stage_inputs(p,args.partition,'raw'),roi,weight,out,protocol=p,
            device=args.device,max_trials=args.max_trials,clip_batch=args.clip_batch)
    else:
        if not args.features_ref:parser.error('head/predict requires --features-ref')
        features=read_ref(args.features_ref)
        if args.phase=='select':
            selection=select_visual_head(load_stage_inputs(p,'train12','select'),
                load_stage_inputs(p,'development2','select'),features,protocol=p)
            fit_class_prior(load_stage_inputs(p,'train12','select'),'select',protocol=p)
            ref=selection.fit_artifact
        elif args.phase=='refit':
            body=json.loads((p.run_root/'A1/select/selection.json').read_text())
            body['fit_artifact']=ArtifactRef(Path(body['fit_artifact']['record_path']),body['fit_artifact']['sha256'])
            selection=Selection(**body)
            inputs=load_stage_inputs(p,'refit14','refit')
            ref=fit_visual_head(inputs,features,selection,protocol=p);fit_class_prior(inputs,'refit',protocol=p)
        else:
            if not args.partition or not args.model_phase:parser.error('predict requires --partition and --model-phase')
            model=read_ref(p.run_root/'A1'/args.model_phase/'artifact.json')
            prior=read_ref(p.run_root/'A1'/args.model_phase/'prior_artifact.json')
            rows=row_index(read_public_rows(load_stage_inputs(p,args.partition,'predict').public_manifest))
            ref=predict_visual_teacher(model,features,rows,prior,protocol=p).prediction.artifact
        if args.phase in ('select','refit'):
            write_json(p.run_root/'A1'/args.phase/'artifact.json',{'record_path':str(ref.record_path),'sha256':ref.sha256})
    print(json.dumps({'phase':args.phase,'record_path':str(ref.record_path),'sha256':ref.sha256}),flush=True)


if __name__=='__main__':main()
