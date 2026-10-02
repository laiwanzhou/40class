"""Generate label-free pose and ROI records; --max-trials produces partial artifacts."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import load_stage_inputs
from src.experiments.artifact_record import ArtifactRegistry
from src.experiments.pose_roi_adapter import build_pose_roi

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--partition',choices=['train12','development2','refit14','final4'],required=True)
    parser.add_argument('--max-trials',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();p=load_protocol(args.config)
    inputs=load_stage_inputs(p,args.partition,'raw')
    weight=ArtifactRegistry(p).register(stage='pose_initializer',kind='public_weights',phase='public',
        files=[p.weights['yolo']],config={'role':'yolo','asset_identity':p.recipe['asset_bindings']})
    suffix='smoke' if args.max_trials else 'full'
    build_pose_roi(inputs,weight,p.run_root/'pose_roi'/args.partition/suffix,protocol=p,
                   device=args.device,max_trials=args.max_trials)

if __name__=='__main__':main()
