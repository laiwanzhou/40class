"""Build label-free pixels once, with optional partial real acceptance."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import load_stage_inputs
from src.experiments.no_vote_types import ArtifactRef
from src.experiments.pixel_cache import build_pixels


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--partition',choices=['train12','development2','refit14','final4'],required=True)
    parser.add_argument('--max-trials',type=int,default=0)
    args=parser.parse_args();p=load_protocol(args.config)
    if shutil.disk_usage(p.run_root).free<20*1024**3:raise RuntimeError('less than20GiB free before pixel cache')
    roi_part='final4' if args.partition=='final4' else 'refit14'
    body=json.loads((p.run_root/'pose_roi'/roi_part/'full/summary.json').read_text())['artifacts']['p29']
    roi=ArtifactRef(Path(body['record_path']),body['sha256'])
    mode=f'acceptance_{args.max_trials}' if args.max_trials else 'full'
    ref=build_pixels(load_stage_inputs(p,args.partition,'raw'),roi,p.run_root/'pixels'/args.partition/mode,
        protocol=p,max_trials=args.max_trials)
    print(json.dumps({'record_path':str(ref.record_path),'sha256':ref.sha256}),flush=True)


if __name__=='__main__':main()
