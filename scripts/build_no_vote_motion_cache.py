"""Build source P31/P86 motion without labels, learned normalization or teachers."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_types import canonical_data
from src.experiments.motion_cache import run_motion_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--partition',choices=['all','train12','development2','refit14','final4'],default='all')
    parser.add_argument('--max-trials',type=int,default=0)
    args=parser.parse_args();protocol=load_protocol(args.config,inference_only=True)
    if shutil.disk_usage(protocol.run_root).free<20*1024**3:raise RuntimeError('less than20GiB free before motion cache')
    from src.experiments.visual_student import phase_lock
    with phase_lock(protocol.run_root/'motion'):
        refs=run_motion_pipeline(protocol,partition=args.partition,max_trials=args.max_trials)
    print(json.dumps({'event':'motion_pipeline_complete','partial':bool(args.max_trials),'artifacts':canonical_data(refs)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
