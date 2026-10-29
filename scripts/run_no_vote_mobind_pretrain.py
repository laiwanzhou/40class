"""Source MoBind: own-phase A1/RF targets, fixed24 select/refit, no final labels."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.mobind_pretrain import run_motion_pipeline


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',default='cuda');args=parser.parse_args()
    run_motion_pipeline(load_protocol(args.config,inference_only=True),device=args.device)


if __name__=='__main__':main()
