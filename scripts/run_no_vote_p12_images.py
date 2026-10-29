"""Train reconstructed source Depth/Thermal teachers with exact fixed populations."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.p12_visual_reconstruction import run_images
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--roles',nargs='+',choices=['depth','thermal'],default=['depth','thermal'])
    args=parser.parse_args();run_images(load_protocol(args.config,inference_only=True),device=args.device,roles=args.roles)
