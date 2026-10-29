"""Task10 source A7 fixed4+20 and independent mask/shuffle controls."""
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.mobind_fusion import run_pipeline

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run_pipeline(load_protocol(args.config,inference_only=True),device=args.device)
