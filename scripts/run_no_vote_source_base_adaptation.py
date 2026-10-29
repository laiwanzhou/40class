"""Run Task11 own-fit-pool adaptation twelve epochs; never final Task12."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.source_base_adaptation import run_source_base_adaptation
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();run_source_base_adaptation(load_protocol(args.config,inference_only=True),device=args.device)
