"""Finish source safe/bank/group/sequence/P310 generation without final labels."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.source_postprocessing import run_postprocessing
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    run_postprocessing(load_protocol(parser.parse_args().config,inference_only=True))
