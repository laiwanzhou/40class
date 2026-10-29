"""Fit the fixed source-base transition models on train12/refit14 labels only."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.source_base_targets import run_source_base_decoder_fit
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run_source_base_decoder_fit(load_protocol(args.config,inference_only=True))
