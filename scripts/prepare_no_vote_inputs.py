"""Trusted standalone preparation; never invoked by model generation."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.no_vote_manifest import prepare_inputs

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--public-output',type=Path,required=True)
    parser.add_argument('--private-output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    args=parser.parse_args();protocol=load_protocol(args.config)
    paths=prepare_inputs(args.source_manifest,protocol,args.public_output,args.private_output,data_root=args.data_root)
    print(json.dumps({'prepared_partitions':list(paths),'rows':{k:v.expected_rows for k,v in protocol.partitions.items()},
                      'training_executed':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
