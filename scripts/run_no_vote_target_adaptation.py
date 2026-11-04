"""Run fixed independent final-input adaptation endpoints; never reveal truth."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.target_adaptation import run_target_adaptation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    import torch
    torch.set_num_threads(4)
    from datetime import datetime
    from src.experiments.no_vote_types import write_json
    p=load_protocol(args.config,inference_only=True)
    status=p.run_root/'protocol/task12_status.json'
    original=Path.open;blocked={Path(x).resolve() for x in p.supervised_labels.values()};attempts=[]
    def guarded(path,*a,**kw):
        if path.resolve() in blocked:
            attempts.append(str(path));raise RuntimeError('supervised label payload opened during Task12')
        return original(path,*a,**kw)
    Path.open=guarded
    write_json(status,dict(status='running',started_at=datetime.now().astimezone().isoformat(),
        protocol_sha256=p.identity(),endpoints=[40,12],task13_started=False))
    try:
        run_target_adaptation(p,device=args.device)
        write_json(status,dict(status='generation_complete_awaiting_review',
            completed_at=datetime.now().astimezone().isoformat(),protocol_sha256=p.identity(),
            supervised_label_open_attempts=len(attempts),task13_started=False))
    except Exception as error:
        write_json(status,dict(status='failed',error=repr(error),
            supervised_label_open_attempts=len(attempts),task13_started=False))
        raise
    finally:
        Path.open=original
