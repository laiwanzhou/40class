"""Build intermediate structured targets for train12/refit14 only after P85 KNN."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.experiments.no_vote_protocol import load_protocol
from src.experiments.artifact_record import VerificationContext
from src.experiments.no_vote_types import canonical_bytes
from src.experiments.no_vote_cache_import import immutable_bytes
from src.experiments.p12_reconstruction import REVISION,ref
from src.experiments.source_base_targets import make_source_base_targets

def main(config):
    p=load_protocol(config,inference_only=True);context=VerificationContext(p);root=p.run_root/'protocol'/REVISION
    decoder=json.loads((root/'source_base_decoder_fit.json').read_text(encoding='utf-8'))
    teachers=json.loads((root/'p12_global_knn_component.json').read_text(encoding='utf-8'))
    if any(x['base_protocol_sha256']!=p.identity() or x['revision']!=REVISION for x in [decoder,teachers]):raise ValueError('source-base component protocol mismatch')
    outputs={}
    for phase,pop in [('select','train12'),('refit','refit14')]:
        outputs[pop]=make_source_base_targets(p,ref(decoder['models'][phase]),ref(teachers['knn_outputs'][pop]),pop,context)
        print(json.dumps({'event':'source_base_structured_targets','phase':phase,'partition':pop}),flush=True)
    body={'revision':REVISION,'base_protocol_sha256':p.identity(),'outputs':outputs,'fit_pool_targets_complete':True,
        'source_base_adapt12_complete':False,'task11_complete':False,'task12_started':False,'final_labels_read':False}
    immutable_bytes(root/'source_base_structured_targets.json',canonical_bytes(body)+b'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    main(parser.parse_args().config)
