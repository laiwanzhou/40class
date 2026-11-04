"""Review registration normalizes relocated roots without changing frozen bytes."""
import hashlib
import json


def test_register_review_through_historical_directory_alias(tmp_path):
    from test_no_vote_evaluation import fake_frozen,module
    from test_no_vote_path_relocation import _alias
    from scripts.review_no_vote_frozen import register_frozen_review
    runtime=tmp_path/'runtime';runtime.mkdir();alias=tmp_path/'historical';_alias(alias,runtime)
    frozen,truth=fake_frozen(runtime);body=json.loads(frozen.read_text());body['state_path']=str(alias/'run_state.json');body['execution_kind']='formal'
    frozen.write_text(json.dumps(body));digest=hashlib.sha256(frozen.read_bytes()).hexdigest()
    (runtime/'protocol').mkdir(exist_ok=True);frozen=frozen.rename(runtime/'protocol/frozen.json')
    state=runtime/'run_state.json';s=json.loads(state.read_text());s['generation']['path']=str(frozen);s['generation']['sha256']=digest;state.write_text(json.dumps(s))
    review=runtime/'protocol/review.json'
    review.write_text(json.dumps(dict(verdict='GO',generation_sha256=digest,protocol_sha256='fixture',unresolved_p1_p2=[])))
    before=frozen.read_bytes();register_frozen_review(frozen,review)
    assert frozen.read_bytes()==before
    assert module().evaluate_frozen(frozen,truth,runtime/'evaluation').exists()
