import numpy as np
import pytest
from src.data.visual90_feature_cache import FeatureCacheDataset
from scripts.smoke_visual90_resources import parse_args


def test_incomplete_cache_cannot_be_used_for_benchmark(tmp_path):
    with pytest.raises(ValueError,match='complete'):
        FeatureCacheDataset(tmp_path)


def test_smoke_cli_does_not_offer_formal_training():
    with pytest.raises(SystemExit): parse_args(['--phase','train'])


def test_smoke_cli_accepts_benchmark_phase():
    args=parse_args(['--phase','benchmark'])
    assert args.phase=='benchmark'


def test_cache_rejects_an_altered_array_before_any_loader_worker_starts(tmp_path):
    import hashlib,json
    from src.data.visual90_feature_cache import FEATURE_KEYS
    hashes={}
    for key in FEATURE_KEYS:
        path=tmp_path/(key+'.npy')
        np.save(path,np.zeros((1,1),dtype=np.float32))
        hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    marker={'sample_ids':['train_a'],'labels':[0],'users':[1],'array_sha256':hashes}
    (tmp_path/'complete.json').write_text(json.dumps(marker))
    np.save(tmp_path/'video.npy',np.ones((1,1),dtype=np.float32))
    with pytest.raises(ValueError,match='hash'):
        FeatureCacheDataset(tmp_path)
