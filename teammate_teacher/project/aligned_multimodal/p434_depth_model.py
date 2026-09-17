"""Offline exact external ImageNet initialization of the original depth model."""
import hashlib
from pathlib import Path
import torch
from .aligned_model import AlignedMultimodalModel

WEIGHTS=Path('C:/Users/ncy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth')
WEIGHTS_SHA256='f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec'


def build_model(weights_path=WEIGHTS):
    path=Path(weights_path)
    if not path.is_file():raise FileNotFoundError(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=WEIGHTS_SHA256:raise ValueError('P434 external initialization hash differs')
    state=torch.load(path,map_location='cpu',weights_only=True)
    # Same architecture and RNG consumption as canonical construction. Only the
    # external ResNet stem/BN/layers are populated; task/temporal heads stay fresh.
    model=AlignedMultimodalModel(['depth'],num_classes=40,dropout=.3,imagenet_pretrained=False)
    destination=model.state_dict();mapped={}
    for key,value in state.items():
        if key.startswith('fc.'):continue
        target='visual.depth_stem.weight' if key=='conv1.weight' else 'visual.'+key
        if target not in destination or destination[target].shape!=value.shape or destination[target].dtype!=value.dtype:
            raise ValueError('P434 backbone schema differs')
        mapped[target]=value
    wanted={k for k in destination if k=='visual.depth_stem.weight' or k.startswith(('visual.bn1.','visual.layer1.','visual.layer2.','visual.layer3.','visual.layer4.'))}
    # Official old checkpoint omits BN batch counters; torchvision supplies
    # their freshly initialized zero buffers during its compatibility load.
    for key in wanted-set(mapped):
        if not key.endswith('.num_batches_tracked') or torch.count_nonzero(destination[key]):
            raise ValueError('P434 missing non-counter backbone state')
        mapped[key]=destination[key]
    if set(mapped)!=wanted:raise ValueError('P434 pretrained backbone coverage differs')
    destination.update(mapped);model.load_state_dict(destination,strict=True)
    return model
