import torch
import pytest

from src.models.visual90_fusion import Visual90Fusion
from src.training.visual90_training import cross_user_supcon


def example(batch=1):
    g = torch.Generator().manual_seed(8)
    return {
        'video': torch.randn(batch, 2, 4, 4, 8, 4, 768, generator=g),
        'appearance': torch.randn(batch, 4, 4, 4, 17, 1024, generator=g),
        'video_mask': torch.ones(batch, 2, 4, 4, dtype=torch.bool),
        'appearance_mask': torch.ones(batch, 4, 4, dtype=torch.bool),
        'roi': torch.rand(batch, 4, 4, 8, 4, generator=g),
        'times': torch.linspace(0, 1, 32).reshape(1, 4, 8).expand(batch, -1, -1),
    }


def test_initial_b_matches_a_and_missing_depth_is_invariant():
    torch.set_num_threads(2)
    a, b = Visual90Fusion(False).eval(), Visual90Fusion(True).eval()
    data = example()
    torch.testing.assert_close(a(data)['logits'], b(data)['logits'], rtol=0, atol=0)
    data['video_mask'][:, 1] = False
    before = b(data)['logits']
    data['video'][:, 1] += 100
    torch.testing.assert_close(before, b(data)['logits'], rtol=0, atol=0)


def test_all_empty_is_finite_and_unsupported():
    model = Visual90Fusion(True).eval()
    data = example()
    data['video_mask'].zero_()
    data['appearance_mask'].zero_()
    out = model(data)
    assert not out['supported'].any()
    assert torch.isfinite(out['logits']).all()


def test_coordinates_and_frame_time_reach_output():
    model = Visual90Fusion(False).eval()
    data = example()
    original = model(data)['logits']
    data['times'] = data['times'].flip(-1)
    assert not torch.allclose(original, model(data)['logits'])
    original = model(data)['logits']
    data['roi'] = data['roi'] + .2
    assert not torch.allclose(original, model(data)['logits'])


def test_appearance_path_gets_gradients_after_zero_projection_step():
    model = Visual90Fusion(True).train()
    data = example(2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(data)['logits'], torch.tensor([1, 2])).backward()
        optimizer.step()
    assert model.appearance_projection.weight.grad.abs().sum() > 0


def test_invalid_nan_fails_instead_of_silent_batch_skip():
    data = example()
    data['video'][0, 0, 0, 0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        Visual90Fusion(False)(data)


def test_supcon_duplicate_ids_do_not_reweight_candidates():
    z = torch.eye(4, requires_grad=True)
    y = torch.tensor([0, 0, 0, 1])
    u = torch.tensor([1, 2, 1, 3])
    loss = cross_user_supcon(z, y, u, ['a', 'b', 'c', 'd'])
    expected = (2 * torch.log(torch.tensor(2.)) + torch.log(torch.tensor(3.))) / 3
    torch.testing.assert_close(loss, expected)
    duplicate = cross_user_supcon(torch.cat([z, z[1:2]]), torch.cat([y,y[1:2]]), torch.cat([u,u[1:2]]), ['a','b','c','d','b'])
    torch.testing.assert_close(loss, duplicate)


def test_no_positive_supcon_returns_differentiable_zero():
    z = torch.randn(2, 8, requires_grad=True)
    loss = cross_user_supcon(z, torch.tensor([0, 0]), torch.tensor([1, 1]), ['a', 'b'])
    loss.backward()
    assert loss == 0
    assert torch.isfinite(z.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA RNG ownership regression')
def test_model_initialization_preserves_callers_cpu_and_cuda_rng():
    torch.manual_seed(71)
    torch.cuda.init()
    cpu_state=torch.get_rng_state().clone()
    gpu_state=torch.cuda.get_rng_state().clone()
    Visual90Fusion(True)
    assert torch.equal(cpu_state,torch.get_rng_state())
    assert torch.equal(gpu_state,torch.cuda.get_rng_state())
