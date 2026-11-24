"""Colab-only Step 3 loss/gradient checks on tiny synthetic tensors."""
import math
import pytest

pytestmark = pytest.mark.torch
torch = pytest.importorskip('torch')
import torch.nn.functional as F
from src.losses import adaptive_loss, AdaptiveEpochSummary, COMPONENTS


def fixture_logits():
    weak = torch.tensor([[4., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                         [0., 5., 0., 0., 0., 0., 0., 0., 0., 0.]])
    strong = torch.arange(30, dtype=torch.float32).reshape(3, 10) / 10
    return weak.requires_grad_(), strong.requires_grad_(), torch.tensor([2, 1, 0]), torch.tensor([.5, 1., 0.])


def reference(weak, strong, labels, scores):
    # Independent scalar log-sum-exp/probability calculations, no shared loss helper.
    components = [[], [], []]
    for w, v, y, s in zip(weak.tolist(), strong.tolist(), labels.tolist(), scores.tolist()):
        pw = [math.exp(x) / sum(math.exp(z) for z in w) for x in w]
        ps = [math.exp(x) / sum(math.exp(z) for z in v) for x in v]
        target = max(range(10), key=pw.__getitem__)
        components[0].append((.1 + .9 * (1-s)) * (math.log(sum(math.exp(x) for x in w)) - w[y]))
        components[1].append(.5 * s * (max(pw) >= .7) * (math.log(sum(math.exp(x) for x in v)) - v[target]))
        components[2].append(.5 * 4*s*(1-s) * sum((a-b)**2 for a, b in zip(pw, ps)) / 10)
    return components


def test_scalar_reference_full_batch_reductions_and_all_diagnostics():
    weak, strong, labels, scores = fixture_logits()
    loss, logs = adaptive_loss(weak, strong, labels, scores)
    expected = reference(weak, strong, labels, scores)
    total = [sum(row) for row in zip(*expected)]
    for name, values in zip(COMPONENTS, [*expected, total]):
        assert logs[name + '_loss'] == pytest.approx(sum(values) / 3, rel=1e-6)
        assert logs[name + '_zero_fraction'] == pytest.approx(sum(x == 0 for x in values) / 3)
        assert logs[name + '_near_zero_fraction'] == pytest.approx(sum(x <= 1e-8 for x in values) / 3)
    assert loss.item() == pytest.approx(sum(total) / 3)
    assert logs['confidence_fraction'] == pytest.approx(2/3)
    assert logs['mean_c'] == pytest.approx(.55)
    assert logs['mean_d'] == pytest.approx(1/3)
    assert all(isinstance(x, (int, float)) for x in logs.values())
    loss.backward()
    assert torch.isfinite(weak.grad).all() and torch.isfinite(strong.grad).all()


def test_all_unconfident_suspicious_examples_keep_supervised_floor():
    weak = torch.zeros(3, 10, requires_grad=True)
    strong = torch.zeros(3, 10, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    loss, logs = adaptive_loss(weak, strong, labels, torch.ones(3))
    assert logs['confidence_fraction'] == logs['pseudo_loss'] == 0
    assert logs['pseudo_zero_fraction'] == logs['pseudo_near_zero_fraction'] == 1
    assert logs['consistency_loss'] == 0
    assert loss.item() == pytest.approx(.1 * math.log(10))
    loss.backward()
    assert weak.grad.abs().sum() > 0 and torch.isfinite(weak.grad).all()


def test_pseudo_targets_and_suspicion_are_detached():
    weak, strong, labels, _ = fixture_logits()
    scores = torch.ones(3, requires_grad=True)
    loss, _ = adaptive_loss(weak, strong, labels, scores, consistency_coefficient=0)
    loss.backward()
    expected_weak = weak.detach().clone().requires_grad_()
    (.1 * F.cross_entropy(expected_weak, labels)).backward()
    torch.testing.assert_close(weak.grad, expected_weak.grad)
    assert scores.grad is None
    assert strong.grad.abs().sum() > 0


def test_consistency_has_both_gradient_paths_and_class_mean_factor():
    weak, strong, labels, _ = fixture_logits()
    base = [weak.detach().clone().requires_grad_(), strong.detach().clone().requires_grad_()]
    scores = torch.full((3,), .5)
    loss, _ = adaptive_loss(weak, strong, labels, scores, pseudo_coefficient=0)
    loss.backward()
    (.55 * F.cross_entropy(base[0], labels)).backward()
    expected_w = weak.detach().clone().requires_grad_()
    expected_s = strong.detach().clone().requires_grad_()
    (.5 * (expected_w.softmax(1) - expected_s.softmax(1)).square().mean()).backward()
    torch.testing.assert_close(weak.grad - base[0].grad, expected_w.grad, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(strong.grad, expected_s.grad)
    assert expected_w.grad.abs().sum() > 0 and expected_s.grad.abs().sum() > 0


def test_epoch_summary_matches_unsplit_batch_including_partial_last_batch():
    weak, strong, labels, scores = fixture_logits()
    _, full = adaptive_loss(weak, strong, labels, scores)
    summary = AdaptiveEpochSummary()
    for index in (slice(0, 2), slice(2, 3)):
        _, logs = adaptive_loss(weak[index], strong[index], labels[index], scores[index])
        summary.update(logs)
    for key, value in full.items():
        assert summary.compute()[key] == pytest.approx(value, abs=1e-7, rel=1e-6)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_inputs_produce_float32_finite_loss_and_gradients(dtype):
    weak, strong, labels, scores = fixture_logits()
    weak = weak.detach().to(dtype).requires_grad_()
    strong = strong.detach().to(dtype).requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, _ = adaptive_loss(weak, strong, labels, scores)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(weak.grad).all() and torch.isfinite(strong.grad).all()


@pytest.mark.parametrize('failure', ['shape', 'nan', 'labels', 'scores', 'range', 'empty'])
def test_invalid_inputs(failure):
    weak, strong, labels, scores = fixture_logits()
    if failure == 'shape': strong = strong[:, :9]
    if failure == 'nan': weak = torch.full_like(weak, float('nan'))
    if failure == 'labels': labels = labels.float()
    if failure == 'scores': scores = scores[:2]
    if failure == 'range': scores = torch.full_like(scores, 1.1)
    if failure == 'empty': weak, strong, labels, scores = weak[:0], strong[:0], labels[:0], scores[:0]
    with pytest.raises(ValueError):
        adaptive_loss(weak, strong, labels, scores)


@pytest.mark.parametrize('option,value', [('pseudo_threshold', 0), ('clean_weight_floor', 0),
    ('pseudo_coefficient', -1), ('consistency_coefficient', float('inf')), ('near_zero_epsilon', -1)])
def test_invalid_configuration(option, value):
    with pytest.raises(ValueError):
        adaptive_loss(*fixture_logits(), **{option: value})


def test_confidence_threshold_is_inclusive_and_near_zero_includes_nonzero_values():
    weak = torch.zeros(1, 10)
    strong = torch.zeros(1, 10)
    _, logs = adaptive_loss(weak, strong, torch.tensor([0]), torch.tensor([.5]),
                            pseudo_threshold=.1, near_zero_epsilon=10.)
    assert logs['confidence_fraction'] == 1
    assert logs['pseudo_loss'] == pytest.approx(.25 * math.log(10))
    assert logs['pseudo_zero_fraction'] == 0
    assert logs['pseudo_near_zero_fraction'] == 1
    assert logs['total_zero_fraction'] == 0
    assert logs['total_near_zero_fraction'] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Colab GPU required for CUDA AMP check')
def test_cuda_autocast_and_extreme_logits_have_finite_loss_and_gradients():
    weak = torch.tensor([[1000., -1000.] + [0.] * 8, [0.] * 10], device='cuda', requires_grad=True)
    strong = torch.tensor([[-1000., 1000.] + [0.] * 8, [0.] * 10], device='cuda', requires_grad=True)
    labels = torch.tensor([1, 0], device='cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss, logs = adaptive_loss(weak, strong, labels, torch.tensor([.5, 1.], device='cuda'))
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    assert logs['confidence_fraction'] == .5
    loss.backward()
    assert torch.isfinite(weak.grad).all() and torch.isfinite(strong.grad).all()
