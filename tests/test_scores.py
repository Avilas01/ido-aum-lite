"""Step 3 rank/weight tests: synthetic NumPy inputs, no torch or downloads."""
import numpy as np
import pytest

from src.data import prepare_labels
from src.scores import heuristic_weights, percentile_rank_by_noisy_class, suspicion_scores


def test_hand_ties_singletons_empty_classes_and_stable_order():
    values = np.array([8, 2, 2, 5, -9, 7, 7])
    labels = np.array([0, 0, 0, 0, 3, 8, 8])
    expected = np.array([.875, .25, .25, .625, .5, .5, .5])
    np.testing.assert_array_equal(percentile_rank_by_noisy_class(values, labels), expected)
    order = np.array([4, 0, 6, 2, 1, 5, 3])
    np.testing.assert_array_equal(percentile_rank_by_noisy_class(values[order], labels[order]), expected[order])
    assert percentile_rank_by_noisy_class(np.array([]), np.array([], dtype=int)).shape == (0,)


def test_direction_fusion_and_independent_storage():
    result = suspicion_scores(np.array([0, 1, 2]), np.array([-3., 0., 3.]),
                              np.zeros(3, dtype=int), observation_counts=np.full(3, 2))
    np.testing.assert_allclose(result['ido_lite'], [1/6, .5, 5/6])
    np.testing.assert_allclose(result['aum'], [5/6, .5, 1/6])
    np.testing.assert_allclose(result['hybrid'], [.5, .5, .5])
    assert not np.shares_memory(result['ido_lite'], result['hybrid'])


@pytest.mark.parametrize('ratio', [.2, .4])
def test_complete_class_means_bounds_and_reproducibility_for_both_noise_conditions(ratio):
    metadata, _ = prepare_labels(np.repeat(np.arange(10), 5000), ratio=ratio)
    rng = np.random.default_rng(42)
    we = rng.integers(0, 7, len(metadata.noisy_labels))
    aum = rng.normal(size=len(we))
    args = (we, aum, metadata.noisy_labels)
    scores = suspicion_scores(*args, observation_counts=np.full(len(we), 6))
    again = suspicion_scores(*args, observation_counts=np.full(len(we), 6))
    for name, values in scores.items():
        assert np.all((values > 0) & (values < 1))
        np.testing.assert_array_equal(values, again[name])
        for label in range(10):
            assert values[metadata.noisy_labels == label].mean() == pytest.approx(.5)
    c, _ = heuristic_weights(scores['hybrid'])
    assert c.mean() == pytest.approx(.55)


def test_endpoint_weights_and_parabolic_difficulty():
    c, d = heuristic_weights([0., .5, 1.])
    np.testing.assert_allclose(c, [1., .55, .1])
    np.testing.assert_array_equal(d, [0., 1., 0.])


@pytest.mark.parametrize('values,labels', [([np.nan], [0]), ([np.inf], [0]),
    ([[1]], [0]), ([1j], [0]), ([1], [1.]), ([1], [10]), ([1], [-1]), ([1, 2], [0])])
def test_bad_rank_inputs(values, labels):
    with pytest.raises(ValueError):
        percentile_rank_by_noisy_class(values, labels)


@pytest.mark.parametrize('we,aum,counts', [([0], [0.], [0]), ([0, 1], [0., 1.], [1, 2]),
    ([2], [0.], [1]), ([-1], [0.], [1]), ([.5], [0.], [1]),
    ([0], [np.nan], [1]), ([0], [0., 1.], [1]), ([0], [0.], [1.])])
def test_unobserved_partial_or_invalid_statistics_rejected(we, aum, counts):
    with pytest.raises(ValueError):
        suspicion_scores(we, aum, np.zeros(len(we), dtype=int), observation_counts=counts)


@pytest.mark.parametrize('values', [[-1], [1.1], [np.nan], [np.inf], [[.5]]])
def test_bad_suspicion(values):
    with pytest.raises(ValueError):
        heuristic_weights(values)
