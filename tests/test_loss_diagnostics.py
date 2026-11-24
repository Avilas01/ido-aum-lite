"""Torch-free logging reductions and import contract."""
import subprocess
import sys
from pathlib import Path

import pytest
from src.losses import AdaptiveEpochSummary, COMPONENTS


def logs(size, supervised, pseudo, consistency, fraction):
    result = {'sample_count': size, 'mean_c': .55, 'mean_d': .5, 'confidence_fraction': fraction}
    for name, loss in zip(COMPONENTS, [supervised, pseudo, consistency, supervised + pseudo + consistency]):
        result[name + '_loss'] = loss
        result[name + '_zero_fraction'] = fraction
        result[name + '_near_zero_fraction'] = fraction
    return result


def test_partial_batch_sample_weighted_reduction_and_no_mutation():
    first, last = logs(4, 1., 2., 3., .25), logs(1, 4., 5., 6., 1.)
    summary = AdaptiveEpochSummary()
    summary.update(first)
    summary.update(last)
    result = summary.compute()
    assert result['sample_count'] == 5
    for key in first.keys() - {'sample_count'}:
        assert result[key] == pytest.approx((4 * first[key] + last[key]) / 5)
    result['supervised_loss'] = 100
    assert summary.compute()['supervised_loss'] == pytest.approx(1.6)


def test_empty_epoch_and_invalid_update_are_rejected_atomically():
    summary = AdaptiveEpochSummary()
    with pytest.raises(ValueError, match='empty'):
        summary.compute()
    valid = logs(2, 1., 0., 0., 0.)
    summary.update(valid)
    for key, value in [('sample_count', 0), ('total_loss', 9), ('confidence_fraction', 1.1),
                       ('mean_c', float('nan')), ('pseudo_near_zero_fraction', -.1)]:
        with pytest.raises(ValueError):
            summary.update(dict(valid, **{key: value}))
        assert summary.compute() == valid


def test_modules_import_without_torch_or_torchvision():
    script = """
import sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torchvision'):
            raise AssertionError('unexpected torch import')
sys.meta_path.insert(0, BlockTorch())
import src.scores
import src.losses
"""
    subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1], check=True)
