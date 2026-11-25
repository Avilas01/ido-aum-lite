"""Hosted-only tiny deterministic fixtures; no CIFAR or full model training.

Run in Colab with the existing PyTorch installation. No local torch install.
"""
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
pytestmark = pytest.mark.torch
torch = pytest.importorskip('torch')

from config import load_config
from src.checkpoints import (CheckpointStore, IdentityMismatch, capture_rng, model_state_hash,
    restore_rng, run_identity, seed_all, seed_streams)
from src.statistics import SampleStatistics
from src.train import run_stage, run_stages

CODE = {'git_sha': 'fixture', 'source_sha256': 'fixture'}
RUNTIME = {'python': 'fixture', 'torch': str(torch.__version__), 'device_type': 'cpu', 'gpu': None}


class FixtureDataset:
    def __init__(self, kind):
        self.kind = kind
        self.metadata = SimpleNamespace(noisy_labels=np.arange(12, dtype=np.int64) % 10)

    def __len__(self):
        return 12

    def __getitem__(self, index):
        image = torch.arange(8, dtype=torch.float32) / 10 + index / 20
        label = int(self.metadata.noisy_labels[index])
        def view():
            return image + .05 * (torch.rand(8) + random.random() + np.random.random())
        if self.kind == 'statistics':
            return image, label, index
        weak = view()
        return (weak, view(), label, index) if self.kind == 'adaptive' else (weak, label, index)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.BatchNorm1d(8),
                                           torch.nn.ReLU(), torch.nn.Dropout(.2), torch.nn.Linear(8, 10))

    def forward(self, x):
        return self.layers(x)


def factory(settings):
    return Tiny()


def setup(workers=0):
    # Deliberately tiny hosted unit fixture, not an allowed full-run configuration.
    settings = dict(load_config(), train_size=12, warmup_epochs=2, continuation_epochs=2,
                    batch_size=5, num_workers=workers, amp=False)
    datasets = {kind: FixtureDataset(kind) for kind in ('standard', 'adaptive', 'statistics')}
    metadata = {'train_size': 12, 'subset_checksum': 'ids', 'noisy_label_checksum': 'labels'}
    identity = run_identity(settings, metadata, CODE, RUNTIME, 'unit_fixture')
    identity['schedule'] = {'warmup_epochs': 2, 'continuation_epochs': 2}
    return settings, datasets, metadata, identity


def load_stage(root, identity, stage):
    return CheckpointStore(root / stage, {**identity, 'stage': stage, 'target_epochs': 2}).load()[0]


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def run_fixture(root, *, resume=False, workers=0):
    settings, datasets, metadata, _ = setup(workers)
    return run_stages(settings, datasets, metadata, root, torch.device('cpu'), resume=resume,
                      kind='unit_fixture', code=CODE, runtime=RUNTIME, model_factory=factory)


def test_model_hash_includes_batchnorm_buffers_and_scalar_counters():
    seed_all(42)
    model = Tiny()
    baseline = model_state_hash(model.state_dict())
    model.layers[1].num_batches_tracked.add_(1)
    assert model_state_hash(model.state_dict()) != baseline
    model.layers[1].num_batches_tracked.sub_(1)
    model.layers[1].running_mean.add_(1)
    assert model_state_hash(model.state_dict()) != baseline


def test_every_rng_and_loader_generator_round_trips():
    seed_all(42)
    generators = {'train': torch.Generator().manual_seed(17), 'statistics': torch.Generator().manual_seed(19)}
    saved = capture_rng(generators)
    def draw():
        return (random.random(), np.random.rand(3), torch.rand(3),
                torch.rand(3, generator=generators['train']), torch.rand(3, generator=generators['statistics']),
                [torch.rand(3, device='cuda')] if torch.cuda.is_available() else [])
    expected = draw()
    draw()
    restore_rng(saved, generators)
    assert_nested_equal(draw(), expected)


@pytest.mark.parametrize('workers', [0, 2])
def test_interrupted_matches_uninterrupted_and_branch_reset_is_complete(tmp_path, workers):
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    run_fixture(full, workers=workers)
    settings, datasets, _, identity = setup(workers)
    streams = seed_streams(settings['seed'])
    common = dict(resume=False, streams=streams, runtime=RUNTIME, model_factory=factory)
    # Simulate restart after one warm-up epoch; rebuild every object on resume.
    run_stage(settings, datasets, interrupted / 'warmup', identity, 'warmup', 2,
              torch.device('cpu'), stop_after=1, **common)
    shared = run_stage(settings, datasets, interrupted / 'warmup', identity, 'warmup', 2,
              torch.device('cpu'), **dict(common, resume=True))
    # Interrupt adaptive continuation too; its own optimizer/statistics/RNG must resume.
    run_stage(settings, datasets, interrupted / 'ido_lite', identity, 'ido_lite', 2,
              torch.device('cpu'), shared=shared, stop_after=1, **common)
    # run_stages deliberately refuses checkpoints without its parent manifest.
    from src.checkpoints import atomic_json
    atomic_json(interrupted / 'run.json', {'identity': identity})
    run_fixture(interrupted, resume=True, workers=workers)
    for stage in ('warmup', 'standard', 'ido_lite', 'hybrid'):
        a, b = load_stage(full, identity, stage), load_stage(interrupted, identity, stage)
        assert a['model_sha256'] == b['model_sha256']
        for key in ('model', 'optimizer', 'scaler', 'statistics', 'rng', 'warmup_history'):
            assert_nested_equal(a[key], b[key])
        for row_a, row_b in zip(a['logs'], b['logs']):
            assert_nested_equal({k:v for k,v in row_a.items() if k != 'epoch_seconds_excluding_checkpoint'},
                                {k:v for k,v in row_b.items() if k != 'epoch_seconds_excluding_checkpoint'})
    warmup = load_stage(full, identity, 'warmup')
    for method in ('standard', 'ido_lite', 'hybrid'):
        branch = load_stage(full, identity, method)
        assert branch['initial_model_sha256'] == warmup['model_sha256']
        assert len(branch['warmup_history']) == 2
        assert branch['statistics']['tracked_epochs'] == (2 if method == 'standard' else 4)
        # Three batches/epoch: fresh optimizer has 6 steps, not warm-up's 6 + 6.
        assert all(float(value['step']) == 6 for value in branch['optimizer']['state'].values())
    assert len(list((interrupted / 'warmup' / 'history').glob('*.npz'))) == 2
    assert not list(interrupted.rglob('metrics.csv'))


def test_completed_resume_is_idempotent_and_fresh_overwrite_rejected(tmp_path):
    identity = run_fixture(tmp_path)
    pointers = {p: p.read_bytes() for p in tmp_path.glob('*/latest.json')}
    run_fixture(tmp_path, resume=True)
    assert {p: p.read_bytes() for p in pointers} == pointers
    with pytest.raises(FileExistsError):
        run_fixture(tmp_path)
    shared = load_stage(tmp_path, identity, 'warmup')
    left = SampleStatistics.from_state_dict(shared['statistics'])
    right = SampleStatistics.from_state_dict(shared['statistics'])
    assert not np.shares_memory(left._we, right._we)
    assert not np.shares_memory(left._margin_sum, right._margin_sum)
    assert not np.shares_memory(left._observation_counts, right._observation_counts)


@pytest.mark.parametrize('mismatch', ['config', 'code', 'dataset'])
def test_incompatible_resume_rejected_before_model_construction(tmp_path, mismatch):
    run_fixture(tmp_path)
    settings, datasets, metadata, _ = setup()
    code = CODE
    if mismatch == 'config': settings = dict(settings, learning_rate=.1)
    if mismatch == 'code': code = dict(CODE, git_sha='different')
    if mismatch == 'dataset': metadata = dict(metadata, noisy_label_checksum='different')
    def forbidden(settings):
        pytest.fail('Must validate identity before constructing/restoring model')
    with pytest.raises(IdentityMismatch):
        run_stages(settings, datasets, metadata, tmp_path, torch.device('cpu'), resume=True,
                   kind='unit_fixture', code=code, runtime=RUNTIME, model_factory=forbidden)


def test_checkpoint_commit_then_failed_exports_recovers_history(tmp_path, monkeypatch):
    import src.train as module
    original = module._export_state
    def interrupted(*args):
        raise OSError('Drive log publication interrupted')
    monkeypatch.setattr(module, '_export_state', interrupted)
    with pytest.raises(OSError):
        run_fixture(tmp_path)
    monkeypatch.setattr(module, '_export_state', original)
    identity = run_fixture(tmp_path, resume=True)
    assert load_stage(tmp_path, identity, 'warmup')['completed_epoch'] == 2
    assert len(list((tmp_path / 'warmup' / 'history').glob('*.npz'))) == 2


def test_corrupt_latest_epoch_repeats_only_uncommitted_work(tmp_path):
    identity = run_fixture(tmp_path)
    store = CheckpointStore(tmp_path / 'hybrid', {**identity, 'stage': 'hybrid', 'target_epochs': 2})
    previous, latest = store.load()
    expected_hash = previous['model_sha256']
    (tmp_path / 'hybrid' / latest['file']).write_bytes(b'interrupted payload')
    run_fixture(tmp_path, resume=True)
    assert store.load()[0]['model_sha256'] == expected_hash
    assert len(store.load()[0]['logs']) == 2
