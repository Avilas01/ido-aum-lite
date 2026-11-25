"""Synthetic evaluation integrity/caching tests; no torch, CIFAR or training.

All test accuracies below are controlled fixture values in temporary directories,
never scientific experiment outputs.
"""
import csv
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from config import load_config
from src.checkpoints import CheckpointStore, IdentityMismatch, atomic_json, run_identity
from src.data import array_checksum, prepare_labels, save_metadata
from src.evaluate import (cached_accuracy, detector_metrics, evaluate_runs, json_digest,
                          training_summary)
from src.statistics import SampleStatistics

CODE = {'git_sha': 'unit-fixture', 'source_sha256': 'unit-fixture'}
RUNTIME = {'python': 'unit-fixture', 'numpy': np.__version__, 'device_type': 'fixture', 'gpu': None}


def model_hash(model):
    return json_digest({key: array_checksum(value) for key,value in model.items()})


def writer(state, path):
    with Path(path).open('wb') as stream:
        pickle.dump(state, stream)


def reader(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def tracker(mask, epoch):
    size = len(mask)
    return SampleStatistics.from_state_dict({'schema_version': 1, 'size': size,
        'tracked_epochs': epoch, 'we': mask.astype(np.int32)*min(epoch, 2),
        'margin_sum': np.where(mask, -epoch, epoch).astype(np.float32),
        'observation_counts': np.full(size, epoch, dtype=np.int32)})


def scientific_fixture(tmp_path, ratios=(.2,.4), incomplete=None):
    settings = dict(load_config(), train_size=20, warmup_epochs=2, continuation_epochs=2, num_workers=0)
    runs = tmp_path/'runs'
    for ratio in ratios:
        root = runs/f'noise_{round(100*ratio)}'
        training, analysis = prepare_labels(np.repeat(np.arange(10), 5), train_size=20, ratio=ratio, seed=42)
        save_metadata(training, analysis, root/'data')
        metadata = {**training.summary(), 'images_sha256': '0'*64, 'source': 'unit fixture'}
        identity = run_identity(settings, metadata, CODE, RUNTIME, 'experiment')
        identity['schedule'] = {'warmup_epochs': 2, 'continuation_epochs': 2}
        atomic_json(root/'run.json', {'identity': identity})
        shared_model = {'weight': np.array([1.,2.], dtype=np.float32)}
        history = [tracker(analysis.is_corrupted, epoch).state_dict() for epoch in (1,2)]
        for index, stage in enumerate(('warmup','standard','ido_lite','hybrid')):
            target = 2
            store = CheckpointStore(root/stage, {**identity, 'stage': stage, 'target_epochs': target}, writer=writer, reader=reader)
            if incomplete == (ratio,stage,'missing'):
                continue
            count = 1 if incomplete == (ratio,stage,'partial') else target
            for epoch in range(1,count+1):
                tracked = epoch if stage == 'warmup' else 2 + (0 if stage == 'standard' else epoch)
                model = shared_model if stage == 'warmup' else {'weight': np.array([index, epoch], dtype=np.float32)}
                state = {'schema_version': 1, 'identity': store.identity, 'stage': stage,
                    'completed_epoch': epoch, 'model': model, 'model_sha256': model_hash(model),
                    'initial_model_sha256': model_hash(shared_model),
                    'statistics': tracker(analysis.is_corrupted, tracked).state_dict(),
                    'warmup_history': history[:epoch] if stage == 'warmup' else history,
                    'logs': [{'epoch': e, 'sample_count': 20, 'common_noisy_ce': 1.,
                              'epoch_seconds_excluding_checkpoint': .5, 'training_ce': 1.}
                             for e in range(1,epoch+1)]}
                store.save(state)
    return settings, runs


class FixtureEvaluator:
    def __init__(self):
        self.calls = []

    def __call__(self, state, settings):
        self.calls.append((state['identity']['dataset']['noise_ratio'], state['stage']))
        return {'correct': 6000, 'total': 10000, 'accuracy_pct': 60.,
                'evaluation_seconds': .25, 'test_data_sha256': '1'*64, 'environment': RUNTIME}


def aggregate(runs, output, settings, evaluator):
    return evaluate_runs(runs, output, settings, CODE, RUNTIME, evaluator,
                         checkpoint_reader=reader, model_hasher=model_hash)


def test_hand_detector_direction_ties_and_missing_classes():
    metrics, _ = detector_metrics(np.array([0,0,2,2]), np.array([2.,2.,-2.,-2.]),
                                 np.zeros(4,dtype=int), np.array([False,False,True,True]), np.full(4,2))
    assert metrics == {'we_roc_auc': 1., 'negative_aum_roc_auc': 1., 'hybrid_roc_auc': 1.}
    metrics, _ = detector_metrics(np.ones(4,dtype=int), np.ones(4), np.zeros(4,dtype=int),
                                 np.array([False,False,True,True]), np.full(4,2))
    assert all(value == .5 for value in metrics.values())
    for mask in (np.zeros(4,dtype=bool), np.ones(4,dtype=bool)):
        with pytest.raises(ValueError, match='both clean and corrupted'):
            detector_metrics(np.zeros(4,dtype=int), np.zeros(4), np.zeros(4,dtype=int), mask, np.ones(4,dtype=int))


def test_six_rows_history_plot_arrays_and_cached_repeat(tmp_path):
    settings, runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    first = aggregate(runs, output, settings, evaluator)
    assert first['complete'] and first['row_count'] == 6
    assert len(evaluator.calls) == 6
    assert len(first['detectors']) == 2
    publication = json.loads((output/'metrics_manifest.json').read_text())
    from src.checkpoints import file_digest
    assert publication['metrics_csv_sha256'] == file_digest(output/'metrics.csv')
    assert publication['metrics_json_sha256'] == file_digest(output/'metrics.json')
    assert publication['complete'] and publication['row_count'] == 6
    for detection in first['detectors']:
        assert detection['primary_epoch'] == settings['warmup_epochs']
        assert [row['epoch'] for row in detection['history']] == [1,2]
        with np.load(output/detection['arrays'], allow_pickle=False) as arrays:
            assert arrays['we_history'].shape == (2,20)
            assert arrays['aum_history'].shape == (2,20)
            assert arrays['is_corrupted'].sum() == round(detection['noise_ratio']*20)
            assert arrays['observation_counts_history'][-1].min() == 2
    before = (output/'metrics.csv').read_bytes()
    second = aggregate(runs, output, settings, evaluator)
    assert len(evaluator.calls) == 6  # no repeat inference for unchanged final checkpoints
    assert second == first
    assert (output/'metrics.csv').read_bytes() == before
    with (output/'metrics.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len({(row['configuration_sha256'],row['noise_ratio'],row['method']) for row in rows}) == 6
    assert all(row['test_accuracy_pct'] == '60.0' for row in rows)
    assert all(row['checkpoint_publication_seconds'] == '' for row in rows)  # missing time is never invented


@pytest.mark.parametrize('kind', ['partial','missing'])
def test_missing_or_incomplete_final_stage_is_pending_not_evaluated(tmp_path, kind):
    settings, runs = scientific_fixture(tmp_path, incomplete=(.4,'hybrid',kind))
    evaluator = FixtureEvaluator()
    summary = aggregate(runs, tmp_path/'results', settings, evaluator)
    assert not summary['complete'] and summary['row_count'] == 5
    assert (.4,'hybrid') not in evaluator.calls
    assert any('hybrid' in message for message in summary['pending'])


def test_partial_noise_then_new_condition_upserts_without_duplicates(tmp_path):
    settings, runs = scientific_fixture(tmp_path, ratios=(.2,))
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    summary = aggregate(runs, output, settings, evaluator)
    assert summary['row_count'] == 3 and not summary['complete']
    scientific_fixture(tmp_path, ratios=(.4,))
    summary = aggregate(runs, output, settings, evaluator)
    assert summary['row_count'] == 6 and summary['complete']
    assert len(evaluator.calls) == 6


def test_new_complete_checkpoint_changes_cache_identity_and_replaces_row(tmp_path):
    settings, runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    aggregate(runs, output, settings, evaluator)
    pointer = runs/'noise_20'/'hybrid'/'latest.json'
    journal = json.loads(pointer.read_text())
    store = CheckpointStore(pointer.parent, journal['identity'], writer=writer, reader=reader)
    state, _ = store.load()
    from src.checkpoints import file_digest
    state['model']['weight'] += 1
    state['model_sha256'] = model_hash(state['model'])
    # Fixture models a new immutable final-checkpoint publication, not an epoch update.
    path = pointer.parent/'replacement.pt'
    writer(state,path)
    atomic_json(pointer, {'identity': store.identity, 'entries': [{'epoch':2,'file':path.name,
                'sha256':file_digest(path),'size_bytes':path.stat().st_size}]})
    summary = aggregate(runs, output, settings, evaluator)
    assert summary['row_count'] == 6 and len(evaluator.calls) == 7


def test_failed_csv_publication_reuses_completed_accuracy_caches(tmp_path, monkeypatch):
    import src.evaluate as module
    settings, runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    original = module._write_csv
    def failed(*args):
        raise OSError('Drive publication interrupted')
    monkeypatch.setattr(module, '_write_csv', failed)
    with pytest.raises(OSError):
        aggregate(runs,tmp_path/'results',settings,evaluator)
    assert len(evaluator.calls) == 6
    assert not (tmp_path/'results'/'metrics_manifest.json').exists()
    monkeypatch.setattr(module, '_write_csv', original)
    aggregate(runs,tmp_path/'results',settings,evaluator)
    assert len(evaluator.calls) == 6


@pytest.mark.parametrize('change', ['smoke','code','configuration','subset','history'])
def test_incompatible_scientific_sources_are_rejected(tmp_path, change):
    settings, runs = scientific_fixture(tmp_path)
    manifest = runs/'noise_20'/'run.json'
    value = json.loads(manifest.read_text())
    if change == 'smoke': value['identity']['kind'] = 'synthetic_smoke'
    if change == 'code': value['identity']['code'] = {'git_sha':'other'}
    if change == 'configuration': value['identity']['configuration_sha256'] = 'other'
    if change == 'subset': value['identity']['dataset']['subset_checksum'] = 'other'
    if change != 'history':
        atomic_json(manifest,value)
    else:
        pointer = runs/'noise_20'/'warmup'/'latest.json'
        journal = json.loads(pointer.read_text())
        path = pointer.parent/journal['entries'][-1]['file']
        state = reader(path)
        state['warmup_history'] = []
        writer(state,path)
        from src.checkpoints import file_digest
        journal['entries'][-1]['sha256'] = file_digest(path)
        journal['entries'][-1]['size_bytes'] = path.stat().st_size
        atomic_json(pointer,journal)
    with pytest.raises((IdentityMismatch,ValueError)):
        aggregate(runs,tmp_path/'results',settings,FixtureEvaluator())


def test_corrupt_cache_is_not_silently_accepted_or_remeasured(tmp_path):
    settings, runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    aggregate(runs,output,settings,evaluator)
    cache = next((output/'accuracy_cache').glob('*.json'))
    record = json.loads(cache.read_text())
    record['value']['accuracy_pct'] = 99.
    atomic_json(cache,record)
    with pytest.raises(ValueError, match='cache identity/checksum'):
        aggregate(runs,output,settings,evaluator)
    assert len(evaluator.calls) == 6


def test_evaluation_cli_imports_and_refuses_cpu_without_torch(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = """
import sys
class BlockTorch:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch','torchvision'):
   raise AssertionError('unexpected torch import')
sys.meta_path.insert(0,BlockTorch())
import src.evaluate
"""
    subprocess.run([sys.executable,'-c',script],cwd=repo,check=True)
    result = subprocess.run([sys.executable,'-m','src.evaluate','--device','cpu',
               '--runs',str(tmp_path/'runs'),'--output-root',str(tmp_path/'results')],
               cwd=repo,capture_output=True,text=True)
    assert result.returncode != 0 and 'Colab CUDA' in result.stderr
    assert not (tmp_path/'results').exists()


def test_primary_detector_is_configured_last_epoch_even_when_earlier_auc_is_better(tmp_path):
    from src.data import load_training_metadata, load_analysis_metadata
    from src.evaluate import evaluate_detectors
    settings,runs = scientific_fixture(tmp_path,ratios=(.2,))
    root = runs/'noise_20'
    identity = json.loads((root/'run.json').read_text())['identity']
    saved = CheckpointStore(root/'warmup',{**identity,'stage':'warmup','target_epochs':2},reader=reader)
    state,entry = saved.load()
    analysis = load_analysis_metadata(root/'data')
    last = state['warmup_history'][-1]
    last['we'] = np.where(analysis.is_corrupted,1,2).astype(np.int32)
    last['margin_sum'] = np.where(analysis.is_corrupted,2,-2).astype(np.float32)
    state['statistics'] = last
    summary = evaluate_detectors(state,entry,load_training_metadata(root/'data'),analysis,tmp_path/'results',settings)
    assert summary['history'][0]['we_roc_auc'] == 1.
    assert summary['primary']['we_roc_auc'] == 0.
    assert summary['primary']['negative_aum_roc_auc'] == 0.
    assert summary['primary_epoch'] == 2


def test_corrupt_final_checkpoint_cannot_reuse_earlier_complete_result_row(tmp_path):
    settings,runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    aggregate(runs,output,settings,evaluator)
    pointer = runs/'noise_20'/'hybrid'/'latest.json'
    journal = json.loads(pointer.read_text())
    (pointer.parent/journal['entries'][-1]['file']).write_bytes(b'truncated')
    summary = aggregate(runs,output,settings,evaluator)
    assert summary['row_count'] == 5 and not summary['complete']
    assert len(evaluator.calls) == 6
    assert any('incomplete' in message for message in summary['pending'])


def test_empty_runs_report_pending_without_fabricated_metrics(tmp_path):
    settings = load_config()
    evaluator = FixtureEvaluator()
    summary = aggregate(tmp_path/'runs',tmp_path/'results',settings,evaluator)
    assert not summary['complete'] and summary['row_count'] == 0
    assert len(summary['pending']) == 2 and not evaluator.calls
    assert len((tmp_path/'results'/'metrics.csv').read_text().splitlines()) == 1


def test_results_from_another_configuration_are_preserved_and_rejected(tmp_path):
    settings,runs = scientific_fixture(tmp_path)
    evaluator = FixtureEvaluator()
    output = tmp_path/'results'
    aggregate(runs,output,settings,evaluator)
    before = (output/'metrics.csv').read_bytes()
    with pytest.raises(IdentityMismatch,match='Results root'):
        aggregate(runs,output,dict(settings,seed=43),evaluator)
    assert (output/'metrics.csv').read_bytes() == before and len(evaluator.calls) == 6


@pytest.mark.parametrize('failure',['count','percent','duration','fingerprint'])
def test_invalid_accuracy_not_published_to_cache(tmp_path,failure):
    settings = load_config()
    state = {'identity':{'fixture':True},'model_sha256':'0'*64}
    def invalid(state,settings):
        value = FixtureEvaluator()(dict(state,stage='fixture',identity={'dataset':{'noise_ratio':.2}}),settings)
        if failure == 'count': value['total'] = 100
        if failure == 'percent': value['accuracy_pct'] = 99.
        if failure == 'duration': value['evaluation_seconds'] = float('nan')
        if failure == 'fingerprint': value['test_data_sha256'] = ''
        return value
    with pytest.raises(ValueError):
        cached_accuracy(state,{'sha256':'1'*64},settings,invalid,tmp_path)
    assert not list(tmp_path.glob('accuracy_cache/*.json'))
