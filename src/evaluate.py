"""Final test/detector evaluation and idempotent results aggregation.

This module alone owns test accuracy, detection ROC-AUC and metrics.csv. Full
model inference is Colab CUDA-only; NumPy orchestration/metric tests need no
PyTorch, downloads or training. No quality metric feeds back into training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
from config import configuration_fingerprint, load_config
from .checkpoints import (CheckpointStore, IdentityMismatch, atomic_json, code_identity,
                          environment, file_digest, model_state_hash)
from .data import (CIFAR_MEAN, CIFAR_STD, array_checksum, load_analysis_metadata,
                   load_training_metadata)
from .scores import suspicion_scores
from .statistics import SampleStatistics
from .train import METHODS, computational_counts, require_device, validate_frozen

SCHEMA_VERSION = 1
CSV_FIELDS = ('configuration_sha256', 'noise_ratio', 'method', 'seed', 'train_size',
              'test_size', 'corruptions', 'warmup_epochs', 'continuation_epochs', 'model_epochs',
              'test_correct', 'test_accuracy_pct', 'we_roc_auc', 'negative_aum_roc_auc',
              'hybrid_roc_auc', 'checkpoint_sha256', 'model_sha256', 'evaluation_seconds',
              'training_seconds_excluding_checkpoint', 'checkpoint_publication_seconds',
              'timing_complete', 'final_common_noisy_ce')


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def test_protocol(settings):
    return {'schema_version': SCHEMA_VERSION, 'dataset': 'CIFAR10',
            'split': 'untouched official test', 'sample_count': settings['test_size'],
            'classes': 10, 'normalization_mean': list(CIFAR_MEAN),
            'normalization_std': list(CIFAR_STD), 'augmentation': False, 'amp': False}


def _finite_number(value, name, *, minimum=0, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < minimum
            or (maximum is not None and value > maximum)):
        raise ValueError(f'Invalid {name}')
    return value


def detector_metrics(we, aum, noisy_labels, mask, observation_counts):
    """Retrospective only: raw WE and -AUM are the prespecified ROC scores."""
    from sklearn.metrics import roc_auc_score
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.shape != np.asarray(we).shape:
        raise ValueError('Corruption mask must be a boolean vector aligned with saved statistics')
    if not mask.any() or mask.all():
        raise ValueError('Detection ROC-AUC requires both clean and corrupted examples; no default AUC is substituted')
    scores = suspicion_scores(we, aum, noisy_labels, observation_counts=observation_counts)
    return {'we_roc_auc': float(roc_auc_score(mask, we)),
            'negative_aum_roc_auc': float(roc_auc_score(mask, -np.asarray(aum))),
            'hybrid_roc_auc': float(roc_auc_score(mask, scores['hybrid']))}, scores


def _snapshots(state, settings):
    history = state['warmup_history']
    if len(history) != settings['warmup_epochs']:
        raise ValueError('Missing historical warm-up snapshots')
    snapshots = []
    for epoch, value in enumerate(history, 1):
        tracker = SampleStatistics.from_state_dict(value)
        if tracker.size != settings['train_size'] or tracker.tracked_epochs != epoch:
            raise ValueError('Warm-up history has missing/invalid epoch coverage')
        snapshots.append(tracker)
    return snapshots


def _validate_checkpoint(state, identity, stage, settings, hasher):
    target = settings['warmup_epochs'] if stage == 'warmup' else settings['continuation_epochs']
    expected = {**identity, 'stage': stage, 'target_epochs': target}
    if state['schema_version'] != SCHEMA_VERSION or state['identity'] != expected or state['stage'] != stage:
        raise IdentityMismatch('Evaluation checkpoint identity/stage mismatch')
    completed = state['completed_epoch']
    if type(completed) is not int or not 1 <= completed <= target:
        raise ValueError('Invalid completed checkpoint epoch')
    tracker = SampleStatistics.from_state_dict(state['statistics'])
    expected_epochs = completed if stage == 'warmup' else settings['warmup_epochs'] + (0 if stage == 'standard' else completed)
    if tracker.size != settings['train_size'] or tracker.tracked_epochs != expected_epochs:
        raise ValueError('Checkpoint statistics coverage differs from completed epochs')
    if hasher(state['model']) != state['model_sha256']:
        raise ValueError('Complete model parameter/buffer hash mismatch')
    logs = state['logs']
    if len(logs) != completed or [row['epoch'] for row in logs] != list(range(1, completed+1)):
        raise ValueError('Missing or duplicate training log epochs')
    for row in logs:
        if row['sample_count'] != settings['train_size']:
            raise ValueError('Training log sample count mismatch')
        for key, value in row.items():
            _finite_number(value, f'training log {key}')
    return completed == target


def _load_stage(directory, identity, stage, settings, reader, hasher):
    target = settings['warmup_epochs'] if stage == 'warmup' else settings['continuation_epochs']
    store = CheckpointStore(directory, {**identity, 'stage': stage, 'target_epochs': target}, reader=reader)
    if not store.exists():
        return None, None, f'{stage}: checkpoint not yet present'
    state, entry = store.load()
    if not _validate_checkpoint(state, identity, stage, settings, hasher):
        return None, None, f'{stage}: incomplete ({state["completed_epoch"]}/{target} epochs); resume training first'
    return state, entry, None


def _validate_run_identity(identity, settings, code, ratio):
    expected_schedule = {key: settings[key] for key in ('warmup_epochs', 'continuation_epochs')}
    if (identity['schema_version'] != SCHEMA_VERSION or identity['kind'] != 'experiment'
            or identity['configuration'] != settings
            or identity['configuration_sha256'] != configuration_fingerprint(settings)
            or identity['code'] != code or identity['schedule'] != expected_schedule
            or identity['dataset']['noise_ratio'] != ratio):
        raise IdentityMismatch('Only matching frozen scientific runs can enter evaluation; smoke/benchmark/mixed identities are excluded')


def _save_arrays(path, arrays):
    """Publish verified plot inputs. Reuse identical existing arrays without writes."""
    path = Path(path)
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(arrays) or any(not np.array_equal(previous[key], value) for key,value in arrays.items()):
                raise ValueError('Existing detector arrays conflict with their immutable source identity')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as checked:
        if set(checked.files) != set(arrays) or any(not np.array_equal(checked[key], value) for key,value in arrays.items()):
            raise OSError('Detector arrays read-back failed')
    temporary.replace(path)


def evaluate_detectors(warmup, entry, training, analysis, output_root, settings):
    snapshots = _snapshots(warmup, settings)
    if any(not np.array_equal(snapshots[-1].state_dict()[key], warmup['statistics'][key])
           for key in ('we', 'margin_sum', 'observation_counts')):
        raise ValueError('Primary warm-up snapshot differs from final warm-up statistics')
    history = []
    for epoch, tracker in enumerate(snapshots, 1):
        metrics, scores = detector_metrics(tracker.we, tracker.aum, training.noisy_labels,
                                           analysis.is_corrupted, tracker.observation_counts)
        history.append({'epoch': epoch, **metrics})
    source = {'schema_version': SCHEMA_VERSION, 'warmup_checkpoint_sha256': entry['sha256'],
              'configuration_sha256': configuration_fingerprint(settings),
              'clean_labels_sha256': array_checksum(analysis.clean_labels),
              'corruption_mask_sha256': array_checksum(analysis.is_corrupted)}
    key = json_digest(source)
    relative = Path('detectors') / f'noise_{round(training.noise_ratio*100)}' / f'{key}.npz'
    arrays = {'subset_positions': np.arange(settings['train_size']),
              'original_indices': training.original_indices, 'noisy_labels': training.noisy_labels,
              'clean_labels': analysis.clean_labels, 'is_corrupted': analysis.is_corrupted,
              'epochs': np.arange(1, len(snapshots)+1),
              'we_history': np.stack([t.we for t in snapshots]),
              'margin_sum_history': np.stack([t.margin_sum for t in snapshots]),
              'observation_counts_history': np.stack([t.observation_counts for t in snapshots]),
              'aum_history': np.stack([t.aum for t in snapshots]),
              'primary_we': snapshots[-1].we, 'primary_negative_aum': -snapshots[-1].aum,
              'primary_we_rank': scores['ido_lite'], 'primary_negative_aum_rank': scores['aum'],
              'primary_hybrid': scores['hybrid']}
    _save_arrays(Path(output_root) / relative, arrays)
    summary = {'source': source, 'primary_epoch': settings['warmup_epochs'],
               'primary': history[-1], 'history': history,
               'arrays': relative.as_posix(), 'arrays_sha256': file_digest(Path(output_root)/relative),
               'interpretation': 'Retrospective synthetic-noise detection; primary epoch fixed by configuration. No tuning or checkpoint selection.'}
    atomic_json((Path(output_root)/relative).with_suffix('.json'), summary)
    return summary


def _accuracy_value(value, settings):
    for key in ('correct', 'total'):
        if type(value[key]) is not int:
            raise ValueError('Accuracy counts must be integers')
    if value['total'] != settings['test_size'] or not 0 <= value['correct'] <= value['total']:
        raise ValueError('Accuracy must cover the untouched complete official test split')
    accuracy = _finite_number(value['accuracy_pct'], 'test accuracy', maximum=100)
    if not math.isclose(accuracy, 100*value['correct']/value['total'], abs_tol=1e-10):
        raise ValueError('Accuracy percent inconsistent with counts')
    _finite_number(value['evaluation_seconds'], 'evaluation duration')
    if not isinstance(value['test_data_sha256'], str) or len(value['test_data_sha256']) != 64:
        raise ValueError('Missing evaluated test-data fingerprint')


def cached_accuracy(state, entry, settings, evaluator, output_root):
    source = {'schema_version': SCHEMA_VERSION, 'checkpoint_sha256': entry['sha256'],
              'identity': state['identity'], 'model_sha256': state['model_sha256'],
              'protocol': test_protocol(settings)}
    key = json_digest(source)
    path = Path(output_root) / 'accuracy_cache' / f'{key}.json'
    if path.exists():
        record = json.loads(path.read_text(encoding='utf-8'))
        if (record['source'] != source or record['record_sha256'] != json_digest({'source': record['source'], 'value': record['value']})):
            raise ValueError('Accuracy cache identity/checksum mismatch; preserve and inspect the artifact')
        _accuracy_value(record['value'], settings)
        return record['value'], path.relative_to(output_root).as_posix()
    value = evaluator(state, settings)
    _accuracy_value(value, settings)
    record = {'source': source, 'value': value}
    atomic_json(path, {**record, 'record_sha256': json_digest(record)})
    return value, path.relative_to(output_root).as_posix()


def accuracy_on_loader(model, loader, device):
    """Tiny CPU unit fixtures allowed; production caller enforces CUDA/10k split."""
    import torch
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler) or loader.drop_last:
        raise ValueError('Test loader must preserve every official test example exactly once')
    modes = [(module, module.training) for module in model.modules()]
    correct, total = 0, 0
    try:
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(device))
                if logits.shape != (len(labels), 10) or not logits.is_floating_point() or not torch.isfinite(logits).all():
                    raise ValueError('Expected finite ten-class evaluation logits')
                if labels.shape != (len(images),) or labels.dtype != torch.long or torch.any((labels < 0) | (labels > 9)):
                    raise ValueError('Invalid official test labels')
                correct += int((logits.argmax(1) == labels.to(device)).sum().item())
                total += len(labels)
    finally:
        for module, training in modes:
            module.training = training
    if total != len(loader.dataset) or not total:
        raise ValueError('Incomplete/empty official test evaluation')
    return {'correct': correct, 'total': total, 'accuracy_pct': 100*correct/total}


class TestEvaluator:
    """Lazily restore/read official test data once, only for missing cache entries."""
    def __init__(self, data_root, archive_root, device, runtime):
        self.data_root, self.archive_root = data_root, archive_root
        self.device, self.runtime = device, runtime
        self.dataset = None

    def __call__(self, state, settings):
        import torch
        from .data import load_cifar10_test
        from .model import build_model
        from .statistics import seed_worker
        require_device(str(self.device))
        started = perf_counter()
        if self.dataset is None:
            self.dataset = load_cifar10_test(self.data_root, archive_root=self.archive_root)
            self.data_sha256 = json_digest({'images': array_checksum(self.dataset.images),
                                            'labels': array_checksum(self.dataset.labels)})
        if len(self.dataset) != settings['test_size']:
            raise ValueError('Official test size mismatch')
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=settings['batch_size'],
                    shuffle=False, drop_last=False, num_workers=settings['num_workers'],
                    generator=torch.Generator().manual_seed(settings['seed']),
                    worker_init_fn=seed_worker, persistent_workers=False)
        model = build_model(settings).to(self.device)
        model.load_state_dict(state['model'])
        torch.cuda.synchronize(self.device)
        result = accuracy_on_loader(model, loader, self.device)
        torch.cuda.synchronize(self.device)
        return {**result, 'evaluation_seconds': perf_counter()-started,
                'test_data_sha256': self.data_sha256, 'environment': self.runtime}


def training_summary(directory, state):
    publication, missing = 0., []
    for row in state['logs']:
        path = Path(directory) / f'timing_{row["epoch"]:03d}.json'
        if not path.exists():
            missing.append(row['epoch'])
            continue
        timing = json.loads(path.read_text(encoding='utf-8'))
        if timing['epoch'] != row['epoch'] or timing['epoch_seconds_excluding_checkpoint'] != row['epoch_seconds_excluding_checkpoint']:
            raise ValueError('Timing sidecar does not match committed epoch logs')
        publication += _finite_number(timing['publication_seconds'], 'checkpoint publication time')
    summary = {}
    keys = set.intersection(*(set(row) for row in state['logs'])) - {'epoch', 'sample_count', 'epoch_seconds_excluding_checkpoint', 'amp_skipped_steps'}
    denominator = sum(row['sample_count'] for row in state['logs'])
    for key in sorted(keys):
        summary[key] = sum(row['sample_count']*row[key] for row in state['logs'])/denominator
    return {'training_seconds_excluding_checkpoint': sum(row['epoch_seconds_excluding_checkpoint'] for row in state['logs']),
            'checkpoint_publication_seconds': publication if not missing else None,
            'timing_complete': not missing, 'missing_timing_epochs': missing,
            'sample_weighted_diagnostics': summary,
            'amp_skipped_steps': sum(row.get('amp_skipped_steps', 0) for row in state['logs']),
            'logs': state['logs']}


def _write_csv(path, rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        stream.write(output.getvalue())
        stream.flush()
        os.fsync(stream.fileno())
    with temporary.open(encoding='utf-8', newline='') as stream:
        checked = list(csv.DictReader(stream))
    if len(checked) != len(rows):
        raise OSError('Metrics CSV read-back failed')
    temporary.replace(path)


def evaluate_runs(runs, output_root, settings, code, runtime, evaluator, *,
                  checkpoint_reader=None, model_hasher=None):
    """Rebuild one row per current configuration/noise/method from verified sources.

    Reader/hasher/evaluator injection is for synthetic, torch-free tests. CLI
    always uses trusted PyTorch checkpoints and the CUDA TestEvaluator.
    """
    runs, output_root = Path(runs), Path(output_root)
    hasher = model_hasher or model_state_hash
    scope = {'schema_version': SCHEMA_VERSION, 'configuration_sha256': configuration_fingerprint(settings),
             'configuration': settings, 'code': code, 'protocol': test_protocol(settings)}
    manifest = output_root / 'evaluation_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != scope:
        raise IdentityMismatch('Results root belongs to a different code/configuration/protocol')
    atomic_json(manifest, scope)
    rows, reports, pending, detections = [], [], [], []
    runtimes, subsets, test_fingerprints = [], [], []
    for ratio in settings['noise_ratios']:
        name = f'noise_{round(ratio*100)}'
        directory = runs / name
        if not (directory / 'run.json').exists():
            pending.append(f'{name}: no scientific run manifest yet')
            continue
        identity = json.loads((directory/'run.json').read_text(encoding='utf-8'))['identity']
        _validate_run_identity(identity, settings, code, ratio)
        if identity['runtime'] != {key:value for key,value in runtime.items() if key != 'gpu'}:
            raise IdentityMismatch('Evaluation runtime differs from scientific run versions')
        runtimes.append(identity['runtime'])
        subsets.append(identity['dataset']['subset_checksum'])
        if any(value != subsets[0] for value in subsets):
            raise IdentityMismatch('Noise conditions selected different training image IDs')
        if any(value != runtimes[0] for value in runtimes):
            raise IdentityMismatch('Noise conditions used different training runtime versions')
        training = load_training_metadata(directory/'data')
        if any(identity['dataset'].get(key) != value for key,value in training.summary().items()):
            raise IdentityMismatch('Training metadata/configuration checksum mismatch')
        if training.seed != settings['seed'] or len(training.noisy_labels) != settings['train_size']:
            raise IdentityMismatch('Training seed/size differs from frozen settings')
        analysis = load_analysis_metadata(directory/'data')
        warmup, warmup_entry, reason = _load_stage(directory/'warmup', identity, 'warmup', settings, checkpoint_reader, hasher)
        if reason:
            pending.append(f'{name}: {reason}')
            continue
        detection = evaluate_detectors(warmup, warmup_entry, training, analysis, output_root, settings)
        detections.append({'noise_ratio': ratio, **detection})
        warmup_summary = training_summary(directory/'warmup', warmup)
        snapshots = _snapshots(warmup, settings)
        for method in METHODS:
            state, entry, reason = _load_stage(directory/method, identity, method, settings, checkpoint_reader, hasher)
            if reason:
                pending.append(f'{name}: {reason}')
                continue
            if state['initial_model_sha256'] != warmup['model_sha256']:
                raise IdentityMismatch('Continuation started from a different shared model state')
            branch_history = _snapshots(state, settings)
            for left,right in zip(snapshots, branch_history):
                if any(not np.array_equal(left.state_dict()[key], right.state_dict()[key]) for key in ('we','margin_sum','observation_counts')):
                    raise IdentityMismatch('Continuation has different shared warm-up statistics')
            if method == 'standard' and any(not np.array_equal(state['statistics'][key], warmup['statistics'][key]) for key in ('we','margin_sum','observation_counts')):
                raise ValueError('Standard continuation must not update WE/AUM')
            accuracy, cache_path = cached_accuracy(state, entry, settings, evaluator, output_root)
            test_fingerprints.append(accuracy['test_data_sha256'])
            if any(value != test_fingerprints[0] for value in test_fingerprints):
                raise IdentityMismatch('Final methods were evaluated on different test data')
            branch_summary = training_summary(directory/method, state)
            complete_timing = warmup_summary['timing_complete'] and branch_summary['timing_complete']
            row = {'configuration_sha256': scope['configuration_sha256'], 'noise_ratio': ratio, 'method': method,
                   'seed': settings['seed'], 'train_size': settings['train_size'], 'test_size': settings['test_size'],
                   'corruptions': int(analysis.is_corrupted.sum()),
                   'warmup_epochs': settings['warmup_epochs'], 'continuation_epochs': settings['continuation_epochs'],
                   'model_epochs': settings['warmup_epochs']+settings['continuation_epochs'],
                   'test_correct': accuracy['correct'], 'test_accuracy_pct': accuracy['accuracy_pct'],
                   **{key:value for key,value in detection['primary'].items() if key != 'epoch'},
                   'checkpoint_sha256': entry['sha256'], 'model_sha256': state['model_sha256'],
                   'evaluation_seconds': accuracy['evaluation_seconds'],
                   'training_seconds_excluding_checkpoint': warmup_summary['training_seconds_excluding_checkpoint']+branch_summary['training_seconds_excluding_checkpoint'],
                   'checkpoint_publication_seconds': warmup_summary['checkpoint_publication_seconds']+branch_summary['checkpoint_publication_seconds'] if complete_timing else None,
                   'timing_complete': complete_timing, 'final_common_noisy_ce': state['logs'][-1]['common_noisy_ce']}
            rows.append(row)
            reports.append({'row': row, 'checkpoint': f'{name}/{method}/{entry["file"]}',
                            'accuracy_cache': cache_path, 'evaluation': accuracy,
                            'warmup_training': warmup_summary, 'continuation_training': branch_summary})
    keys = {(row['configuration_sha256'],row['noise_ratio'],row['method']) for row in rows}
    expected_rows = len(settings['noise_ratios'])*len(METHODS)
    if len(keys) != len(rows) or len(rows) > expected_rows:
        raise ValueError('Duplicate/mixed result keys')
    complete = len(rows) == expected_rows and not pending
    summary = {'schema_version': SCHEMA_VERSION, 'scope': scope, 'complete': complete,
               'row_count': len(rows), 'expected_rows': expected_rows,
               'counts': computational_counts(settings), 'pending': pending,
               'methods': reports, 'detectors': detections,
               'interpretation': 'Single-seed exploratory results; IDO-inspired simplified implementation. No statistically reliable superiority claim. Per-method training time includes its shared warm-up; do not sum rows as total executed runtime.'}
    atomic_json(output_root/'metrics.json', summary)
    _write_csv(output_root/'metrics.csv', rows)
    atomic_json(output_root/'metrics_manifest.json',
                {'schema_version': SCHEMA_VERSION, 'scope_sha256': json_digest(scope),
                 'metrics_json_sha256': file_digest(output_root/'metrics.json'),
                 'metrics_csv_sha256': file_digest(output_root/'metrics.csv'),
                 'row_count': len(rows), 'complete': complete})
    print(f'Evaluation aggregation: {len(rows)}/{expected_rows} final rows; complete={complete}', flush=True)
    for reason in pending:
        print(f'PENDING: {reason}', flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/content/data'))
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--acknowledge-gpu-change', action='store_true')
    args = parser.parse_args()
    device = require_device(args.device)
    if args.runs.name != 'runs':
        raise ValueError('Evaluate the frozen experiment runs/ root, not a smoke/benchmark directory')
    if args.output_root.resolve() == args.runs.resolve():
        raise ValueError('Results must be separate from scientific run/checkpoint inputs')
    settings = load_config(args.runs.parent/'experiment_config.json')
    repo = Path(__file__).resolve().parents[1]
    code, runtime = code_identity(repo), environment(device)
    if subprocess.run(['git', 'status', '--porcelain'], cwd=repo, check=True,
                      capture_output=True, text=True).stdout.strip():
        raise ValueError('Final evaluation requires the clean, frozen experiment checkout.')
    validate_frozen(args.runs.parent/'experiment_config.json', args.runs, code, runtime,
                    acknowledge_gpu_change=args.acknowledge_gpu_change)
    archive_root = args.archive_root or args.runs.parent.parent.parent/'cache'/'cifar10'
    evaluator = TestEvaluator(args.data_root, archive_root, device, runtime)
    evaluate_runs(args.runs, args.output_root, settings, code, runtime, evaluator)


if __name__ == '__main__':
    main()
