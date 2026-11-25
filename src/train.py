"""Shared warm-up, fair continuation branches and epoch-boundary recovery.

Full training/benchmarks are Colab CUDA-only. CPU is allowed solely for explicit
synthetic smoke/unit fixtures, never the CIFAR experiment. No test-set access,
noise-quality metrics, final evaluation or metrics.csv writing occurs here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
from config import DEFAULT_CONFIG, configuration_fingerprint, load_config
from .checkpoints import (CheckpointStore, IdentityMismatch, atomic_json, capture_rng,
                          code_identity, environment, file_digest, model_state_hash,
                          restore_rng, run_identity, seed_all, seed_streams)
from .statistics import SampleStatistics, collect_epoch_statistics, seed_worker

METHODS = ('standard', 'ido_lite', 'hybrid')
LOSS_KEYS = ('pseudo_threshold', 'clean_weight_floor', 'pseudo_coefficient',
             'consistency_coefficient', 'near_zero_epsilon')


def require_device(device, *, smoke=False):
    # Refuse CPU/Windows production before any torch import or dataset side effect.
    if not smoke and (not str(device).startswith('cuda') or platform.system() == 'Windows'):
        raise RuntimeError('Full training/benchmarks require Colab CUDA; CPU only for explicit synthetic --smoke.')
    import torch
    result = torch.device(device)
    if result.type not in ('cpu', 'cuda') or (result.type == 'cuda' and not torch.cuda.is_available()):
        raise RuntimeError('Requested device unavailable; no automatic CPU fallback.')
    return result


def validate_frozen(config_path, output_root, code, runtime, *, acknowledge_gpu_change=False):
    """Require the notebook's frozen pair and exact selected scientific settings."""
    experiment = Path(output_root).resolve().parent
    frozen = experiment / 'experiment_config.json'
    manifest_path = experiment / 'manifest.json'
    if Path(config_path).resolve() != frozen or not frozen.exists() or not manifest_path.exists():
        raise ValueError('Full runs require the frozen experiment_config.json and manifest.json beside runs/.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['config_sha256'] != file_digest(frozen) or manifest['git_sha'] != code['git_sha']:
        raise IdentityMismatch('Frozen code/configuration identity mismatch')
    receipt_path = experiment / 'benchmark' / 'benchmark_report.json'
    if not receipt_path.is_file():
        raise ValueError('Complete the matching benchmark before full experiments.')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt['kind'] != 'benchmark' or receipt['identity']['code'] != code
            or receipt['configuration_sha256'] != configuration_fingerprint(load_config(frozen))):
        raise IdentityMismatch('Frozen configuration/code lacks a matching completed benchmark')
    if receipt['identity']['runtime'] != {key: value for key, value in runtime.items() if key != 'gpu'}:
        raise IdentityMismatch('Runtime differs from the completed benchmark')
    previous = manifest['environment']
    for key in ('python', 'torch', 'torchvision', 'cuda'):
        if key in previous and previous[key] != runtime[key]:
            raise IdentityMismatch(f'Restore frozen runtime version: {key}')
    if previous.get('gpu') != runtime['gpu'] and not acknowledge_gpu_change:
        raise ValueError('GPU changed: review reproducibility limits and pass --acknowledge-gpu-change.')
    return manifest


def computational_counts(settings):
    w, c = settings['warmup_epochs'], settings['continuation_epochs']
    return {'model_epochs': w + c, 'training_epochs': 2 * (w + 3*c),
            'training_image_view_passes': 2 * (w + 5*c),
            'statistics_passes': 2 * (w + 2*c), 'diagnostic_only_passes': 2*c,
            'final_test_passes': 6}


def runtime_projection(settings, warmup_seconds, standard_seconds, adaptive_seconds):
    times = (warmup_seconds, standard_seconds, adaptive_seconds)
    if any(not math.isfinite(t) or t <= 0 for t in times):
        raise ValueError('Benchmark epoch times must be finite and positive')
    w, c = settings['warmup_epochs'], settings['continuation_epochs']
    return {'main_training_seconds': 2 * (w*times[0] + c*times[1] + 2*c*times[2]),
            'counts': computational_counts(settings),
            'caveat': 'One measured epoch per type is uncertain. Add actual setup/download, benchmark, six final test passes, detection aggregation and debugging margin to fit the 4.5-hour run/evaluation budget. Freeze fallback using runtime only, never test/detection quality.'}


def _sync(device):
    if device.type == 'cuda':
        import torch
        torch.cuda.synchronize(device)


def _generators(streams):
    import torch
    return {name: torch.Generator().manual_seed(streams[key]) for name, key in
            (('train', 'shuffle'), ('statistics', 'statistics'))}


def _loaders(datasets, settings, generators, adaptive):
    import torch
    common = {'batch_size': settings['batch_size'], 'num_workers': settings['num_workers'],
              'drop_last': False, 'worker_init_fn': seed_worker, 'persistent_workers': False}
    training = torch.utils.data.DataLoader(datasets['adaptive' if adaptive else 'standard'],
                 shuffle=True, generator=generators['train'], **common)
    stats = torch.utils.data.DataLoader(datasets['statistics'], shuffle=False,
                 generator=generators['statistics'], **common)
    return training, stats


def train_epoch(model, loader, optimizer, scaler, settings, device, *, scores=None):
    import torch
    from .losses import AdaptiveEpochSummary, adaptive_loss
    model.train()
    summary = AdaptiveEpochSummary() if scores is not None else None
    total, count, skipped_steps = 0., 0, 0
    seen = np.zeros(len(loader.dataset), dtype=bool)
    for batch in loader:
        if scores is None:
            weak, labels, ids = batch
        else:
            weak, strong, labels, ids = batch
        positions = ids.cpu().numpy()
        if (np.any((positions < 0) | (positions >= len(seen)))
                or len(np.unique(positions)) != len(positions) or seen[positions].any()):
            raise ValueError('Training loader repeated/invalid stable IDs')
        seen[positions] = True
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits = model(weak.to(device))
            if scores is None:
                loss = torch.nn.functional.cross_entropy(logits.float(), labels)
            else:
                strong_logits = model(strong.to(device))
                loss, logs = adaptive_loss(logits, strong_logits, labels, scores[positions],
                                          **{key: settings[key] for key in LOSS_KEYS})
                summary.update(logs)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if not scaler.is_enabled() and any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise ValueError('Nonfinite training gradients')
        # With AMP, GradScaler handles overflow by skipping the unsafe step.
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped_steps += int(scaler.get_scale() < previous_scale)
        total += loss.detach().item() * len(labels)
        count += len(labels)
    if count != len(seen) or not seen.all():
        raise ValueError('Training epoch did not cover every sample exactly once')
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite model state after training')
    result = summary.compute() if summary is not None else {'sample_count': count, 'training_ce': total / count}
    return {**result, 'amp_skipped_steps': skipped_steps}


def _validate_state(state, identity, stage, target_epochs, size, warmup_epochs):
    if state['identity'] != identity or state['stage'] != stage or state['schema_version'] != 1:
        raise IdentityMismatch('Stage checkpoint identity mismatch')
    epoch = state['completed_epoch']
    if type(epoch) is not int or not 1 <= epoch <= target_epochs:
        raise ValueError('Invalid completed stage epoch')
    if len(state['logs']) != epoch or [row['epoch'] for row in state['logs']] != list(range(1, epoch + 1)):
        raise ValueError('Duplicate/missing checkpoint log epochs')
    tracker = SampleStatistics.from_state_dict(state['statistics'])
    expected = epoch if stage == 'warmup' else warmup_epochs + (0 if stage == 'standard' else epoch)
    if tracker.size != size or tracker.tracked_epochs != expected:
        raise ValueError('Checkpoint statistics epoch/count mismatch')
    history = state['warmup_history']
    if len(history) != (epoch if stage == 'warmup' else warmup_epochs):
        raise ValueError('Warm-up history missing from checkpoint')
    for index, snapshot in enumerate(history, 1):
        saved = SampleStatistics.from_state_dict(snapshot)
        if saved.tracked_epochs != index or saved.size != size:
            raise ValueError('Invalid warm-up snapshot sequence')
    if model_state_hash(state['model']) != state['model_sha256']:
        raise ValueError('Full model-state hash mismatch')
    return tracker


def _export_state(directory, state):
    # Derived views: checkpoint is authoritative. Regenerate after an interrupted
    # log/snapshot publication, without duplicate epoch entries or recomputation.
    directory = Path(directory)
    atomic_json(directory / 'epochs.json', state['logs'])
    if state['stage'] == 'warmup':
        for epoch, snapshot in enumerate(state['warmup_history'], 1):
            SampleStatistics.from_state_dict(snapshot).save_snapshot(directory / 'history' / f'epoch_{epoch:03d}.npz')


def run_stage(settings, datasets, directory, identity, stage, target_epochs, device,
              *, resume, streams, runtime, model_factory, shared=None, stop_after=None):
    import torch
    from .scores import suspicion_scores
    size = len(datasets['standard'])
    adaptive = stage in ('ido_lite', 'hybrid')
    stage_identity = {**identity, 'stage': stage, 'target_epochs': target_epochs}
    store = CheckpointStore(directory, stage_identity)
    if store.exists() and not resume:
        raise FileExistsError('Stage already has checkpoints; use --resume or a distinct isolated run root.')
    saved = store.load()[0] if store.exists() else None
    tracker = _validate_state(saved, stage_identity, stage, target_epochs, size,
                              identity['schedule']['warmup_epochs']) if saved else None
    if shared is not None:
        expected_shared = {**identity, 'stage': 'warmup', 'target_epochs': identity['schedule']['warmup_epochs']}
        _validate_state(shared, expected_shared, 'warmup', identity['schedule']['warmup_epochs'],
                        size, identity['schedule']['warmup_epochs'])
        if saved and saved['initial_model_sha256'] != shared['model_sha256']:
            raise IdentityMismatch('Branch did not start from this verified shared warm-up state')
    if saved and saved['completed_epoch'] == target_epochs:
        _export_state(directory, saved)
        print(f'{stage}: already complete; skipped', flush=True)
        return saved
    # Object construction may consume random state. Restore resumed RNGs last,
    # after model/optimizer/scaler/loaders are all constructed and loaded.
    seed_all(streams['model'])
    model = model_factory(settings).to(device)
    if shared is not None:
        model.load_state_dict(shared['model'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=settings['amp'] and device.type == 'cuda')
    generators = _generators(streams)
    training, statistics = _loaders(datasets, settings, generators, adaptive)
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        logs, history = saved['logs'], saved['warmup_history']
        initial_hash = saved['initial_model_sha256']
        restore_rng(saved['rng'], generators)
    else:
        tracker = SampleStatistics.from_state_dict(shared['statistics']) if shared else SampleStatistics(size)
        history = [SampleStatistics.from_state_dict(item).state_dict() for item in shared['warmup_history']] if shared else []
        logs = []
        initial_hash = model_state_hash(model.state_dict())
        if shared and initial_hash != shared['model_sha256']:
            raise ValueError('New branch failed complete-state warm-up reset')
        seed_all(streams['warmup'] if stage == 'warmup' else streams['branch'])
    for epoch in range(len(logs) + 1, target_epochs + 1):
        if stop_after is not None and epoch > stop_after:
            break  # deterministic fault-injection boundary for hosted unit tests
        _sync(device)
        started = perf_counter()
        scores = None
        if adaptive:
            scores = suspicion_scores(tracker.we, tracker.aum, datasets['statistics'].metadata.noisy_labels,
                                       observation_counts=tracker.observation_counts)[stage]
        diagnostics = train_epoch(model, training, optimizer, scaler, settings, device, scores=scores)
        tracked = tracker if stage != 'standard' else None
        common = collect_epoch_statistics(model, statistics, device=device, tracker=tracked,
                                          epoch=tracker.tracked_epochs + 1 if tracked is not None else None)
        if stage == 'warmup':
            history.append(tracker.state_dict())
        _sync(device)
        row = {'epoch': epoch, 'common_noisy_ce': common['noisy_ce'], **diagnostics,
               'epoch_seconds_excluding_checkpoint': perf_counter() - started}
        logs.append(row)
        checkpoint_started = perf_counter()
        state = {'schema_version': 1, 'identity': stage_identity, 'stage': stage,
                 'completed_epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scaler': scaler.state_dict(), 'statistics': tracker.state_dict(), 'warmup_history': history,
                 'rng': capture_rng(generators), 'logs': logs, 'environment': runtime, 'seed_streams': streams,
                 'model_sha256': model_state_hash(model.state_dict()), 'initial_model_sha256': initial_hash}
        _validate_state(state, stage_identity, stage, target_epochs, size, identity['schedule']['warmup_epochs'])
        entry = store.save(state)
        _export_state(directory, state)
        _sync(device)
        # Includes hashing, verification, checkpoint and derived Drive publication.
        atomic_json(Path(directory) / f'timing_{epoch:03d}.json',
                    {'epoch': epoch, 'checkpoint_sha256': entry['sha256'],
                     'publication_seconds': perf_counter() - checkpoint_started,
                     'epoch_seconds_excluding_checkpoint': row['epoch_seconds_excluding_checkpoint']})
        print(f'{stage} epoch {epoch}/{target_epochs}: CE={common["noisy_ce"]:.6f}; checkpoint={entry["sha256"][:12]}', flush=True)
        saved = state
    if saved is None:
        raise ValueError('No completed epoch')
    return saved


def run_stages(settings, datasets, metadata, output_root, device, *, resume=False,
               kind='experiment', code=None, runtime=None, model_factory=None,
               schedule=None, methods=METHODS):
    from .model import build_model
    output_root = Path(output_root)
    size = len(datasets['standard'])
    if size != settings['train_size'] or metadata['train_size'] != size or any(len(dataset) != size for dataset in datasets.values()):
        raise ValueError('Configured training size and all stable-ID datasets must match')
    reference_labels = datasets['statistics'].metadata.noisy_labels
    if any(not np.array_equal(dataset.metadata.noisy_labels, reference_labels) for dataset in datasets.values()):
        raise ValueError('All views must use identical provided labels in stable-ID order')
    code = code if code is not None else code_identity(Path(__file__).resolve().parents[1])
    runtime = runtime if runtime is not None else environment(device)
    schedule = schedule or {key: settings[key] for key in ('warmup_epochs', 'continuation_epochs')}
    identity = run_identity(settings, metadata, code, runtime, kind)
    identity['schedule'] = schedule
    manifest_path = output_root / 'run.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding='utf-8'))['identity'] != identity:
            raise IdentityMismatch('Run identity mismatch; refusing mixed outputs')
        if not resume:
            raise FileExistsError('Existing run: use --resume; frozen outputs are never replaced by a fresh start.')
    elif any(output_root.glob('*/latest*.json')):
        raise ValueError('Checkpoint directories without run manifest; inspect before recovery')
    else:
        atomic_json(manifest_path, {'identity': identity, 'seed_streams': seed_streams(settings['seed']),
                                   'environment': runtime, 'candidate_counts_both_noise_conditions': computational_counts(settings),
                                   'executed_stage_epochs': schedule['warmup_epochs'] + len(methods)*schedule['continuation_epochs']})
    streams = seed_streams(settings['seed'])
    factory = model_factory or build_model
    shared = run_stage(settings, datasets, output_root / 'warmup', identity, 'warmup',
                       schedule['warmup_epochs'], device, resume=resume, streams=streams,
                       runtime=runtime, model_factory=factory)
    for method in methods:
        if method not in METHODS:
            raise ValueError('Unknown continuation method')
        run_stage(settings, datasets, output_root / method, identity, method,
                  schedule['continuation_epochs'], device, resume=resume, streams=streams,
                  runtime=runtime, model_factory=factory, shared=shared)
    atomic_json(output_root / 'completion.json', {'identity': identity, 'completed_methods': list(methods),
                                                'warmup_epochs': schedule['warmup_epochs'],
                                                'continuation_epochs': schedule['continuation_epochs']})
    return identity


def synthetic_data(settings, noise):
    from .data import TrainingData, TrainingMetadata
    rng = np.random.default_rng(seed_streams(settings['seed'])['synthetic'])
    return TrainingData(rng.integers(0, 256, (512, 32, 32, 3), dtype=np.uint8),
                        TrainingMetadata(np.arange(512), rng.integers(0, 10, 512), settings['seed'], noise))


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--noise', type=float, choices=(.2, .4), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--smoke', action='store_true')
    mode.add_argument('--benchmark', action='store_true')
    parser.add_argument('--device', required=True)
    parser.add_argument('--data-root', type=Path, default=Path('/content/data'))
    parser.add_argument('--archive-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--acknowledge-gpu-change', action='store_true')
    args = parser.parse_args()
    settings = load_config(args.config)
    device = require_device(args.device, smoke=args.smoke)
    repo = Path(__file__).resolve().parents[1]
    code, runtime = code_identity(repo), environment(device)
    if not args.smoke and subprocess.run(['git', 'status', '--porcelain'], cwd=repo, check=True,
                                         capture_output=True, text=True).stdout.strip():
        raise ValueError('Benchmarks/full experiments require a clean, reviewed checkout.')
    from .data import array_checksum, make_training_datasets, prepare_cifar10_training
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    kind = 'synthetic_smoke' if args.smoke else ('benchmark' if args.benchmark else 'experiment')
    if kind == 'experiment':
        if args.output_root.name != 'runs':
            raise ValueError('Full experiments must use <experiment-root>/runs')
        validate_frozen(args.config, args.output_root, code, runtime,
                        acknowledge_gpu_change=args.acknowledge_gpu_change)
        root = args.output_root / f'noise_{round(args.noise * 100)}'
    elif args.benchmark:
        if args.resume:
            raise ValueError('Benchmark trials are isolated fresh measurements, not resumed experiments.')
        if (args.output_root.parent / 'experiment_config.json').exists():
            raise FileExistsError('Frozen experiment already exists; do not rebenchmark/replace its candidate.')
        root = args.output_root / 'trials' / stamp / f'noise_{round(args.noise * 100)}'
    else:
        root = args.output_root / 'synthetic_smoke' / f'noise_{round(args.noise * 100)}'
        settings = {**settings, 'train_size': 512, 'warmup_epochs': 1, 'continuation_epochs': 1, 'num_workers': 0}
    if args.smoke:
        data = synthetic_data(settings, args.noise)
    else:
        # Construction may use original labels only inside the separate data API.
        # Training receives images/provided labels/IDs; never load analysis files.
        # Standard layout: .../experiments/RUN_ID/runs -> .../cache/cifar10.
        # Nonstandard layouts should always supply --archive-root explicitly.
        archive_root = args.archive_root or args.output_root.parent.parent.parent / 'cache' / 'cifar10'
        data = prepare_cifar10_training(settings, args.noise, args.data_root, root / 'data', archive_root=archive_root)
    datasets = make_training_datasets(data)
    metadata = {**data.metadata.summary(), 'images_sha256': array_checksum(data.images),
                'source': 'generated random images/labels; no noise-quality interpretation' if args.smoke else 'official CIFAR-10 training split'}
    if args.benchmark:
        # Candidate subset, full two-view epoch + deterministic diagnostics + Drive
        # checkpoint I/O. One representative warm-up/standard/adaptive epoch each.
        before = perf_counter()
        identity = run_stages(settings, datasets, metadata, root, device, kind=kind,
                              code=code, runtime=runtime, schedule={'warmup_epochs': 1, 'continuation_epochs': 1},
                              methods=('standard', 'hybrid'))
        times, predecessor_reads = {}, {}
        for stage in ('warmup', 'standard', 'hybrid'):
            timing = json.loads((root / stage / 'timing_001.json').read_text())
            # Epoch 1 has no predecessor. Measure an additional verified read
            # so the projection includes the journal/read-back overhead incurred
            # when publishing subsequent epochs of a real stage.
            read_started = perf_counter()
            CheckpointStore(root / stage, {**identity, 'stage': stage, 'target_epochs': 1}).load()
            predecessor_reads[stage] = perf_counter() - read_started
            times[stage] = (timing['epoch_seconds_excluding_checkpoint'] + timing['publication_seconds']
                            + predecessor_reads[stage])
        report = {'kind': kind, 'identity': identity, 'epoch_seconds': times, 'predecessor_verification_seconds': predecessor_reads,
                  'benchmark_stages_seconds': perf_counter() - before,
                  'setup_and_benchmark_seconds': perf_counter() - started,
                  'projection': runtime_projection(settings, times['warmup'], times['standard'], times['hybrid']),
                  'candidate_root': str(root), 'configuration_sha256': configuration_fingerprint(settings),
                  'selection_rule': 'Review remaining time/overhead; fallback uniformly before both noise runs. No automatic scientific-setting changes.'}
        atomic_json(root.parent / 'benchmark_report.json', report)
        atomic_json(args.output_root / 'benchmark_report.json', report)
        atomic_json(args.output_root / 'experiment_config.json', settings)
        print(json.dumps(report, indent=2), flush=True)
    else:
        run_stages(settings, datasets, metadata, root, device, resume=args.resume,
                   kind=kind, code=code, runtime=runtime)
        print(f'{kind} stages complete. Final evaluation is separate; no final metrics computed.', flush=True)


if __name__ == '__main__':
    main()
