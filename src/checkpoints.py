"""Verified epoch-boundary checkpoints, with lazy PyTorch imports.

Only load trusted project checkpoints: torch's full-state loader uses pickle.
Immutable payloads are published before a small journal pointer. Keep previous
verified generations and pointer backup; temporary/unpublished files are ignored.
Drive is not a database: use one writer per experiment and retain backups.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import uuid

import numpy as np
from config import configuration_fingerprint


class IdentityMismatch(ValueError):
    pass


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Read back a complete temporary JSON before replacing the published file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with temporary.open('w', encoding='utf-8') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    if json.loads(temporary.read_text(encoding='utf-8')) != value:
        raise OSError('JSON read-back verification failed')
    temporary.replace(path)


def code_identity(repo):
    """Git commit plus implementation digest catches stale or edited imports."""
    repo = Path(repo)
    sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, check=True,
                         capture_output=True, text=True).stdout.strip()
    digest = hashlib.sha256()
    paths = [repo / 'config.py', *sorted((repo / 'src').glob('*.py')),
             *sorted(repo.glob('requirements*.txt'))]
    for path in paths:
        digest.update(path.relative_to(repo).as_posix().encode())
        digest.update(path.read_bytes())
    return {'git_sha': sha, 'source_sha256': digest.hexdigest()}


def environment(device):
    import torch
    import torchvision
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'device_type': device.type,
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}


def run_identity(settings, metadata, code, runtime, kind):
    # GPU changes are recorded on each checkpoint, not treated as bitwise-equivalent.
    return {'schema_version': 1, 'kind': kind, 'configuration': settings,
            'configuration_sha256': configuration_fingerprint(settings),
            'dataset': metadata, 'code': code,
            'runtime': {key: value for key, value in runtime.items() if key != 'gpu'}}


def seed_streams(seed):
    def derive(tag):
        return int(np.random.SeedSequence([seed, tag]).generate_state(1)[0])
    return {'model': derive(10), 'warmup': derive(11), 'branch': derive(12),
            'shuffle': derive(13), 'statistics': derive(14), 'synthetic': derive(15)}


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(generators):
    import torch
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'generators': {key: generator.get_state() for key, generator in generators.items()}}


def restore_rng(state, generators):
    import torch
    if set(state['generators']) != set(generators):
        raise ValueError('DataLoader generator set differs from checkpoint')
    if len(state['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError('CUDA RNG device count differs from checkpoint')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])
    for key, generator in generators.items():
        generator.set_state(state['generators'][key].cpu())


def model_state_hash(state):
    """Hash every parameter AND buffer, including scalar BatchNorm counters."""
    import torch
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _write_torch(state, path):
    import torch
    with Path(path).open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _read_torch(path):
    import torch
    return torch.load(path, map_location='cpu', weights_only=False)


class CheckpointStore:
    """Journal per noise/stage/branch. Identity checked before deserialisation.

    writer/reader injection allows publication-failure tests without PyTorch.
    Completed epochs must be sequential. The latest two verified entries remain
    in the pointer; older immutable files are retained, never silently deleted.
    """
    def __init__(self, directory, identity, *, writer=None, reader=None):
        self.directory = Path(directory)
        self.identity = identity
        self.writer = writer or _write_torch
        self.reader = reader or _read_torch
        self.pointer = self.directory / 'latest.json'
        self.backup = self.directory / 'latest.previous.json'

    def exists(self):
        return self.pointer.exists() or self.backup.exists()

    def _journal(self, path):
        journal = json.loads(path.read_text(encoding='utf-8'))
        if journal['identity'] != self.identity:
            raise IdentityMismatch('Checkpoint code/configuration/dataset/runtime identity mismatch')
        entries = journal['entries']
        if not entries or len(entries) > 2:
            raise ValueError('Invalid checkpoint journal')
        for entry in entries:
            if (type(entry['epoch']) is not int or entry['epoch'] < 1
                    or Path(entry['file']).name != entry['file']
                    or len(entry['sha256']) != 64
                    or type(entry['size_bytes']) is not int or entry['size_bytes'] <= 0):
                raise ValueError('Invalid checkpoint entry')
        return journal

    def load(self):
        errors = []
        for pointer in (self.pointer, self.backup):
            if not pointer.exists():
                continue
            try:
                journal = self._journal(pointer)
            except IdentityMismatch:
                raise
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(str(error))
                continue
            for entry in reversed(journal['entries']):
                try:
                    path = self.directory / entry['file']
                    if path.stat().st_size != entry['size_bytes'] or file_digest(path) != entry['sha256']:
                        raise ValueError('Checkpoint checksum mismatch')
                    state = self.reader(path)
                    if state['identity'] != self.identity:
                        raise IdentityMismatch('Payload identity mismatch')
                    if state['completed_epoch'] != entry['epoch']:
                        raise ValueError('Checkpoint epoch mismatch')
                    return state, entry
                except IdentityMismatch:
                    raise
                except (ValueError, KeyError, TypeError, OSError, EOFError, RuntimeError) as error:
                    errors.append(str(error))
        raise ValueError('No verified checkpoint generation: ' + '; '.join(errors))

    def save(self, state):
        if state['identity'] != self.identity:
            raise IdentityMismatch('Refusing checkpoint with different identity')
        previous = None
        if self.exists():
            _, previous = self.load()
        epoch = state['completed_epoch']
        if type(epoch) is not int or epoch != (previous['epoch'] + 1 if previous else 1):
            raise ValueError('Checkpoints must publish the next completed epoch exactly once')
        self.directory.mkdir(parents=True, exist_ok=True)
        name = f'epoch_{epoch:03d}_{uuid.uuid4().hex}.pt'
        path = self.directory / name
        temporary = path.with_suffix('.pt.tmp')
        self.writer(state, temporary)
        digest = file_digest(temporary)
        size = temporary.stat().st_size
        checked = self.reader(temporary)
        if checked['identity'] != self.identity or checked['completed_epoch'] != epoch:
            raise ValueError('Checkpoint read-back verification failed')
        temporary.replace(path)
        if path.stat().st_size != size or file_digest(path) != digest:
            raise OSError('Published checkpoint checksum mismatch')
        entry = {'epoch': epoch, 'file': name, 'sha256': digest, 'size_bytes': size}
        entries = ([previous] if previous else []) + [entry]
        # Backup records the actual verified predecessor, even after recovery.
        if previous:
            atomic_json(self.backup, {'identity': self.identity, 'entries': [previous]})
        atomic_json(self.pointer, {'identity': self.identity, 'entries': entries})
        return entry
