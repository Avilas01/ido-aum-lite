"""Torch-free execution guards, frozen provenance and runtime arithmetic."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
from config import load_config, configuration_fingerprint
from src.checkpoints import IdentityMismatch, atomic_json, file_digest, seed_streams
from src.train import computational_counts, require_device, runtime_projection, validate_frozen


def test_full_cpu_refusal_happens_before_importing_torch():
    with pytest.raises(RuntimeError, match='Colab CUDA'):
        require_device('cpu')


def test_modules_help_and_cpu_refusal_without_torch(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    script = """
import sys
class BlockTorch:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in ('torch', 'torchvision'):
   raise AssertionError('unexpected torch import')
sys.meta_path.insert(0, BlockTorch())
import src.train
import src.checkpoints
try:
 src.train.require_device('cpu')
except RuntimeError:
 pass
else:
 raise AssertionError('CPU run allowed')
"""
    subprocess.run([sys.executable, '-c', script], cwd=repo, check=True)
    result = subprocess.run([sys.executable, '-m', 'src.train', '--help'], cwd=repo,
                             capture_output=True, text=True)
    assert result.returncode == 0
    for flag in ('--smoke', '--benchmark', '--resume', '--archive-root', '--device'):
        assert flag in result.stdout
    result = subprocess.run([sys.executable, '-m', 'src.train', '--noise', '.2',
                             '--device', 'cpu', '--output-root', str(tmp_path / 'runs')],
                             cwd=repo, capture_output=True, text=True)
    assert result.returncode != 0 and 'Colab CUDA' in result.stderr
    assert not (tmp_path / 'runs').exists()


@pytest.mark.parametrize('w,c,epochs,views,stats,diagnostics', [(6,6,48,72,36,12), (6,4,36,52,28,8), (4,4,32,48,24,8)])
def test_default_and_fallback_runtime_accounting(w,c,epochs,views,stats,diagnostics):
    settings = dict(load_config(), warmup_epochs=w, continuation_epochs=c)
    counts = computational_counts(settings)
    assert counts == dict(model_epochs=w+c, training_epochs=epochs,
        training_image_view_passes=views, statistics_passes=stats,
        diagnostic_only_passes=diagnostics, final_test_passes=6)
    projection = runtime_projection(settings, 2., 3., 5.)
    assert projection['main_training_seconds'] == 2 * (w*2 + c*3 + 2*c*5)
    assert 'margin' in projection['caveat']


def test_invalid_timing_and_reproducible_recorded_streams():
    with pytest.raises(ValueError):
        runtime_projection(load_config(), 0., 1., 1.)
    with pytest.raises(ValueError):
        runtime_projection(load_config(), float('nan'), 1., 1.)
    assert seed_streams(42) == seed_streams(42)
    assert seed_streams(42) != seed_streams(43)
    assert len(set(seed_streams(42).values())) == len(seed_streams(42))


def frozen_fixture(tmp_path):
    frozen = tmp_path / 'experiment_config.json'
    atomic_json(frozen, load_config())
    code = {'git_sha': 'abc', 'source_sha256': '123'}
    runtime = {'python': '3', 'torch': '2', 'torchvision': '1', 'cuda': '12', 'gpu': 'T4'}
    atomic_json(tmp_path / 'manifest.json', {'git_sha': 'abc', 'config_sha256': file_digest(frozen), 'environment': runtime})
    atomic_json(tmp_path / 'benchmark' / 'benchmark_report.json', {'kind': 'benchmark',
                'identity': {'code': code, 'runtime': {k:v for k,v in runtime.items() if k != 'gpu'}}, 'configuration_sha256': configuration_fingerprint(load_config())})
    return frozen, code, runtime


def test_frozen_config_and_benchmark_code_checks(tmp_path):
    frozen, code, runtime = frozen_fixture(tmp_path)
    validate_frozen(frozen, tmp_path / 'runs', code, runtime)
    with pytest.raises(IdentityMismatch):
        validate_frozen(frozen, tmp_path / 'runs', dict(code, git_sha='different'), runtime)
    with pytest.raises(IdentityMismatch):
        validate_frozen(frozen, tmp_path / 'runs', dict(code, source_sha256='edited'), runtime)
    with pytest.raises(IdentityMismatch):
        validate_frozen(frozen, tmp_path / 'runs', code, dict(runtime, torch='new'))
    with pytest.raises(ValueError, match='GPU changed'):
        validate_frozen(frozen, tmp_path / 'runs', code, dict(runtime, gpu='A100'))
    validate_frozen(frozen, tmp_path / 'runs', code, dict(runtime, gpu='A100'), acknowledge_gpu_change=True)
    atomic_json(frozen, dict(load_config(), train_size=10000))
    with pytest.raises(IdentityMismatch):
        validate_frozen(frozen, tmp_path / 'runs', code, runtime)


def test_no_training_test_split_or_final_metric_access():
    source = (Path(__file__).resolve().parents[1] / 'src' / 'train.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    names = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({'load_cifar10_test', 'load_analysis_metadata', 'clean_labels', 'is_corrupted', 'roc_auc_score'})
    assert all('metrics.csv' not in node.value for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str) and not node.value.startswith('Shared warm-up'))



def test_full_run_requires_a_completed_benchmark_receipt(tmp_path):
    frozen, code, runtime = frozen_fixture(tmp_path)
    (tmp_path / 'benchmark' / 'benchmark_report.json').unlink()
    with pytest.raises(ValueError, match='Complete the matching benchmark'):
        validate_frozen(frozen, tmp_path / 'runs', code, runtime)
