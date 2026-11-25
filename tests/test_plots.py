"""Step 6 uses verified synthetic exports; no torch, downloads or model runs."""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from src.checkpoints import atomic_json, file_digest
from src.plots import (FIGURES, LABELS, _context, _logs, comparison_rows,
                       generate_plots, load_bundle, render_figures)
from test_evaluate import FixtureEvaluator, aggregate, scientific_fixture


@pytest.fixture(scope='module')
def original_export(tmp_path_factory):
    base = tmp_path_factory.mktemp('plot-export')
    settings,runs = scientific_fixture(base)
    aggregate(runs,base/'results',settings,FixtureEvaluator())
    atomic_json(base/'experiment_config.json',settings)
    return base


@pytest.fixture
def exported(original_export,tmp_path):
    destination = tmp_path/'export'
    shutil.copytree(original_export,destination)
    return destination


def amend_metrics(base,change):
    path=base/'results'/'metrics.json'
    metrics=json.loads(path.read_text())
    change(metrics)
    atomic_json(path,metrics)
    receipt=base/'results'/'metrics_manifest.json'
    value=json.loads(receipt.read_text())
    value['metrics_json_sha256']=file_digest(path)
    atomic_json(receipt,value)


def test_embedded_logs_fallback_schedule_and_measured_table(original_export):
    bundle=load_bundle(original_export/'results')
    assert bundle['settings']['warmup_epochs']==2
    assert bundle['settings']['continuation_epochs']==2
    assert '2 warm-up + 2 continuation' in _context(bundle['settings'])
    assert 'non-default size/schedule' in _context(bundle['settings'])
    for key,curve in bundle['ce'].items():
        assert len(curve)==4
        np.testing.assert_array_equal(curve,[1,1,1,1])
    rows=comparison_rows(bundle)
    assert len(rows)==6
    assert all(row['test_accuracy_pct']==60. and row['delta_vs_standard_pp']==0 for row in rows)


def test_real_png_outputs_tables_and_source_hashes(original_export,tmp_path):
    manifest=generate_plots(original_export/'results',output_root=tmp_path/'outputs')
    assert manifest['warmup_boundary']==2.5
    assert manifest['primary_detector_epoch']==2
    assert manifest['representative_noise_ratio']==.2
    assert len(manifest['artifacts'])==6
    for name in FIGURES:
        path=tmp_path/'outputs'/'figures'/name
        with Image.open(path) as image:
            assert image.format=='PNG' and min(image.size)>=1000
            image.verify()
    for name,digest in manifest['artifacts'].items():
        assert file_digest(tmp_path/'outputs'/name)==digest
    with (tmp_path/'outputs'/'comparison_table.csv').open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==6 and all(row['test_accuracy_pct']=='60.00' for row in rows)
    text=(tmp_path/'outputs'/'comparison_table.md').read_text(encoding='utf-8')
    assert 'single' in text.lower() and '20 train' in text and '2 warm-up + 2 continuation' in text


def test_rendered_artists_follow_export_and_configured_epoch_boundary(original_export,tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    import src.plots as module
    bundle=load_bundle(original_export/'results')
    captured={}
    monkeypatch.setattr(module,'_save_figure',lambda fig,path:captured.setdefault(path.name,fig))
    try:
        render_figures(bundle,tmp_path)
        accuracy=captured[FIGURES[0]].axes[0]
        assert [bar.get_height() for bar in accuracy.patches]==[60.]*6
        for ax,ratio in zip(captured[FIGURES[3]].axes,bundle['settings']['noise_ratios']):
            np.testing.assert_array_equal(ax.lines[0].get_xdata(),[2.5,2.5])
            branches={line.get_label():line for line in ax.lines}
            for method in ('standard','ido_lite','hybrid'):
                line=branches[LABELS[method]]
                np.testing.assert_array_equal(line.get_xdata(),[2,3,4])
                np.testing.assert_array_equal(line.get_ydata(),bundle['ce'][ratio,method][1:])
    finally:
        for fig in captured.values():
            plt.close(fig)


def test_weighted_objectives_cannot_replace_common_deterministic_ce():
    logs=[{'epoch':1,'sample_count':20,'common_noisy_ce':1.9,'total_loss':.01,'training_ce':99.},
          {'epoch':2,'sample_count':20,'common_noisy_ce':1.2,'total_loss':.02,'training_ce':98.}]
    report={'continuation_training':{'logs':logs}}
    np.testing.assert_array_equal(_logs(report,'continuation',20,2,None,.2,'hybrid'),[1.9,1.2])
    del logs[0]['common_noisy_ce']
    with pytest.raises(ValueError,match='Common deterministic CE'):
        _logs(report,'continuation',20,2,None,.2,'hybrid')


def test_checksum_failure_prevents_any_output_generation(exported,tmp_path):
    path=exported/'results'/'metrics.csv'
    path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='publication checksum'):
        generate_plots(exported/'results',output_root=tmp_path/'should-not-exist')
    assert not (tmp_path/'should-not-exist').exists()


def test_corrupt_detector_file_rejected(exported):
    path=next((exported/'results'/'detectors').rglob('*.npz'))
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Detector arrays checksum'):
        load_bundle(exported/'results')


def test_inconsistent_auc_even_with_updated_publication_checksum_rejected(exported):
    amend_metrics(exported,lambda m:m['detectors'][0]['primary'].update(we_roc_auc=.123))
    with pytest.raises(ValueError,match='Saved detector AUC'):
        load_bundle(exported/'results')


def test_inconsistent_aum_cumulative_history_rejected(exported):
    path=next((exported/'results'/'detectors'/'noise_20').glob('*.npz'))
    with np.load(path,allow_pickle=False) as previous:
        arrays={key:previous[key] for key in previous.files}
    arrays['aum_history'][0]=0
    with path.open('wb') as stream:
        np.savez_compressed(stream,**arrays)
    amend_metrics(exported,lambda m:m['detectors'][0].update(arrays_sha256=file_digest(path)))
    with pytest.raises(ValueError,match='AUM does not match'):
        load_bundle(exported/'results')


@pytest.mark.parametrize('failure',['partial','duplicate','epoch','ce','path'])
def test_partial_or_malformed_exports_fail_clearly(exported,failure):
    def change(m):
        if failure=='partial':m['complete']=False
        if failure=='duplicate':m['methods'][2]=m['methods'][1]
        if failure=='epoch':m['detectors'][0]['primary_epoch']=1
        if failure=='ce':del m['methods'][0]['continuation_training']['logs'][0]['common_noisy_ce']
        if failure=='path':m['detectors'][0]['arrays']='../outside.npz'
    amend_metrics(exported,change)
    with pytest.raises(ValueError):
        load_bundle(exported/'results')


def test_supplied_config_mismatch_and_external_log_mismatch(exported):
    cfg=exported/'experiment_config.json'
    settings=json.loads(cfg.read_text())
    atomic_json(cfg,dict(settings,seed=43))
    with pytest.raises(ValueError,match='Supplied experiment configuration'):
        load_bundle(exported/'results',config=cfg)
    atomic_json(cfg,settings)
    log=exported/'runs'/'noise_20'/'warmup'/'epochs.json'
    log.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(log,[{'epoch':1,'common_noisy_ce':99}])
    with pytest.raises(ValueError,match='Embedded and external'):
        load_bundle(exported/'results',runs=exported/'runs')


def test_csv_row_order_does_not_change_method_mapping(exported):
    path=exported/'results'/'metrics.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream)
        rows=list(reader)
        names=reader.fieldnames
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names)
        writer.writeheader()
        writer.writerows(reversed(rows))
    receipt=exported/'results'/'metrics_manifest.json'
    value=json.loads(receipt.read_text())
    value['metrics_csv_sha256']=file_digest(path)
    atomic_json(receipt,value)
    assert len(load_bundle(exported/'results')['reports'])==6


def test_plotter_import_and_cli_help_without_torch():
    repo=Path(__file__).resolve().parents[1]
    script="""
import sys
class BlockTorch:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','torchvision'):
   raise AssertionError('plotting must not import torch')
sys.meta_path.insert(0,BlockTorch())
import src.plots
"""
    subprocess.run([sys.executable,'-c',script],cwd=repo,check=True)
    result=subprocess.run([sys.executable,'-m','src.plots','--help'],cwd=repo,capture_output=True,text=True)
    assert result.returncode==0 and '--results' in result.stdout and '--output-root' in result.stdout
