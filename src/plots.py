"""Read verified evaluation exports and render four scientific PNG figures.

No torch, checkpoint/model loading, dataset download, training or evaluation.
Curves use saved common deterministic CE, never adaptive weighted objectives.
The saved primary epoch and configuration drive all labels, tables and bounds.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path

import numpy as np
from config import configuration_fingerprint, load_config
from .checkpoints import atomic_json, file_digest
from .data import array_checksum
from .evaluate import json_digest
from .scores import suspicion_scores
from .statistics import SampleStatistics
from .train import METHODS

LABELS = {'standard': 'Standard', 'ido_lite': 'IDO-lite (WE)', 'hybrid': 'Hybrid (WE + AUM)'}
COLORS = {'standard': '#475569', 'ido_lite': '#2563eb', 'hybrid': '#0f766e'}
FIGURES = ('accuracy_comparison.png', 'noise_detection_roc.png',
           'score_distributions.png', 'common_ce_curves.png')


def _expect(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, minimum=0, maximum=None):
    _expect(type(value) in (int, float) and math.isfinite(value) and value >= minimum
            and (maximum is None or value <= maximum), f'Invalid {name}')
    return value


def _artifact(root, relative):
    path = (root / relative).resolve()
    _expect(path.is_relative_to(root.resolve()), 'Artifact path leaves results directory')
    return path


def _logs(report, phase, size, count, root, ratio, method):
    logs = report.get(phase + '_training', {}).get('logs')
    name = 'warmup' if phase == 'warmup' else method
    external = root / f'noise_{round(100*ratio)}' / name / 'epochs.json' if root else None
    if external and external.exists():
        saved = json.loads(external.read_text(encoding='utf-8'))
        if logs is None:
            logs = saved
        else:
            _expect(logs == saved, 'Embedded and external epoch logs disagree')
    _expect(isinstance(logs, list) and len(logs) == count, f'Missing complete {phase} epoch logs')
    _expect([row['epoch'] for row in logs] == list(range(1, count+1)), 'Duplicate/missing log epochs')
    _expect(all(row['sample_count'] == size for row in logs), 'Log sample counts do not match configuration')
    # Intentionally no fallback to training_ce or weighted adaptive total_loss.
    _expect(all('common_noisy_ce' in row for row in logs), 'Common deterministic CE is required')
    return np.array([_number(row['common_noisy_ce'], 'common deterministic CE') for row in logs])


def _csv_matches(csv_row, json_row):
    _expect(set(csv_row) == set(json_row), 'CSV/JSON result columns differ')
    for key, value in json_row.items():
        text = csv_row[key]
        if value is None:
            valid = text == ''
        elif type(value) is bool:
            valid = text.lower() == str(value).lower()
        elif type(value) in (int, float):
            parsed = float(text)
            valid = math.isfinite(parsed) and math.isclose(parsed, value, rel_tol=0, abs_tol=1e-12)
        else:
            valid = text == value
        _expect(valid, f'CSV and JSON disagree on {key}')


def _detector(root, record, settings):
    from sklearn.metrics import roc_auc_score, roc_curve
    size, warmup = settings['train_size'], settings['warmup_epochs']
    _expect(record['primary_epoch'] == warmup and record['primary']['epoch'] == warmup,
            'Primary detector epoch must be the configured last warm-up epoch')
    _expect([row['epoch'] for row in record['history']] == list(range(1,warmup+1)),
            'Incomplete detector metric history')
    _expect(record['source']['configuration_sha256'] == configuration_fingerprint(settings),
            'Detector configuration fingerprint mismatch')
    path = _artifact(root, record['arrays'])
    _expect(file_digest(path) == record['arrays_sha256'], 'Detector arrays checksum mismatch')
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    _expect(np.array_equal(arrays['epochs'], np.arange(1,warmup+1)), 'Detector epoch IDs mismatch')
    _expect(np.array_equal(arrays['subset_positions'], np.arange(size)), 'Detector stable IDs mismatch')
    ids = arrays['original_indices']
    _expect(ids.shape == (size,) and np.issubdtype(ids.dtype, np.integer)
            and len(np.unique(ids)) == size and np.all((ids >= 0) & (ids < 50000)), 'Invalid original CIFAR IDs')
    for name in ('clean_labels','noisy_labels'):
        labels = arrays[name]
        _expect(labels.shape == (size,) and np.issubdtype(labels.dtype,np.integer)
                and np.all((labels >= 0) & (labels < 10)), 'Invalid detector analysis/provided labels')
    mask = arrays['is_corrupted']
    _expect(mask.shape == (size,) and mask.dtype == np.bool_ and mask.any() and not mask.all(),
            'Detector mask needs both clean and corrupted samples')
    _expect(np.array_equal(mask,arrays['clean_labels'] != arrays['noisy_labels']), 'Corruption mask/labels mismatch')
    _expect(int(mask.sum()) == round(record['noise_ratio']*size), 'Incorrect configured corruption count')
    for key, source_key in (('clean_labels','clean_labels_sha256'),('is_corrupted','corruption_mask_sha256')):
        _expect(array_checksum(arrays[key]) == record['source'][source_key], 'Detector analysis checksum mismatch')
    for name in ('we_history','margin_sum_history','observation_counts_history','aum_history'):
        _expect(arrays[name].shape == (warmup,size), 'Detector history shape mismatch')
    for index in range(warmup):
        tracker = SampleStatistics.from_state_dict({'schema_version':1,'size':size,'tracked_epochs':index+1,
            'we':arrays['we_history'][index], 'margin_sum':arrays['margin_sum_history'][index],
            'observation_counts':arrays['observation_counts_history'][index]})
        _expect(np.array_equal(tracker.aum,arrays['aum_history'][index]), 'AUM does not match cumulative sums/counts')
    changes = np.diff(arrays['we_history'],axis=0)
    _expect(np.all((changes >= 0) & (changes <= 1)), 'WE must increase by at most one per completed epoch')
    _expect(np.array_equal(arrays['primary_we'],arrays['we_history'][-1])
            and np.array_equal(arrays['primary_negative_aum'],-arrays['aum_history'][-1]), 'Primary scores differ from last warm-up')
    ranks = suspicion_scores(arrays['primary_we'],arrays['aum_history'][-1],arrays['noisy_labels'],
                              observation_counts=arrays['observation_counts_history'][-1])
    curves, aucs = {}, {}
    for name, values, metric in (('we',arrays['primary_we'],'we_roc_auc'),
                                 ('negative_aum',arrays['primary_negative_aum'],'negative_aum_roc_auc'),
                                 ('hybrid',ranks['hybrid'],'hybrid_roc_auc')):
        aucs[metric] = float(roc_auc_score(mask,values))
        _expect(math.isclose(aucs[metric],record['primary'][metric],abs_tol=1e-12,rel_tol=0),
                'Saved detector AUC disagrees with source arrays')
        _expect(record['history'][-1][metric] == record['primary'][metric], 'Primary/history AUC mismatch')
        fpr,tpr,_ = roc_curve(mask,values,drop_intermediate=False)
        curves[name] = (fpr,tpr)
    for key,rank in (('primary_we_rank',ranks['ido_lite']),('primary_negative_aum_rank',ranks['aum']),('primary_hybrid',ranks['hybrid'])):
        _expect(np.allclose(arrays[key],rank,atol=1e-12,rtol=0), 'Saved relative ranks disagree with provided-label grouping')
    return {'record':record,'arrays':arrays,'curves':curves,'aucs':aucs}


def load_bundle(results, *, runs=None, config=None):
    """Require a complete, checksum-verified six-row evaluation export."""
    root = Path(results)
    receipt = json.loads((root/'metrics_manifest.json').read_text(encoding='utf-8'))
    for name,key in (('metrics.json','metrics_json_sha256'),('metrics.csv','metrics_csv_sha256')):
        _expect(file_digest(root/name) == receipt[key], f'{name} publication checksum mismatch')
    metrics = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    scope, settings = metrics['scope'], metrics['scope']['configuration']
    _expect(scope == json.loads((root/'evaluation_manifest.json').read_text(encoding='utf-8')), 'Evaluation manifest mismatch')
    _expect(receipt['scope_sha256'] == json_digest(scope), 'Evaluation scope checksum mismatch')
    _expect(configuration_fingerprint(settings) == scope['configuration_sha256'], 'Scientific configuration fingerprint mismatch')
    config_path = Path(config) if config else (root/'experiment_config.json' if (root/'experiment_config.json').exists() else root.parent/'experiment_config.json')
    if config_path.exists():
        _expect(load_config(config_path) == settings, 'Supplied experiment configuration differs from evaluation')
    elif config:
        raise FileNotFoundError(config_path)
    for name in ('train_size','test_size','warmup_epochs','continuation_epochs'):
        _expect(type(settings[name]) is int and settings[name] > 0, 'Invalid configured size/schedule')
    _expect(settings['noise_ratios'] == [.2,.4] and settings['test_size'] == 10000
            and settings['train_size'] % 10 == 0 and settings['train_size'] <= 50000, 'Unexpected scientific dataset/noise scope')
    expected = {(ratio,method) for ratio in settings['noise_ratios'] for method in METHODS}
    _expect(metrics['schema_version'] == 1 and metrics['complete'] is True and not metrics['pending']
            and metrics['row_count'] == len(expected) and metrics['expected_rows'] == len(expected)
            and receipt['complete'] is True and receipt['row_count'] == len(expected), 'Plots require all six completed scientific results')
    with (root/'metrics.csv').open(encoding='utf-8',newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    _expect(len(csv_rows) == len(expected) and len(metrics['methods']) == len(expected), 'Missing/duplicate final result rows')
    reports, curves, warmups = {}, {}, {}
    csv_by_key = {(float(row['noise_ratio']),row['method']):row for row in csv_rows}
    _expect(set(csv_by_key) == expected and len(csv_by_key) == len(csv_rows), 'Duplicate/missing CSV method keys')
    for report in metrics['methods']:
        row = report['row']
        key = (row['noise_ratio'],row['method'])
        _expect(key in expected and key not in reports, 'Missing/duplicate noise/method key')
        _csv_matches(csv_by_key[key],row)
        _expect(row['configuration_sha256'] == scope['configuration_sha256'], 'Mixed result configurations')
        for name in ('seed','train_size','test_size','warmup_epochs','continuation_epochs'):
            _expect(row[name] == settings[name], 'Result metadata differs from saved configuration')
        _expect(row['model_epochs'] == settings['warmup_epochs']+settings['continuation_epochs'], 'Model epoch total mismatch')
        _number(row['test_accuracy_pct'],'test accuracy',maximum=100)
        _expect(type(row['test_correct']) is int and 0 <= row['test_correct'] <= settings['test_size']
                and math.isclose(row['test_accuracy_pct'],100*row['test_correct']/settings['test_size'],abs_tol=1e-12), 'Test accuracy/count mismatch')
        warmup = _logs(report,'warmup',settings['train_size'],settings['warmup_epochs'],Path(runs) if runs else None,*key)
        continuation = _logs(report,'continuation',settings['train_size'],settings['continuation_epochs'],Path(runs) if runs else None,*key)
        _expect(math.isclose(row['final_common_noisy_ce'],continuation[-1],rel_tol=0,abs_tol=1e-12), 'Final CE differs from committed logs')
        if key[0] in warmups:
            _expect(np.array_equal(warmups[key[0]],warmup), 'Methods have different shared warm-up CE')
        warmups[key[0]] = warmup
        curves[key] = np.r_[warmup,continuation]
        reports[key] = report
    _expect(set(reports) == expected, 'Missing required methods/noise conditions')
    _expect(len(metrics['detectors']) == len(settings['noise_ratios']), 'Missing detector conditions')
    detectors = {}
    for record in metrics['detectors']:
        ratio = record['noise_ratio']
        _expect(ratio in settings['noise_ratios'] and ratio not in detectors, 'Duplicate/unknown detector condition')
        detectors[ratio] = _detector(root,record,settings)
        for method in METHODS:
            row = reports[ratio,method]['row']
            _expect(row['corruptions'] == int(detectors[ratio]['arrays']['is_corrupted'].sum()), 'Result corruption count mismatch')
            for key,value in detectors[ratio]['aucs'].items():
                _expect(math.isclose(row[key],value,rel_tol=0,abs_tol=1e-12), 'CSV detector AUC mismatch')
    reference = detectors[settings['noise_ratios'][0]]['arrays']['original_indices']
    _expect(all(np.array_equal(reference,value['arrays']['original_indices']) for value in detectors.values()), 'Noise conditions selected different training images')
    clean_reference = detectors[settings['noise_ratios'][0]]['arrays']['clean_labels']
    _expect(all(np.array_equal(clean_reference,value['arrays']['clean_labels']) for value in detectors.values()), 'Original clean labels differ between noise conditions')
    sources = {'evaluation_code':scope['code'],'configuration_sha256':scope['configuration_sha256'],
               'metrics_json_sha256':file_digest(root/'metrics.json'), 'metrics_csv_sha256':file_digest(root/'metrics.csv'),
               'metrics_manifest_sha256':file_digest(root/'metrics_manifest.json'),
               'detector_arrays_sha256':{str(ratio):value['record']['arrays_sha256'] for ratio,value in detectors.items()}}
    return {'settings':settings,'metrics':metrics,'reports':reports,'ce':curves,
            'warmups':warmups,'detectors':detectors,'sources':sources}


def comparison_rows(bundle):
    rows = []
    for ratio in bundle['settings']['noise_ratios']:
        baseline = bundle['reports'][ratio,'standard']['row']['test_accuracy_pct']
        for method in METHODS:
            accuracy = bundle['reports'][ratio,method]['row']['test_accuracy_pct']
            rows.append({'noise_ratio':ratio,'method':method,'test_accuracy_pct':accuracy,
                         'delta_vs_standard_pp':round(accuracy-baseline,10)})
    return rows


def _context(settings):
    defaults = load_config()
    nondefault = any(settings[key] != defaults[key] for key in ('train_size','warmup_epochs','continuation_epochs'))
    label = (f"CIFAR-10 | {settings['train_size']:,} train / {settings['test_size']:,} test | "
             f"{settings['warmup_epochs']} warm-up + {settings['continuation_epochs']} continuation epochs | "
             f"seed {settings['seed']} (single seed)")
    return label + (' | non-default size/schedule' if nondefault else '')


def _footer(fig,bundle,note):
    fig.text(.5,.105,_context(bundle['settings']),ha='center',fontsize=8)
    fig.text(.5,.065,note,ha='center',fontsize=8,color='#475569')
    fig.text(.5,.025,f"Source: verified evaluation exports | experiment {bundle['sources']['evaluation_code']['git_sha'][:7]} | IDO-inspired simplified experiment",ha='center',fontsize=8,color='#475569')


def _save_figure(fig,path):
    import matplotlib.pyplot as plt
    from PIL import Image
    temporary = path.with_name(path.name+'.tmp')
    try:
        fig.savefig(temporary,format='png',dpi=200,facecolor='white',metadata={'Software':'IDO-AUM Lite src.plots'})
        with Image.open(temporary) as saved:
            saved.verify()
        temporary.replace(path)
    finally:
        plt.close(fig)


def render_figures(bundle, output_root):
    import matplotlib
    matplotlib.use('Agg',force=True)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    root = Path(output_root)/'figures'
    root.mkdir(parents=True,exist_ok=True)
    cfg = bundle['settings']
    ratios = cfg['noise_ratios']
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold','axes.axisbelow':True}):
        fig,ax = plt.subplots(figsize=(8,5.3))
        x,width = np.arange(len(ratios)),.23
        for index,method in enumerate(METHODS):
            values = [bundle['reports'][ratio,method]['row']['test_accuracy_pct'] for ratio in ratios]
            bars = ax.bar(x+(index-1)*width,values,width,label=LABELS[method],color=COLORS[method])
            ax.bar_label(bars,labels=[f'{value:.2f}%' for value in values],padding=4,fontsize=10)
        ax.set(xticks=x,xticklabels=[f'{ratio:.0%} label noise' for ratio in ratios],
               ylabel='Test accuracy (%)',ylim=(0,100))
        ax.yaxis.grid(True,alpha=.18)
        fig.suptitle('Final accuracy on the untouched CIFAR-10 test split',y=.97,fontsize=14,fontweight='bold')
        fig.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False)
        _footer(fig,bundle,'One observed run per method; no error bars or statistical superiority claim.')
        fig.tight_layout(rect=(0,.16,1,.83))
        _save_figure(fig,root/FIGURES[0])

        fig,axes = plt.subplots(1,len(ratios),figsize=(11,5.2),sharex=True,sharey=True,squeeze=False)
        for ax,ratio in zip(axes[0],ratios):
            detector = bundle['detectors'][ratio]
            for key,label,color,metric in (('we','WE','#d97706','we_roc_auc'),('negative_aum','-AUM','#2563eb','negative_aum_roc_auc')):
                fpr,tpr = detector['curves'][key]
                ax.plot(fpr,tpr,color=color,linewidth=2,label=f'{label}: AUC {detector["aucs"][metric]:.4f}')
            ax.plot([0,1],[0,1],color='#94a3b8',linestyle=':',linewidth=1)
            ax.set(title=f'{ratio:.0%} label noise',xlabel='False positive rate (clean flagged suspicious)',xlim=(0,1),ylim=(0,1.02))
            ax.grid(alpha=.15)
            ax.legend(loc='lower right',frameon=True,fontsize=9)
        axes[0,0].set_ylabel('True positive rate (corrupted detected)')
        fig.suptitle(f"Noise detection at the prespecified warm-up epoch {cfg['warmup_epochs']}",y=.98,fontsize=14,fontweight='bold')
        _footer(fig,bundle,'Raw WE and negative AUM scores; retrospective corruption masks, no threshold tuning.')
        fig.tight_layout(rect=(0,.16,1,.91))
        _save_figure(fig,root/FIGURES[1])

        ratio = ratios[0]  # Prespecified representative: first configured condition, not best result.
        arrays = bundle['detectors'][ratio]['arrays']
        mask = arrays['is_corrupted']
        fig,axes = plt.subplots(1,2,figsize=(11,5.2))
        values = np.arange(cfg['warmup_epochs']+1)
        groups = ((~mask,'Clean','#2563eb'),(mask,'Corrupted','#d97706'))
        aum = -arrays['primary_negative_aum']
        bins = np.linspace(float(aum.min()),float(aum.max()),41) if np.ptp(aum) > 0 else np.linspace(float(aum[0])-1,float(aum[0])+1,41)
        for index,(group,label,color) in enumerate(groups):
            label = f'{label} (n={int(group.sum()):,})'
            counts = np.bincount(arrays['primary_we'][group],minlength=len(values))/group.sum()
            axes[0].bar(values+(index-.5)*.38,counts,width=.38,color=color,label=label)
            axes[1].hist(aum[group],bins=bins,density=True,alpha=.5,color=color,label=label)
        axes[0].set(title='Wrong events (WE)',xlabel='Wrong predictions across tracked epochs',ylabel='Fraction within group',xticks=values)
        axes[0].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1].set(title='Area under margin (AUM)',xlabel='Mean assigned-label minus alternative logit',ylabel='Density within group')
        for ax in axes:
            ax.yaxis.grid(True,alpha=.15)
        fig.suptitle(f"Warm-up score distributions | {ratio:.0%} label noise | epoch {cfg['warmup_epochs']}",y=.98,fontsize=14,fontweight='bold')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
        _footer(fig,bundle,'Groups normalized separately. Higher WE / lower AUM indicates greater suspicion; analysis labels only.')
        fig.tight_layout(rect=(0,.16,1,.82))
        _save_figure(fig,root/FIGURES[2])

        fig,axes = plt.subplots(1,len(ratios),figsize=(11,5.2),sharex=True,sharey=True,squeeze=False)
        w,c = cfg['warmup_epochs'],cfg['continuation_epochs']
        maximum = max(float(series.max()) for series in bundle['ce'].values())
        for ax,ratio in zip(axes[0],ratios):
            ax.axvspan(.5,w+.5,color='#94a3b8',alpha=.1)
            ax.axvline(w+.5,color='#64748b',linestyle=':',linewidth=1.5)
            ax.plot(np.arange(1,w+1),bundle['warmups'][ratio],color='#94a3b8',linewidth=2,label='Shared warm-up')
            for method,style in zip(METHODS,('-','--','-.')):
                series = bundle['ce'][ratio,method]
                ax.plot(np.arange(w,w+c+1),series[w-1:],label=LABELS[method],color=COLORS[method],
                        linestyle=style,linewidth=1.9,marker='o',markersize=3)
            ax.set(title=f'{ratio:.0%} label noise',xlabel='Completed training epoch',xlim=(.5,w+c+.5),ylim=(0,max(.1,maximum*1.12)))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True,nbins=7))
            ax.grid(alpha=.15)
        axes[0,0].set_ylabel('Deterministic unweighted noisy-label CE')
        fig.suptitle('Common training-subset diagnostic across methods',y=.98,fontsize=14,fontweight='bold')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.92),ncol=4,frameon=False)
        _footer(fig,bundle,f'Dashed boundary: after warm-up epoch {w}. Weighted adaptive objectives are not plotted as CE.')
        fig.tight_layout(rect=(0,.16,1,.82))
        _save_figure(fig,root/FIGURES[3])
    return [root/name for name in FIGURES]


def _publish_text(path,text):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(text,encoding='utf-8')
    _expect(temporary.read_text(encoding='utf-8') == text, 'Table write read-back failed')
    temporary.replace(path)


def write_tables(bundle,root):
    rows = comparison_rows(bundle)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output,fieldnames=('noise_ratio','method','test_accuracy_pct','delta_vs_standard_pp'),lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({**row,'test_accuracy_pct':f'{row["test_accuracy_pct"]:.2f}',
                         'delta_vs_standard_pp':f'{row["delta_vs_standard_pp"]:.2f}'})
    _publish_text(root/'comparison_table.csv',output.getvalue())
    lines = ['# Measured experiment comparison','',_context(bundle['settings']),
             '', 'Single-seed exploratory observations from verified saved evaluation outputs. No significance test or reliable-superiority claim.',
             '', '| Noise | Method | Test accuracy (%) | Difference vs Standard (pp) |',
             '| --- | --- | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["noise_ratio"]:.0%} | {LABELS[row["method"]]} | {row["test_accuracy_pct"]:.2f} | {row["delta_vs_standard_pp"]:+.2f} |')
    lines += ['', 'Detector metrics use the shared, prespecified final warm-up epoch; they are identical across continuation methods by design.',
              '', '| Noise | WE ROC-AUC | -AUM ROC-AUC | Hybrid ROC-AUC |','| --- | ---: | ---: | ---: |']
    for ratio in bundle['settings']['noise_ratios']:
        aucs = bundle['detectors'][ratio]['aucs']
        lines.append(f'| {ratio:.0%} | {aucs["we_roc_auc"]:.6f} | {aucs["negative_aum_roc_auc"]:.6f} | {aucs["hybrid_roc_auc"]:.6f} |')
    lines += ['',f'Source experiment commit: `{bundle["sources"]["evaluation_code"]["git_sha"]}`.',
              f'Configuration SHA256: `{bundle["sources"]["configuration_sha256"]}`.',
              '', 'Standard-versus-adaptive differences compare whole training strategies. WE-only versus Hybrid uses the same adaptive objective and views; only suspicion scoring differs.', '']
    _publish_text(root/'comparison_table.md','\n'.join(lines))
    return [root/'comparison_table.csv',root/'comparison_table.md']


def generate_plots(results, *, runs=None, config=None, output_root=None):
    bundle = load_bundle(results,runs=runs,config=config)
    output_root = Path(output_root) if output_root else Path(results)
    output_root.mkdir(parents=True,exist_ok=True)
    tables = write_tables(bundle,output_root)
    figures = render_figures(bundle,output_root)
    manifest = {'schema_version':1,'sources':bundle['sources'],'configuration':bundle['settings'],
                'plotter_sha256':file_digest(Path(__file__)), 'representative_noise_ratio':bundle['settings']['noise_ratios'][0],
                'primary_detector_epoch':bundle['settings']['warmup_epochs'],
                'warmup_boundary':bundle['settings']['warmup_epochs']+.5,
                'learning_curve_metric':'common_noisy_ce: deterministic unweighted provided-label CE',
                'artifacts':{path.relative_to(output_root).as_posix():file_digest(path) for path in [*figures,*tables]}}
    atomic_json(output_root/'plots_manifest.json',manifest)
    print(f'Generated {len(figures)} PNG figures and 2 comparison tables in {output_root}',flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--runs',type=Path,help='Optional external epoch logs; embedded metrics.json logs suffice')
    parser.add_argument('--config',type=Path,help='Optional frozen configuration; otherwise infer from results parent or embedded scope')
    parser.add_argument('--output-root',type=Path,help='Default: write figures/tables beside input metrics')
    args = parser.parse_args()
    generate_plots(args.results,runs=args.runs,config=args.config,output_root=args.output_root)


if __name__ == '__main__':
    main()
