# IDO-AUM Lite

An **IDO-inspired simplified implementation** comparing Standard cross-entropy,
WE-only IDO-lite and fixed 50:50 WE+AUM scoring on noisy CIFAR-10. This is not a
faithful reproduction of original IDO or AUM threshold calibration.

Implementation, both full training conditions, evaluation, four figures and an
editable report are available. The Word report can be revised for final course
submission rules. Group names on its cover do not assign individual contributions.
Development used Codex assistance; no unconfirmed individual roles are claimed.

## Measured results

| Label noise | Standard | IDO-lite | Hybrid | Hybrid minus IDO-lite |
| --- | ---: | ---: | ---: | ---: |
| 20% | 70.42% | 75.26% | 74.76% | -0.50 pp |
| 40% | 63.80% | 68.61% | 68.48% | -0.13 pp |

| Label noise | Warm-up WE ROC-AUC | Warm-up negative-AUM ROC-AUC | Hybrid rank ROC-AUC |
| --- | ---: | ---: | ---: |
| 20% | 0.899201 | 0.943642 | 0.943767 |
| 40% | 0.867467 | 0.921886 | 0.919509 |

These are **one-seed exploratory observations**, not statistically reliable
superiority claims. Both adaptive strategies exceed Standard in this run, while
adding AUM does not exceed WE-only accuracy. Stronger detection and better
classification are different outcomes. Detection uses shared warm-up epoch 6;
its repeated values across continuation methods do not represent independent
measurements. Standard versus adaptive changes loss/views; WE-only versus Hybrid
is the intended scoring comparison. GradScaler recorded three skipped updates
(one in each Standard continuation and one in 40% Hybrid), so realised update
counts also differ slightly despite equal epoch schedules.

Training/evaluation source: **695f50fef1915f0ab2d370ba6c8cda1282e9dae2**.
Scientific configuration: **a949134949d25219405ea573349acee3f46191429c5542b7b46a4a33d517acc0**.

## Artifacts

- [Editable Word report](report/IDO_AUM_Lite_Report.docx), [Markdown source](report/report.md), and [editing notes](report/EDITING.md).
- [Measured comparison table](results/comparison_table.md) and [compact CSV](results/comparison_table.csv).
- [Accuracy bars](results/figures/accuracy_comparison.png), [ROC curves](results/figures/noise_detection_roc.png), [score distributions](results/figures/score_distributions.png), [common CE curves](results/figures/common_ce_curves.png).
- `results/metrics.csv`, `metrics.json`, evaluation/publication manifests, six accuracy caches, detector NPZ/JSON files, frozen `experiment_config.json`, and plot provenance.
- Full model/optimizer checkpoints remain on Drive. No images, CIFAR archive or `.pt` files belong in Git.

All numbers come from verified saved outputs. Plotting checks original publication
hashes, CSV/JSON agreement, detector arrays, configuration and epoch logs. The
result files retain exact bytes through Git attributes so checksum verification
survives a fresh checkout. Report tables are generated from those same metrics;
Word text, tables and OMML equations are editable. Rebuilding the report replaces
manual Word/Markdown edits, so preserve the edited copy first.

## Completed run archive and teammate handoff

The [completed Colab notebook](notebooks/colab_runner_completed.ipynb) preserves
saved benchmark, freeze, training, evaluation and accuracy-table outputs at
695f50f. It is an execution archive; use the separate clean runner for new work.
Its original cells and outputs are unchanged, with an added archive notice.
No hosted pytest count is inferred from the saved cell's absent summary.

See [the teammate handoff](handoff/README.md) for the completed warm-up folders,
frozen configuration, required original Drive manifests and a checksum/inventory
of the downloaded archive members. Large checkpoints stay on Drive, not in Git.
A shared Drive URL has not been supplied; folder access must be shared separately.

## Method and executed configuration

The same deterministic stratified 20,000-image training subset (2,000 per original
class) is used for both conditions. Exactly 4,000/8,000 labels change uniformly to
one of the nine other classes. Selection uses seed 42; noise streams are
`SeedSequence([seed, noise_percent, 1])`. Stable subset positions map to original
CIFAR IDs. Clean labels and corruption masks are analysis-only after construction.
The official 10,000-image test split is untouched and is never used for tuning.

A non-pretrained ResNet-18 uses a 3x3 stride-1 stem, no initial maxpool and ten
outputs. AdamW uses lr 0.0003, weight decay 0.0001, batch 128, two workers and
CUDA mixed precision. Each noise condition executes one six-epoch warm-up and
three independent six-epoch continuations. Each final model has 12 epochs of
history: **48 executed training epochs**, 72 training image-view passes, 36
statistics passes, 12 additional CE-only diagnostic passes and six final test
passes. No subset/schedule fallback or threshold amendment was used.

WE counts wrong predictions once per completed epoch. AUM averages assigned-label
minus strongest-alternative raw logit margins across all tracked epochs. Both
are keyed by stable IDs and collected with deterministic transforms, `eval()`
and no gradients. All warm-up snapshots are retained.

Within each provided-label class, one-based average tie rank r gives
`s=(r-0.5)/n`. Higher WE/lower AUM means suspicious. IDO-lite uses WE ranks;
Hybrid averages WE and negative-AUM ranks. Ranks are relative, not corruption
probabilities: complete-class mean suspicion is 0.5. The experimental weights are
`c=0.1+0.9*(1-s)` and `d=4*s*(1-s)`.

Both adaptive branches use one shared full-batch mean of
`c*CE(weak,noisy_y) + 0.5*s*mask*CE(strong,detached_weak_pseudo) + 0.5*d*class_mean((p_weak-p_strong)^2)`.
Confidence threshold is 0.7; masked samples remain in the denominator. Consistency
differentiates both views. Scores are frozen from prior completed epochs and
updated after the next statistics pass. Standard continuation collects common
unweighted deterministic noisy-label CE without WE/AUM updates. Cross-method
curves use that common diagnostic, never the different weighted objectives.

All continuations reset from the same complete warm-up state including BatchNorm,
with fresh optimizer/scaler and matched branch RNG/loader streams. Resume instead
restores the interrupted branch's complete state. Original IDO BMM estimation,
base-model label-wave selection and AUM threshold samples are omitted. The
0.1 floor, parabolic difficulty, fixed coefficients and short schedule are
experimental simplifications. Crop size is 32 with padding 4; strong ColorJitter
uses brightness/contrast/saturation 0.2 and hue 0.05. Channel mean is
(0.4914, 0.4822, 0.4465), standard deviation (0.2470, 0.2435, 0.2616).
Mean-loss magnitudes do not establish equal
gradient influence. See the report for confidence/near-zero diagnostics and limits.

## Windows development

Use the existing `.venv`; do not install torch, torchvision or CUDA locally,
download CIFAR locally, or run model training/evaluation on the laptop. All full
training, benchmarking and evaluation run on Colab GPU. Local saved-result plots
need no GPU.

```powershell
./.venv/Scripts/python.exe -m pytest -q -m "not torch"
./.venv/Scripts/python.exe -m pip check
./.venv/Scripts/python.exe -m src.plots --results results --config results/experiment_config.json
```

Last scientific-code validation: **209 passed, six PyTorch-dependent modules
skipped locally**; pip check passed. PyTorch tests use `torch` markers and
`importorskip`. The hosted test command, integrated GPU smoke and completed-stage
resume were observed successful. An interrupted real epoch recovered in a fresh
VM was not demonstrated. These checks do not measure statistical reliability.
Report authoring uses the separate Codex document runtime with `python-docx`;
scientific `.venv` dependencies are unchanged. See `report/EDITING.md`.

Reusable modules own data, archive cache, statistics, scores, losses, checkpoint
publication, training, evaluation and plotting. The notebook only orchestrates.
`src.evaluate` alone owns test/detection metrics and `metrics.csv`; `src.plots`
only reads saved results. Neither training nor plotting chooses settings from
clean-label detection or test accuracy.

## Colab setup and persistent paths

Open [the notebook](https://colab.research.google.com/github/Avilas01/ido-aum-lite/blob/main/notebooks/colab_runner.ipynb).
Saved Drive notebook copies, source checkouts and imported Python modules can
become stale independently. Refresh source and restart Python when updating an
unfrozen checkout; a browser refresh alone is insufficient. Public GitHub checkout
needs no token. All action flags start False.

| Purpose | Path |
| --- | --- |
| Disposable checkout | `/content/ido-aum-lite` |
| Disposable local extraction | `/content/data` |
| Shared verified archive | `/content/drive/MyDrive/ido-aum-lite/cache/cifar10` |
| Experiment root | `/content/drive/MyDrive/ido-aum-lite/experiments/<RUN_ID>` |
| Frozen settings and code/environment | `<experiment-root>/experiment_config.json`, `manifest.json` |
| Benchmark / runs / results | `<experiment-root>/benchmark`, `runs`, `results` |

For a cache-only session select a standard CPU runtime, set
`SESSION_MODE='data_only'`, and run Blocks 1,2,3,5,6 with `RUN_CACHE_DATA=True`.
Wait for the verified Drive archive before switching hardware. One official
archive is shared across both conditions and RUN_IDs. Valid local archive ->
valid Drive archive -> internet download; copies are checksum-verified before
publication. Partial HTTP resume is not implemented. Block 6 timeout is 3,600 s;
Block 8 preparation-check timeout is 900 s. Training/evaluation have no code timeout.

For model checks/new full experiments select GPU and `SESSION_MODE='verify'` or
`'train'`, run Blocks 1-5, then hosted tests in Block 7. Follow an explicit restart
request if PyTorch repair changes native packages. Existing compatible pairs are
preserved; record actual versions. The completed experiment used Tesla T4,
Python 3.13.16, NumPy 2.4.6, torch 2.14.1+cu130, torchvision 0.29.1+cu130 and
CUDA 13.0. GPU availability and session duration are not promised.

An isolated generated-data smoke, when needed, is:

```python
run_module('src.train', '--smoke', '--noise', 0.2, '--device', 'cuda',
           '--output-root', EXPERIMENT_ROOT / 'smoke_step4')
run_module('src.train', '--smoke', '--noise', 0.2, '--device', 'cuda',
           '--output-root', EXPERIMENT_ROOT / 'smoke_step4', '--resume')
```

Use a distinct smoke root if its saved code/environment identity differs. Smoke
uses 512 generated images, one warm-up and one epoch per continuation; it is not
a scientific run or runtime estimate. Do not repeat successful unchanged checks
for every noise condition. Block 7 may hide inherited pytest output; that deferred
notebook issue has not been changed.

## Benchmark freeze training and evaluation

For a NEW experiment: Block 10 measures a full representative warm-up, standard
and adaptive epoch on the candidate subset, including diagnostics and persistent
checkpoint I/O. Review its runtime projection, evaluation/setup/debugging margin
and remaining budget before Block 11 freeze. Choose fallback only from runtime,
never quality: first reduce subset to 10,000; then continuation 6->4; warm-up 6->4
only last. Rebenchmark and apply one configuration to every method and both
conditions. No automatic fallback is applied. The planned project budget was
15 hours, including 4.5 hours for runs/evaluation/debugging; a complete actual
wall-clock development-time audit is not available.

Inside the Colab checkout, with `EXPERIMENT_ROOT` set in the same shell:

```bash
export EXPERIMENT_ROOT=/content/drive/MyDrive/ido-aum-lite/experiments/run-001
python -m src.train --benchmark --config configs/default.json --noise 0.2 --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --output-root "$EXPERIMENT_ROOT/benchmark"
```

Block 11 publishes the reviewed candidate config and code/environment manifest
once when `FREEZE_APPROVED=True`. The benchmark candidate alone is not a frozen
experiment. Full runs require that pair, a matching completed benchmark and a
clean pinned checkout. Then Block 12 or these commands execute/resume:

```bash
python -m src.train --config "$EXPERIMENT_ROOT/experiment_config.json" --noise 0.2 --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --output-root "$EXPERIMENT_ROOT/runs" --resume
python -m src.train --config "$EXPERIMENT_ROOT/experiment_config.json" --noise 0.4 --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --output-root "$EXPERIMENT_ROOT/runs" --resume
python -m src.evaluate --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --runs "$EXPERIMENT_ROOT/runs" --output-root "$EXPERIMENT_ROOT/results"
```

`NOISES_TO_RUN` in Block 5 can be `[0.2]`, `[0.4]` or both. Keep RUN_ID and the
frozen settings unchanged across sessions. Evaluation is enabled separately in
Block 13; wait for `6/6 final rows; complete=True`. Accuracy caches prevent repeat
inference for unchanged final checkpoint/protocol identities; partial conditions
are explicitly pending. `metrics_manifest.json` verifies the CSV/JSON publication.
Detector archives retain all warm-up WE/margin sums/counts/AUM and analysis masks.
Missing/corrupt artifacts are diagnosed, not replaced with invented values.

Archived training/evaluation used commit 695f50f. Keep that revision for recovery
of `run-001`; adding new code to its frozen checkout is rejected. Newer plotting
code can read exported results in a separate CPU/development checkout:

```bash
python -m src.plots --results "$EXPERIMENT_ROOT/results" --runs "$EXPERIMENT_ROOT/runs"
```

This generates the four PNGs, compact tables and plot manifest without reopening
model checkpoints. An intact committed `results/` directory also supports offline
local plotting using the Windows command above. No frozen experiment needs to be
unfrozen or retrained to plot/report its outputs.

## Restarting an interrupted Colab session

1. Reopen the same notebook, select GPU and mount the same Drive account. Ensure no other session is writing this RUN_ID.
2. Reuse the exact RUN_ID and experiment-root path. Read saved Git SHA, configuration and environment manifest; check out that code rather than the latest branch head. Reinstall dependencies and verify GPU/imports/versions. Document hardware changes; bitwise equality across GPU/kernel versions is not guaranteed.
3. Recreate `/content/ido-aum-lite` as needed; restore the verified CIFAR archive from the shared Drive cache to `/content/data` and extract locally rather than downloading again. Only a missing/corrupt archive needs another internet download. Source/local extraction are disposable; never delete persistent outputs, regenerate different corrupted labels, rerun fallback selection or overwrite frozen configuration during recovery.
4. Rerun configuration/status setup. Validate scientific configuration, code identity, subset/corrupted-label checksums and latest completed checkpoint/manifest. Ignore partial temporary files; fall back to the previous verified checkpoint if publication was interrupted. The runner restores model/optimizer/scaler/statistics/RNG/DataLoader state, not notebook code.
5. Rerun identical `--resume` commands above. Missing stages start, interrupted stages continue from the last durably completed epoch and finished branches are skipped. An unfinished epoch may repeat; completed statistics/logs must not be duplicated. New branches reset optimizer/RNGs after loading shared warm-up state; interrupted branches restore their own complete state.
6. Rerun evaluation/plotting after training. Reuse cached metrics for unchanged final checkpoints and compute only missing evaluations, retaining one final row per configuration/noise/method. Confirm all outputs exist on Drive before ending the runtime. Use a separate newer CPU checkout for plotting if the frozen code predates `src.plots`.

An acknowledged GPU change requires `ACKNOWLEDGE_GPU_CHANGE=True` in the notebook
or `--acknowledge-gpu-change` on train/evaluate CLI calls. Version/config/code/data
mismatches remain errors. Use one writer. Epoch journals retain checksum-verified
immutable payloads and a previous verified generation. Logs/history are derived
from those payloads after interrupted publication; missing timing sidecars remain
explicitly incomplete rather than fabricated. Checkpoint files use the full-state
PyTorch loader and must come from trusted project storage.

## Primary references

- [IDO paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/429e7b31625a8b7839f9e4d6e2aa9bb9-Abstract-Conference.html) and [official implementation](https://github.com/iTheresaApocalypse/IDO).
- [AUM paper](https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html).
- [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [ResNet](https://arxiv.org/abs/1512.03385), [AdamW](https://arxiv.org/abs/1711.05101).
- [Colab resource and recovery guidance](https://research.google.com/colaboratory/faq.html).
