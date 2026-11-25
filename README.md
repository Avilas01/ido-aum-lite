# IDO-AUM Lite

Develop with VS Code and Codex on Windows. Run all full training, benchmarking and final evaluation on a Google Colab Free GPU. A local GPU is not required.

**Current stage: Step 6 figures/tables implemented and generated from completed real experiments.** Benchmarking, 48 training epochs across both noise conditions, and all six final evaluations completed in Colab at commit 695f50f. The exported default configuration uses 20,000 training images, the untouched 10,000-image test split, seed 42 and 6+6 epochs. Four PNG figures, compact CSV/Markdown tables, verified source exports and a plot provenance manifest are in `results/`. The final report remains Step 7. No PyTorch installation, model training or CIFAR download has run locally; only saved-result analysis and plotting run here.

## Repository responsibilities

- `config.py`: CPU-safe configuration loader/validator and scientific-settings fingerprint.
- `configs/default.json`: portable scientific defaults (20k images, 6+6 epochs, seed 42).
- `notebooks/colab_runner.ipynb`: Drive mount, pinned checkout, dependencies/GPU checks, configuration, benchmark/run/resume CLI orchestration.
- `src/data.py`: NumPy-only reproducible label preparation, observable cached CIFAR loading and stable-ID datasets.
- `src/archive_cache.py`: torch-free, checksum-verified CIFAR archive backup/restore shared across Colab sessions and noise conditions.
- `src/commands.py`: unbuffered project-module execution with streamed stdout/stderr, heartbeat and child cleanup.
- `src/preparation_progress.py`: preparation stage timings and liveness messages.
- `src/model.py`: non-pretrained CIFAR-stem ResNet-18 factory, imported lazily.
- `src/statistics.py`: stable-ID WE/AUM/CE, complete-pass validation, branch copies and epoch snapshots.
- `src/smoke.py`: isolated GPU-only model/tracking verification on 512 generated examples.
- `src/scores.py`: NumPy-only within-provided-label-class WE/inverted-AUM ranks and 50:50 hybrid suspicion.
- `src/losses.py`: shared IDO-inspired adaptive objective and sample-weighted component diagnostics; torch imported only when computing loss.
- `src/checkpoints.py`: verified checkpoint journal, full-state hashing, code/configuration/data identity, RNG capture/restore.
- `src/train.py`: shared warm-up, three continuation branches, epoch-boundary resume, isolated smoke and representative-epoch benchmark.
- `src/evaluate.py`: sole owner of cached final test accuracy, historical detector ROC-AUC, plot-input arrays and JSON/CSV aggregation.
- `src/plots.py`: CPU-safe verified-export reader, four PNG figures, compact comparison tables and source/artifact provenance.
- Remaining work: Step 7 final report and reproducibility polish.
- `requirements.txt`: lightweight Windows dependencies, excluding torch/torchvision/CUDA.
- `requirements-colab.txt`: explicit training/evaluation dependencies including torch and torchvision.
- `src/colab_dependencies.py`: Colab-only GPU dependency inspection/repair, safe to import locally without torch.
- `requirements-dev.txt`: Windows requirements plus pytest, nbformat and packaging for lightweight verification.
- `tests/`: runnable configuration and notebook checks; no data download or GPU required.

## Windows development (existing .venv, no PyTorch)

Keep the configured `.venv`; do not recreate it or install torch, torchvision or CUDA locally. Select `.venv/Scripts/python.exe` as the VS Code interpreter. From the repository root in PowerShell:

```powershell
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m config
./.venv/Scripts/python.exe -m pytest -q -m "not torch"
./.venv/Scripts/python.exe -m pip check
```

Existing PyYAML, NumPy, pandas, Matplotlib, SciPy and scikit-learn are retained when they satisfy requirements. Neither Windows dependency file includes torch or CUDA. Never install requirements-colab.txt on Windows. Local tests validate configuration, requirement syntax, notebook structure and mocked GPU setup without importing real torch, downloading CIFAR-10 or training.

Future PyTorch-dependent dataset/model/statistics/loss tests run **in Colab**. Mark them `torch` and use `pytest.importorskip("torch")` / `pytest.importorskip("torchvision")` before importing PyTorch-dependent project modules so local collection remains safe. Do not run a local model smoke. Reusable modules use pathlib, expose CLI paths and avoid unconditional Colab imports.

## Colab walkthrough: choose the blocks you need

Open the [latest notebook in Colab](https://colab.research.google.com/github/Avilas01/ido-aum-lite/blob/main/notebooks/colab_runner.ipynb). Each code block is numbered; the first cell contains the same quick guide. Saved Drive copies do not automatically receive GitHub notebook updates. The source checkout cell fetches main for unfrozen runs; frozen experiments preserve their recorded SHA. After source updates, restart Python and rerun setup to avoid stale imported modules.

| Goal | Runtime / Block 1 setting | Run blocks | Flags to set True |
| --- | --- | --- | --- |
| Save the actual dataset once | Standard CPU, `SESSION_MODE='data_only'` | 1, 2, 3, 5, 6; skip 4 | `RUN_CACHE_DATA` in 6 |
| Test Step 2 model/tracker with generated inputs | GPU, `SESSION_MODE='verify'` | 1-5, 7, 9 | `RUN_COLAB_TESTS` in 7, `RUN_SMOKE` in 9 |
| Verify Step 1 using real CIFAR | GPU, `SESSION_MODE='verify'`; cache on CPU first | 1-5, 7 if not already passed for this source/environment, 8 | `RUN_COLAB_TESTS` if needed, `RUN_DATA_CHECK` in 8 |
| Full training (hosted checks pending) | GPU, `SESSION_MODE='train'` | 1-5, 6 as needed, 10 then 11 once, 12 | Later: `RUN_BENCHMARK`, `FREEZE_APPROVED`, `RUN_TRAINING` |
| Resume / final evaluation (real runs pending) | GPU, `SESSION_MODE='train'`, same RUN_ID | 1-5, 6 if local archive missing, 12 / 13 | Later: `RUN_TRAINING` / `RUN_EVALUATION`; skip 10-11 for a frozen run |

All action flags default to False. Enable only the actions you need. `ACKNOWLEDGE_GPU_CHANGE` in Block 3 is only for reviewing and recording a different GPU on a frozen experiment, never bypassing a dependency mismatch. Leave all future-stage flags False until those modules exist. Block 9 is an engineering smoke, not actual CIFAR model training. Blocks 10-13 safely skip by default.

Run setup once per new/restarted Python session. Retry only the failed action in a healthy session; rerun the helper if it changed. Do not repeat tests/smoke/real-data verification for every noise condition when code/environment/configuration are unchanged. Runtime selection is manual: changing SESSION_MODE does not allocate/release a GPU. Use a standard runtime for a slow download; switch to GPU only for model checks/training, then disconnect when finished. This follows the [Colab resource guidance](https://research.google.com/colaboratory/faq.html).

### Persistent CIFAR archive

Block 6 uses the official archive (50,000 training and 10,000 test images), not a synthetic test dataset. It verifies the official MD5 and saves one archive at `/content/drive/MyDrive/ido-aum-lite/cache/cifar10/cifar-10-python.tar.gz`, independent of RUN_ID. Both 20% and 40% configurations reuse it. Keep this file on Drive across sessions (roughly 170 MB).

Local valid archive -> valid Drive archive -> internet download, in that order. Only complete verified copies are published; failed Drive writes stop clearly and preserve valid local input. On a fresh VM, Block 6 restores the archive; Block 8 also restores it automatically before extraction. Images are extracted/read under `/content/data` for speed, rather than repeatedly read from Drive. Existing metadata on Drive are verified/reused. Deleting the runtime removes local extraction but does not remove the Drive archive. This follows [Colab's archive/local-I/O recommendation](https://research.google.com/colaboratory/faq.html).

Block 6 needs no torch, GPU, extraction or label preparation. It has a 3,600-second timeout and visible progress; the first download may be slow. Interrupted partial downloads are not saved as valid archives and may need restarting; HTTP byte-range resume is not implemented. A corrupt Drive archive is replaced only after a valid local/downloaded archive has been verified. Use one writer for the shared cache. Do not add archives, datasets or checkpoints to Git.

From the Colab checkout, the cache-only CLI is:

```bash
python -u -m src.archive_cache --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10
```

Add `--no-download` to that cache-only command to require an already valid local/Drive archive and forbid internet access. It does not extract images. Ordinary `src.data --prepare --no-download` still requires an already extracted local dataset.

## Colab installation and execution

1. Open the latest notebook and select the runtime from the walkthrough above. Use standard CPU for cache-only work; select **Runtime > Change runtime type > GPU** for model verification and future training/evaluation.
2. Set REPO_URL, REPO_REF and stable RUN_ID. Setup mounts Drive, checks out the recorded Git SHA and adds the checkout root to sys.path. Source/data caches live under /content; persistent configuration/checkpoints/logs/results live under /content/drive/MyDrive/ido-aum-lite/experiments/<RUN_ID>.
3. The repository is public: source checkout uses anonymous HTTPS clone/fetch, with no GitHub token or sign-in prompt. Rerun the source cell on an existing runtime to fetch the latest main before using new modules; existing frozen experiments keep their recorded Git SHA.
4. Run Block 3. CPU `data_only` mode installs requirements.txt only and skips torch/GPU setup. In `verify`/`train` mode it reads requirements-colab.txt, installs missing common packages without upgrading satisfied ones, and probes torch/torchvision in a fresh subprocess. Checks include installed versions, actual imports, torchvision's torch dependency metadata, CUDA, GPU name and a tiny compiled CUDA operation. A working GPU pair is kept even if it differs from the repair pair. No dataset is downloaded.
5. Missing/incompatible/broken pairs are repaired **together** using torch==2.14.1 and torchvision==0.29.1 from the official CUDA 12.6 wheel index, as listed in [PyTorch's installation matrix](https://pytorch.org/get-started/previous-versions/). General requirements are intentionally not independently pinned to force replacement of a working Colab pair; metadata and runtime checks enforce compatibility. Actual versions are recorded in the persistent environment manifest. If no GPU is assigned, setup stops before installation; select a GPU rather than installing a CUDA toolkit.
6. **If PyTorch installations change, setup raises RestartRequired. Select Runtime > Restart session (Restart runtime in some UIs), then rerun setup with the same RUN_ID.** Do not continue with cached native libraries. The next pass preserves the repaired pair. If post-repair CUDA/import checks fail, stop and review driver/package compatibility or select a fresh GPU runtime before proceeding.
7. Run the optional hosted-test cell to install requirements-dev.txt and run python -m pytest -q in Colab. This includes the implemented PyTorch-dependent checks; Steps 1–2 include synthetic-image transform/DataLoader/model/collection tests; Steps 3-4 add loss/gradient and tiny staged-resume checks.
8. Run the Step 2 smoke cell now when ready. Keep benchmark/freeze/full-training disabled until Step 4 hosted tests and integrated smoke pass. Step 5 evaluation requires completed final checkpoints; plotting remains a later stage. Notebook commands use sys.executable -m, checked subprocess argument lists and explicit repository cwd. Missing research modules produce an explicit error; their implementation remains later work.

The non-torch installation cannot replace torch. A healthy GPU pair with a broken common import is reported separately rather than triggering PyTorch replacement. The tiny CUDA probe is dependency verification, not model training. The bootstrap is safe to import locally, but its installation function refuses Windows/non-Colab runtimes before any pip call.

For a module session **inside Colab only**, use the same setup entry point:

```python
from pathlib import Path
import sys
repo = Path('/content/ido-aum-lite')
sys.path.insert(0, str(repo))
from src.colab_dependencies import ensure_colab_dependencies
report = ensure_colab_dependencies(repo)
# If RestartRequired is raised, restart the runtime and rerun setup first.
```

Experimental settings, the 15-hour budget and uniform fallback rules are unchanged. Default counts remain 48 executed training epochs, 72 training image-view passes, 36 statistics passes and 12 diagnostic-only inference passes. Step 1 data preparation and Step 2 synthetic model/tracker verification are available below; the runner and evaluator are implemented, while actual benchmark/training/evaluation remain pending.

## Step 1: data and exact noisy labels

Local, synthetic-only verification (no torch or CIFAR download):

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_noise.py -q
./.venv/Scripts/python.exe -m src.data --synthetic-check
```

The synthetic check reports subset sizes, original-label class counts, 4,000/8,000 changed labels at the default size, stable checksums and sample shapes using generated images. These are sanity checks, not experimental results.

In Colab, run the hosted tests (including tests/test_data_torch.py), then optionally enable RUN_DATA_CHECK in the notebook. Real CIFAR-10 is restored from the shared verified Drive archive when available; otherwise this opt-in may download it and save a verified archive. Cache it on CPU with Block 6 first to avoid spending GPU allocation on network I/O. A direct CLI equivalent for one noise condition, from the repository checkout, is:

```bash
python -m src.data --prepare --config configs/default.json --noise 0.2 --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --metadata-root /content/drive/MyDrive/ido-aum-lite/experiments/run-001/data_checks/manual/noise_20
```

Use a distinct metadata directory per noise/configuration; both conditions always select the same image IDs. Exact corruption draws use independent recorded SeedSequence([seed, noise_percent, 1]) streams derived from seed 42. All replacements uniformly exclude the original class. Subset positions are 0..N-1; TrainingMetadata.original_indices maps them to original CIFAR IDs.

Training preparation loads only train=True, saves analysis separately and returns TrainingData containing images, provided noisy labels and IDs. It never retains the raw torchvision object or its original targets. make_training_datasets returns one-view standard, independent two-view adaptive and deterministic statistics datasets. Evaluation uses the separate load_cifar10_test entry point with all 10,000 official labels unchanged. Model construction is a separate Step 2 interface.

Persisted files under each metadata directory:

- training.npz: original_indices and noisy_labels only; load_training_metadata never opens analysis.npz.
- analysis.npz: clean_labels and is_corrupted, for retrospective evaluation/sanity checks only.
- metadata.json: subset/noisy-label checksums, sample count, ratio, seed streams and fixed transform settings. Existing compatible files are verified/reused, valid partial metadata are completed after interruption, and conflicting/corrupt final files are rejected rather than silently overwritten. Temporary files are verified before publication.

The weak view uses crop (32, padding 4), horizontal flip, tensor conversion and normalization. The strong view independently repeats crop/flip, adds ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05) before tensor conversion/normalization. Evaluation/statistics use only tensor conversion/normalization. Fixed channel mean=(0.4914, 0.4822, 0.4465) and std=(0.2470, 0.2435, 0.2616) are recorded with metadata. Standard/warm-up samples never compute an unused strong view. Training uses only noisy-label classes; clean originals and masks must never enter loss/score/selection functions.

## Slow preparation: progress, cache reuse and safe retries

The traceback in subprocess.wait()/os.waitpid() shows the notebook waiting for its child, not which child operation is slow. Previously python -m used buffered output and inherited notebook OS file descriptors without an explicit relay. The command printed its summary only at the end. Two separate --prepare commands also imported torchvision, verified/read the training batches and copied selected images twice. It did not augment all 20,000 images; only one weak/strong example per condition was checked. Torchvision skips a valid extracted download, so repeated downloads were not expected unless the cache was missing/corrupt or the runtime/data-root changed.

The notebook now performs one preparation command for both noise conditions, loading/validating the source once and sharing one selected-image array. The exact subset/noise RNG streams and checksums are unchanged. Source downloads, archives, extracted batches and the final test check all use the same DATA_ROOT (/content/data), while the verified source archive and small metadata files go to Drive; images remain local. The download/extraction wrapper uses torchvision's original archive URL, archive MD5, batch checksums and extraction utility; it does not substitute another dataset or weaken integrity checks. A verified instance avoids hashing every batch twice during its own constructor.

run_module launches python -u with unbuffered output, merges stderr/stdout into a pipe, and forwards each line through Python into the visible Colab cell. It prints start/exit timing, a child PID heartbeat when quiet and recent output on failure. A reader thread keeps supervision responsive; interrupts/timeouts stop the child session instead of leaving an orphan process. Download progress is rate-limited to roughly one update per second when supported by tqdm. Preparation stages additionally print START/END/FAILED and elapsed seconds, with one heartbeat for the currently active stage. Initialization totals include nested download/check/extraction timing; those durations should not be added twice.

To rerun Step 1 safely:

1. Open the latest notebook. If you interrupted the old runner, stop any remaining preparation first and restart the Python session to clear cached imports. If you cannot confirm the old child stopped, use a fresh Colab runtime to avoid concurrent writers; /content/data can then be restored from the verified Drive archive. Remount the same Drive and reuse RUN_ID. Do not delete Drive checkpoints/configuration or the local CIFAR cache.
2. Rerun source checkout, dependency setup and Module command helper; a browser refresh alone does not update /content source. Respect any frozen experiment's recorded Git SHA rather than silently changing its methodology.
3. Run hosted tests, then RUN_DATA_CHECK=True. The cell prints the exact cache/metadata locations and uses PREPARATION_TIMEOUT_SECONDS=900. That timeout applies only to this preparation check; it is not a training limit and assumes the archive was cached first with Block 6 on CPU. Increase it explicitly if needed for unusually slow extraction/Drive reads.
4. Watch the latest active stage. Download/verify archive means network/archive I/O; extracted-batch verification/loading means local cache reads; save metadata means Drive I/O. The final output includes per-stage and total preparation timings. This is data preparation, not ResNet training; a GPU does not speed the download.
5. Valid extracted files skip download/extraction. If only the verified archive exists, torchvision reuses it for extraction. An interrupted/incomplete archive is checksum-checked and may need downloading again. A replacement Colab VM can lose /content/data; the archive is restored from Drive without another internet download if its checksum is valid. The shared archive persists separately from per-run metadata.
6. Compatible metadata are reused/completed without altering label assignments. Corrupt or conflicting final files are preserved with a clear error; inspect them or choose a new directory under data_checks for a clean verification attempt. Never remove the runs directory or change noise/seed/size to get past a mismatch. Use only one writer for each verification directory.

The combined CLI equivalent, from the Colab checkout, is:

```bash
python -u -m src.data --prepare --config configs/default.json --noise 0.2 0.4 --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --metadata-root /content/drive/MyDrive/ido-aum-lite/experiments/run-001/data_checks/manual
```

It writes separate noise_20 and noise_40 subdirectories. Existing single-noise commands retain their direct metadata-root layout. Add --no-download only when the local extracted cache is already complete; a missing/corrupt cache then fails instead of attempting network access. Timing/progress changes do not modify scientific configuration, dataset size, noise levels or training settings.

## Step 2: model, WE/AUM and diagnostic verification

The model uses weights=None, a 3x3 stride-1/padding-1 stem, no initial maxpool and 10 outputs. No ImageNet weights are fetched. build_model(settings) imports torch/torchvision only when called in Colab.

Run the hand-logit tracking tests locally without torch:

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_statistics.py -q
```

In the latest Colab notebook, rerun Source and dependencies so the printed Step 2 module-availability flags are True. Run **Hosted tests (Steps 0–2)** with RUN_COLAB_TESTS=True, then **Step 2 verification: GPU model and tracking smoke** with RUN_SMOKE=True. This uses src.smoke, not the separate src.train staged runner. The equivalent smoke command, from the Colab checkout, is:

```bash
python -m src.smoke --config configs/default.json --device cuda --output-root /content/drive/MyDrive/ido-aum-lite/experiments/run-001/smoke_step2
```

This is one plain-CE epoch on 512 generated tensors, not a CIFAR experiment or accuracy result. Default batch size 128 gives four training batches, followed by a complete deterministic inference pass. Reported training/diagnostic CE values and timings are real only after this command runs; synthetic timings cannot size the final experiment budget. Each invocation writes a new timestamped smoke_report.json and history/epoch_001.npz on Drive. No model/optimizer checkpoint or full-run resume behavior is claimed at this stage.

SampleStatistics(train_size) counts wrong predictions once per completed epoch and accumulates assigned-label-minus-best-alternative raw logit margins. update_epoch(logits, noisy_labels, subset_positions, epoch=...) validates exactly-once full coverage before committing; repeated/skipped epochs fail. Its we and observation_counts are int32, margin_sum and AUM float32. AUM cannot be queried before an observation. IDs can arrive in any order; they are scattered into stable subset positions. copy()/state_dict()/from_state_dict() preserve history with independent branch arrays.

make_statistics_loader supplies deterministic eval transforms, shuffle=False and drop_last=False. collect_epoch_statistics uses model.eval() and torch.no_grad(), restores all previous module modes, and computes sample-mean unweighted noisy-label CE. Passing no tracker performs only that common diagnostic, without WE/AUM updates. No clean labels/masks are inputs to these APIs.

The warm-up runner calls save_snapshot with a distinct epoch filename after every completed epoch (six by default). The .npz stores WE, cumulative margin sums, per-example observation counts, tracked epochs and AUM. Snapshots are verified before publication, identical repeat saves are idempotent, conflicting history is rejected, and AUM is validated against saved sums/counts on reload. Dataset/configuration/code checksums and full model/optimizer/RNG resume state are handled by src.checkpoints and src.train.

## Step 3: relative scoring and shared adaptive objective

Local synthetic checks (no torch, download or training):

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_scores.py tests/test_loss_diagnostics.py -q
```

In Colab, the existing opt-in hosted-test block discovers the new tests automatically. To run the focused checks from the checkout:

```bash
python -m pytest tests/test_scores.py tests/test_loss_diagnostics.py tests/test_losses.py -q
```

`percentile_rank_by_noisy_class(values, noisy_labels)` uses one-based average tie ranks and `(rank-0.5)/class_size`, preserving stable-ID array order. Singleton/all-tied classes return 0.5; empty classes are skipped. `suspicion_scores(tracker.we, tracker.aum, noisy_labels, observation_counts=tracker.observation_counts)` returns `ido_lite`, `aum` (ranked negative AUM), and their 50:50 `hybrid`. Positive, equal observation counts are required. Each complete noisy-label class has mean suspicion 0.5; these ranks are not corruption probabilities. Only provided labels and completed statistics are inputs.

`heuristic_weights(s)` returns classification weight `0.1+0.9*(1-s)` and parabolic difficulty `4*s*(1-s)`. The classification floor and objective are experimental IDO-inspired simplifications, not original IDO/BMM.

`adaptive_loss(weak_logits, strong_logits, noisy_labels, batch_scores, **loss_settings)` returns a differentiable scalar plus detached diagnostic numbers. Gather `batch_scores` from the frozen full-subset score vector by stable sample IDs. Both adaptive methods call this same function; only score inputs differ. Pass the central configuration's `pseudo_threshold`, `clean_weight_floor`, `pseudo_coefficient`, `consistency_coefficient` and `near_zero_epsilon` as `loss_settings`; function defaults match `configs/default.json`.

The objective is `mean(c*CE(weak,noisy_y) + 0.5*s*mask*CE(strong,detached_weak_pseudo) + 0.5*d*class_mean((p_weak-p_strong)^2))`. Confidence uses the inclusive 0.7 threshold. Masked samples remain in the full-batch denominator, and consistency retains both gradient paths. Softmax/loss computations use float32 with autocast disabled internally. Frozen scores and pseudo-label creation are detached.

Diagnostics include weighted supervised/pseudo/consistency losses, total, mean c/d, confidence fraction, and exact-zero/near-zero fractions (including zeros) for each component and total. `AdaptiveEpochSummary.update(logs)` and `.compute()` reduce every field by sample count, including partial final batches, and check that components sum to total. These record loss values, not gradient norms. The common deterministic unweighted noisy-label CE remains a separate Step 2 diagnostic.

The Step 4 runner freezes scores from all prior completed epochs, trains, collects deterministic statistics and computes scores for the next epoch. Step 3 does not add the staged runner, checkpoints, benchmark or full training. Blocks 10-13 remain disabled.

## Step 4: staged runner, checkpoints and runtime benchmark

Step 4 hosted test command, integrated GPU smoke and completed-stage resume passed at 9892aee, as confirmed by the user. Step 5 adds new hosted checks; run those against the final reviewed code before real experiments. No GPU allocation is needed for lightweight code development.

Local lightweight verification:

```powershell
./.venv/Scripts/python.exe -m pytest -q -m "not torch"
./.venv/Scripts/python.exe -m pip check
```

When Colab GPU becomes available, rerun source/setup for the reviewed commit, then hosted tests (Block 7). These include the pending Step 3 loss/gradient checks and Step 4 tiny-fixture tests for interrupted versus uninterrupted training with zero/two workers, complete BatchNorm resets, fresh branch optimizers, RNG/generator restoration, immutable warm-up history and idempotent resume. Checkpoints use the full-state PyTorch loader; load only trusted project artifacts.

From the Colab checkout, run the integrated synthetic smoke via the existing command helper (these are notebook Python statements):

```python
run_module('src.train', '--smoke', '--noise', 0.2, '--device', 'cuda',
           '--output-root', EXPERIMENT_ROOT / 'smoke_step4')
# Rerun with --resume to verify completed stages are skipped.
run_module('src.train', '--smoke', '--noise', 0.2, '--device', 'cuda',
           '--output-root', EXPERIMENT_ROOT / 'smoke_step4', '--resume')
```

This uses 512 generated random images/labels, one warm-up and one epoch per continuation, in a separate `synthetic_smoke/noise_20` tree. It is an engineering check, not a CIFAR experiment, measured accuracy or runtime estimate. CPU is permitted only for explicit synthetic smoke/unit fixtures; do not run model smoke/training on the Windows development environment.

After hosted checks pass, Block 10 benchmarks one complete warm-up, standard and hybrid adaptive epoch on the candidate real subset, including deterministic inference and persistent checkpoint read-back/publication. An extra verified payload read estimates the predecessor-verification cost of later epochs; its measured duration is included conservatively in the projection. It uses an isolated short warm-up, not the scientific six-epoch warm-up. Each trial lives under `benchmark/trials/<timestamp>/noise_20`. The root `benchmark_report.json` records candidate identity, measured epoch times, elapsed setup/benchmark time and the Section I projection. `benchmark/experiment_config.json` is a candidate, not a frozen experiment. Benchmark counts are distinct from the projected full experiment counts. No automatic fallback or clean-label/test-quality decision is made. Review the projection plus six test passes, setup, benchmark, detection aggregation and debugging margin within the 4.5-hour allocation. Apply any planned fallback uniformly, rebenchmark and freeze once using Block 11.

Blocks 10/12 now pass the explicit archive cache path. A CLI equivalent (inside Colab, with `EXPERIMENT_ROOT` set) is:

```bash
python -m src.train --benchmark --config configs/default.json --noise 0.2 --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --output-root "$EXPERIMENT_ROOT/benchmark"
python -m src.train --config "$EXPERIMENT_ROOT/experiment_config.json" --noise 0.2 --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --output-root "$EXPERIMENT_ROOT/runs" --resume
# Then repeat the same frozen command for --noise 0.4.
```

Full runs refuse CPU/Windows execution and require a clean reviewed checkout, the notebook's frozen config/manifest pair, and a completed benchmark matching code, scientific settings and runtime. GPU changes require explicit acknowledgement (`ACKNOWLEDGE_GPU_CHANGE` in the notebook, or CLI `--acknowledge-gpu-change`); changed package/config/code/data identities are rejected. Same-environment deterministic fixture equivalence does not imply bitwise reproducibility across GPU/kernel versions.

Each noise condition has `run.json`, a shared `warmup/`, and `standard/`, `ido_lite/`, `hybrid/` directories. New branches load the exact warm-up model including BatchNorm buffers, copy statistics, reset optimizer/scaler and reset the same branch RNG/loader streams. Adaptive scores are frozen from prior completed epochs and refreshed only after the next deterministic statistics pass. Standard continuation performs CE diagnostics without WE/AUM updates. Training never loads the test split or analysis metadata and never writes final metrics.

Each stage journal (`latest.json` plus `latest.previous.json`) points to checksum-verified immutable `.pt` payloads. A payload includes model, optimizer, scaler, cumulative statistics, complete warm-up history, all completed logs, configuration/code/data/runtime identity, Python/NumPy/CPU/CUDA RNGs and both loader-generator states. Identity is checked before deserialisation and state restoration. Object construction/loading finishes before RNG restoration and before the next loader iterator. Workers are recreated each epoch with `persistent_workers=False`.

Temporary/orphan payloads do not commit an epoch. If the latest payload/pointer is corrupt, recovery uses the previous verified generation. A durably completed epoch is not repeated; an unfinished epoch can repeat. `epochs.json` and `warmup/history/epoch_*.npz` are derived from the checkpoint and regenerated after interrupted publication. Per-epoch timing files record training/inference and checkpoint publication separately; a missing timing sidecar after a crash does not invalidate a verified training epoch. No historical time is invented during recovery. `completion.json` is derived; resume verifies stage journals before skipping completed branches.

Epoch payloads are retained, so budget Drive storage for model and optimizer states across both conditions. Use one writer per experiment. Recover with the same RUN_ID, pinned code, frozen configuration and persistent paths; runtime source/extraction can be rebuilt from GitHub and the verified shared archive. Do not delete persistent checkpoints, alter labels or repeat fallback selection during recovery. Final evaluation and result aggregation are owned by src.evaluate; plotting remains Step 6.

## Step 5: final evaluation and result aggregation

Step 5 full experiments and all six final measurements completed in Colab at 695f50f; the exported measurements are preserved in results/. Local tests use controlled synthetic fixtures, never experimental result claims. The evaluator imports torch only for checkpoint/model operations; pure detector, cache and aggregation tests remain lightweight.

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_evaluate.py -q
```

Before real runs, commit/push the reviewed Step 5 code, refresh Colab source/setup, and run hosted tests including `tests/test_evaluate_torch.py`. Benchmark/freeze must use that same reviewed code identity, so the pinned experiment includes the evaluator. The Step 4 smoke outputs remain engineering evidence and are never accepted as scientific runs. Repeating their checks is only needed when relevant code/environment changes warrant it. This update does not alter training configuration, noise, seeds, transforms, loss or the deferred Block 7 output behavior.

For the real experiment, use Block 10 to benchmark, review remaining time and any uniform Section I fallback, then Block 11 to freeze once. Block 12 executes/resumes both noise conditions. Do not change configuration, code or label metadata after freezing. Allow for evaluation and setup/Drive/debugging overhead in the 4.5-hour allocation; the 512-image smoke timing is not a runtime estimate for real training.

Block 13 now has separate `RUN_EVALUATION=False` and `RUN_PLOTTING=False` controls. Once final checkpoints exist, enable evaluation only. Plotting is now implemented and CPU-safe; use a checkout containing src.plots without changing the frozen training/evaluation checkout. The equivalent CLI from the frozen checkout is:

```bash
python -m src.evaluate --device cuda --data-root /content/data --archive-root /content/drive/MyDrive/ido-aum-lite/cache/cifar10 --runs "$EXPERIMENT_ROOT/runs" --output-root "$EXPERIMENT_ROOT/results"
```

Full inference refuses CPU/Windows execution and checks the frozen config/manifest, completed benchmark, source identity and runtime before loading official test data. GPU changes use the same explicit acknowledgement as training. Evaluation reads only matching scientific `noise_20` / `noise_40` runs; synthetic smoke and benchmark identities are rejected. All three methods must share the same full warm-up state and historical statistics. Both conditions must share configuration, runtime versions and selected training image IDs. Model hashes include all BatchNorm buffers.

Each completed final model is tested on the entire untouched official 10,000-image test split, with deterministic transforms and no AMP. No checkpoint/epoch/hyperparameter selection uses these accuracies. A lazy test-data loader is shared across missing evaluations. Each completed measurement is atomically cached under `results/accuracy_cache/` using full checkpoint hash, model hash, run identity/configuration and evaluation protocol. Cached records verify count/accuracy consistency, identity and payload checksum. Unchanged repeat aggregation does not load the test dataset or rerun model inference. An interrupted measurement without a published cache may repeat; a corrupt published cache is reported for inspection instead of silently accepted or remeasured.

Detection uses the saved shared warm-up history embedded in the verified checkpoint, including cumulative WE, raw-logit margin sums and observation counts. The prespecified primary epoch is `warmup_epochs`; earlier epochs are retrospective diagnostics only and are never selected for quality. Primary scores are raw WE and negative AUM, plus an optional rank-fused hybrid. Missing history, unobserved samples, inconsistent masks or absent clean/corrupted classes fail explicitly rather than substituting ROC-AUC 0.5.

Outputs under the persistent results root:

- `evaluation_manifest.json`: scientific configuration, code identity and test protocol; incompatible existing results are preserved and rejected.
- `accuracy_cache/<identity>.json`: one verified completed test measurement per final checkpoint identity, including correct/total, accuracy percent, data fingerprint, runtime and evaluation duration.
- `detectors/noise_20/` and `detectors/noise_40/`: source-identified NPZ arrays and JSON summaries containing every warm-up epoch's WE/margin sums/counts/AUM, original/stable IDs, provided/analysis labels, corruption mask, primary detector scores and per-epoch ROC-AUC. These are evaluation-only plot inputs.
- `metrics.json`: completion/pending status, full method/configuration provenance, checkpoint/cache references, warm-up and continuation logs, component summaries, measured available timings and detector summaries.
- `metrics_manifest.json`: final publication marker with CSV/JSON checksums and completion count. Verify it before consuming results; a missing/mismatched marker indicates an interrupted aggregation. Rerun evaluation to rebuild from caches.
- `metrics.csv`: one row per configuration/noise/method, rebuilt from currently verified completed checkpoints and their caches. Repeating or incrementally completing conditions replaces rows without duplicates. All six rows are required for `complete=true`; a missing/incomplete/corrupt-final-fallback branch is reported pending and contributes no stale final row.

CSV accuracy is a percentage; ROC-AUC values lie in [0,1]. Each method's reported training duration includes the shared warm-up, so summing rows would count shared work three times. Missing checkpoint-publication timing sidecars remain null/explicitly incomplete; no duration is invented. JSON retains sample-weighted adaptive component/confidence/zero/near-zero diagnostics and common deterministic CE separately from the weighted training objective. CSV/JSON are derived from immutable checkpoint/cache records and can be rebuilt after interrupted aggregation without repeating completed test measurements. Use one evaluator writer per results directory.

After both conditions finish, confirm `metrics.json` says `complete: true`, `row_count: 6`, with two detector summaries and primary WE/-AUM AUC for both noise levels. These are exploratory single-seed results from an IDO-inspired simplification; negative outcomes are valid. Actual real-data numbers must come from those persisted outputs. Save source/frozen metadata and small reviewed results to GitHub later; keep datasets and model/optimizer checkpoints on Drive. Step 6 figures/tables are generated below; the Step 7 report remains unimplemented.

## Step 6: figures, comparison tables and measured source exports

All four required plots are generated from the supplied `run-001` evaluation export. No GPU, model checkpoint, CIFAR download or model inference is needed for plotting. The separate epoch files are redundant: `metrics.json` already embeds every common deterministic warm-up/continuation CE log. `--runs` remains an optional cross-check/fallback for external `epochs.json` files.

Regenerate all figures/tables locally with the existing lightweight environment:

```powershell
./.venv/Scripts/python.exe -m src.plots --results results --config results/experiment_config.json
./.venv/Scripts/python.exe -m pytest tests/test_plots.py -q
```

The same CLI works in a CPU Colab session using `python -m src.plots --results <exported-results> --runs <runs-root>`. `--config` can name an explicit frozen configuration; otherwise the reader checks `experiment_config.json` inside/beside results or uses its verified embedded configuration. `--output-root` writes derived files elsewhere while preserving source inputs.

A frozen training checkout at 695f50f predates `src.plots`. Preserve its code/configuration identity for training/evaluation recovery; run the newer plotting code against exported results in the development checkout or a separate CPU checkout. There is no reason to retrain or unfreeze the experiment to generate figures. Block 13 keeps independent evaluation/plotting flags, both disabled by default; its plotting action requires a checkout containing the plotter.

Generated artifacts:

| File under `results/` | Content |
| --- | --- |
| `figures/accuracy_comparison.png` | Three measured final accuracies at each noise ratio; a zero-based 0-100% axis, individual value labels and no fabricated uncertainty bars. |
| `figures/noise_detection_roc.png` | Raw WE versus negative-AUM ROC curves at the configured primary warm-up epoch, one panel per noise condition; plotted AUCs are recalculated from the saved arrays and checked against evaluation. |
| `figures/score_distributions.png` | WE and raw AUM separated by actual clean/corrupted labels at the first configured noise level (20% here), selected independently of quality. Each group is normalized separately. |
| `figures/common_ce_curves.png` | Shared warm-up followed by three continuations for each noise condition. Only common deterministic unweighted provided-label CE is used; the boundary is after configured epoch W (6 here), never hardcoded or inferred from curve shape. |
| `comparison_table.csv` / `comparison_table.md` | Six measured accuracy rows and changes versus Standard in percentage points; Markdown also includes the two shared warm-up detector summaries. |
| `plots_manifest.json` | Original experiment code identity, scientific configuration, input checksums, plotter-source checksum, configured boundary/representative noise and hashes of all six generated artifacts. |

The supplied `experiment_config.json`, original evaluation CSV/JSON/publication manifests, six accuracy-cache records and detector summaries/NPZ arrays are preserved under `results/` for a portable offline reproduction. These small per-sample statistic arrays are not images or model checkpoints. The untouched import copy is also retained under ignored `runs/colab-export/run-001/`. All large checkpoints and CIFAR files remain on Drive.

Before plotting, the reader verifies the evaluation publication marker and CSV/JSON agreement; requires all six completed rows; checks configuration, counts, stable IDs, corruption masks, historical WE/margin sums/counts/AUM and detector checksums; recalculates primary detector AUCs; and validates complete common-CE logs. Missing or inconsistent sources fail before creating figures. The original `metrics.csv` and evaluation caches are never rewritten by plotting. Every figure records actual subset/schedule/seed, source and single-seed scope. Non-default size/schedule is labelled when applicable.

The measured table records IDO-lite/Hybrid accuracies of 75.26%/74.76% at 20% noise and 68.61%/68.48% at 40%, versus Standard 70.42%/63.80%. These are exploratory observations from one seed: AUM's stronger detection AUC does not establish that hybrid scoring improves test accuracy. Standard versus adaptive compares whole strategies; WE-only versus Hybrid holds the adaptive objective/views fixed. The final report must retain these qualifications.

## Restart and resume

1. Reopen the notebook with GPU enabled and mount the same Drive account.
2. Use the **same `RUN_ID`** and persistent directory. Do not create a new ID for each session or run two writers against one experiment.
3. Rerun setup. It reads the recorded Git SHA, reclones/checks out that version, checks/installs dependencies as needed and verifies the saved configuration/environment. Restore the recorded dependency/runtime versions if checks report a mismatch. If only the assigned GPU changed, the notebook requires explicit `ACKNOWLEDGE_GPU_CHANGE=True` and records the new hardware before continuing; bitwise equivalence is not guaranteed. Restore the disposable `/content/data` archive using Block 6 (or let Block 8 restore it before preparation). The verified Drive archive survives runtime replacement.
4. Skip benchmark/freeze cells for an existing frozen experiment. Never overwrite its configuration or change the selected image IDs/noisy labels during recovery.
5. Rerun training cells with `--resume` (already included). The runner starts missing stages, restores interrupted branches from their last complete persistent checkpoint, and skips completed branches. Incomplete epoch work can repeat; completed statistics/log rows must not repeat. Resume restores optimizer/scaler/RNG/loader/statistics state; starting a new branch resets optimizer/RNGs after loading the shared warm-up state.
6. Rerun evaluation/plotting cells. The evaluation module caches final checkpoint identities and avoids duplicate evaluation/rows. Confirm results are present on Drive.

Colab runtime-local files are disposable. Persistent writes must retain a previous verified checkpoint and ignore partially published files on restart. A final copy at session end is insufficient. GPU/environment changes may prevent bitwise reproducibility; record them rather than silently claiming equivalence. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Keep credentials, datasets and checkpoints out of Git. Notebook outputs should remain cleared. Export only reviewed small configuration/metrics/figures/report artifacts to GitHub.
