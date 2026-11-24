# IDO-AUM Lite

Develop with VS Code and Codex on Windows. Run all full training, benchmarking and final evaluation on a Google Colab Free GPU. A local GPU is not required.

**Current stage: Step 2 implemented; hosted verification pending.** The CIFAR-adapted model, WE/AUM/common-CE tracker, history snapshots and isolated GPU smoke are implemented alongside Step 1 data preparation. NumPy tracker tests run locally without PyTorch. Model/collection tests and the GPU smoke must be run in Colab. Adaptive scoring/loss, the full experiment runner and final evaluation/plotting remain later stages; no model training or data download has run locally.

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
- Other research modules (planned): scoring, loss, full checkpoint/resume, training, evaluation and plotting.
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
| Future full training (not implemented yet) | GPU, `SESSION_MODE='train'` | 1-5, 6 as needed, 10 then 11 once, 12 | Later: `RUN_BENCHMARK`, `FREEZE_APPROVED`, `RUN_TRAINING` |
| Future resume / final evaluation (not implemented yet) | GPU, `SESSION_MODE='train'`, same RUN_ID | 1-5, 6 if local archive missing, 12 / 13 | Later: `RUN_TRAINING` / `RUN_EVALUATION`; skip 10-11 for a frozen run |

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
7. Run the optional hosted-test cell to install requirements-dev.txt and run python -m pytest -q in Colab. This includes the implemented PyTorch-dependent checks; Steps 1–2 include synthetic-image transform/DataLoader/model/collection tests; the full training pipeline remains later work.
8. Run the Step 2 smoke cell now when ready. The benchmark/freeze/full-training/evaluation cells require the later experiment runner and must wait for those stages. Notebook commands use sys.executable -m, checked subprocess argument lists and explicit repository cwd. Missing research modules produce an explicit error; their implementation remains later work.

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

Experimental settings, the 15-hour budget and uniform fallback rules are unchanged. Default counts remain 48 executed training epochs, 72 training image-view passes, 36 statistics passes and 12 diagnostic-only inference passes. Step 1 data preparation and Step 2 synthetic model/tracker verification are available below; full experiments remain unimplemented.

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

In the latest Colab notebook, rerun Source and dependencies so the printed Step 2 module-availability flags are True. Run **Hosted tests (Steps 0–2)** with RUN_COLAB_TESTS=True, then **Step 2 verification: GPU model and tracking smoke** with RUN_SMOKE=True. This uses src.smoke, not the future src.train runner. The equivalent smoke command, from the Colab checkout, is:

```bash
python -m src.smoke --config configs/default.json --device cuda --output-root /content/drive/MyDrive/ido-aum-lite/experiments/run-001/smoke_step2
```

This is one plain-CE epoch on 512 generated tensors, not a CIFAR experiment or accuracy result. Default batch size 128 gives four training batches, followed by a complete deterministic inference pass. Reported training/diagnostic CE values and timings are real only after this command runs; synthetic timings cannot size the final experiment budget. Each invocation writes a new timestamped smoke_report.json and history/epoch_001.npz on Drive. No model/optimizer checkpoint or full-run resume behavior is claimed at this stage.

SampleStatistics(train_size) counts wrong predictions once per completed epoch and accumulates assigned-label-minus-best-alternative raw logit margins. update_epoch(logits, noisy_labels, subset_positions, epoch=...) validates exactly-once full coverage before committing; repeated/skipped epochs fail. Its we and observation_counts are int32, margin_sum and AUM float32. AUM cannot be queried before an observation. IDs can arrive in any order; they are scattered into stable subset positions. copy()/state_dict()/from_state_dict() preserve history with independent branch arrays.

make_statistics_loader supplies deterministic eval transforms, shuffle=False and drop_last=False. collect_epoch_statistics uses model.eval() and torch.no_grad(), restores all previous module modes, and computes sample-mean unweighted noisy-label CE. Passing no tracker performs only that common diagnostic, without WE/AUM updates. No clean labels/masks are inputs to these APIs.

During the future warm-up runner, call save_snapshot with a distinct epoch filename after every completed epoch (six by default). The .npz stores WE, cumulative margin sums, per-example observation counts, tracked epochs and AUM. Snapshots are verified before publication, identical repeat saves are idempotent, conflicting history is rejected, and AUM is validated against saved sums/counts on reload. Dataset/configuration/code checksums and full model/optimizer/RNG resume state belong to the future outer checkpoint layer.

## Restart and resume

1. Reopen the notebook with GPU enabled and mount the same Drive account.
2. Use the **same `RUN_ID`** and persistent directory. Do not create a new ID for each session or run two writers against one experiment.
3. Rerun setup. It reads the recorded Git SHA, reclones/checks out that version, checks/installs dependencies as needed and verifies the saved configuration/environment. Restore the recorded dependency/runtime versions if checks report a mismatch. If only the assigned GPU changed, the notebook requires explicit `ACKNOWLEDGE_GPU_CHANGE=True` and records the new hardware before continuing; bitwise equivalence is not guaranteed. Restore the disposable `/content/data` archive using Block 6 (or let Block 8 restore it before preparation). The verified Drive archive survives runtime replacement.
4. Skip benchmark/freeze cells for an existing frozen experiment. Never overwrite its configuration or change the selected image IDs/noisy labels during recovery.
5. Rerun training cells with `--resume` (already included). The future runner must start missing stages, restore interrupted branches from their last complete persistent checkpoint, and skip completed branches. Incomplete epoch work can repeat; completed statistics/log rows must not repeat. Resume restores optimizer/scaler/RNG/loader/statistics state; starting a new branch resets optimizer/RNGs after loading the shared warm-up state.
6. Rerun evaluation/plotting cells. The future evaluation module caches final checkpoint identities and avoids duplicate evaluation/rows. Confirm results are present on Drive.

Colab runtime-local files are disposable. Persistent writes must retain a previous verified checkpoint and ignore partially published files on restart. A final copy at session end is insufficient. GPU/environment changes may prevent bitwise reproducibility; record them rather than silently claiming equivalence. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Keep credentials, datasets and checkpoints out of Git. Notebook outputs should remain cleared. Export only reviewed small configuration/metrics/figures/report artifacts to GitHub.
