# IDO-AUM Lite

Develop with VS Code and Codex on Windows. Run all full training, benchmarking and final evaluation on a Google Colab Free GPU. A local GPU is not required.

**Current stage: Step 0 complete.** Portable configuration, a thin Colab entry notebook and CPU-only scaffold tests are implemented. Research modules (`src.train`, `src.evaluate`, `src.plots`, etc.) and research-method tests belong to later steps and have not been implemented. The notebook checks for these modules before launching experiments; it does not contain a substitute training implementation. Follow the existing implementation plan for the remaining stages. No experiments or results have been produced.

## Repository responsibilities

- `config.py`: CPU-safe configuration loader/validator and scientific-settings fingerprint.
- `configs/default.json`: portable scientific defaults (20k images, 6+6 epochs, seed 42).
- `notebooks/run_colab.ipynb`: Drive mount, pinned checkout, dependencies/GPU checks, configuration, benchmark/run/resume CLI orchestration.
- `src/` (planned): reusable data, model, statistics, scores, loss, checkpoint, training, evaluation and plotting modules. All research and resume logic belongs here.
- `requirements.txt`: common runtime dependencies; leaves Colab's torch/torchvision pair intact.
- `requirements-dev.txt`: pytest and nbformat for CPU-only scaffold validation.
- `tests/`: runnable configuration and notebook checks; no data download or GPU required.

## Windows development

From the repository root in the VS Code PowerShell terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Run the Step 0 checks now:

```powershell
.\.venv\Scripts\python.exe -m config
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

If a compatible CPU-only torch/torchvision pair is already installed, an optional `python -m venv --system-site-packages .venv` reuses it; this is how the checked local environment was created. Otherwise use the isolated setup above. No global packages are modified.

The 25 scaffold checks validate portable configuration (including 10,000 official test images and the CIFAR stem), permitted fallbacks, invalid settings, the config CLI without site-packages, notebook schema/syntax, path quoting, configuration publication and the no-GPU guard. Colab/Drive calls are mocked; an actual Colab GPU session remains a hosted check.

Once src.train is implemented, a tiny local smoke can use `python -m src.train --smoke --device cpu --data-root ./data --output-root ./runs/smoke`. Use small synthetic fixtures and `num_workers=0` for CPU tests. No full data download or local CUDA is needed for unit tests. Research modules must use `pathlib`, accept storage paths via CLI, guard executable entry points for Windows multiprocessing, and avoid importing `google.colab`.

## Colab execution

1. Commit/push the repository to GitHub and open `notebooks/run_colab.ipynb` in Colab. Select **Runtime > Change runtime type > GPU**.
2. Set `REPO_URL`, `REPO_REF` and a stable `RUN_ID` in its first setup cell. Mount Drive when prompted. Source checkout and CIFAR cache live under `/content`; persistent artifacts live under `/content/drive/MyDrive/ido-aum-lite/experiments/<RUN_ID>`.
3. Run setup, installation and CUDA checks. Full experiments stop if no GPU is available; do not switch full runs to local CPU. Installed torch/torchvision are import-checked and preserved.
4. Once the research modules are implemented, run the isolated benchmark. Choose fallback by runtime only, using the central configuration. Freeze the candidate configuration and code/environment manifest once before either full noise-condition run.
5. An optional smoke cell is provided for checking the training runner after it is implemented. Then run both noise conditions, evaluation and plotting through notebook CLI cells. `src.train` must save complete verified epoch-boundary checkpoints, score histories, logs and completion manifests to the supplied persistent output root throughout training. The notebook does not implement checkpoint saving.

Default compute accounting is 48 executed training epochs, 72 training image-view passes, 36 statistics passes and 12 additional diagnostic-only inference passes. The benchmark, Drive I/O, evaluation, development and reporting all count toward the 15-hour budget. See the plan for uniform fallback rules.

## Restart and resume

1. Reopen the notebook with GPU enabled and mount the same Drive account.
2. Use the **same `RUN_ID`** and persistent directory. Do not create a new ID for each session or run two writers against one experiment.
3. Rerun setup. It reads the recorded Git SHA, reclones/checks out that version, reinstalls dependencies and verifies the saved configuration/environment. Restore the recorded dependency/runtime versions if checks report a mismatch. If only the assigned GPU changed, the notebook requires explicit `ACKNOWLEDGE_GPU_CHANGE=True` and records the new hardware before continuing; bitwise equivalence is not guaranteed. Recreate the disposable `/content/data` cache if needed.
4. Skip benchmark/freeze cells for an existing frozen experiment. Never overwrite its configuration or change the selected image IDs/noisy labels during recovery.
5. Rerun training cells with `--resume` (already included). The future runner must start missing stages, restore interrupted branches from their last complete persistent checkpoint, and skip completed branches. Incomplete epoch work can repeat; completed statistics/log rows must not repeat. Resume restores optimizer/scaler/RNG/loader/statistics state; starting a new branch resets optimizer/RNGs after loading the shared warm-up state.
6. Rerun evaluation/plotting cells. The future evaluation module caches final checkpoint identities and avoids duplicate evaluation/rows. Confirm results are present on Drive.

Colab runtime-local files are disposable. Persistent writes must retain a previous verified checkpoint and ignore partially published files on restart. A final copy at session end is insufficient. GPU/environment changes may prevent bitwise reproducibility; record them rather than silently claiming equivalence. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

Keep credentials, datasets and checkpoints out of Git. Notebook outputs should remain cleared. Export only reviewed small configuration/metrics/figures/report artifacts to GitHub.
