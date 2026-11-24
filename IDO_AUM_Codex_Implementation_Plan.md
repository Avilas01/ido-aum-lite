# IDO-AUM Lite: Solo Codex Implementation Plan

**Course:** BITS F471 — Introduction to Large Language Models  
**Project:** WE+AUM for robust classification with synthetic noisy labels  
**Target:** One implementer; at most 15 hours total (coding, training, testing, plots, report)  
**Development:** VS Code + Codex on Windows; lightweight CPU tests, code development and debugging only.  
**Execution:** All full-model training, benchmarking and final evaluation run on Google Colab Free with GPU acceleration; PyTorch + torchvision. No local GPU is required.  
**Research status:** A deliberately simplified **IDO-inspired** experiment, **not** an exact reproduction of the original IDO paper or the original AUM threshold-sample procedure.

## A. Scope, research questions, and success criteria

**Research questions:**
1. Can Wrong Events (WE) identify mislabeled training examples?
2. How does AUM compare with WE for noisy-label detection?
3. Does a fixed 50:50 WE+AUM combination improve generalisation relative to WE-only IDO-lite?

**Datasets:** CIFAR-10, 20,000 stratified training examples by default from the official 50,000-image training split; untouched official 10,000-image test split. No manual image download: `torchvision.datasets.CIFAR10(download=True)` fetches the archive. The two corruptions use the same configured selected image IDs (20,000 default; 10,000 only under the Section I fallback).

**Noise:** exactly 20% and 40% symmetric *flip-to-a-different-class* label noise (4,000 and 8,000 changed labels at the default subset size; derive counts under fallback). A different random wrong class is chosen uniformly among the 9 incorrect classes. Keep the original label for analysis only. Seed = 42; noise generation is deterministic for each level.

**Network:** ResNet-18, trained from scratch (`weights=None`) with CIFAR-10 adaptation: first 7x7 stride-2 convolution becomes 3x3 stride-1; remove first maxpool; final linear layer has 10 outputs. Do not download ImageNet weights. Training on free GPU is not guaranteed; measure epoch time early and use fallback if needed.

**Final methods (3):**
- **Standard:** cross-entropy on provided (possibly noisy) labels.
- **IDO-lite:** per-instance WE-based ranking used in an IDO-inspired adaptive loss.
- **Hybrid IDO-AUM:** exact same model and adaptive loss, but noise-suspicion ranking uses 50% WE + 50% inverted AUM.

**Training schedule per noise level:**
1. **Stage 1 shared warm-up**: 6 epochs standard cross-entropy with WE + AUM tracking after each epoch. Save a separate statistics snapshot after every warm-up epoch for retrospective detection analysis.
2. Save one shared warm-up checkpoint per noise condition: complete model state including BatchNorm parameters and running-statistics buffers, WE arrays, cumulative margin sums and observation counts.
3. **Stage 2 three branches:** each starts from *exactly the same complete saved model state*, resets its optimizer at the same learning rate, and runs 6 additional epochs: standard, IDO-lite, hybrid. Reset branch RNGs and controlled DataLoader generators before each continuation. WE/AUM dynamic arrays are initialised from the shared warm-up for each adaptive branch, **copied separately**, and refreshed each epoch. Never allow one branch to reuse another branch's updated statistics. Do not collect Stage 2 WE/AUM in standard; run a deterministic training-subset inference pass only for the common CE diagnostic. Adaptive statistics passes also collect that diagnostic.
4. Each resulting method has seen 12 epochs by default (6 shared + 6 continuation; derive actual totals from configuration under fallback). The warm-up is **shared computation**, not counted as three independent extra runs.
5. Run all of the above for both 20% and 40% noise: **6 final trained variants, 2 shared warm-ups**.

**Comparison interpretation:** Standard Training (plain CE) versus IDO-lite compares overall training strategies, including loss and augmentation procedures. IDO-lite versus Hybrid WE+AUM-IDO (also called Hybrid IDO-AUM here) is the primary controlled comparison: only WE-only versus fused suspicion differs. Do not add AUM-only IDO as a required experiment. One-seed differences are exploratory observations, not statistically reliable evidence of superiority.

**Fixed training defaults:** batch size 128; AdamW `lr=3e-4`, `weight_decay=1e-4` for all branches, reinitialised at branch start; one experimental seed 42; deterministic train/eval splits. Use explicit recorded RNG streams derived from 42 for subsets, each noise level, model initialisation, branch shuffling and augmentations; these are reproducibility streams, not additional experimental seeds. Seed Python, NumPy, PyTorch CPU/CUDA and DataLoader workers; control DataLoader generators and reset model-initialisation seed for each noise condition. Use default autocast mixed precision only when GPU and PyTorch setup supports it reliably. `num_workers=2` (reduce to zero if notebooks have worker issues). Full-run and benchmark commands require an explicit CUDA device and fail early if CUDA is unavailable; permit CPU only for unit tests and explicitly requested lightweight smoke runs. Train augmentations: random crop with padding 4, random horizontal flip, normalise. Deterministic statistics/eval transform: tensor conversion and normalise only. Adaptive second view: independent crop/flip plus modest colour jitter (not extreme), then tensor conversion and normalisation; apply jitter before normalisation. Do not turn a new tuning sweep into part of this project. Optimizer and learning-rate choices are pragmatic defaults rather than reproductions of the paper.

**Primary outcome:** test accuracy (%) on the untouched test split, measured **once per completed final method**; no selection of checkpoints/hyperparameters by test accuracy. **Detection outcome:** ROC-AUC against synthetic corruption mask for WE, inverted AUM, and optional combined score, computed from saved shared warm-up statistics, using the final warm-up epoch as the prespecified primary detector outcome. Where practical, retrospectively save ROC-AUC after every warm-up epoch; never use detection quality to change the configuration. The evaluation module alone owns test/detection metrics, aggregation and results/metrics.csv; training saves checkpoints, statistics and logs. Save common diagnostic curves and adaptive component logs. No promises about accuracy or beating baselines.

**Central configuration:** Defaults: train_size=20000, warmup_epochs=6, continuation_epochs=6, seed=42, pseudo_threshold=0.7, clean_weight_floor=0.1, near_zero_epsilon=1e-8. Derive per-class size as train_size/10, corruption counts as round(p*train_size), total model epochs as warmup_epochs+continuation_epochs, and all array lengths/assertions from configuration. The benchmark in Section I selects one frozen configuration for both noise levels before full experiments. Commands load that saved configuration; plot labels, tables, report and completion checklist must show actual configured values, including fallbacks.

## B. Paper fidelity and deliberate omissions

Original IDO (Kuan Zhang et al., NeurIPS 2025) accumulates wrong events over training; uses a two-component **Beta Mixture Model** and BMM posterior / CDF to estimate clean, noisy, and difficult samples; trains a two-stage model with clean-label classification, pseudo-label, and consistency losses; updates statistics dynamically. Our reduced scope keeps the two-stage structure, WE collection, heuristically estimated cleanliness/difficulty, and three roles in the adaptive loss but **replaces BMM with within-class score ranks**, simplifies the losses, does not use the paper's base-model label-wave selection, and uses a small dataset/short schedule. Its results cannot be compared numerically as an exact reimplementation.

Original AUM (Geoff Pleiss et al., NeurIPS 2020) also describes **threshold samples** for calibrating a noisy-label cutoff. Here we calculate the averaged per-example margins only, to compare and fuse them with WE; the full threshold-sample procedure is out of scope.

Sources:
- IDO paper: https://proceedings.neurips.cc/paper_files/paper/2025/hash/429e7b31625a8b7839f9e4d6e2aa9bb9-Abstract-Conference.html
- Official IDO code: https://github.com/iTheresaApocalypse/IDO
- AUM paper: https://proceedings.neurips.cc/paper_files/paper/2020/hash/c6102b3727b2a7d8b1bb6981147081ef-Abstract.html
- CIFAR-10 dataset API: https://docs.pytorch.org/vision/master/generated/torchvision.datasets.CIFAR10.html

## C. Exact labels and noise procedure

1. Load `train=True` and `train=False` CIFAR-10 using torchvision. Only `train=True` is ever selected for corruption.
2. For each class (0..9), select exactly train_size/10 random original training image IDs (2,000 by default), without replacement, using NumPy RNG seed 42. Concatenate to train_size (20,000 by default), then shuffle deterministically. Maintain a `subset_position` ranging 0..train_size-1 **and** the corresponding original CIFAR index. Preserve these IDs through all Dataset and DataLoader calls.
3. Make copies `clean_labels` and `noisy_labels`. For noise ratio p, sample exactly `round(p * train_size)` distinct subset positions without replacement. For each picked position with true class `y`, choose a uniform integer in `[0,8]` and map it into `[0,9] \ {y}` (e.g., `replacement = draw + (draw >= y)`). Thus **every selected label changes**.
4. Save the corruption mask `noisy_labels != clean_labels` to metadata (analyses only) and assert its sum is exactly round(p*train_size): 4,000/8,000 by default or 2,000/4,000 for the 10,000-image fallback. Generate independent but reproducible masks for the two levels using explicit distinct RNG seeds; keep selected image IDs constant.
5. Training dataset returns `(weak_image, strong_image, noisy_label, subset_position)` where needed; standard and warm-up construct only the weak view, avoiding unused augmentation and RNG consumption. Evaluating scores uses `(deterministically_transformed_image, noisy_label, subset_position)` with `shuffle=False`, `model.eval()`, and `torch.no_grad()`. Never rely on batch ordering to assign WE/AUM statistics.
6. Official 10,000-image test split remains **unchanged**; do not use it to tune losses, select epochs, pick thresholds, or alter models. No clean validation set is necessary because the schedule is fixed in advance. Initial subset/noise construction may use original labels; after that clean labels and masks are for retrospective evaluation/sanity checks only, never training or hyperparameter selection.
7. Store image files under ignored `data/` or a Colab runtime-local data path, not in GitHub. The download archive is roughly 170 MB; it is not a separate 20,000-file download.

**Never** expose `clean_labels`, `is_corrupted`, or true-label agreement to model fitting, score normalisation, sample weights, pseudo-label creation, or any training-stage decisions. Only evaluation and sanity-checking may see them.

## D. Tracking WE and AUM correctly

For sample i at completed epoch t, compute a single deterministic full-subset inference pass using current network predictions `logits_i[t]` against the **provided noisy label** `y_noisy[i]`.

`WE_i += int(argmax(logits_i[t]) != y_noisy[i])` (one increment per completed epoch at most).

`margin_i[t] = logits_i[t, y_noisy[i]] - max_{k != y_noisy[i]} logits_i[t,k]`.

`margin_sum_i += margin_i[t]`, `observation_count_i += 1`, `AUM_i = margin_sum_i / observation_count_i`. Assert each full pass visits every subset ID exactly once; all counts then equal the number of tracked epochs. Never rank unobserved examples.

Track `WE`, `margin_sum`, and per-example observation counts in train_size-length arrays keyed by `subset_position`. Store float32 margin sum and int32 WE, optionally generated records for ROC curves. Use **raw logits** for margins, not probabilities. AUM's low values indicate higher suspicion, so use `-AUM` as the ROC score.

Save WE, margin sums, observation counts and AUM after every warm-up epoch. src.evaluate computes ROC-AUC against is_corrupted separately for WE and -AUM from these snapshots, optionally also the fused score. Use the last warm-up snapshot as the primary comparison and earlier snapshots as retrospective diagnostics only. Historical statistics cannot be reconstructed from model weights alone. Six warm-up epochs give WE only seven possible counts (0..6), leaving many ties; AUM may help by breaking ties. Do not select epochs or hyperparameters from detection ROC-AUC. AUM calculation and WE are allowed to use *noisy training labels*, because those are the labels that a real model would receive; they must not refer to the true labels.

## E. Simplified per-sample scores (our proposal, not original IDO)

Make `percentile_rank_with_ties(array, grouping_by_noisy_class)` group by the **assigned noisy label**, with higher output for larger input. Within a nonempty class of n examples, assign one-based ranks 1..n, average ranks for ties, and define:

`s_i = (r_i - 0.5) / n`.

A singleton or all-tied class receives 0.5; skip empty classes. Use deterministic stable tie handling. Observed scores lie strictly between 0 and 1, hence within [0,1]; test limiting s=0 and s=1 separately in loss tests.

For each example i:

- `s_we[i] = percentile_rank_by_noisy_class(WE[i])`
- `s_aum[i] = percentile_rank_by_noisy_class(-AUM[i])`
- **IDO-lite:** `s[i] = s_we[i]`
- **Hybrid:** `s[i] = 0.5*s_we[i] + 0.5*s_aum[i]`

Higher WE and lower AUM mean greater suspicion. These are **relative rankings, not calibrated probabilities of label corruption**. Average within-class suspicion is approximately 0.5 at both 20% and 40% corruption (exactly 0.5 with this average-rank formula on complete groups), including for hybrid. Do not interpret weights as estimates of the true clean-label fraction. Mean classification weight below is consequently 0.55 for a complete class; minibatch means can differ.

Define heuristic classification/clean-label weight `c_i = 0.1 + 0.9*(1-s_i)` and difficulty `d_i = 4*s_i*(1-s_i)`. Difficulty is **parabolic**, peaking at intermediate suspicion. The 0.1 classification floor is an experimental heuristic intended to prevent suspicious examples from receiving negligible supervised influence; it does not guarantee nonzero gradients for already-confident predictions. Neither heuristic is part of original IDO or its BMM weighting.

At each Stage 2 epoch, freeze the scores computed from all prior epochs (do not change them batch-by-batch); run the epoch; run the deterministic statistics pass; update WE/AUM; recompute `s` for the next epoch. The first Stage 2 scores come from the common Stage 1 arrays. Do not accidentally use current-epoch statistics before the epoch has finished.

## F. Stage 2 adaptive loss (deliberately simplified)

Given two independently augmented views, compute logits_weak and logits_strong. Use float32 softmax/loss calculations where needed for AMP stability. Per example:

- `c_i = 0.1 + 0.9*(1-s_i)`; `d_i = 4*s_i*(1-s_i)`.
- `L_clean_i = CE(logits_weak_i, provided_noisy_label_i)` with unreduced per-example CE; the name refers to weighting, labels remain noisy.
- `p_weak = softmax(logits_weak)`; `p_strong = softmax(logits_strong)`.
- `pseudo_label_i = argmax(p_weak_i.detach())`; `conf_i = max(p_weak_i.detach())`; `mask_i = float(conf_i >= 0.7)`.
- `L_pseudo_i = mask_i * CE(logits_strong_i, pseudo_label_i)` with unreduced per-example CE.
- `L_consistency_i = mean_over_classes((p_weak_i - p_strong_i)**2)`; gradients flow through both views for consistency.
- `L_i = c_i*L_clean_i + 0.5*s_i*L_pseudo_i + 0.5*d_i*L_consistency_i`.
- `L_batch = mean_over_all_examples_in_batch(L_i)`.

The mask zeros only the pseudo-label contribution for unconfident examples. Never divide by the number of confident examples: all components use the same full-batch mean in both adaptive methods. Epoch summaries are sample-weighted means, including partial final batches. Standard uses plain mean CE. Both adaptive methods share one loss implementation, architecture, optimizer, two-view transforms, score refresh and coefficients; only suspicion arrays differ. This is an **IDO-inspired simplification**, not Eq. (8) or the original algorithm.

**Required adaptive-epoch logs:** sample-weighted means of weighted supervised classification loss c*L_clean, weighted pseudo-label loss 0.5*s*L_pseudo, weighted consistency loss 0.5*d*L_consistency, total adaptive loss, mean c, mean d, and fraction with conf>=0.7. For each weighted component and total, record fractions of examples with exactly zero contribution and contribution <=near_zero_epsilon (default 1e-8, including zeros). Inactive masked pseudo terms are expected. These measure loss values, not gradient norms. Assert component means sum to total.

Coefficients do **not** imply equal gradient influence. With ten classes, class-averaged probability MSE is bounded by 0.2 and its weighted contribution by 0.1 per example. Interpret component logs accordingly.

**Common diagnostic:** after every epoch of every method, including shared warm-up, measure unweighted noisy-label CE on the full training subset with deterministic transforms, model.eval() and no gradients. Reuse warm-up/adaptive statistics passes; standard continuation uses a diagnostic-only pass without WE/AUM updates. Use this sample-mean CE for cross-method learning curves. Raw Standard CE and adaptive weighted objectives are different quantities; plot adaptive components separately only if useful.

**Risk note and pseudo-label diagnostics:** six warm-up epochs on 20,000 images can still yield few predictions above 0.7, so the pseudo-label term may contribute little. Keep logging coverage, validate finite losses/gradients, and ensure the supervised floor remains active when every mask is zero. Keep threshold 0.7 for the planned experiment. Never adapt it automatically or use test accuracy, original clean labels or corruption masks to justify changes. Any exceptional amendment must use training-only diagnostics, be documented before final runs, apply identically to both adaptive methods and both noise levels, and require consistent reruns if necessary; do not add a tuning sweep. Report negative results honestly.

## G. Repository structure

```text
ido-aum-lite/
├── README.md
├── requirements.txt       # common non-torch runtime dependencies; preserve Colab torch pair
├── requirements-dev.txt   # lightweight development/test extras
├── configs/
│   └── default.json       # experiment parameters, no machine-specific absolute paths
├── .gitignore
├── config.py
├── src/
│   ├── __init__.py
│   ├── checkpoints.py   # complete state, checksums, persistent save/restore
│   ├── data.py          # download, stratified indices, noise, stable IDs, transformations
│   ├── model.py         # CIFAR-adapted ResNet-18
│   ├── statistics.py    # epoch-end WE and AUM tracking
│   ├── scores.py        # within-class ranks; WE and hybrid suspicion
│   ├── losses.py        # CE and IDO-lite / hybrid adaptive loss
│   ├── train.py         # Stage 1 + three Stage 2 branches; checkpointing/logging
│   ├── evaluate.py      # sole owner: test/detection metrics, aggregation/metrics.csv
│   └── plots.py         # results tables and saved PNG plots
├── notebooks/
│   └── run_colab.ipynb  # thin install/run notebook, no duplicate research logic
├── tests/
│   ├── test_noise.py
│   ├── test_statistics.py
│   ├── test_scores.py
│   ├── test_losses.py
│   └── test_resume.py     # full-state branch reset and epoch-boundary resume
├── results/              # small .csv/.json and images can be tracked in git
│   └── figures/
└── report/
    └── report.md        # content can later be transferred to LaTeX or PDF
```

Ignore in `.gitignore`: `data/`, `.venv/`, `__pycache__/`, `*.pt`, `*.pth`, temporary checkpoints, tokens, notebooks' large outputs. Save **final result metrics, plot PNGs, configuration and README to GitHub**. Persist complete checkpoints and required histories to Google Drive after each epoch, never GitHub; preservation is required for Colab recovery.

### Windows development and Colab execution contract

Use reusable Python modules as the sole implementation of data preparation, statistics, scoring, losses, training, evaluation and plotting. The notebook orchestrates CLI calls only; it must not contain model definitions, epoch loops or a second copy of research logic. config.py validates/loads experiment settings; configs/default.json holds portable defaults. Keep storage paths separate from scientific hyperparameters so relocating a run on restart does not change its scientific configuration identity.

Use pathlib.Path throughout, create output directories explicitly, and pass data/output locations through CLI arguments. No drive-letter paths, Windows separators, PowerShell commands, user-specific directories or unconditional google.colab imports in src/. Imports of Colab helpers belong only in the notebook. Keep command entry points under if __name__ == '__main__' so Windows DataLoader worker spawning is safe. Windows tests default to num_workers=0; Colab may use the configured two workers. Do not install CUDA or require local CUDA availability for CPU tests.

| Location | Colab example | Lifetime/purpose |
|---|---|---|
| Source checkout | /content/ido-aum-lite | Ephemeral; clone the recorded Git commit each session |
| Dataset cache | /content/data | Ephemeral; download again after runtime replacement; do not stream images from Drive |
| Persistent experiment root | /content/drive/MyDrive/ido-aum-lite/experiments/<RUN_ID> | Reuse unchanged on every restart |
| Frozen experiment configuration | <experiment-root>/experiment_config.json | Persistent; create/freeze once after benchmark, never overwrite on resume |
| Benchmark artifacts | <experiment-root>/benchmark | Persistent, isolated from scientific results |
| Checkpoints, histories and logs | <experiment-root>/runs | Persistent; separate noise/stage/branch directories |
| Final metrics/plots | <experiment-root>/results | Persistent; evaluation/plotting outputs |

RUN_ID identifies one frozen experiment configuration, not each Colab session. Store Git commit SHA, configuration fingerprint, environment versions, noise metadata/checksums and run manifest beside the persistent configuration. Retain the shared warm-up state, per-epoch warm-up score snapshots, latest complete branch-resume state and final checkpoints. Use epoch-granularity writes rather than per-batch Drive writes. Write checkpoints to a temporary file, flush/close, verify size/hash, and publish the complete checkpoint/manifest only after the Drive write succeeds. Keep the previous verified checkpoint until publication completes; on restart reject partial/corrupt saves and fall back to that previous state. Do not mark an epoch or branch complete before durable artifacts exist. Do not assume a Drive write is guaranteed atomic; test interrupted publication. Persistent storage is mandatory, not an optional end-of-session copy. Only one Colab session may write a given RUN_ID at a time.

### Lightweight Colab notebook specification

Create notebooks/run_colab.ipynb with empty outputs and a small number of setup/orchestration cells:

1. Explain the Windows/Colab split and select Runtime > Change runtime type > GPU. Mount Google Drive using google.colab.drive.mount('/content/drive'); request normal interactive authorisation rather than storing credentials.
2. Define REPO_URL, an explicit Git commit/ref, RUN_ID, REPO_DIR=/content/ido-aum-lite, DATA_ROOT=/content/data and persistent experiment root using pathlib. Clone the repository or access an existing clean checkout; verify its Git SHA. On restart use the SHA recorded in the persistent manifest. Never silently pull a new version into an interrupted experiment; use a new RUN_ID for intentional methodology/code changes.
3. Install common requirements using sys.executable -m pip with subprocess.run([...], check=True, cwd=REPO_DIR). Preserve and import-check Colab's installed compatible torch/torchvision pair rather than blindly replacing it. Print Python, torch, torchvision, CUDA and GPU versions. Fail clearly if torch.cuda.is_available() is false; never silently launch a full CPU run.
4. Load/validate configs/default.json for a new benchmark, or the persistent frozen experiment_config.json for a restart. Show subset size, noise levels, W/C and all storage paths. Verify persistent storage is writable and enough space is available. Keep the frozen configuration immutable after full runs start.
5. For a NEW experiment only, call the benchmark CLI, choose any uniform fallback by runtime only, and publish the frozen configuration and code/environment manifest to the persistent root before training. For a restart skip the benchmark/fallback selection and reuse those artifacts.
6. Run src.train for noise 0.2 and 0.4 via checked subprocess argument lists, passing --device cuda, --data-root, --output-root, --config and --resume. --resume must safely start missing stages and skip completed stages; the same commands are used on first execution and restart. Keep notebook controls for selecting either noise condition if GPU time requires separate sessions.
7. Run src.evaluate on GPU from persistent checkpoints, then src.plots, writing to the persistent results directory. Display the saved table/figures and print exact persistent paths. The notebook itself must not calculate metrics. Evaluation caching prevents duplicate evaluation of completed identical checkpoints.
8. Include the restart checklist in Section J and explain that runtime-local files are disposable. Restarting setup cells must not delete persistent artifacts or regenerate a frozen configuration.

Use checked subprocess argument lists and explicit cwd instead of relying on notebook !cd, shell variable expansion or quoting paths with spaces. Put data, checkpoints, run artifacts and notebook outputs in .gitignore; deliberately export only small reviewed metrics, figures and the frozen configuration to GitHub. No tokens, datasets or model checkpoints in Git.

## H. Step-by-step Codex prompts and checkpoints

Send **one prompt at a time**, inspect its tests, and commit when passing. Use Python package/module invocation to avoid path issues. Make code runnable from project root.

### Step 0 — Repo, configuration and Colab GPU entry point (0.75 h)
**Codex prompt:**
> Create a minimal PyTorch research repo named `ido-aum-lite` matching the directory structure in this plan. Add `config.py` with 20,000 CIFAR-10 train images, 10,000 original test images, seed 42, noise ratios 0.2/0.4, ResNet-18 CIFAR stem, batch 128, 6 Stage 1 epochs, 6 Stage 2 continuation epochs, AdamW lr 3e-4, weight decay 1e-4. Add common requirements for numpy, scipy (if needed), pandas, scikit-learn and matplotlib, with pytest in development extras. Document a compatible CPU-only torch/torchvision install for Windows tests separately; preserve Colab's installed torch pair and verify imports before using it. Add `.gitignore`, simple README and `notebooks/run_colab.ipynb` that installs requirements and runs smoke/full commands. Do not commit data or checkpoints. Centralise train_size, warmup_epochs, continuation_epochs, threshold 0.7, classification floor 0.1 and near_zero_epsilon 1e-8. Derive all counts/display values from configuration; support loading a saved experiment configuration with --config. Create configs/default.json and the thin notebook exactly as specified in Section G: checked module commands, Drive persistence, pinned checkout, CUDA guard and restart flow. Keep Colab imports out of reusable modules and use pathlib paths. No training logic yet.

**Check:** `python -m pytest` can be invoked, the notebook is valid and code cells parse; local unit tests import on Windows without CUDA or google.colab. In Colab, setup mounts Drive, installs dependencies, verifies the Git SHA and reports CUDA/GPU; full-run cells refuse CPU fallback. Test paths containing spaces and requirements without replacing a platform-provided compatible torch/torchvision pair.

**Commit:** `chore: project scaffold and GPU notebook`.

### Step 1 — Dataset and exact synthetic noise (1.75 h)
**Codex prompt:**
> Implement `src/data.py`: torchvision CIFAR-10 automatic download; deterministic class-stratified choice of train_size/10 training examples per class (2,000 default), maintaining original dataset IDs and 0..train_size-1 subset IDs; deterministic 20% and 40% symmetric noise where exactly round(p*train_size) training labels change (4,000/8,000 by default; 2,000/4,000 at 10,000 images) to a uniformly sampled different class; keep clean labels and corruption mask for EVALUATION ONLY; return required weak/strong/eval transforms and sample IDs; untouched official test split. Implement tests for fixed seeds, no duplicate image IDs, exact noise counts, altered labels != clean labels, no test corruption, stable indexing, and 10-class output. Parameterise tests over both noise conditions and default/fallback subset sizes; identical seeds/inputs must reproduce subset and corruption checksums. Standard/warm-up construct no strong view. Ensure true labels/masks cannot leak into training or hyperparameter selection.

**Check:** run `python -m pytest tests/test_noise.py -q`, print dataset lengths, per-class counts, noise corruption totals and sample shapes. Confirm no `data/` tracked.

**Commit:** `feat: reproducible cifar10 and synthetic noise`.

### Step 2 — Model, WE and AUM tracking (2 h)
**Codex prompt:**
> Implement CIFAR-adapted non-pretrained ResNet-18 in `src/model.py`, output shape `[batch, 10]`. Implement `src/statistics.py` for stable-ID per-example end-of-epoch tracking on a deterministic, non-shuffled inference loader: increment WE once per epoch when argmax != noisy label; accumulate raw-logit margin (assigned-label logit minus largest other-class logit) and return AUM as mean over tracked epochs. Keep arrays and per-example observation counts keyed by subset positions, not loader order. Save WE, cumulative margin sums, counts and AUM after each default warm-up epoch (six); preserve historical snapshots on checkpoint reload. Collect the common deterministic unweighted noisy-label CE in the same pass, with a CE-only pass for standard continuation. Restore train mode before the next training epoch. Implement toy-logit tests that hand-calculate margins and WE and test shuffled input IDs, exactly-once coverage, observation counts and history persistence. Provide a 512-image CPU/GPU smoke test (1 epoch, minimal mini-batches). Do not compute WE only on augmented minibatches during gradient training.

**Check:** `python -m pytest tests/test_statistics.py -q`; verify finite AUM, WE in `[0, tracked_epochs]`, correct output shapes.

**Commit:** `feat: resnet18 and sample-level WE AUM tracking`.

### Step 3 — WE-only and hybrid scoring, adaptive objective (2.5 h)
**Codex prompt:**
> Implement within-provided-label-class scoring in src/scores.py using one-based average tie ranks and s=(r-0.5)/n; singleton/all-tied classes return 0.5 and empty classes are skipped. Rank WE and negative AUM; hybrid is their 50:50 average. Scores are relative rankings, not corruption probabilities; their complete-class mean is 0.5 for both noise settings. Define c=0.1+0.9*(1-s), d=4*s*(1-s), a parabolic heuristic. Implement one shared src/losses.py objective: c*CE(weak,noisy_y) + 0.5*s*mask*CE(strong,detached_weak_pseudo) + 0.5*d*mean_over_classes((weak_softmax-strong_softmax)**2), threshold 0.7, unreduced per-example CE, then full-batch mean; never confident-only normalisation. Detach pseudo targets but retain both consistency gradient paths. Implement all Section F component/total logs, confidence/zero/near-zero fractions, mean c/d and sample-weighted epoch reductions. Test ties, ordering, singleton/empty groups, bounds and mean invariance under both noise conditions, endpoint weights c(0)=1 and c(1)=0.1, with d(0)=d(1)=0 and d(0.5)=1, all-zero confidence masks, reductions, detached targets and finite gradients. Both adaptive methods must use identical loss code except score inputs. Label the classification floor and objective as experimental IDO-inspired simplifications.

**Check:** `python -m pytest tests/test_scores.py tests/test_losses.py -q`. No clean-label use in scoring/loss/hyperparameter selection; component means sum to total and unconfident suspicious examples retain supervised weight.

**Commit:** `feat: IDO-lite and hybrid AUM scoring and loss`.

### Step 4 — Two-stage experiment runner (part of Step 3/4 budget, then GPU budget)
**Codex prompt:**
> Implement src/train.py with --config, --noise 0.2|0.4, --smoke, --benchmark, --device, --data-root, --output-root and --resume. Require CUDA for full training/benchmarks, allowing CPU only for explicit smoke runs. Use pathlib and the Section G persistence contract through src/checkpoints.py; notebook orchestration must contain no training logic. Load central defaults (6 warm-up + 6 continuation epochs, train_size=20000) or the single saved fallback configuration. Per noise condition train one shared CE warm-up, save every warm-up WE/AUM snapshot and one final shared checkpoint. Restore the complete state_dict, including BatchNorm parameters and running buffers, into standard, ido_lite and hybrid; deep-copy WE, cumulative margins and counts for adaptive branches. For each NEW branch reset Python/NumPy/PyTorch CPU/CUDA RNGs and controlled DataLoader generators to the same recorded branch-start seeds for all three new continuations, seed workers, and create a fresh identical optimizer and AMP scaler. Do not reuse warm-up optimizer momentum. All continuations have equal configured epoch counts; update adaptive statistics after each epoch, scores only for the next epoch. Standard has no continuation WE/AUM updates. Save common deterministic CE for all methods, reusing statistics passes, and all Section F adaptive logs. Save timings, GPU/environment, configuration, seed streams, subset/corrupted-label checksums and full-model-state hashes. Training must not access test data, compute corruption-mask metrics or write results/metrics.csv.
>
> Support epoch-boundary resume only. After every completed epoch atomically save: model state_dict; optimizer state_dict; AMP GradScaler state if enabled; stage/branch and completed epoch; WE tracking arrays; AUM cumulative margin sums; AUM observation counts; Python RNG state; NumPy RNG state; PyTorch CPU RNG state; CUDA RNG states when applicable; all DataLoader generator states; experiment configuration; dataset/subset checksum; corrupted-label checksum. For interrupted-branch resume, validate configuration/checksums FIRST, then restore the interrupted model, epoch, WE/margin/count arrays, optimizer/scaler and all RNG/generator states after object construction/loading and before fetching the next batch. Continue the next epoch without recomputing or double-counting completed statistics. New-branch initialisation resets optimizer/RNGs; resume restores them. Recreate deterministically seeded workers each epoch with persistent_workers=False, avoiding worker-local checkpoint state. Make --resume idempotent: start absent stages, continue interrupted stages and skip completed branches using durable manifests. Reject code/configuration/checksum mismatches and never overwrite a frozen run with a fresh start. Preserve all historical warm-up snapshots, which weights alone cannot reconstruct. Identical inputs/seeds should reproduce subsets, corruptions, initial weights and initial statistics in a supported fixed environment; disclose remaining cross-device/kernel nondeterminism.
>
> Include an isolated 512-image smoke configuration and --benchmark mode: perform a complete adaptive epoch at candidate configured subset size, including two-view forward/backward, data loading, statistics/common CE and checkpointing. Also time representative warm-up/standard epochs with inference diagnostics. Benchmark paths must be separate from full runs and excluded from metrics. Use no clean labels, corruption masks or test quality to choose fallback. Freeze one configuration for both noise conditions before either full run.

**Check:** smoke one noise condition. Add tests/test_resume.py verifying equal complete-state hashes including BatchNorm buffers, separate statistic-array storage, reproducible subset/corruption/initial weights/statistics, RNG/generator restoration, interrupted versus uninterrupted epoch-boundary equivalence on a small deterministic fixture, rejected configuration/checksum mismatches and preserved warm-up snapshots. Assert finite logs, no duplicated statistics/epochs, no training test access or metrics.csv output. Test persistence/publication failure recovery and full-run CUDA refusal using small fixtures/mocks without a local GPU. On Colab verify restart finds Drive state even when /content source/data caches were recreated. Run the Section I benchmark and persist the chosen configuration before full experiments.

**Commit:** `feat: reproducible staged training and resume`.

### Step 5 — Actual GPU runs and evaluation (4.5 h budget including debugging)
**Codex prompt:**
> Implement src.evaluate with --device cuda and --data-root options, using pathlib paths and persistent run roots. Full-model evaluation is Colab GPU-only; toy/unit tests remain CPU-compatible. It is the sole owner of final test accuracy, noise-detection ROC-AUC, result aggregation and results/metrics.csv. Evaluate the untouched official test split once per completed final method; cache by full checkpoint/configuration identity so repeated aggregation does not reevaluate unchanged models. Load saved historical WE, cumulative margins, counts and AUM snapshots, not just warm-up weights. Compute primary ROC-AUC at the configured last warm-up epoch for WE and -AUM against the analysis-only corruption mask; where practical also save retrospective per-warm-up-epoch AUC and optional fused-score AUC. Never use evaluation results for training, fallback, epoch or hyperparameter selection. Diagnose absent positive/negative labels or missing artifacts rather than supplying 0.5. Save detector arrays for ROC/distribution plots. Aggregate configuration, training durations/log summaries, noise counts, accuracy and detection metrics into JSON/CSV; upsert by configuration/noise/method and verify six final rows once both noise conditions finish. Report exploratory single-seed observations honestly, including negative outcomes; no faithful-reproduction claim.

**Benchmark and freeze configuration first (within the 4.5 h run/debugging budget):**

```bash
export EXPERIMENT_ROOT="/content/drive/MyDrive/ido-aum-lite/experiments/run-001"
python -m src.train --benchmark --noise 0.2 --device cuda --data-root /content/data --output-root "$EXPERIMENT_ROOT/benchmark"
```
Benchmark uses an isolated short smoke/warm-up state, no test or corruption-mask evaluation, prints the Section I projection and saves candidate central defaults to <experiment-root>/benchmark/experiment_config.json. Bash examples run from the repository checkout in one shell; notebook cells pass the same paths as Python subprocess arguments, avoiding shell state between cells. If fallback is necessary, change central configuration fields only, rerun the benchmark at the candidate size/schedule, then publish the chosen JSON once as <experiment-root>/experiment_config.json and freeze it for both full runs. Never overwrite the frozen file on restart; check it against the recorded manifest. Exclude benchmark artifacts from aggregation.

**Run after configuration is frozen:**

```bash
export EXPERIMENT_ROOT="/content/drive/MyDrive/ido-aum-lite/experiments/run-001"
python -m src.train --config "$EXPERIMENT_ROOT/experiment_config.json" --noise 0.2 --device cuda --data-root /content/data --output-root "$EXPERIMENT_ROOT/runs" --resume
python -m src.train --config "$EXPERIMENT_ROOT/experiment_config.json" --noise 0.4 --device cuda --data-root /content/data --output-root "$EXPERIMENT_ROOT/runs" --resume
python -m src.evaluate --device cuda --data-root /content/data --runs "$EXPERIMENT_ROOT/runs" --output-root "$EXPERIMENT_ROOT/results"
python -m src.plots --results "$EXPERIMENT_ROOT/results" --runs "$EXPERIMENT_ROOT/runs"
```

**Check:** six final rows from 2 noise levels x 3 methods, no duplicates on repeat aggregation; at least four primary detection AUC entries; warm-up history loaded; test labels unchanged. Verify benchmark decision predates both full runs, saved configuration/checksums match, fallback counts are derived, and evaluation overhead fits the projection. Only evaluation owns metrics.csv. Use subprocess.run([sys.executable, "-m", ...], cwd=REPO_DIR, check=True) in Colab with the same explicit paths. Confirm fresh-runtime recovery finds Drive state and cached evaluation.

**Commit:** `exp: log two noise-level comparisons` (results JSON/CSV only; do not commit bulky checkpoint files).

### Step 6 — Results figures and tables (1 h)
**Codex prompt:**
> Implement `src/plots.py` to read CSV and saved per-sample score arrays. Generate and save publication-readable PNGs: (1) grouped bars showing test accuracy for three methods at 20% and 40%; (2) WE vs -AUM ROC curves for both noise levels; (3) distributions of WE or AUM separated by actual clean/corrupted labels for a representative noise level; (4) per-epoch common deterministic unweighted noisy-label CE curves for all methods, with warm-up boundary read from saved configuration (6 default). Do not compare raw standard CE with adaptive weighted objectives as the same loss; plot adaptive components separately only if time permits. Annotate figures with data source, noise ratio and single-seed caveat and actual configured subset/epoch counts, including fallback labels and boundaries from saved metadata. Generate one compact results table CSV and a Markdown table for report. No invented numbers; data must come only from saved run outputs.

**Check:** `python -m src.plots --results "$EXPERIMENT_ROOT/results" --runs "$EXPERIMENT_ROOT/runs"` (Colab shell with EXPERIMENT_ROOT set; notebook uses Python path variables); inspect axis labels, legends and figure files; make sure results are consistent with logged metrics.

**Commit:** `docs: comparative result plots and tables`.

### Step 7 — README and written report (2 h)
**Codex prompt:**
> Create an honest report in `report/report.md` (approximately 5–7 well-structured pages when formatted) and final README based solely on saved experiment outputs. Sections: abstract; problem/motivation; original IDO and AUM background and citations; clear IDO-lite simplifications; dataset selection and precise noise injection; scoring formulas; training settings and controls (full-state branch resets, epoch-boundary resume, benchmark/fallback); exact percentile formula and 0.1 classification floor; component/confidence/near-zero-loss diagnostics; measured test accuracy and noise-detection ROC-AUC; four plots and discussion; unexpected outcomes and failure modes; threats to validity (actual configured subset/schedule, single-seed exploratory results, coarse WE with warmup_epochs+1 possible counts, rank-imposed mean suspicion, possible low pseudo-label coverage, no original BMM, no AUM threshold procedure, synthetic symmetric noise only); distinguish overall Standard-versus-adaptive comparisons from controlled WE-only-versus-hybrid comparison; development changes made with Codex; future work; references and run/replication instructions. Do not fabricate improvements, accuracy figures, GPU usage or methodology claims. Final README must explain Windows VS Code/Codex development and CPU-only tests, and include exact Colab benchmark/config/full-run/evaluation/resume commands, Drive paths, pinned-code/environment recovery and the complete Section J restart checklist. All full-model training/evaluation must run on Colab GPU; no local GPU is needed. Generate tables/completion counts from saved configuration, disclose fallback or exceptional threshold amendments, and do not claim statistically reliable superiority or faithful reproduction.

**Check:** all citations valid, all table values match CSV, plot files exist, reproducibility instructions match implemented CLI, the report explicitly says 'IDO-inspired simplified implementation'; subset sizes, noise totals, epoch counts and plot boundaries match saved configuration, and evaluation has one clear owner.

**Commit:** `docs: final report and reproducibility guide`.

### Buffer (0.5 h)
Use only for unexpected platform or runtime issues; omit optional features rather than overrunning the 15-hour cap.

## I. Total 15-hour work budget

| Work | Hours |
|---|---:|
| Scaffold repo, Windows CPU tests and Colab/Drive setup | 0.75 |
| Dataset, precise noise generator and tests | 1.75 |
| ResNet-18 and WE/AUM collection | 2.00 |
| Fusion, adaptive loss and staged-runner coding | 2.50 |
| GPU smoke/benchmark, six branches, evaluation, checkpointing and debugging | 4.50 |
| Plotting and main results table | 1.00 |
| README and written report | 2.00 |
| Contingency | 0.50 |
| **Total** | **15.00** |

**Runtime accounting (default):** Let W=warmup_epochs and C=continuation_epochs. Across two noise conditions, executed training epochs are `2*(W+3*C)`. Training image-view passes are `2*(W+C+2*C+2*C)=2*(W+5*C)`: standard/warm-up use one view, each adaptive branch uses two. Statistics passes are `2*(W+2*C)`, with no standard-continuation statistics updates. At W=C=6 this gives **48 executed training epochs, approximately 72 training image-view passes and 36 full-subset statistics passes**. Common deterministic CE adds `2*C=12` standard-continuation diagnostic-only passes, hence **48 full-subset inference passes total** for statistics/diagnostics, plus six final test passes. Warm-up history ROC-AUC uses saved scores, adding no model inference. Image-view passes are accounting units, not equal wall-clock durations; adaptive backward/loading costs must be measured.

**Explicit benchmark before either full experiment:** First run an isolated smoke, then time at least one complete representative adaptive epoch on the candidate subset, including both augmented views, forward/backward, loading, statistics/common CE and checkpoint saving. Also time representative warm-up and standard epochs including their inference diagnostics/checkpointing; synchronise CUDA around timing. Let t_w, t_s and t_a be these measured epoch times. Estimate main training runtime as `2*(W*t_w + C*t_s + 2*C*t_a)`. Add six test evaluations, detection aggregation, download/setup, benchmark itself, persistent Drive checkpoint/sync overhead not already timed and debugging margin. All belong within the 4.5-hour run/evaluation/debugging allocation, not in addition to it. Other tasks plus that allocation and the 0.5-hour contingency total **15 hours**; do not spend contingency twice. Prefer correct essential results over optional metrics/plots. One timed epoch is an estimate; record uncertainty and allow a margin.

**Fallback decision before full runs:** Freeze one central configuration for BOTH noise conditions using runtime and time remaining only, never test accuracy or clean-label detection quality. Preferred order: (1) reduce train_size from 20,000 to 10,000 (1,000/class, exactly 2,000/4,000 corrupt labels), keeping W=C=6; (2) if necessary reduce C uniformly from 6 to 4 for every method and both noise levels; (3) only as a final fallback reduce W from 6 to 4, documenting that WE resolution falls from seven to five possible counts. Rebenchmark candidate configurations within the budget. At W=6,C=4 the counts are 36 training epochs, 52 training image-view passes, 28 statistics passes plus 8 diagnostic-only passes; at W=4,C=4 they are 32, 48, 24 plus 8 respectively. These reduced-C fallbacks give W+C=10 or 8 epochs per final model, so report actual totals rather than claiming 12. Subset reduction changes per-pass cost, not pass counts.

All fallback fields live in central configuration; the frozen JSON drives both full-run commands. Automatically propagate subset/per-class sizes, expected corruption counts, array lengths, W/C, per-model/executed totals, assertions, metadata, plot labels/boundaries, final report tables and completion checklist. Never shorten a single method or use different schedules between noise levels. If later runtime changes force a new configuration, invalidate incompatible outputs and rerun both conditions consistently; do not mix configurations in the six-row table. Colab Free GPU availability is not guaranteed, so feasibility within 15 hours remains conditional on the benchmark and access, not a runtime promise.

## J. User's manual actions

1. Develop in VS Code with Codex on Windows. Create the GitHub repository, commit source/configuration/notebook changes and push reviewed commits. Use a local virtual environment, CPU-compatible torch/torchvision and development requirements for lightweight tests/debugging only. Run python -m pytest; optional tiny smoke tests explicitly use --smoke --device cpu and num_workers=0. Unit tests use synthetic inputs/fixtures, not full data downloads. No local GPU is required; do not run full training/evaluation on the laptop.
2. Open notebooks/run_colab.ipynb in Google Colab, select GPU and run setup. Mount Drive, choose RUN_ID and the repository ref, clone into /content, install dependencies and verify CUDA. Use /content/data for disposable CIFAR-10 cache; no Windows paths in Colab.
3. For a new RUN_ID, benchmark a full adaptive epoch including persistent checkpoint I/O, choose a uniform fallback if needed and save frozen configuration plus Git SHA/environment manifest to Drive. Persist checkpoints, score histories and logs after each epoch, not only at session end.
4. Execute noise 20% then 40% with the same frozen configuration, --device cuda and --resume through src.train. Separate Colab sessions may finish different conditions. Inspect training diagnostics/files without choosing changes from test/detection quality. Run src.evaluate and src.plots on Colab afterwards with persistent output paths.
5. Review six final rows and four essential figures; commit only small reviewed metrics, figures, configuration and report. Keep checkpoints on Drive and data/tokens/checkpoints out of Git.

**Restarting an interrupted Colab session:**

1. Reopen the same notebook, select GPU and mount the same Drive account. Ensure no other session is writing this RUN_ID.
2. Reuse the exact RUN_ID and experiment-root path. Read saved Git SHA, configuration and environment manifest; check out that code rather than the latest branch head. Reinstall dependencies and verify GPU/imports/versions. Document hardware changes; bitwise equality across GPU/kernel versions is not guaranteed.
3. Recreate /content/ido-aum-lite and redownload data to /content/data as needed. These caches are disposable; never delete persistent outputs, regenerate different corrupted labels, rerun fallback selection or overwrite frozen configuration during recovery.
4. Rerun configuration/status setup. Validate scientific configuration, code identity, subset/corrupted-label checksums and latest completed checkpoint/manifest. Ignore partial temporary files; fall back to the previous verified checkpoint if publication was interrupted. The runner restores model/optimizer/scaler/statistics/RNG/DataLoader state, not notebook code.
5. Rerun identical --resume commands from Step 5. Missing stages start, interrupted stages continue from the last durably completed epoch and finished branches are skipped. An unfinished epoch may repeat; completed statistics/logs must not be duplicated. New branches reset optimizer/RNGs after loading shared warm-up state; interrupted branches restore their own complete state.
6. Rerun evaluation/plotting after training. Reuse cached metrics for unchanged final checkpoints and compute only missing evaluations, retaining one final row per configuration/noise/method. Confirm all outputs exist on Drive before ending the runtime.

The notebook orchestrates; reusable modules own all substantive training, integrity, resume and evaluation behavior. Google documents temporary Colab virtual machines and recommends avoiding many small Drive operations; use local data caches and epoch-granularity persistent artifacts. Source: [Colab FAQ](https://research.google.com/colaboratory/faq.html).

## K. Acceptance checklist

- [ ] Runnable GitHub repository with reusable Python modules, thin Colab notebook, portable configuration and runtime/development requirements.
- [ ] Windows CPU-only tests/debugging require neither local CUDA nor google.colab; full training/benchmark/evaluation run on Colab GPU and refuse CPU fallback.
- [ ] pathlib/CLI paths are Linux-compatible; notebook handles spaces and duplicates no research logic.
- [ ] Drive stores frozen configuration, code/environment manifest, checkpoints, histories, logs and results; fresh-runtime resume recovers the previous verified epoch.
- [ ] Configured deterministic stratified training IDs (20,000 default; 10,000 fallback); immutable official 10,000-image test set.
- [ ] Exactly round(0.2*train_size) / round(0.4*train_size) changed labels: 4,000/8,000 default or 2,000/4,000 fallback.
- [ ] No original clean labels or corruption mask in train-time score/loss/selection functions.
- [ ] ID-indexed WE and logit AUM tracked once per completed epoch on deterministic transforms.
- [ ] Exact (average_rank-0.5)/class_size scores; high WE/low AUM mean suspicious; ties/singletons/bounds and both noise conditions tested. Rank means are not corruption probabilities.
- [ ] c=0.1+0.9*(1-s), parabolic d=4*s*(1-s), threshold 0.7 and full-batch/class-mean reductions match Section F; all required logs recorded.
- [ ] Three methods per noise condition start from identical complete shared state including BatchNorm buffers; new-branch RNG/DataLoader resets verified.
- [ ] Six warm-up plus six continuation epochs by default; saved configuration drives actual W+C and computational counts.
- [ ] Every warm-up WE/margin-sum/count/AUM snapshot preserved; complete epoch-boundary resume state and checksum/configuration validation tested.
- [ ] Adaptive methods use the same three-term loss and settings; only WE-only versus fused suspicion differs.
- [ ] Six measured final accuracy entries; at least four detection ROC-AUC entries (two scores x two noise settings).
- [ ] Four essential PNG plots from real outputs with configured labels/boundaries; one results table.
- [ ] Cross-method curves use common deterministic unweighted noisy-label CE; adaptive component losses labelled separately.
- [ ] Evaluation alone writes test/detection metrics and results/metrics.csv; repeat aggregation does not duplicate evaluation or rows.
- [ ] No test-set tuning or hyperparameter search.
- [ ] Report includes exact simplifications, scope, single-seed limitation and negative results if applicable.
- [ ] Complete adaptive-epoch benchmark precedes both full runs; one frozen configuration/fallback propagates to commands, assertions, metadata, plots, report and this configuration-driven checklist.
- [ ] Coding/training/evaluation/debugging/plots/report remain within 15 hours, including benchmark and overhead; fallback disclosed.

## L. High-priority methodological failure checks

- **Noise definition:** Many implementations randomly reassign among all 10 classes, so some selected labels stay unchanged; that does *not* yield exactly 20%/40% incorrect labels. Exclude the original class during replacement.
- **Counting WE:** Wrong Events count **epochs with wrong predictions**, not only transitions from correct to incorrect (those are different forgetting events). Count no more than once per sample per epoch.
- **Stable IDs:** Do not align persistent statistics with shuffled DataLoader order; use explicit sample index.
- **Margin sign:** AUM is `assigned-label logit - strongest alternative`. Larger *negative* AUM signals more suspicious; invert for noise ROC-AUC.
- **AUM temporal definition:** Average over all completed tracked epochs, not just the final-epoch margin.
- **No peeking:** Clean ground-truth labels and mask are held only for retrospective synthetic-noise detection evaluation; never compute train-time quality weights from them.
- **Fairness:** Same images and corrupted labels per noise condition, full common Stage 1 state, equal configured W+C epochs (12 default), architecture, optimizer settings and recorded seed resets. Adaptive transforms/loss are identical; only scoring differs. Standard uses one-view CE, so its comparison measures overall strategy, not scoring alone.
- **Reproducibility:** New branches reset optimizer/RNGs; resumed branches restore optimizer/scaler/RNG/generator states. Validate configuration/checksums, preserve historical arrays and avoid duplicated epoch statistics.
- **Weighting:** Percentile ranks impose mean suspicion 0.5; the 0.1 supervised floor and parabolic difficulty are experimental heuristics. Coefficients do not equalise gradient influence.
- **Colab recovery:** /content source/data caches are ephemeral. Persist state/results to Drive during execution; reuse RUN_ID, frozen configuration and pinned Git SHA after restart. Reject partial saves; never launch full local CPU runs.
- **Runtime/evaluation ownership:** Include both views, statistics, diagnostic-only inference and evaluation/checkpoint overhead. Freeze fallback before both full runs; evaluation alone owns final metrics and aggregation.
- **Scientific labelling:** Do not call WE-only score ranks an official IDO BMM; do not claim reproduction of AUM's threshold-calibration component.
- **Result integrity:** Use plots from logged arrays; no hardcoded bar heights or hypothetical metric values.
