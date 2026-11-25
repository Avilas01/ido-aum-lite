# Completed warm up handoff

The completed experiment is `run-001`, trained/evaluated at source commit
`695f50fef1915f0ab2d370ba6c8cda1282e9dae2`. Both warm-ups completed **six epochs**.
The same 20,000-image subset, seed 42 and frozen 6+6 schedule apply to both noise
conditions. Individual team contributions are not assigned by this handoff.

## What is already on GitHub

- [Frozen scientific configuration](../results/experiment_config.json).
- [Original measured results and detector arrays](../results/).
- [Completed notebook with saved outputs](../notebooks/colab_runner_completed.ipynb).
- [Notebook archive provenance](notebook_archive.json).
- [Warm-up inventory and checkpoint checksums](warmup_inventory.json).

The completed notebook is an output archive, not the clean reusable runner. It
preserves the original executed cells, flags, counts and outputs, with one added
Markdown notice. It has no captured pytest summary, so no hosted test count is
inferred. Nothing was rerun to create it. Use the clean runner and the experiment's
pinned code for actual recovery. Plotting/report changes made after execution do
not alter the archived experiment.

## Files to share through Drive

Share copies of these paths from:

`My Drive/ido-aum-lite/experiments/run-001/`

```text
experiment_config.json
manifest.json                         # original frozen root manifest
benchmark/benchmark_report.json       # required by original runner's freeze checks
runs/
  noise_20/
    run.json
    data/
      metadata.json
      training.npz
      analysis.npz                    # evaluation only; never feed to training
    warmup/                           # copy the entire completed folder
      latest.json
      latest.previous.json
      epochs.json
      history/epoch_001.npz ... epoch_006.npz
      timing_001.json ... timing_006.json
      epoch_*.pt
  noise_40/
    run.json
    data/
      metadata.json
      training.npz
      analysis.npz
    warmup/                           # same complete structure
```

The two folders to share are:

- `runs/noise_20/warmup/`
- `runs/noise_40/warmup/`

The original root `manifest.json` and benchmark report were not present in the
uploaded run ZIPs. Obtain those files from Drive if the teammate will use the
original frozen runner. They have not been reconstructed or fabricated here.
A warm-up folder plus scientific config is useful as an initialization artifact,
but is not by itself a complete frozen experiment root for `src.train --resume`.

**Shared Drive URL: not supplied.** Share the copied folders/config with the
teammate separately and send the resulting link. This repository does not grant
access to the Drive files. The large model/optimizer checkpoints remain outside
Git, as intended by the repository's `.pt`/`.pth` ignore rules.

## Verified final checkpoints

| Noise | Completed epoch | Final checkpoint SHA256 |
| --- | ---: | --- |
| 20% | 6 | `6cb42633eac3a7922e34a3ce37c69035e71e1f9d28f4b15cb7573ef68325ee4a` |
| 40% | 6 | `89c78c575078b985e8ada977ea913300bfd98dd0ac57fdc196e6d3f011e438b9` |

`latest.json` identifies the exact epoch-6 filename, payload size and checksum in
each folder. It also retains epoch 5 for rollback. Both epoch-6 and epoch-5
payloads were streamed directly from the downloaded ZIPs and SHA256-verified,
without loading torch or deserialising models locally. Older checkpoints are
listed by ZIP size/CRC in the inventory, not claimed as SHA256-verified.

The four downloaded `runs-20261125T181843Z-1-001.zip` through `-004.zip` files
contain different members of these folders. If using ZIP downloads instead of
Drive folder sharing, merge their selected `runs/noise_*/warmup/`, `data/` and
`run.json` paths while preserving their structure. Do not overwrite conflicting
files silently. `warmup_inventory.json` maps each member to its source ZIP.

## Using the handoff

For an unchanged continuation/recovery, use the pinned experiment code,
configuration, original manifests, matching runtime and data metadata in a
separate copy of the experiment root. The original runner validates identities
and checkpoint coverage before restoring state. Restore model/optimizer/scaler,
statistics, RNGs and loader generators for interrupted stages; reset the optimizer
and branch RNGs only for a genuinely new branch from the shared warm-up.

For a new method or changed code, create a distinct experiment identity and record
this warm-up's provenance. Do not edit the original `run-001` frozen manifests,
pretend modified code is an unchanged resume, or bypass identity validation. New
initialisation/import support is a separate implementation task; it is not
provided by simply copying files or by this handoff README.

Checkpoint loading and all model work belong in Colab with the project's verified
PyTorch environment. Checkpoints contain full trusted Python/PyTorch state.
Images are not included; use the official CIFAR archive and the matching saved
IDs/noisy labels. Clean labels and masks remain analysis-only.
