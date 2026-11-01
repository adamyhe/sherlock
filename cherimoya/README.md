# Building cherimoya apptainer

Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. To get around this, we build an apptainer image to sidestep this.

This image is intended to expose the Cherimoya Python API (the `Cherimoya` model class, `EMA`, output wrappers, `fit`/`predict`) as well as the `cherimoya` end-to-end CLI pipeline. It's built `FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` so that torch/torchvision come from the base image's own pinned build (torch 2.13.0, CUDA 12.6). The other dependencies (`tangermeme`, `modisco`, `figwig`/`bam2bw`, `lightning`, ...) are installed normally, and `bpnet-lite` and Cherimoya itself are installed with `--no-deps`, so a dependency-resolving `pip install` never gets a chance to replace that torch build. See the comments in each `.def` for details.

## Versions

There is one definition file per upstream Cherimoya release. Each one pins Cherimoya (`https://github.com/jmschrei/cherimoya`) to a git tag or commit, never to an unpinned branch:

| Definition | Cherimoya | Pinned to | Notes |
| --- | --- | --- | --- |
| `cherimoya-0.3.1.def` (**recommended**) | v0.3.1 | commit `f6eb5d2` (the `v0.3.1` tag) | Faster DeepLIFT/SHAP for log-counts (requires `tangermeme>=1.6.0`); the pipeline converts BAMs to bigWigs with `figwig bam2bw` instead of `bam2bw`; new `n_jobs` / `num_threads` CPU caps |
| `cherimoya-0.3.0.def` | v0.3.0 | the `v0.3.0` tag | PyTorch Lightning training (multi-GPU), held-out test evaluation, DeepLIFT/SHAP as the default attribution |
| `cherimoya-8e4283f.def` | v0.2.0 (post-rewrite) | commit `8e4283f` | The commit upstream force-moved the `v0.2.0` tag to (EMA weights, `valid_count_corr` checkpointing). The PyPI 0.2.0 release predates this rewrite |

The examples below use 0.3.1. For another version, substitute its `.def` and `.sif` names.

## Build

```bash
apptainer build cherimoya-0.3.1.sif cherimoya-0.3.1.def
```

If the build node requires fakeroot or a writable cache, use Sherlock scratch:

```bash
mkdir -p "$SCRATCH/apptainer-cache" "$SCRATCH/apptainer-tmp"
APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache" \
APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp" \
apptainer build --fakeroot cherimoya-0.3.1.sif cherimoya-0.3.1.def
```

The build test checks that the `cherimoya` CLI runs and that `cherimoya`, `bpnetlite.performance`, and `tangermeme` all import. The 0.3.x images also check that `lightning` imports, and 0.3.1 also checks `figwig bam2bw --help`.

## Use the Python API

Request a GPU node before running anything training- or inference-heavy (`salloc -c 2 --mem=16G --gpus=1 --partition=dev -t 2:00:00`), then use `--nv` so the container can see the GPU:

```bash
apptainer exec --nv cherimoya-0.3.1.sif \
  python -c "import cherimoya; print(cherimoya.__file__)"
```

Run one of your own Python scripts against the containerized environment:

```bash
apptainer exec --nv \
  --bind "$PWD:/work" \
  --bind "$SCRATCH:/scratch" \
  cherimoya-0.3.1.sif \
  python /work/my_cherimoya_script.py
```

A minimal script loading the model directly (Cherimoya models are saved as config + state_dict bundles, not pickled modules, so they load safely with `weights_only=True`):

```python
from cherimoya import Cherimoya

model = Cherimoya(...)
y_profile, y_counts = model(X)  # X: (N, 4, L) one-hot DNA
```

For notebooks or interactive exploration, `pytest`, `ipython`, and `jupyter` are installed in the image alongside the core dependencies:

```bash
apptainer shell --nv cherimoya-0.3.1.sif
ipython
```

## Use the CLI

The `cherimoya` CLI drives the full pipeline — peak calling, signal extraction, training, attribution, seqlet discovery, and motif annotation — either end to end or one stage at a time:

```bash
apptainer exec --nv cherimoya-0.3.1.sif cherimoya --help
```

Key subcommands (see `cherimoya <subcommand> --help` for each one's options):

- `pipeline-json` — generate a pipeline config from raw genomic data pointers (BAM/BED/FASTA)
- `pipeline` — run the full workflow from that config
- `negatives`, `fit`, `evaluate`, `attribute`, `marginalize`, `seqlets` — run an individual stage in isolation
- `install-skill` — install a Claude Code agent skill for guided Cherimoya analysis

Bind any data, genome, or scratch directories that aren't visible by default:

```bash
apptainer exec --nv \
  --bind /path/to/data:/data \
  --bind "$SCRATCH:/scratch" \
  cherimoya-0.3.1.sif \
  cherimoya pipeline --help
```

## Caches

Each `.def`'s `%environment` points `NUMBA_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `XDG_CACHE_HOME` at paths under `/tmp`, which is ephemeral per job on Sherlock. If you want compiled kernels/cache artifacts to persist across jobs, bind a scratch directory over one of these paths, e.g. `--bind "$SCRATCH/torchinductor-cache:/tmp/torchinductor_cache"`.
