# Building cherimoya apptainer

Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. To get around this, we build an apptainer image to sidestep this.

This image is intended to expose the Cherimoya Python API (the `Cherimoya` model class, `EMA`, output wrappers, `fit`/`predict`) as well as the `cherimoya` end-to-end CLI pipeline. It's built `FROM pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` so that torch/torchvision come from the base image's own pinned build (torch 2.13.0, CUDA 12.6); Cherimoya and its runtime dependencies (`bpnet-lite`, `tangermeme`, `modisco`, `bam2bw`) are then layered on top with `--no-deps`, per the comments in `cherimoya.def`, so a plain dependency-resolving `pip install` never gets a chance to replace that torch build.

The upstream Cherimoya repo is `https://github.com/jmschrei/cherimoya`, pinned to a specific commit rather than a tag or the PyPI release — see the comment in `cherimoya.def` for why.

## Build

```bash
apptainer build cherimoya.sif cherimoya.def
```

If the build node requires fakeroot or a writable cache, use Sherlock scratch:

```bash
mkdir -p "$SCRATCH/apptainer-cache" "$SCRATCH/apptainer-tmp"
APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache" \
APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp" \
apptainer build --fakeroot cherimoya.sif cherimoya.def
```

The build test also checks that the `cherimoya` CLI works and that `cherimoya`, `bpnetlite.performance`, and `tangermeme` all import successfully.

## Use the Python API

Request a GPU node before running anything training- or inference-heavy (`salloc -c 2 --mem=16G --gpus=1 --partition=dev -t 2:00:00`), then use `--nv` so the container can see the GPU:

```bash
apptainer exec --nv cherimoya.sif \
  python -c "import cherimoya; print(cherimoya.__file__)"
```

Run one of your own Python scripts against the containerized environment:

```bash
apptainer exec --nv \
  --bind "$PWD:/work" \
  --bind "$SCRATCH:/scratch" \
  cherimoya.sif \
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
apptainer shell --nv cherimoya.sif
ipython
```

## Use the CLI

The `cherimoya` CLI drives the full pipeline — peak calling, signal extraction, training, attribution, seqlet discovery, and motif annotation — either end to end or one stage at a time:

```bash
apptainer exec --nv cherimoya.sif cherimoya --help
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
  cherimoya.sif \
  cherimoya pipeline --help
```

## Caches

`cherimoya.def`'s `%environment` points `NUMBA_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, and `XDG_CACHE_HOME` at paths under `/tmp`, which is ephemeral per job on Sherlock. If you want compiled kernels/cache artifacts to persist across jobs, bind a scratch directory over one of these paths, e.g. `--bind "$SCRATCH/torchinductor-cache:/tmp/torchinductor_cache"`.
