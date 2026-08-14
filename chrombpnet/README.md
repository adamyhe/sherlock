# Building ChromBPNet Apptainer

Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. To get around this, build an Apptainer image from the upstream ChromBPNet Docker image.

This image is intended to expose the ChromBPNet Python API as well as the `chrombpnet` command-line entry point. During the build, the upstream editable install is copied from `/scratch/chrombpnet` into `/opt/chrombpnet` and reinstalled from there so binding Sherlock scratch onto `/scratch` does not hide the package source.

The upstream Docker image is `kundajelab/chrombpnet:latest`.

## Build

```bash
apptainer build chrombpnet.sif chrombpnet.def
```

If the build node requires fakeroot or a writable cache, use Sherlock scratch:

```bash
mkdir -p "$SCRATCH/apptainer-cache" "$SCRATCH/apptainer-tmp"
APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache" \
APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp" \
apptainer build --fakeroot chrombpnet.sif chrombpnet.def
```

## Use the Python API

Request a GPU node before running anything training- or inference-heavy (`salloc -c 2 --mem=16G --gpus=1 --partition=dev -t 2:00:00`), then run Python inside the image and import ChromBPNet directly:

```bash
apptainer exec --nv chrombpnet.sif \
  python -c "import chrombpnet; print(chrombpnet.__file__)"
```

Run one of your own Python scripts against the containerized ChromBPNet environment:

```bash
apptainer exec --nv \
  --bind "$PWD:/work" \
  --bind "$SCRATCH:/scratch" \
  chrombpnet.sif \
  python /work/my_chrombpnet_script.py
```

For notebooks or interactive exploration, start Python from an interactive shell:

```bash
apptainer shell --nv chrombpnet.sif
python
```

The build test also checks that `import chrombpnet` succeeds.

## Use the CLI

Use `--nv` when running GPU-backed ChromBPNet jobs:

```bash
apptainer exec --nv chrombpnet.sif chrombpnet --help
```

For an interactive shell:

```bash
apptainer shell --nv chrombpnet.sif
```

Bind any project, genome, or scratch directories that are not visible by default:

```bash
apptainer exec --nv \
  --bind /path/to/data:/data \
  --bind "$SCRATCH:/scratch" \
  chrombpnet.sif \
  chrombpnet pipeline --help
```

## Environment

`chrombpnet.def`'s `%environment` sets `CHROMBPNET_SOURCE=/opt/chrombpnet` and prepends it to `PYTHONPATH`, so the Python API resolves to the copy under `/opt` (see the note above about why it isn't imported straight from `/scratch/chrombpnet`) regardless of what's bound onto `/scratch` at runtime. It also points `NUMBA_CACHE_DIR`, `XDG_CACHE_HOME`, and `MPLCONFIGDIR` at paths under `/tmp`, which is ephemeral per job — bind a scratch directory over one of these (e.g. `--bind "$SCRATCH/numba-cache:/tmp/numba_cache"`) if you want cached artifacts to persist across jobs.
