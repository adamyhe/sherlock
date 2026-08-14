# Building AlphaGenome apptainer

Sherlock's compiler and OS are very old, which makes installing newer packages a major challenge. This used to be worked around with a bare venv (`ml cuda/...`, `ml cudnn/...`, then `pip install "jax[cuda12-local]"`), but that approach is fragile: it only works if the loaded module versions exactly match what JAX/TensorFlow were built against. This apptainer image sidesteps that by using `jax[cuda12]` and `tensorflow[and-cuda]`, which bundle their own CUDA/cuDNN userspace libraries as pip wheels — the image only needs the NVIDIA driver from the host, mounted via `--nv`.

## Build

```bash
apptainer build alphagenome.sif alphagenome.def
```

If the build node requires fakeroot or a writable cache, use Sherlock scratch:

```bash
mkdir -p "$SCRATCH/apptainer-cache" "$SCRATCH/apptainer-tmp"
APPTAINER_CACHEDIR="$SCRATCH/apptainer-cache" \
APPTAINER_TMPDIR="$SCRATCH/apptainer-tmp" \
apptainer build --fakeroot alphagenome.sif alphagenome.def
```

## Use the Python API

Request a GPU node before running anything that needs a device (`salloc -c 2 --mem=16G --gpus=1 --partition=dev -t 2:00:00`), then use `--nv` so the container can see the GPU:

```bash
apptainer exec --nv alphagenome.sif \
  python -c "import jax; print(jax.devices())"
```

Run one of your own Python scripts against the containerized environment:

```bash
apptainer exec --nv \
  --bind "$PWD:/work" \
  --bind "$SCRATCH:/scratch" \
  alphagenome.sif \
  python /work/my_alphagenome_script.py
```

For notebooks or interactive exploration, start Python from an interactive shell:

```bash
apptainer shell --nv alphagenome.sif
python
```

The build test also checks that `import jax`, `import tensorflow`, `import alphagenome`, and `import alphagenome_research` all succeed.

## Environment and caches

`alphagenome.def`'s `%environment` points `XDG_CACHE_HOME` and `MPLCONFIGDIR` at paths under `/tmp`, which is ephemeral per job on Sherlock. `huggingface_hub` and `kagglehub` (both installed for pulling model weights) will cache downloads under paths derived from this, so a fresh job re-downloads everything from scratch. If you're re-running against the same weights repeatedly, bind a persistent scratch directory over the cache path instead, e.g. `--bind "$SCRATCH/hf-cache:/tmp/.cache"`.
