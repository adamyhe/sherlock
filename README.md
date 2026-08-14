# sherlock

Reproducible environment recipes for running genomic sequence-to-function ML tools on Stanford's [Sherlock](https://www.sherlock.stanford.edu/) HPC cluster.

Sherlock's system compiler and OS are old enough that installing modern ML tooling directly — matching CUDA/cuDNN module versions to whatever JAX/PyTorch/TensorFlow wheel you need — is fragile and breaks often. Every tool in this repo is packaged as an [Apptainer](https://apptainer.org/) image instead, built from a Docker base, with CUDA/cuDNN delivered as pip wheels bundled inside the image wherever possible. The only thing that needs to come from the host at runtime is the NVIDIA driver, via `apptainer ... --nv`.

## Tools

| Directory | Tool | Base image |
| --- | --- | --- |
| [`alphagenome/`](alphagenome/README.md) | [AlphaGenome](https://github.com/google-deepmind/alphagenome) + `alphagenome_research` | `python:3.12-slim` |
| [`cherimoya/`](cherimoya/README.md) | [Cherimoya](https://github.com/jmschrei/cherimoya) | `pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime` |
| [`chrombpnet/`](chrombpnet/README.md) | [ChromBPNet](https://github.com/kundajelab/chrombpnet) | `kundajelab/chrombpnet:latest` |

Each directory has a `<tool>.def` Apptainer definition and a `README.md` with build and usage instructions specific to that tool. Nothing in these `.def` files is Sherlock-specific, so these images can be built on other machines (e.g., a lab workstation, or in an interactive Sherlock session) — any Linux host with Apptainer/Singularity installed and network access to Docker Hub/PyPI/GitHub will do. A GPU is only needed later, at run time with `--nv`.

## Quick start

```bash
cd <tool>
apptainer build <tool>.sif <tool>.def
apptainer exec --nv <tool>.sif python -c "import <tool>"
```

See the per-tool READMEs for `--fakeroot`/scratch-cache build options, GPU node requests, CLI usage, and cache-persistence notes.

## Other files

- `torch_source.sh` — loads Sherlock environment modules and activates a `torch` miniforge env, for non-containerized PyTorch work outside of these images. Dev/personal (Adam)-use only.
