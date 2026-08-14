# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

This is not an application codebase — it's a collection of reproducible environment-build recipes for running genomic sequence-to-function ML tools (AlphaGenome, Cherimoya, ChromBPNet) on Stanford's Sherlock HPC cluster. There is no build/lint/test tooling; changes here are Markdown docs, Apptainer `.def` files, and small shell snippets that get run manually on Sherlock login/dev nodes.

## Repo layout

- `alphagenome/` — `alphagenome.def` (Apptainer definition, built `FROM python:3.12-slim`) + `README.md` with build/run invocations, for JAX+CUDA, TensorFlow, and the `alphagenome`/`alphagenome_research` packages.
- `cherimoya/` — `cherimoya.def` (Apptainer definition, built `FROM pytorch/pytorch`) + `README.md` with the `apptainer build` invocation.
- `chrombpnet/` — `chrombpnet.def` (Apptainer definition, built `FROM kundajelab/chrombpnet:latest`) + `README.md` with build/run invocations.
- `torch_source.sh` — a snippet that loads Sherlock environment modules (`ml cuda/...`, `ml gcc/...`, etc.) and activates a `torch` miniforge env. Meant to be sourced at the start of a Sherlock session for non-containerized PyTorch work, not executed as a standalone script.

All three tools used to have (or, for AlphaGenome, still had until recently) instructions for building a bare Sherlock venv via `ml cuda/...`/`ml cudnn/...` module loads plus `pip install jax[cuda12-local]`/similar. That approach is fragile: it only works if the loaded module versions exactly match what the pip wheels were built against, and Sherlock's modules change over time. The project has moved (and should keep moving) toward Apptainer images that bundle their own CUDA/cuDNN userspace libraries as pip wheels (e.g. `jax[cuda12]`, `tensorflow[and-cuda]`) or via the base Docker image, so the only thing that has to come from the host is the NVIDIA driver, mounted with `apptainer ... --nv`.

## Working with the Apptainer `.def` files

`cherimoya/cherimoya.def`, `chrombpnet/chrombpnet.def`, and `alphagenome/alphagenome.def` exist because Sherlock's system compiler/OS is too old to install these packages' dependencies directly — the container sidesteps the host toolchain entirely. Keep that constraint in mind when editing:

- Prefer `--no-deps` pip installs for the target package itself when the base Docker image (or an earlier pip install in the same `.def`) already provides a specific, load-bearing version of a dependency (e.g. cherimoya's base image pins `torch==2.13.0+cu126`; a dependency-resolving install would silently replace it).
- When pinning a package to a git commit instead of a released tag/PyPI version (as `cherimoya.def` and `alphagenome.def` do for `cherimoya` and `alphagenome_research` respectively), that's deliberate — check the surrounding comment before "simplifying" it back to a version pin, since upstream tags can be force-moved and unpinned branches aren't reproducible.
- Prefer CUDA delivery via pip wheels bundled with the framework (`jax[cuda12]`, `tensorflow[and-cuda]`) over extras like `cuda12-local` that expect a matching system CUDA toolkit — that's the whole point of containerizing instead of relying on Sherlock's `ml cuda/...` modules.
- `chrombpnet.def` copies the upstream editable install from `/scratch/chrombpnet` to `/opt/chrombpnet` and reinstalls from there — this is so the Python API stays importable even when Sherlock scratch is bind-mounted over `/scratch` at runtime (which would otherwise shadow the package source). Don't remove this copy step.
- Rebuild with `apptainer build <name>.sif <name>.def` from inside each tool's directory; use `--fakeroot` with `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` pointed at `$SCRATCH` if the build node lacks a writable default cache (documented in each tool's `README.md`).
- Each `.def` has a `%test` section that apptainer runs automatically as part of `build` (import checks, and `chrombpnet --help` for the CLI) — keep it passing when editing the install steps. These only check imports, not GPU functionality, since build nodes typically don't have `--nv`/a GPU attached.
