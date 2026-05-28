# AlphaGenome install

This README documents a reproducible workflow for building a Python virtual environment on Stanford Sherlock that supports:

* alphagenome
* alphagenome_research
* JAX with local CUDA
* TensorFlow
* common AlphaGenome-related dependencies

The workflow is designed for Sherlock's HPC environment, where source builds from pip can fail because of compiler/toolchain incompatibilities. The guiding principle is:

> Prefer binary wheels whenever possible, manually stage fragile dependencies, and only let pip resolve dependencies when it is safe to do so.

## Start an interactive GPU development session

Request a GPU development node:

```bash
salloc -c 2 --mem=16G --gpus=1 --partition=dev -t 2:00:00
```

This is useful because the environment should be built and smoke-tested in a context where CUDA/JAX/TensorFlow can see a GPU.

## Load required Sherlock modules

```bash
ml cuda/12.8.0
ml cudnn/9.14.0
ml gcc/12.4.0
ml cmake/3.31.4
ml openblas/0.3.28
ml xsimd/8.1.0
ml xz/5.8.1
ml hdf5/1.14.4
ml arrow/22.0.0
ml load py-pyarrow/18.1.0_py312
ml lz4/1.8.0
ml biology
ml htslib
ml ucsc-utils
```

## Create and activate the virtual environment

```bash
mamba create -n jax python==3.12.1
```

## Pip installs

Install all pypi dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: "scipy==1.12.0" "ml-dtypes==0.5.0"
python -m pip install "jax[cuda12-local]==0.7.0"
python -m pip install --only-binary=:all: "h5py==3.12.1"
python -m pip install --only-binary=:all: "tensorflow==2.18.1"

python -m pip install --only-binary=:all: \
  absl-py \
  anndata \
  einshape \
  "etils[epath]" \
  huggingface_hub \
  jaxtyping \
  kagglehub \
  pandas \
  pyBigWig \
  pyarrow \
  pyfaidx \
  immutabledict \
  intervaltree \
  matplotlib \
  seaborn \
  zstandard \
  typeguard \
  typing_extensions \
  tqdm \
  "protobuf>=5.28.3" \
  "grpcio>=1.67.1"

python -m pip install cython
python -m pip install "sorted-nearest>=0.0.41"
python -m pip install "pyranges==0.1.4"
python -m pip install --only-binary=:all: chex dm-haiku optax orbax
python -m pip install --no-deps alphagenome
```

## Install `alphagenome_research`

```bash
git clone https://github.com/google-deepmind/alphagenome_research.git
python -m pip install --no-deps -e ./alphagenome_research
```

