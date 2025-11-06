#! /bin/bash

set -ex

export HUGGINGFACE_TOKEN="TOKEN"

apt-get update 
apt-get -y install software-properties-common ca-certificates python3 python3-pip python-is-python3 wget git vim net-tools jq zip tmux pciutils
mkdir -p /mnt/disks/data/tmp /mnt/disks/data/pip-cache /mnt/disks/data/.cache

export SETUPTOOLS_ENABLE_FEATURES=legacy-editable
export PYTHONPATH="$PWD:$PWD/finemoe/ops"
export TMPDIR=/mnt/disks/data/tmp
export PIP_CACHE_DIR=/mnt/disks/data/pip-cache
export XDG_CACHE_HOME=/mnt/disks/data/.cache
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ln -s finemoe/ops/op_builder op_builder

/mnt/disks/data/.venv/bin/pip install -e . --no-build-isolation --no-cache-dir

# /mnt/disks/data/.venv/bin/pip wheel --no-build-isolation --no-cache-dir -w /mnt/disks/data/wheels "flash-attn==2.8.3"

/mnt/disks/data/.venv/bin/pip install flash-attn --no-build-isolation --no-cache-dir 

/mnt/disks/data/.venv/bin/huggingface-cli login --token $HUGGINGFACE_TOKEN
