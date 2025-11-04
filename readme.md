
```

sudo apt-get update
sudo apt-get -y install \
  software-properties-common ca-certificates \
  python3 python3-pip python-is-python3 python3-venv \
  wget git vim net-tools jq zip tmux pciutils \
  build-essential ninja-build cmake

```


```

git clone https://github.com/IntelliSys-Lab/FineMoE-EuroSys26
cd FineMoE-EuroSys26

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel

```


```
# NumPy first (prevents Torch→NumPy warnings during build)
pip install numpy

# PyTorch that matches your CUDA runtime (common for GCE CUDA 12.x)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision

# (CPU-only fallback)
# pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

```



```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

```



```
ln -s finemoe/ops/op_builder op_builder

# Confirm it imports
python -c "import op_builder, inspect; print(op_builder.__file__)"
```


```
# (If you haven't added NVIDIA's repo yet)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu$(lsb_release -rs | tr -d .)/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install only what's needed for building CUDA extensions
sudo apt-get install -y \
  cuda-compiler-12-4 \
  cuda-cudart-12-4 \
  cuda-cudart-dev-12-4 \
  cuda-libraries-12-4 \
  cuda-libraries-dev-12-4



export CUDA_HOME=/usr/local/cuda-12.4
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

which nvcc
nvcc --version
```


```
SETUPTOOLS_ENABLE_FEATURES=legacy-editable \
PYTHONPATH="$PWD:$PWD/finemoe/ops" \
pip install -e . --no-build-isolation
```


```
# Adjust /mnt/disks/scratch to your actual large volume
mkdir -p /mnt/disks/scratch/{hf_cache,transformers,datasets,torch,tmp}
# (sudo chown if needed)

export HF_HOME=/mnt/disks/scratch/hf_cache
export HF_HUB_CACHE=/mnt/disks/scratch/hf_cache/hub
export TRANSFORMERS_CACHE=/mnt/disks/scratch/transformers
export HF_DATASETS_CACHE=/mnt/disks/scratch/datasets
export TORCH_HOME=/mnt/disks/scratch/torch
export TMPDIR=/mnt/disks/scratch/tmp

```


```
export HUGGINGFACE_TOKEN=hf_xxx...xxx
huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
```



```
cd demo
python prepare_data.py
python process_data.py
python eval.py

```