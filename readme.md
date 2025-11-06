## Set Up Instructions 

1. Create a virual environment in `/mnt/disks/data` using:
  ```
  python3 -m venv .venv
  ```
2. Go back to the root directory of the project and source this environment.
3. Install `wheel` and `setuptools` and torch:
   ```
   python -m pip install --upgrade pip setuptools wheel
   pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
   ```
4.  Add NVIDIA CUDA repo for Ubuntu 24.04 (noble)
   ```
    sudo apt-get update
    sudo apt-get install -y wget gnupg
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
   ```
 5. Install CUDA 12.4 toolkit (gives nvcc)
    ```
      sudo apt-get install -y cuda-toolkit-12-4 build-essential ninja-build cmake
    ```
  6.  Verify nvcc
    ```
      nvcc --version
      ls -l /usr/local/cuda/bin/nvcc
    ```
7. Set Paths
     ```
      export CUDA_HOME=/usr/local/cuda
      export PATH="$CUDA_HOME/bin:$PATH"
      export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
      ```
8. Run `sudo ./setup.sh`

9. Run each command from `prepare.sh`
10. Run  `python prepare_data.py`
11. Run  `python process_data.py`
12. Run `python eval.py`
