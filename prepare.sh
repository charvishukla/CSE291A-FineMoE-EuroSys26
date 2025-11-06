# 0) Toolchain
sudo apt-get update
sudo apt-get install -y g++-12 gcc-12 ninja-build build-essential

# 1) Force Torch C++ extensions to use GCC 12
export CC=gcc-12
export CXX=g++-12

# 2) Keep builds on the big disk and verbose
export TORCH_EXTENSIONS_DIR=/mnt/disks/data/torch_extensions
export MAX_JOBS=$(nproc)
export VERBOSE=1

# (Optional safety belt) ensure <string> gets included early
export CXXFLAGS="$CXXFLAGS -include string"

# 3) Nuke previous failed builds and retry
rm -rf "$TORCH_EXTENSIONS_DIR"/prefetch*
cd demo
python prepare_data.py
