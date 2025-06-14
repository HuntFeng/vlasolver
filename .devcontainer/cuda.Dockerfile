FROM nvidia/cuda:12.3.0-devel-ubuntu22.04

RUN apt update && apt install -y sudo

# use non-root user, set sudo password to none
RUN useradd -ms /bin/bash ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

# make .config writable
RUN mkdir /home/ubuntu/.config \
    && chown ubuntu:ubuntu /home/ubuntu/.config

# essential things
RUN <<EOF 
sudo apt install -y build-essential cmake nano git wget curl unzip
sudo apt install -y software-properties-common
# install python3 and pip for python packages
sudo apt install -y python3 python3-pip python3-venv
# remove EXTERNALLY-MANAGED so we can do pip install as normal site-packages
sudo rm -f /usr/lib/python3*/EXTERNALLY-MANAGED
pip install GDBKokkos
# install gdb for debugging
sudo apt install -y gdb 
# xclip for system clipboard
sudo apt install -y xclip
# ripgre for searching files
sudo apt install -y ripgrep
# nsight systems for profiling
sudo apt-get install -y nsight-systems-2024.2.3
EOF

# inih for reading .ini file
RUN sudo apt install -y libinih-dev

# hdf5 for better output format
RUN <<EOF
export HDF5_VERSION=1.14.5
wget https://github.com/HDFGroup/hdf5/releases/download/hdf5_${HDF5_VERSION}/hdf5-${HDF5_VERSION}.tar.gz
tar -xvzf hdf5-${HDF5_VERSION}.tar.gz
rm hdf5-${HDF5_VERSION}.tar.gz
cd hdf5-${HDF5_VERSION}
cmake -B build \
-DCMAKE_BUILD_TYPE=Release \
-DHDF5_ENABLE_ZLIB_SUPPORT=Off \
-DHDF5_ENABLE_SZIP_SUPPORT=Off
cmake --build build --parallel 4
sudo cmake --install build --prefix /usr/local
EOF

# highfive for easier hdf5 usage
RUN <<EOF
export HIGHFIVE_VERSION=2.10.1
wget https://github.com/BlueBrain/HighFive/archive/refs/tags/v${HIGHFIVE_VERSION}.tar.gz -O highfive-${HIGHFIVE_VERSION}.tar.gz
tar -xvzf highfive-${HIGHFIVE_VERSION}.tar.gz
rm highfive-${HIGHFIVE_VERSION}.tar.gz
cd HighFive-${HIGHFIVE_VERSION}
cmake -B build \
-DHIGHFIVE_USE_BOOST=Off \
-DHIGHFIVE_EXAMPLES=Off \
-DHIGHFIVE_USE_BOOST=Off \
-DHIGHFIVE_UNIT_TESTS=Off
cmake --build build --parallel 4
sudo cmake --install build --prefix /usr/local
EOF

# Unfortunately, unable to use it together with cuda backend, since it uses nvcc as compiler
# gridformat for generating vtk files for paraview visualization
# it's better to have highfive install
# RUN <<EOF
# export GRIDFORMAT_VERSION=0.4.0
# wget https://github.com/dglaeser/gridformat/archive/refs/tags/v${GRIDFORMAT_VERSION}.tar.gz -O gridformat-${GRIDFORMAT_VERSION}.tar.gz
# tar -xvzf gridformat-${GRIDFORMAT_VERSION}.tar.gz
# rm gridformat-${GRIDFORMAT_VERSION}.tar.gz
# cd gridformat-${GRIDFORMAT_VERSION}
# cmake -B build -DCMAKE_CXX_COMPILER=g++
# cmake --build build --parallel 4
# sudo cmake --install build --prefix /usr/local
# EOF

# Kokkos HPC framework
RUN <<EOF
export KOKKOS_VERSION=4.5.01
wget https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}/kokkos-${KOKKOS_VERSION}.tar.gz
tar -xvzf kokkos-${KOKKOS_VERSION}.tar.gz
rm kokkos-${KOKKOS_VERSION}.tar.gz
cd kokkos-${KOKKOS_VERSION}
# Host Serial: SERIAL
# Host Parallel: OpenMP
# Device Parallel: CUDA
cmake -B build \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_STANDARD=20 \
-DKokkos_ARCH_TURING75=On \
-DKokkos_ENABLE_CUDA=On \
-DKokkos_ENABLE_OPENMP=On \
-DKokkos_ENABLE_SERIAL=On \
-DKokkos_ENABLE_DEPRECATED_CODE_4=Off
cmake --build build --parallel 4
sudo cmake --install build --prefix /usr/local
EOF
# set these env vars to optimize OPENMP 4.0 performance
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=threads

# Neovim
RUN <<EOF
# install nodejs and npm for copilot 
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash - && sudo apt install -y nodejs
wget https://github.com/neovim/neovim/releases/download/v0.11.1/nvim-linux-x86_64.tar.gz
tar xzvf nvim-linux-x86_64.tar.gz
sudo ln -s /home/ubuntu/nvim-linux-x86_64/bin/nvim /usr/bin/nvim
EOF
