# Discontinuous Galerkin Method for Ideal MHD

## Set Up Apptainer Environment (Recommended)

- Apptainer is required for creating isolated and stable environment.

1. Build container

- Build the apptainer image in using the command

```bash
apptainer build .devcontainer/kokkos_cuda.sif .devcontainer/Apptainer.def
```

2. Execution code

- Build and Run the code in container

```bash
apptainer run --nv .devcontainer/kokkos_cuda.sif cmake -B build
apptainer run --nv .devcontainer/kokkos_cuda.sif cmake --build build
apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
```

## Sample Slurm Script for Running on GPU Cluster

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=vlasolver
#SBATCH --output=vlasolver.out

apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
```

3. Development in container

- To install editors and other tools, we can use overlay feature of Apptainer.

```bash
mkdir -p .devcontainer/overlay
apptainer shell --no-home --fakeroot --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

Now we can have root previlege in the overlay container. Once tool installations is done, we can use the overlay container for development.

```bash
apptainer shell --nv \          # --nv for GPU support
  --no-home \                   # avoid mounting home directory
  --bind ~/.ssh \               # for git access
  --bind /usr/share/terminfo \  # for better terminal support
  --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

## Debugging

- Configure CMake to build with debug symbols

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build
```

- Use `gdb` or `cuda-gdb` to debug the executable

```bash
gdb build/vlasolver
```

## Profiling

- If code is built with CUDA backend, `nsys`, `ncu` can be used to profile the code.
