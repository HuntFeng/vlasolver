# Noise‑Free Grid-based Vlasov Modeling of Plasma-Object Interactions via a Unified Ghost‑Fluid Immersed‑Boundary Method

## Set Up Apptainer Environment (Recommended)

### Requirements

- Apptainer is required for creating isolated and stable environment.

We can either install Apptainer on your own machine

```bash
sudo apt install apptainer
```

or load module on HPC

```bash
module load apptainer
```

### Building Container

- Build the apptainer image in using the command

```bash
apptainer build .devcontainer/kokkos_cuda.sif .devcontainer/Apptainer.def
```

### Executing Code

- Build and Run the code in container

```bash
apptainer run --nv .devcontainer/kokkos_cuda.sif cmake -B build
apptainer run --nv .devcontainer/kokkos_cuda.sif cmake --build build
apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
```

- A sample slurm script for running on gpu cluster

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=vlasolver
#SBATCH --output=vlasolver.out

apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
```

### Development in Container (Optional)

- For better coding experience, we may want to install editors and tools (such as LSP, Language-Server-Protocol) in the container.
- Since the apptainer container is immutable once created, we need to create a writable overlay to the existing SIF image.

```bash
mkdir -p .devcontainer/overlay
apptainer shell --no-home --fakeroot --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

Now we can have root previlege in the writable overlay and we can install system-wide tools. Once tool installation is done, we can use the overlay for development.

```bash
apptainer shell --nv \          # --nv for GPU support
  --no-home \                   # avoid mounting home directory
  --bind ~/.ssh \               # for git access
  --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

### Debugging

- Configure CMake to build with debug symbols

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build
```

- Use `gdb` or `cuda-gdb` to debug the executable

```bash
gdb build/vlasolver
```

### Profiling

- If code is built with CUDA backend, `nsys`, `ncu` can be used to profile the code.
