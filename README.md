# Discontinuous Galerkin Method for Ideal MHD

## Set Up Apptainer Environment (Recommended)

- Apptainer is required for creating isolated and stable environment.

1. Build container

- Build the apptainer image in using the command

```bash
apptainer build .devcontainer/kokkos_cuda.sif .devcontainer/Apptainer.def
```

- Start the apptainer shell with writable tmpfs and NVIDIA support

```bash
apptainer shell --nv .devcontainer/kokkos_cuda
```

2. For code execution only

- Build the apptainer image

```bash
apptainer build .devcontainer/kokkos_cuda.sif .devcontainer/Apptainer.def
```

- Build and Run the code in container

```bash
apptainer run --nv --app build .devcontainer/kokkos_cuda.sif
apptainer run --nv --app run .devcontainer/kokkos_cuda.sif
```

3. Development in container

- To install editors and other tools, we can use overlay feature of Apptainer.

```bash
mkdir -p .devcontainer/overlay
apptainer shell --no-home --fakeroot --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

Now we can have root previlege. After installations, we can restart the container with without `--fakeroot` option.

```bash
apptainer shell --no-home --nv --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
```

### Usage on HPC

1. Make sure the Apptainer image, `kokkos_cuda.sif`, is built.
2. Make sure the Vlasolver code is built.
3. Put the following script into a file, e.g., `vlasolver.sh`, and make it executable.

```bash
#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=00:10:00
#SBATCH --job-name=vlasolver
#SBATCH --output=vlasolver.out

apptainer run --nv --app run .devcontainer/kokkos_cuda.sif
```

4. Submit the job script to the scheduler

```bash
sbatch vlasolver.sh
```

## Set Up Docker Environment (Only for Local Development)

- Docker is required for creating isolated and stable environment.

1. Build the docker image then run it as daemon using the command

```bash
cd .devcontainer
# OpenMP build
docker compose -f compose.yaml up --build -d
# or CUDA build together with OpenMP for host parallelization
docker compose -f compose.cuda.yaml up --build -d
```

2. Start and/or attach to the existing container daemon

```bash
# Kokkos container with OpenMP backend
docker start -ai vlasolver
# or Kokkos container with CUDA backend
docker start -ai vlasolver-cuda
```

## Setup Neovim (Optional)

1. In any directory, start `neovim` by

```bash
nvim
```

Plugins will be installed if you have neovim configs under `~/.config/nvim` on your computer, this config folder will be mounted to the container so we don't need to install plugins again.

2. Connect Copilot (Optional)
   In Neovim, type command `:Copilot`, then copy one-time passcode to GitHyb's device login page, https://github.com/login/device. Then it's good to go.

3. Set `DISPLAY` environment variable on Mac for X11 and clipboard (Optional)

- To use xclip for the global clipboard, XQuartz is needed.
- After installing XQuartz, use the XQuartz built-in terminal to check `DISPLAY` variable using echo.

```bash
echo $DISPLAY
# usually :0
```

- In the container, set DISPLAY variable

```bash
export DISPLAY=host.docker.internal:0
```

## Build Project

- CMake is used for building this C++ project.

```bash
cmake -B build
cmake --build build
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
