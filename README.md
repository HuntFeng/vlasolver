# Discontinuous Galerkin Method for Ideal MHD

## Set Up Environment

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

- Use `gdb` to debug the executable

```bash
gdb build/vlasolver
```
