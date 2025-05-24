# Discontinuous Galerkin Method for Ideal MHD

## Set Up Environment

- Docker is required for creating isolated and stable environment.

1. Build the docker image using the command

```bash
devpod up . --provider docker
```

2. Start and/or attach to the existing container daemon

```bash
ssh vlasolver.devpod
```

## Setup Neovim (Optional)

1. In any directory, start `neovim` by

```bash
nvim
```

Plugins will be installed if you have neovim configs under `~/.config/nvim` on your computer, this config folder will be mounted to the container so we don't need to install plugins again.

2. Connect Copilot (Optional)
   In Neovim, type command `:Copilot`, then copy one-time passcode to GitHyb's device login page, https://github.com/login/device. Then it's good to go.

## Build Project

- CMake is used for building this C++ project.

```bash
cmake -B build
cmake --build build
```
