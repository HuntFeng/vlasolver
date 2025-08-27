#!/bin/bash
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=output_%j.log
stdbuf -oL -eL apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
