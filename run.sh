#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=CHEN_CUI
#SBATCH --output=output_%j.log

# stdbuf -oL -eL apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/sheath ./examples/sheath/input.ini
stdbuf -oL -eL apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/sheath_rough_wall ./examples/sheath_rough_wall/input.ini
# stdbuf -oL -eL apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/cylinder ./examples/cylinder/input.ini
