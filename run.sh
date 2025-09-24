#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --account=CHEN_CUI
#SBATCH --output=output_%j.log
stdbuf -oL -eL apptainer run --nv .devcontainer/kokkos_cuda.sif ./build/sheath_full ./examples/sheath_full/input.ini

