apptainer shell --nv \
  --no-home \
  --bind ~/.ssh \
  --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
