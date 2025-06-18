apptainer shell --nv \
  --no-home \
  --bind ~/.config/nvim \
  --bind ~/.ssh \
  --bind ~/.gitconfig \
  --bind /usr/share/terminfo \
  --overlay .devcontainer/overlay .devcontainer/kokkos_cuda.sif
