#!/usr/bin/bash
set -e # exits immediately if 1 command fails

for n in 8 16 32 64 128; do
    echo "Running poisson_cylinder with n=${n}..."
    build/poisson_cylinder "$n"

    echo "Copying results for n=${n}..."
    cp -r data/poisson_cylinder/* examples/poisson_cylinder
done
