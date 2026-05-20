#!/usr/bin/bash
set -e # exits immediately if 1 command fails

for n in 16 32 64 128 256; do
    echo "Running poisson with n=${n}..."
    build/poisson "$n"

    echo "Copying results for n=${n}..."
    cp -r data/poisson/*.h5 examples/poisson/
done
