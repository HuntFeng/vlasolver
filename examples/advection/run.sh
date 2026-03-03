#!/usr/bin/bash
set -e # exits immediately if 1 command fails

for n in 8 16 32 64 128 256; do
    echo "Running advection with n=${n}..."
    build/advection "$n"

    echo "Copying results for n=${n}..."
    rm data/advection/output_${n}_*0.h5
    cp -r data/advection/*.h5 examples/advection/
done
