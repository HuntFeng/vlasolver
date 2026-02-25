#!/usr/bin/bash
set -e # exits immediately if 1 command fails

for n in 8 16 32 64 128; do
    echo "Running advection with n=${n}..."
    build/advection "$n"

    echo "Copying results for n=${n}..."
    cp -r data/advection/* examples/advection
done
