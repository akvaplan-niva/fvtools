#!/bin/bash

module purge
module load intel/2018a
module load Python/3.6.6-intel-2018b
# module load CMake/3.9.1 %needed for pip install cx

source /cluster/home/hes001/python3/bin/activate

cx --boundary="appdata/boundary.txt" \
   --islands="appdata/islands.txt" \
   --view-angle=90.0 \
   --min-distance=0.0 \
   --max-distance=500000.0 \
   --output-dir="$PWD/input"

echo "Finished"
