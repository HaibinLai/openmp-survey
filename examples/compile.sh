# !/bin/bash

# Generate AST dump files for OpenMP code
gcc -O0 -fopenmp -fdump-tree-original run_GIMPLE.c -c

# Generate High GIMPLE dump files for OpenMP code
gcc -O0 -fopenmp -fdump-tree-gimple run_GIMPLE.c -c

# Generate Low GIMPLE dump files for OpenMP code
gcc -O0 -fopenmp -fdump-tree-ompexp run_GIMPLE.c -c

# Generate all GIMPLE dump files for OpenMP code
gcc -O0 -fopenmp -fdump-tree-all run_GIMPLE.c -c