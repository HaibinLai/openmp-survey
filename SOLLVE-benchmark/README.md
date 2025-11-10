# SOLLVE-benchmark

In this openmp survey project, we use 2 workloads originated from SOLLVE benchmark.

The source code of the benchmark can be found in github:

https://github.com/SOLLVE/benchmarks.git



## SOLLVE introduction

Overview

The SOLLVE benchmark suite is a collection of representative applications and microbenchmarks designed to evaluate parallel programming models—especially OpenMP and its extensions (tasks, SIMD, and offloading)—on real scientific and engineering workloads. The suite includes kernels and full applications that exercise common HPC computation patterns such as data-parallel loops, stencil updates, particle/molecular dynamics, irregular/recursive tasks, and compute- vs. memory-bound kernels. The goal of SOLLVE is to provide reproducible scenarios for measuring performance, scalability, portability, and energy efficiency across different OpenMP implementations and hardware platforms.

## Why include SOLLVE in an OpenMP survey

Representativeness: The suite covers multiple programming patterns and bottlenecks typical of HPC codes, making it suitable to illustrate where OpenMP performs well and where it struggles.
Multi-dimensional evaluation: It supports analysis beyond wall-clock time—e.g., strong/weak scaling, memory bandwidth, cache behavior, and energy consumption.
Reproducibility: Provided inputs and run scripts make it straightforward to reproduce experiments on different systems and compilers, enabling fair comparisons.
Repository layout (high level)

`SOLLVE-benchmark/`
MolecularDynamics/ — particle or molecular dynamics examples exercising neighbor lists and pairwise force computations.

`Stencil/` — 1D/2D/3D stencil kernels to investigate memory access patterns, tiling, and vectorization.

`README.md` — overall instructions, build/run hints, and references.
(other subdirectories) — specific applications, input sets, and scripts.
Note: In your survey, mention which sub-benchmarks you actually ran (for example, “I used the MolecularDynamics and Stencil cases”) to be explicit about coverage.

## Compile

All of the source code can be compiled using:

```cpp
g++ -O2 ./bench.cpp -o bench_cpp -fopenmp -std=c++14
```

Feel free to use intel icx, llvm and other compilers.

Run: use small inputs to validate correctness first, then run performance experiments with larger inputs. Typical workflow:

Set OMP_NUM_THREADS (or use explicit affinity controls).

Run warm-up iterations, then timed trials.
Repeat each measurement multiple times (≥5) and report mean and variance.

Collect extra metrics: use perf, PAPI, or vendor tools to capture cache misses, memory bandwidth, FLOPS, and energy (RAPL).