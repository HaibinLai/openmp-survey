# Matmul Benchmark 


Compile script:

```bash
g++ -O3 -march=native -fopenmp -std=c++17 matmul.cpp -o omp_mm_bench
```

Runtime script:

```bash
# 4096 阶，fork-join，16 线程，CPU 频率 3.5GHz（用于把 TSC 周期换算为秒/GFLOPS）
OMP_NUM_THREADS=16 ./omp_mm_bench --n 4096 --mode fj --repeats 3 --ghz 3.5

# 4096 阶，task 模式，每个 task 处理 64 行
OMP_NUM_THREADS=16 numactl --cpunodebind=1  ./omp_mm_bench --n 4096 --mode task --rows_per_task 64 --repeats 3 --ghz 3.5

```