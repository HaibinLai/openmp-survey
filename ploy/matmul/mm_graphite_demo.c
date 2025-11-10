// mm_graphite_demo.c
// 演示 Graphite（多面体）优化与 OpenMP 的交互
// 编译时可用不同编译选项触发：基线 / Graphite 优化 / 自动并行 / 手写 OpenMP
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef ALIGN
#define ALIGN 64
#endif

static inline double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *amalloc(size_t nbytes) {
    void *p = NULL;
    if (posix_memalign(&p, ALIGN, nbytes) != 0) return NULL;
    memset(p, 0, nbytes);
    return p;
}

static void init(float *A, float *B, float *C, int N) {
    #pragma omp parallel for if(N>256)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i*(size_t)N + j] = (float)((i + j) % 17) * 0.01f;
            B[i*(size_t)N + j] = (float)((i * 3 + j) % 13) * 0.02f;
            C[i*(size_t)N + j] = 0.0f;
        }
    }
}

// 1) 朴素版本（不给任何并行/向量化提示）：让 Graphite 自己做活
void mm_plain(const float * __restrict A,
              const float * __restrict B,
              float * __restrict C,
              int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i*(size_t)N + k] * B[k*(size_t)N + j];
            C[i*(size_t)N + j] = sum;
        }
}

// 2) 手写 OpenMP：外两层并行 + 内层 SIMD（便于与 Graphite 对比）
void mm_omp_simd(const float * __restrict A,
                 const float * __restrict B,
                 float * __restrict C,
                 int N)
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; ++k)
                sum += A[i*(size_t)N + k] * B[k*(size_t)N + j];
            C[i*(size_t)N + j] = sum;
        }
}

static double checksum(const float *C, int N) {
    double s = 0.0;
    for (int i = 0; i < N*N; ++i) s += C[i];
    return s;
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int mode = (argc > 2) ? atoi(argv[2]) : 0;
    // mode: 0=plain(给 Graphite 玩), 1=手写 OpenMP+SIMD

    fprintf(stderr, "N=%d, mode=%d\n", N, mode);

    size_t bytes = (size_t)N * N * sizeof(float);
    float *A = (float*)amalloc(bytes);
    float *B = (float*)amalloc(bytes);
    float *C = (float*)amalloc(bytes);
    if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }

    init(A, B, C, N);

    // 预热
    if (mode == 0) mm_plain(A, B, C, N);
    else           mm_omp_simd(A, B, C, N);

    init(A, B, C, N);
    double t0 = now_s();

    if (mode == 0) mm_plain(A, B, C, N);
    else           mm_omp_simd(A, B, C, N);

    double t1 = now_s();
    double gflops = 2.0 * (double)N * N * N / (t1 - t0) / 1e9;

    printf("time = %.6f s, GFLOPS = %.2f, checksum = %.6f\n",
           (t1 - t0), gflops, checksum(C, N));

    free(A); free(B); free(C);
    return 0;
}
