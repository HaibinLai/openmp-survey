// stencil2d_graphite.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static inline double now() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec*1e-9;
}

static void init(float *u, int N) {
    #pragma omp parallel for if(N>256)
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            u[(size_t)i*N + j] = (float)((i+j)%7);
}

int main(int argc,char**argv){
    int N   = (argc>1)?atoi(argv[1]):4096;
    int T   = (argc>2)?atoi(argv[2]):50; // 时间步
    float *u = (float*)aligned_alloc(64, (size_t)N*N*sizeof(float));
    float *v = (float*)aligned_alloc(64, (size_t)N*N*sizeof(float));
    init(u,N); memset(v,0,N*(size_t)N*sizeof(float));

    // 预热
    for (int t=0;t<2;t++)
      for (int i=1;i<N-1;i++)
        for (int j=1;j<N-1;j++)
          v[(size_t)i*N+j] = 0.25f*(u[(i-1)*(size_t)N+j]+u[(i+1)*(size_t)N+j]+
                                    u[i*(size_t)N+j-1]+u[i*(size_t)N+j+1]) - u[i*(size_t)N+j];

    double t0=now();
    for (int t=0;t<T;t++) {
      // —— 多面体易于做 loop interchange/tiling 的规范形 —— 
      for (int i=1;i<N-1;i++)
        for (int j=1;j<N-1;j++)
          v[(size_t)i*N+j] = 0.25f*(u[(i-1)*(size_t)N+j]+u[(i+1)*(size_t)N+j]+
                                    u[i*(size_t)N+j-1]+u[i*(size_t)N+j+1]) - u[i*(size_t)N+j];
      float *tmp=u; u=v; v=tmp;
    }
    double t1=now();
    // 防止优化
    volatile float chk=u[(N/2)*(size_t)N + (N/2)];
    printf("N=%d T=%d time=%.3f s checksum=%.3f\n",N,T,t1-t0,(double)chk);
    free(u); free(v); return 0;
}
