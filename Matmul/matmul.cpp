// omp_mm_bench.cpp
#include <bits/stdc++.h>
#include <omp.h>

#ifdef __x86_64__
static inline void cpuid_serialize() {
    unsigned int a=0, b, c, d;
    __asm__ __volatile__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0));
}
static inline uint64_t rdtscp_read() {
    unsigned int aux, lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux) ::);
    return (uint64_t(hi) << 32) | lo;
}
#else
# error "This benchmark uses RDTSCP and currently supports only x86_64."
#endif

struct Args {
    int    N = 2048;          // 矩阵阶数
    int    repeats = 3;       // 重复次数取最小值
    int    rows_per_task = 64;// task 模式的行块大小
    int    threads = 0;       // 0=沿用 OMP_NUM_THREADS
    double ghz = 0.0;         // 若>0，用来把 TSC 周期换成秒
    std::string mode = "fj";  // "fj" 或 "task"
};

static void parse_args(int argc, char** argv, Args& a) {
    for (int i=1; i<argc; ++i) {
        std::string s = argv[i];
        auto need = [&](int i){ if (i+1>=argc) { fprintf(stderr,"missing value for %s\n", s.c_str()); exit(1);} };
        if (s=="--n") { need(i); a.N = std::stoi(argv[++i]); }
        else if (s=="--repeats") { need(i); a.repeats = std::stoi(argv[++i]); }
        else if (s=="--rows_per_task") { need(i); a.rows_per_task = std::stoi(argv[++i]); }
        else if (s=="--threads") { need(i); a.threads = std::stoi(argv[++i]); }
        else if (s=="--ghz") { need(i); a.ghz = std::stod(argv[++i]); }
        else if (s=="--mode") { need(i); a.mode = argv[++i]; }
        else {
            fprintf(stderr, "Unknown arg: %s\n", s.c_str());
            exit(1);
        }
    }
}

template<typename T>
static void init_matrix(std::vector<T>& M, int N, uint64_t seed=42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<T> dist(T(0.0), T(1.0));
    for (auto &x: M) x = dist(rng);
}

template<typename T>
static T checksum(const std::vector<T>& M) {
    long double s = 0;
    for (auto x: M) s += x;
    return (T)s;
}

template<typename T>
static void zero(std::vector<T>& M) { std::fill(M.begin(), M.end(), T(0)); }

// Fork-Join 版本：简单的 ijk + collapse(2)
template<typename T>
static void matmul_fork_join(const T* A, const T* B, T* C, int N) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<N; ++i) {
        for (int j=0; j<N; ++j) {
            T sum = 0;
            const T* arow = A + (size_t)i * N;
            for (int k=0; k<N; ++k) {
                sum += arow[k] * B[(size_t)k * N + j];
            }
            C[(size_t)i * N + j] = sum;
        }
    }
}

// Task 版本：按行块划分，每个 task 负责 [i0, i1) 的所有 j,k
template<typename T>
static void matmul_tasks(const T* A, const T* B, T* C, int N, int rows_per_task) {
#pragma omp parallel
    {
#pragma omp single nowait
        {
            for (int i0 = 0; i0 < N; i0 += rows_per_task) {
                int i1 = std::min(i0 + rows_per_task, N);
#pragma omp task firstprivate(i0,i1) shared(A,B,C)
                {
                    for (int i = i0; i < i1; ++i) {
                        for (int j = 0; j < N; ++j) {
                            double sum = 0.0;
                            const double* arow = A + (size_t)i * N;
                            for (int k = 0; k < N; ++k) {
                                sum += arow[k] * B[(size_t)k * N + j];
                            }
                            C[(size_t)i * N + j] = (double)sum;
                        }
                    }
                }
            }
        }
#pragma omp taskwait
    }
}

// 计时包装：使用 RDTSCP（主）+ chrono（参考）
template<typename F>
static void measure_rdtscp(const Args& a, const char* tag, F&& fn,
                           double flops_total, // 2*N^3
                           uint64_t& best_cycles,
                           double& best_seconds_rdtsc,
                           double& best_seconds_chrono)
{
    best_cycles = std::numeric_limits<uint64_t>::max();
    best_seconds_rdtsc = std::numeric_limits<double>::infinity();
    best_seconds_chrono = std::numeric_limits<double>::infinity();

    for (int r=0; r<a.repeats; ++r) {
        // chrono 参考
        auto t0 = std::chrono::high_resolution_clock::now();

        // RDTSCP 主计时
        cpuid_serialize();                 // 保证开始前序列化
        uint64_t c0 = rdtscp_read();

        fn(); // 执行被测

        uint64_t c1 = rdtscp_read();       // RDTSCP 是序列化的，确保指令完成
        cpuid_serialize();                 // 防止后续乱序越过

        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t cyc = c1 - c0;
        double secs_chrono = std::chrono::duration<double>(t1 - t0).count();
        double secs_tsc = (a.ghz > 0.0) ? (cyc / (a.ghz * 1e9)) : std::numeric_limits<double>::quiet_NaN();

        best_cycles = std::min(best_cycles, cyc);
        best_seconds_chrono = std::min(best_seconds_chrono, secs_chrono);
        if (!std::isnan(secs_tsc)) best_seconds_rdtsc = std::min(best_seconds_rdtsc, secs_tsc);
    }

    // 输出
    std::cout << "==== " << tag << " ====\n";
    std::cout << "Best cycles (RDTSCP): " << best_cycles << "\n";
    if (a.ghz > 0.0) {
        double secs = best_seconds_rdtsc;
        double gflops = flops_total / 1e9 / secs;
        std::cout << "CPU GHz: " << a.ghz << ", time (RDTSCP): " << secs << " s, GFLOPS: " << gflops << "\n";
    } else {
        std::cout << "Tip: pass --ghz <freq> to convert cycles -> seconds/GFLOPS using TSC.\n";
    }
    std::cout << "Reference time (chrono): " << best_seconds_chrono << " s\n";
}

int main(int argc, char** argv) {
    Args args;
    parse_args(argc, argv, args);
    if (args.threads > 0) omp_set_num_threads(args.threads);

    const int N = args.N;
    const size_t NN = (size_t)N * (size_t)N;
    std::cout << "N=" << N << ", mode=" << args.mode
              << ", repeats=" << args.repeats
              << ", rows_per_task=" << args.rows_per_task
              << ", threads=" << (args.threads>0?args.threads:omp_get_max_threads())
              << ", ghz=" << args.ghz << "\n";

    // 使用 double，内存 N^2 * 8 * 3
    std::vector<double> A(NN), B(NN), C(NN);
    init_matrix(A, N, 1);
    init_matrix(B, N, 2);
    zero(C);

    // 预热一次（避免首次页错/缓存冷启动）
    {
        std::vector<double> Cw(NN, 0.0);
#pragma omp parallel for collapse(2) schedule(static)
        for (int i=0;i<N;++i)
            for (int j=0;j<N;++j) {
                double s=0; const double* arow = A.data() + (size_t)i*N;
                for (int k=0;k<N;++k) s+= arow[k]*B[(size_t)k*N+j];
                Cw[(size_t)i*N+j]=s;
            }
    }

    const double flops_total = 2.0 * (double)N * (double)N * (double)N;

    uint64_t best_cycles;
    double best_secs_tsc, best_secs_ch;

    if (args.mode == "fj") {
        auto run = [&](){
            zero(C);
            matmul_fork_join(A.data(), B.data(), C.data(), N);
        };
        measure_rdtscp(args, "Fork-Join", run, flops_total, best_cycles, best_secs_tsc, best_secs_ch);
    } else if (args.mode == "task") {
        auto run = [&](){
            zero(C);
            matmul_tasks(A.data(), B.data(), C.data(), N, args.rows_per_task);
        };
        measure_rdtscp(args, "Task", run, flops_total, best_cycles, best_secs_tsc, best_secs_ch);
    } else {
        std::cerr << "Unknown mode: " << args.mode << " (use 'fj' or 'task')\n";
        return 1;
    }

    // 简单输出校验（避免编译器激进优化把计算删了）
    std::cout << "Checksum(C) = " << std::setprecision(15) << checksum(C) << "\n";
    return 0;
}
