#include <omp.h>
int add(int a, int b) {
    int arr[4];
    #pragma omp parallel num_threads(4) 
    { 
        int tid = omp_get_thread_num();
        arr[tid] = a + b;
    }
}

int main() {
    return add(1, 2);
}

