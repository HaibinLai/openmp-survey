#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

// #include "timing/ompvv_timing.h"

#define N 500
#define MAXWORK 10

#define MAX_TIMESTEPS 100

static const int num_threads = 4;

//#define CPU_TEST                                                                                                                                            

int main(int argc, char* argv[])
{
    // OMPVV_INIT_TIMERS;
    float lboundary[N];
    float rboundary[N]; // for a reg. mesh  
    // setup threads
    omp_set_num_threads(num_threads);
    
    // const int ndevs = omp_get_num_devices();
    int *devices = NULL;
    double *time_devices = NULL;
    double start_iterations, end_iterations;
    int timestep = 0;
    int probSize = MAXWORK;
    int num_timesteps = 1;
    int numThreads = 1;
    int numTasks = N;
	
    int gsz = 1;

  
    int ndevs = 1;
    // assert(ndevs > 0);
    srand((unsigned) time(NULL));
    if(argc <= 1)
      {
        printf("Usage bench_stencil [pSize] [numBlocks] [chunkSize] [numTimesteps]\n" );
        printf("Using default parameters\n" );
        probSize = MAXWORK;
        num_timesteps = 1;
#pragma omp parallel
        numThreads = omp_get_num_threads();
        numTasks = N;
        gsz = 1;
      }
    else
      {
        if (argc > 1)
          probSize = atoi(argv[1]);
        if (argc > 2)
	  num_timesteps = atoi(argv[2]);
        if (argc > 3)
          numTasks = atoi(argv[3]);
        if (argc > 4)
          gsz = atoi(argv[4]);
      }
    printf("bench_stencil [pSize=%d] [numTasks=%d] [gsz=%d] [num_timesteps=%d] [numThreads=%d] \n", probSize, numTasks, gsz, num_timesteps, numThreads);

    int arrSize = probSize;
    int numBlocks = numTasks;
  float* a = new float[arrSize];
  float* b = new float[arrSize];
  float* c = new float[arrSize];
  int* blockWork = new int[numBlocks];

    for (int i = 0; i< arrSize; i++)
      {
        a[i] = 3.0;
        b[i] = 2.0;
        c[i] = 0.0;
      }

    int ctaskwork;

    for (int i = 0 ; i < numBlocks; i++)
      {
        ctaskwork = (probSize - 1)/(numTasks); // maybe could be MAXWORK/TotWork rather than div by 2                                                         
        blockWork[i] = ctaskwork;
      }

    int numCores = 0;
    double cpu_time = 0.0;
    double task_time = 0.0;

#ifdef CPU_TEST
    cpu_time = -omp_get_wtime();

    float* temp;
#pragma omp parallel
    int numCores = omp_get_num_threads();

#pragma omp parallel
    {
#pragma omp for schedule(static, gsz)
      {
        for (int i = 0; i < numBlocks; i++)
          {
                // compute start/end as prefix sum to avoid division/mod by ndevs and be robust
                int startInd = 0;
                for (int bb = 0; bb < i; ++bb) startInd += blockWork[bb];
                int endInd = startInd + blockWork[i] - 1;
                // safe boundary handling
                if (startInd > 0) b[startInd-1] = lboundary[i];
                if (endInd + 1 < arrSize) b[endInd+1] = rboundary[i];
                for (int j = startInd; j <= endInd ; j++)
                  a[j] = (b[j] + b[j-1] + b[j+1])/3.0;
                //swap pointers a and b for update
                c = b;
                b = a;
                a = c;
                if (startInd > 0) lboundary[i] = a[startInd-1]; else lboundary[i] = a[startInd];
                if (endInd + 1 < arrSize) rboundary[i] = a[endInd+1]; else rboundary[i] = a[endInd];
              }
      }
      cpu_time += omp_get_wtime();
      printf("cpu_time for comp: %f\n", cpu_time);
#endif

while(timestep < num_timesteps)
  {
#pragma omp parallel
    {
#pragma omp for schedule(static, gsz)
        for (int i = 0; i < numBlocks; i++) {
          //const int dev = (int) ((i/numBlocks)*ndevs); // use for static schedule                                                                         
	  
	  const int dev = i%ndevs;   
	  printf("device chosen for iteration %d : %d\n" , i, dev);
          // OMPVV_START_TIMER;
// #pragma omp target distribute parallel for simd device(dev) map(alloc: a[0:arrSize], b[0:arrSize], numBlocks, ndevs) map(tofrom: lboundary[i:1], rboundary[i:1], blockWork[i:1]) nowait
          
// Replace device offload with CPU-only vectorized inner loop
            {
                const int NN = blockWork[i];
                // compute start/end as prefix sum (robust CPU-only mapping)
                int startInd = 0;
                for (int bb = 0; bb < i; ++bb) startInd += blockWork[bb];
                const int endInd = startInd + NN - 1;
                // obtain boundaries for neighboring blocks (safe checks)
                float* temp ; //temp variable
                if (startInd > 0) b[startInd-1] = lboundary[i];
                if (endInd + 1 < arrSize) b[endInd+1] = rboundary[i];
                /* vectorize the inner loop on CPU */
  #pragma omp simd
                for (int j = startInd; j<= endInd ; j++)
                  a[j] = (b[j] + b[j-1] +b[j+1])/3.0;
              //swap pointers a an b for update                                                                                                               
              temp=b;
              b=a;
              a=temp;
              if (startInd > 0) lboundary[i] = a[startInd-1]; else lboundary[i] = a[startInd];
              if (endInd + 1 < arrSize) rboundary[i] = a[endInd+1]; else rboundary[i] = a[endInd];
            } // end inner compute
            // OMPVV_STOP_TIMER;
        } // end for      
    } // end parallel                                                                                                                               

    timestep++;
  } // end while                                                                                                                                              
 delete [] a;
 delete [] b;
 delete [] c;
 delete [] blockWork;
 free(devices);
 free(time_devices);
 return 0;
	    
} // end main 
