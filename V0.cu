  #include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
 #if __DEVICE_EMULATION__

   bool InitCUDA(void){return true;}

   #else
   bool InitCUDA(void)
  {
      int count = 0;
      int i = 0;

      cudaGetDeviceCount(&count);
      if(count == 0) {
       fprintf(stderr, "There is no device./n");
        return false;
       }

     for(i = 0; i < count; i++) {
         cudaDeviceProp prop;
     if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
             if(prop.major >= 1) {
                 break;
              }
       }
      }
    if(i == count) {
            fprintf(stderr, "There is no device supporting CUDA./n");
            return false;
    }
    cudaSetDevice(i);

    printf("CUDA initialized./n");
    return true; 
  }

  #endif
  void matgen(float* a, int lda, int n)
    {
        int i, j;

        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                a[i * lda + j] = (float) rand() / RAND_MAX +
                    (float) rand() / (RAND_MAX * RAND_MAX);
            }
        }
    }
 void matmult(const float* a, int lda, const float* b, int ldb,
        float* c, int ldc, int n)
    {
        int i, j, k;

        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                double t = 0;
                for(k = 0; k < n; k++) {
                    t += a[i * lda + k] * b[k * ldb + j];
                }
                c[i * ldc + j] = t;
            }
        }
    }
    void compare_mat(const float* a, int lda,
        const float* b, int ldb, int n)
    {
        float max_err = 0;
    float average_err = 0;
        int i, j;

        for(i = 0; i < n; i++) {
            for(j = 0; j < n; j++) {
                if(b[i * ldb + j] != 0) {
                    float err = fabs((a[i * lda + j] -
                        b[i * ldb + j]) / b[i * ldb + j]);
                    if(max_err < err) max_err = err;
                    average_err += err;
                }
            }
        }

        printf("Max error: %g Average error: %g\n",
            max_err, average_err / (n * n));
    }
     #define NUM_THREADS 256
        __global__ static void matMultCUDA(const float* a, size_t lda,
        const float* b, size_t ldb, float* c, size_t ldc, int n)
    {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int idx = bid * blockDim.x + tid;
        const int row = idx / n;
        const int column = idx % n;
        int i;

        if(row < n && column < n) {
            float t = 0;
            for(i = 0; i < n; i++) {
                t += a[row * lda + i] * b[i * ldb + column];
            }
            c[row * ldc + column] = t;
        }
    }
    clock_t matmultCUDA(const float* a, int lda,
        const float* b, int ldb, float* c, int ldc, int n)
    {
        float *ac, *bc, *cc;
        clock_t start, end;

        start = clock();
        cudaMalloc((void**) &ac, sizeof(float) * n * n);
        cudaMalloc((void**) &bc, sizeof(float) * n * n);
        cudaMalloc((void**) &cc, sizeof(float) * n * n);

        cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda,
            sizeof(float) * n, n, cudaMemcpyHostToDevice);
        cudaMemcpy2D(bc, sizeof(float) * n, b, sizeof(float) * ldb,
            sizeof(float) * n, n, cudaMemcpyHostToDevice);

        int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
        matMultCUDA<<<blocks * n, NUM_THREADS>>>
            (ac, n, bc, n, cc, n, n);

        cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n,
        sizeof(float) * n, n, cudaMemcpyDeviceToHost);

        cudaFree(ac);
        cudaFree(bc);
        cudaFree(cc);

        end = clock();

        return end - start;
    }

 int main()
    {
        float *a, *b, *c, *d;
        int n = 1000;

        if(!InitCUDA()) return 0;

        a = (float*) malloc(sizeof(float) * n * n);
        b = (float*) malloc(sizeof(float) * n * n);
        c = (float*) malloc(sizeof(float) * n * n);
        d = (float*) malloc(sizeof(float) * n * n);

        srand(0);

        matgen(a, n, n);
        matgen(b, n, n);

        clock_t time = matmultCUDA(a, n, b, n, c, n, n);

        matmult(a, n, b, n, d, n, n);
        compare_mat(c, n, d, n, n);

        double sec = (double) time / CLOCKS_PER_SEC;
        printf("Time used: %.2f (%.2lf GFLOPS)\n", sec,
            2.0 * n * n * n / (sec * 1E9));

        return 0;
    }
