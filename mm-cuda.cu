/**
 * 
 * Matrix Multiplication - CUDA for GPUs
 *
 * CS3210
 *
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

int size;
#define BLOCK_SIZE 32
typedef struct
{
	float ** element;
} matrix;


long long wall_clock_time()
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/**
 * Allocates memory for a matrix of size SIZE
 * The memory is allocated row-major order, i.e. 
 *  elements from the same row are allocated at contiguous 
 *  memory addresses.
 **/
void allocate_matrix(matrix* m)
{
	int i;
	cudaError_t rc;
	
	// allocate array for all the rows
	rc = cudaMallocManaged((void**)&(m->element), sizeof(float*) * size);
	if (rc != cudaSuccess)
	{
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
		exit(1);
	}
	
	// allocate an array for each row of the matrix
	for (i = 0; i < size; i++)
	{
		rc = cudaMallocManaged((void**)&(m->element[i]), sizeof(float) * size);
		if (rc != cudaSuccess)
		{
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(rc));
			exit(1);
		}
	}
}

/**
 * Free the memory allocated for a matrix.
 **/
void free_matrix(matrix* m) {
	int i;
	for (i = 0; i < size; i++)
		cudaFree(m->element[i]);
	cudaFree(m->element);
}

/**
 * Initializes the elements of the matrix with
 * random values between 0 and 9
 **/
void init_matrix(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = rand() % 10;
		}
}

/**
 * Initializes the elements of the matrix with
 * element 0.
 **/
void init_matrix_zero(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
		{
			m.element[i][j] = 0.0;
		}
}


/**
 * Multiplies matrix @a with matrix @b storing
 * the result in matrix @result
 * 
 * The multiplication algorithm is the O(n^3) 
 * algorithm
 */
void mm(matrix a, matrix b, matrix result)
{
	int i, j, k;
	
	// Do the multiplication
	for (i = 0; i < size; i++)
		for (j = 0; j < size; j++)
			for(k = 0; k < size; k++)
				result.element[i][j] += a.element[i][k] * b.element[k][j];
}

/**
 * Each kernel computes the result element (i,j).
 */
__global__ void mm_kernel(matrix a, matrix b, matrix result, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x; 
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k;

	if (i >= size || j >= size)
		return;

	for(k = 0; k < size; k++)
		result.element[i][j] += a.element[i][k] * b.element[k][j];
}
__global__ void mm_improved(matrix a, matrix b, matrix result, int size){
	  /*Use shared memory instead of load from global memory each time.
	  The shared memeory would be shared by all the threads inside one block
	  */
        __shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
        const int tidr = threadIdx.x;
        const int tidc = threadIdx.y;
        const int bidr = blockIdx.x * BLOCK_SIZE;
        const int bidc = blockIdx.y * BLOCK_SIZE;
        float tmp = 0;

        for (int i = 0; i < size; i += BLOCK_SIZE){
            matA[tidr][tidc] = a.element[tidr+bidr][tidc+i];
            matB[tidr][tidc] = b.element[tidr+i][tidc + bidc];
            //Each time, all the threads would form two shared matrix, and use the shared matrix to calculate paritial answers.
		__syncthreads();

            for (int j = 0; j < BLOCK_SIZE; j++){
                tmp += matA[tidr][j] * matB[j][tidc];
            }
		//After all threads finish the calculation of these two sub matrix, they would move on to the next step.
            __syncthreads();
        }
        result.element[tidr+bidr][tidc+bidc] = tmp;


}
void print_matrix(matrix m)
{
	int i, j;
	
	for (i = 0; i < size; i++)
	{
		printf("row %4d: ", i);
		for (j = 0; j < size; j++)
			printf("%6.2f  ", m.element[i][j]);
		printf("\n");
	}
}



void work()
{
	matrix a, b, result1, result2, result3;
	long long before, after;
	int correct, i, j, dim;
	cudaError_t rc;

	// Allocate memory for matrices
	allocate_matrix(&a);
	allocate_matrix(&b);
	allocate_matrix(&result1);
	allocate_matrix(&result2);	
        allocate_matrix(&result3);
	// Initialize matrix elements
	init_matrix(a);
	init_matrix(b);

	// Perform sequential matrix multiplication
	before = wall_clock_time();
	mm(a, b, result1);
	after = wall_clock_time();
        fprintf(stderr, "Matrix multiplication on CPU took %1.2f seconds\n", ((float)(after - before))/1000000000);

	// Perform CUDA matrix  multiplication
	dim3 block(32, 32);			// a block of 32 x 32 CUDA threads
	dim = (size % 32 == 0) ? size / 32 : size / 32 + 1; 
	dim3 grid(dim, dim);	// a grid of CUDA thread blocks
	before = wall_clock_time();
	mm_improved<<<grid, block>>>(a, b, result2, size);
	cudaDeviceSynchronize();
	after = wall_clock_time();
	fprintf(stderr, "Matrix multiplication on GPU took %1.2f seconds\n", ((float)(after - before))/1000000000);
        before = wall_clock_time();
        mm_kernel<<<grid,block>>>(a,b,result3,size);
        cudaDeviceSynchronize();
        after = wall_clock_time();
        fprintf(stderr, "Matrix multiplication on GPU took %1.2f seconds\n", ((float)(after - before))/1000000000);
	// was there any error?
        rc = cudaGetLastError();
        if (rc != cudaSuccess)
                printf("Last CUDA error %s\n", cudaGetErrorString(rc));

	// Compare the results
	correct = 1;
	for (i = 0; correct && i < size; i++)
		for (j = 0; j < size; j++)
			if ((result1.element[i][j] != result3.element[i][j])||(result1.element[i][j] != result2.element[i][j])) {
				correct = 0;
				break;
			}

	if (correct)
		printf("The result matrices are identical!\n");
	else
		printf("Difference in result matrices at element (%d, %d)!\n", i, j);

	free_matrix(&a);
	free_matrix(&b);
	free_matrix(&result1);
	free_matrix(&result2);
}


int main(int argc, char ** argv)
{
	srand(0); 

	printf("Usage: %s <size>\n", argv[0]);
    
	if (argc >= 2)
		size = atoi(argv[1]);
	else
		size = 1024;
		
	fprintf(stderr,"Sequential matrix multiplication of size %d\n", size);
    
	// Multiply the matrices
	work();

	return 0;
}
