#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
  
/*******************************************************************************/
__global__ void mult(const float *A, const float *B, float *C){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0;
    for(int j=0; j<N; j++){
      sum = 0;
      for(int i=0; i<N; i++){
          sum += *(A + index*N +i) * *(B+ i*N + j);
      }
      *(C + index*N + j) = sum;
    }
}
/*******************************************************************************/

void printMatrix(float *A){
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            printf("%f ", *(A+i*N+j));
        }
        printf("\n");
    }
}

int verify(float *A, float *B, float *C){
    int i, j, k;
    float sum;

    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += *(A + i*N + k) * *(B + k*N + j);
            }
            if(fabs(*(C + i*N + j) - sum) > 1e-4){
                printf("ERROR. Verify failed at element: %d\n", i*N+j);
                printf("Expected: %f. Found: %f\n\n", sum, *(C + i*N + j));
                return 0;
            }
        }
    }
    return 1;
}

float* initMatrix(float *A, int fill){
    A = (float*)malloc(N*N*sizeof(float*));
    if(fill)
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          *(A + i*N +j) = (float)rand()/RAND_MAX;
        }
      }
    return A;
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = N;
    size_t size = N*N*sizeof(float);
    printf("[Matrix multiplication of %dx%d elements]\n", numElements, numElements);

    // Allocate the host input vector A
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    h_A = initMatrix(h_A, 1);
    h_B = initMatrix(h_B, 1);
    h_C = initMatrix(h_C, 0);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
 
    // Allocate the device input vector A
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    float *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector C
    float *d_C = NULL;
    err = cudaMalloc((void **)&d_C, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
 
    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    int threadsPerBlock = 64;
    int blocksPerGrid = N/threadsPerBlock;

    mult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);
    err = cudaGetLastError();
 
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("\nCopy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    //verify(h_A, h_B, h_C);

    /*FREE ALL*/
 
    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    printf("Done\n");
    return 0;
}
