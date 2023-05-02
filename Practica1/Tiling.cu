#include <iostream>

const int N = 1024;
const int SHMEM_SIZE = 1024;

// Threads per block
int THREADS = 4;

__global__ void matrixMul(const float *A, const float *B, float *C) {
  // Compute row and column index
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float s_a[SHMEM_SIZE];
  __shared__ float s_b[SHMEM_SIZE];

  float sum = 0;

  // Sweep tile across matrix
  for (int i = 0; i < N; i += blockDim.x) {
    // Load in elements for this tile
    s_a[threadIdx.y * blockDim.x + threadIdx.x] = A[row * N + i + threadIdx.x];
    s_b[threadIdx.y * blockDim.x + threadIdx.x] = B[i * N + threadIdx.y * N + col];

    // Wait for both tiles to be loaded in before doing computation
    __syncthreads();

    // Matrix multiplication
    for (int j = 0; j < blockDim.x; j++) {
      sum += s_a[threadIdx.y * blockDim.x + j] * s_b[j * blockDim.x + threadIdx.x];
    }

    // Wait for all threads loading new
    __syncthreads();
  }

  C[row * N + col] = sum;
}

// Verify result on the CPU
void verify(float *A, float *B, float *C) {
  int offset;
  for (int i = 0; i < N; i++) {
    offset = i*N;
    for (int j = 0; j < N; j++) {
      float sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[offset + k] * B[k * N + j];
      }
      if(fabs(sum - C[offset + j]) > 1e-4){
        printf("ERROR. Verify failed at element: %d\n", offset+j);
        printf("Expected: %f Found: %f\n", sum, C[offset + j]);
        return;
      }
    }
  }
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

int main() {

  size_t bytes = N * N * sizeof(float);

  // Host matrices
  float *h_A = NULL;
  float *h_B = NULL;
  float *h_C = NULL;

  h_A = initMatrix(h_A, 1);
  h_B = initMatrix(h_B, 1);
  h_C = initMatrix(h_C, 0);

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, bytes);
  cudaMalloc(&d_B, bytes);
  cudaMalloc(&d_C, bytes);

  // Copy data from host to the device
  cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

  // Blocks per grid
  int BLOCKS = N / THREADS;

  dim3 threads(THREADS, THREADS);
  dim3 blocks(BLOCKS, BLOCKS);

  // Launch kernel
  matrixMul<<<blocks, threads>>>(d_A, d_B, d_C);

  // Copy back to the host
  cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

  // Check result
  verify(h_A, h_B, h_C);

  printf("DONE\n");

  // Free memory on device
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}