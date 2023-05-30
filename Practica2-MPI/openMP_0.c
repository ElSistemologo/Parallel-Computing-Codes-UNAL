#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

#define N 8

int verify(float* A, float* B, float* C){
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            float sum = 0.0;
            for(int k=0; k<N; k++) {
                sum += A[i*N+k]*B[k*N+j];
            }
            if(fabs(C[i*N+j] - sum) > 1e-5){
                printf("ERROR. Verify failed at element: %d, %d\n", i, j);
                printf("Expected: %f, Found: %f\n", sum, C[i*N+j]);
                return 0;
            }
        }
    }
    return 1;
}

void printMatrix(float *A){
    for(int row = 0; row<N; row++){
      for(int column = 0; column<N; column++){
        printf("%f ", *(A + row*N + column));
      }
      printf("\n");
    }
}

void initMatrix(float **A, int fill) {
    *A = (float *)malloc(N * N * sizeof(float));
    if (fill) {
        for (int i = 0; i < N * N; i++) {
            (*A)[i] = (float)rand() / RAND_MAX;
        }
    }
}

void multiply(float *A, float *B, float *C, int size, int extra) {
    int i, j, k;
    float sum;

    #pragma omp parallel for private(j, k, sum)
    for (i = 0; i < N/size + extra; i++){
        for (j = 0; j < N; j++){
            sum = 0.0;
            for (k = 0; k < N; k++){
                sum += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    float *A, *B, *C, *aa, *cc;
    
    initMatrix(&A, 1);
    initMatrix(&B, 1);
    initMatrix(&C, 0);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int scatter_size = N/size;
    int extra = (rank < (N % size)) ? 1 : 0;

    aa = (float *)malloc((scatter_size + extra)*N*sizeof(float));
    cc = (float *)malloc((scatter_size + extra)*N*sizeof(float));

    MPI_Scatter(A, scatter_size*N, MPI_FLOAT, aa, (scatter_size + extra)*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Bcast(B, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); 

    multiply(aa, B, cc, size, extra);

    MPI_Gather(cc, (scatter_size + extra)*N, MPI_FLOAT, C, (scatter_size + extra)*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        if (verify(A, B, C)) {
            printMatrix(C);
        }
        
        free(A);
        free(B);
        free(C);
    }

    MPI_Finalize();

    return 0;
}
