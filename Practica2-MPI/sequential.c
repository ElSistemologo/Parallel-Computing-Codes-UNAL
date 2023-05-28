#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
    // Function implementation modified to accept a double pointer
    *A = (float *)malloc(N * N * sizeof(float));
    if (fill) {
        for (int i = 0; i < N * N; i++) {
            (*A)[i] = (float)rand() / RAND_MAX;
            //(*A)[i] = i%9 + 1;
        }
    }
}

void multiply(float *aa, float *B, float *cc, int size) {
    int i, j, k;
    float sum;

    for (k = 0; k < N/size; k++){
        for (i = 0; i < N; i++){
            sum = 0;
            for (j = 0; j < N; j++){
                sum = sum + aa[k*N+j] * B[j*N+i];            
            }
            cc[k*N+i] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size, i, j, k;
    float sum;
    float *A, *B, *C, *aa, *cc;
    
    initMatrix(&A, 1);
    initMatrix(&B, 1);
    initMatrix(&C, 0);
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    aa = (float *)malloc(N*N*sizeof(float)/size);
    cc = (float *)malloc(N*N*sizeof(float)/size);

    // Scatter matrix A to all processes
    MPI_Scatter(A, N*N/size, MPI_FLOAT, aa, N*N/size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    MPI_Barrier(MPI_COMM_WORLD); 

    multiply(aa, B, cc, size);

    // Gather the final result to process 0
    MPI_Gather(cc, N*N/size, MPI_FLOAT, C, N*N/size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    

    if (rank == 0) {
        // Verify and print the result
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
