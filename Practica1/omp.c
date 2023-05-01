#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 1024

void transpose(float *A, float *B) {
    int i,j;
    for(i=0; i<N; i++) {
        for(j=0; j<N; j++) {
            B[j*N+i] = A[i*N+j];
        }
    }
}

void gemmT_omp(float *A, float *B, float *C) 
{   
    float *B2;
    B2 = (float*)malloc(sizeof(float)*N*N);
    transpose(B,B2);
    #pragma omp parallel
    {
        int i, j, k;
        #pragma omp for
        for (i = 0; i < N; i++) { 
            for (j = 0; j < N; j++) {
                float dot  = 0;
                for (k = 0; k < N; k++) {
                    dot += A[i*N+k]*B2[j*N+k];
                } 
                C[i*N+j ] = dot;
            }
        }

    }
    free(B2);
}

void printMatrix(float *A){
    for(int row = 0; row<N; row++){
      for(int column = 0; column<N; column++){
        printf("%f ", *(A + row*N + column));
      }
      printf("\n");
    }
}

void multiply(float *A, float *B, float *C) 
{   
    #pragma omp parallel
    {
        int i, j, k, offset;
        #pragma omp for
        for (i = 0; i < N; i++) { 
            offset = i*N;
            for (j = 0; j < N; j++) {
                float dot  = 0;
                for (k = 0; k < N; k++) {
                    dot += A[offset+k]*B[k*N+j];
                } 
                C[offset+j ] = dot;
            }
        }

    }
}

float* initMatrix(float *A, int fill){
    A = (float*)malloc(N*N*sizeof(float*));
    if(fill)
      for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
          *(A + i*N +j) = (int)((float)rand()/RAND_MAX*10);
        }
      }
    return A;
}

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

int main() {
    int i;
    float *A, *B, *C, dtime;

    A = initMatrix(A, 1);
    B = initMatrix(B, 1);
    C = initMatrix(A, 0);

    multiply(A,B,C);
    //verify(A, B, C);

    free(A);
    free(B);
    free(C);

    return 0;

}