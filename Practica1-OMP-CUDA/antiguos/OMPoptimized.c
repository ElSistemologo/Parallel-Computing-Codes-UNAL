#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024
#define BLOCK_SIZE 32

void printMatrix(float *A){
    for(int row = 0; row<N; row++){
      for(int column = 0; column<N; column++){
        printf("%f ", *(A + row*N + column));
      }
      printf("\n");
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

void multiply(float *A, float *B, float *C)
{
    int i, j, k, i_block, j_block, k_block;
    float sum;
    float *Ap, *Bp, *Cp;

    omp_set_num_threads(4);

    #pragma omp parallel for private(Ap, Bp, Cp, i, j, k, i_block, j_block, k_block, sum)
    for (i_block = 0; i_block < N; i_block += BLOCK_SIZE) {
        for (j_block = 0; j_block < N; j_block += BLOCK_SIZE) {
            for (k_block = 0; k_block < N; k_block += BLOCK_SIZE) {
                for (i = i_block; i < i_block + BLOCK_SIZE; i++) {
                    Ap = A + i*N + k_block;
                    Cp = C + i*N + j_block;
                    for (j = j_block; j < j_block + BLOCK_SIZE; j++) {
                        sum = 0.0;
                        Bp = B + k_block*N + j;
                        for (k = 0; k < BLOCK_SIZE; k++) {
                            sum += *(Ap++) * *(Bp);
                            Bp += N;
                        }
                        *(Cp++) += sum;
                    }
                }
            }
        }
    }
}

int main()
{
    int i, j;
    float *A, *B, *C;

    A = initMatrix(A, 1);
    B = initMatrix(B, 1);
    C = initMatrix(A, 0);

    double start_time = omp_get_wtime();
    multiply(A, B, C);
    double end_time = omp_get_wtime();

    printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);

    // Imprimir la matriz resultante C
    //printMatrix(C);

    free(A);
    free(B);
    free(C);

    return 0;
}
