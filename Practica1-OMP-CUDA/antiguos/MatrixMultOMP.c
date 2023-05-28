#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 1024

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
    int i, j, k;
    float sum[4];
    float *Ap;
    int id;
    omp_set_num_threads(4);

      #pragma omp parallel for private(Ap, id, j, k, i) 
      for (i = 0; i < N; i++) {
        id = omp_get_thread_num();
        Ap = A + i*N ;
        for (j = 0; j < N; j++) {
            sum[id] = 0.0;
            for (k = 0; k < N; k++) {
                sum[id] += 0;//*(Ap + k) * *(B + k*N + j);
            }
            *(C + i*N + j) = sum[id];
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

    multiply(A, B, C);

    // Imprimir la matriz resultante C
    //printMatrix(C);

    free(A);
    free(B);
    free(C);

    return 0;
}
