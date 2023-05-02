#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

int verify(float *C){
  FILE *file = fopen("C.txt", "r");

  int i=0;
  float num;
  while(fscanf(file, "%f\t", &num) > 0) {
      if(fabs(*(C +i) - num) > 1e-5){
        printf("ERROR. Failed verify at element: %d\n", i);
        return 0;
      }
      i++;
  }
  fclose(file);
  
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

float* initMatrix(float *A, int fill){
    A = (float*)malloc(N*N*sizeof(float*));
    if(fill)
      for(int i=0; i<N*N; i++){
        *(A + i) = (float)rand()/RAND_MAX;
      }
    return A;
}

void multiply(float *A, float *B, float *C)
{
    int i, j, k, offset;
    float sum;

    for (i = 0; i < N; i++) {
      offset = i*N;
      for (j = 0; j < N; j++) {
        sum = 0.0;
        for (k = 0; k < N; k++) {
            sum += *(A + offset + k) * *(B + k*N + j);
        }
        *(C + offset + j) = sum;
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