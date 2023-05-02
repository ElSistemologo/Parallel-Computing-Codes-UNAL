#include <stdio.h>
#include <stdlib.h>
#include <omp.h> // Incluir la librería OpenMP

int n = 1024;
int nThreads = 2; // Definir la cantidad de hilos a ejecutar
void printMatrix(float **A){
    for(int row = 0; row<n; row++){
      for(int column = 0; column<n; column++){
        printf("%f ", *(*(A+row)+column));
      }
      printf("\n");
    }
}

float** initMatrix(float **A, int fill){
    A = (float **) malloc(n * sizeof(float*));
    for(int row = 0; row<n; row++) {
        A[row] = (float *) malloc(n * sizeof(float));
        if(fill) for(int column=0; column<n; column++) *(*(A+row)+column) = (float)rand()/RAND_MAX;
    }
    return A;
}

float** multMatrix(float **A, float **B, float **C){
  float sum;
  #pragma omp parallel for num_threads(nThreads) shared(A, B, C) private(sum) // Paralelizar el bucle externo
  for(int iter=0; iter<n; iter++){
    for(int i=0; i<n; i++){
      sum = 0;
      for(int j=0; j<n; j++){
        sum += *(*(A+iter)+j) * *(*(B+j)+i);
      }
      *(*(C+iter)+i) = sum;
    }
  }
  return C;
}

int main(){

    float **matrixA, **matrixB, **matrixC;

    matrixA = initMatrix(matrixA, 1);
    matrixB = initMatrix(matrixB, 1);
    matrixC = initMatrix(matrixC, 0);

    // int nThreads = 4; // Definir la cantidad de hilos a ejecutar

    multMatrix(matrixA, matrixB, matrixC);

    // printMatrix(matrixC);


    //Free matrix
    for(int row = 0; row<n; row++) {
        free(matrixA[row]);
        free(matrixB[row]);
        free(matrixC[row]);
    }

    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
