#include <stdio.h>
#include <stdlib.h>

int n = 10; // Global variable to represent the size of the square matrices

void printMatrix(float **A){
    for(int row = 0; row<n; row++){
      for(int column = 0; column<n; column++){
        printf("%f ", *(*(A+row)+column)); // Access and print matrix elements using pointer arithmetic
      }
      printf("\n"); // Print new line after each row
    }
}

float** initMatrix(float **A, int fill){
    A = (float **) malloc(n * sizeof(float*)); // Allocate memory for rows of the matrix
    for(int row = 0; row<n; row++) {
        A[row] = (float *) malloc(n * sizeof(float)); // Allocate memory for columns of the matrix
        if(fill) for(int column=0; column<n; column++) *(*(A+row)+column) = (float)rand()/RAND_MAX; // Fill matrix with random float values if fill is set to 1
    }
    return A;
}

float** multMatrix(float **A, float **B, float **C){
  float sum;
  for(int iter=0; iter<n; iter++){
    for(int i=0; i<n; i++){
      sum = 0;
      for(int j=0; j<n; j++){
        sum += *(*(A+iter)+j) * *(*(B+j)+i); // Perform matrix multiplication using pointer arithmetic
      }
      *(*(C+iter)+i) = sum; // Store the result in the output matrix C
    }
  }
  return C;
}

int main(){

    float **matrixA, **matrixB, **matrixC;

    matrixA = initMatrix(matrixA, 1); // Initialize matrixA with random values
    matrixB = initMatrix(matrixB, 1); // Initialize matrixB with random values
    matrixC = initMatrix(matrixC, 0); // Initialize matrixC with zeros

    multMatrix(matrixA, matrixB, matrixC); // Perform matrix multiplication

    printMatrix(matrixC); // Print the resulting matrix C

    // Free dynamically allocated memory to prevent memory leaks
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

