all: sequential.c ompArray.c ompMatrix.c cuda.cu Tiling.cu
	gcc sequential.c -o sequential
	gcc ompArray.c -o ompArray -fopenmp
	gcc ompMatrix.c -o ompMatrix -fopenmp
	nvcc cuda.cu -o cuda
	nvcc Tiling.cu -o Tiling

clean:
	rm sequential ompArray ompMatrix cuda Tiling

