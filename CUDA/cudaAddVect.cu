//Adding vectors in parralel
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define N (2048*2048) //Vectors of length 4 million
#define THREADS_PER_BLOCK 512

void generateRand(int *a);
__global__ void addVectCuda(int *da, int *db, int*dc){
	//If threads are used, must index at threadIdx.x
	//For fully indexing threads and block index is calculated by:
	//index = threadIdx.x + blockIdx.x*(Num Blocks)
	//Use blockDim.x for the threads per block
	int index = threadIdx.x + blockIdx.x*(blockDim.x);
	dc[index] = da[index] + db[index];
	printf("i=%d a=%d b=%d c=%d\n", index,da[index], db[index], dc[index]); 	
}

int main(){
	int *a, *b, *c;
	int *da, *db, *dc;
	
	int size = N*sizeof(int);
	
	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);
	generateRand(a);
	generateRand(b);

	//Allocate CUDA memory
	cudaMalloc((void **)&da, size);
	cudaMalloc((void **)&db, size);
	cudaMalloc((void **)&dc, size);

	//Copy inputs to CUDA memory
	cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
	
	//X index controls the number of blocks, Y index controls the number of threads
	addVectCuda<<<(N + THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,THREADS_PER_BLOCK>>>(da, db, dc);

	cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);
	
	//int i = 0;
	//for(i; i < N; i++){
	//	printf("i=%d a=%d b=%d c=%d\n", i,a[i], b[i], c[i]); 	
	//}

	free(a);
	free(b);
	free(c);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	return 0;
	
}

void generateRand(int *a){
	int i = 0;
	for(i; i < N; i++){
		*(a+i) = rand()%100 + 1;
	}
}


