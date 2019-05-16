// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
//Since this is matrix multiplication, A.width must be equal to B.height and the final matrix has height A.height and width B.width
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 32 //32 is the max since it means there are 32*32=1024 threads operating for each block
#define ARR_DIM (2048*2048)
#define WIDTH 2048
#define HEIGHT 2048

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(float *A, float *B, float *C);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(Matrix *A, Matrix *B, Matrix *C)
{
    
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A->width; d_A.height = A->height;
    size_t size = A->width * A->height * sizeof(float);
    cudaMalloc((void **)&d_A.elements, size);
    cudaMemcpy(d_A.elements, A->elements, size,
               cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = B->width; d_B.height = B->height;
    size = B->width * B->height * sizeof(float);
    cudaMalloc((void **)&d_B.elements, size);
    cudaMemcpy(d_B.elements, B->elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C->width; d_C.height = C->height;
    size = C->width * C->height * sizeof(float);
    cudaMalloc((void **)&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B->width / dimBlock.x, A->height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A.elements, d_B.elements, d_C.elements);
    //This is causing a segmentation fault
    // Read C from device memory
    cudaMemcpy(C->elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

	

}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(float *A, float *B, float *C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < WIDTH; ++e)
        Cvalue += A[row * WIDTH + e]
                * B[e * WIDTH + col];

    C[row * WIDTH + col] = Cvalue;
    //printf("VAL=%f row=%d col=%d\n", Cvalue, row, col); 
}
__global__ void cudaRandomize(float *arr){
	float val;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
	arr[row*WIDTH + col] = 1.654981;
}
float* generateRandArray(){
	float* a = (float *)malloc(ARR_DIM*sizeof(float));
	int i = 0;
	for(i; i < ARR_DIM; i++){
		a[i] = rand()%100 + 1;
	}
	return a;
}
int main(){
	Matrix *A,*B,*C;
	A = (Matrix *)malloc(sizeof(Matrix));
	B = (Matrix *)malloc(sizeof(Matrix));
	C = (Matrix *)malloc(sizeof(Matrix));

	A->width = WIDTH;
	A->height = HEIGHT;
	B->width=WIDTH;
	B->height = HEIGHT;
	C->width = WIDTH;
	C->height = HEIGHT;
	A->elements = (float *)malloc(ARR_DIM*sizeof(float));
	B->elements = (float *)malloc(ARR_DIM*sizeof(float));
	

	float * d_A, *d_B;
	size_t size = A->width * A->height * sizeof(float);
	cudaMalloc((void**)&d_A, size);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(A->width / dimBlock.x, A->height / dimBlock.y);
        cudaRandomize<<<dimGrid, dimBlock>>>(d_A);
        cudaMemcpy(A->elements, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);	

	size = B->width * B->height * sizeof(float);
	cudaMalloc((void**)&d_B, size);
	dim3 dimBlock2(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid2(B->width / dimBlock.x, B->height / dimBlock.y);
        cudaRandomize<<<dimGrid2, dimBlock2>>>(d_B);
        cudaMemcpy(B->elements, d_B, size, cudaMemcpyDeviceToHost);
	cudaFree(d_B);	

	C->elements = (float *)malloc(ARR_DIM*sizeof(float));
	for(int i = 0; i < 500; i++){
		printf("i=%d\n",i);		
		MatMul(A,B,C);
	}

	free(A->elements);
	free(B->elements);
	free(C->elements);
	return 0;
}
