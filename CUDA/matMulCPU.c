#include <stdio.h>
#include <math.h>
#define ARR_DIM (2048*2048)
#define WIDTH 2048
#define HEIGHT 2048
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;
float* generateRandArray(){
	printf("HERE\n");
	float* a = (float *)malloc(ARR_DIM*sizeof(float));
	int i = 0;
	for(i; i < ARR_DIM; i++){
		a[i] = 1.654981;
	}
	return a;
}
int main(){
	Matrix A,B,C;

	A.width = WIDTH;
	A.height = HEIGHT;
	B.width=WIDTH;
	B.height = HEIGHT;
	C.width = WIDTH;
	C.height = HEIGHT;
	A.elements = generateRandArray();
	B.elements = generateRandArray();
	C.elements = (float *)malloc(ARR_DIM*sizeof(float));
	int i = 0;
	int j = 0;
	int e = 0;
	for(i; i < WIDTH; i++){	
		printf("%d\n",i);	
		j=0;
		for(j; j < HEIGHT; j++){
			//Here for every element in Carray
			float Cvalue = 0;
		        int row = i;
		        int col = j;
		        for (int e = 0; e < WIDTH; ++e)
				Cvalue += A.elements[row * WIDTH + e]* B.elements[e * WIDTH + col];
		        C.elements[row * WIDTH + col] = Cvalue;
		}

	}
	printf("%f\n", C.elements[2048*2048-1]);
	free(A.elements);
	free(B.elements);
	free(C.elements);
	return 0;

}
