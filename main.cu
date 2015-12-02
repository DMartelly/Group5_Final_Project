#include <stdio.h>
#include <stdlib.h>

int* generateAdjMatrix(int count, int* adjMatrix);
void printAdjMatrix(int count, int* adjMatrix);
int* multiplyMatrix(int* in,int* in2, int num,int count);
void matrixMultiplication(int count, int path, int* matrix);

#define NUMTHREADS 1024;

//This is the main function
int main(int argc, char* argv[]){
	int* adjMatrix = NULL;
//	int* gpuMatrix;
//	int* multipliedMatrix = NULL;
	int count;
	int path;
	if(argc > 3){
		 fprintf(stderr,"Usage: %s <node count>\n",argv[0]);
		 return 1;
	}
	if(argc==1){
	 	count = 10;
	 	path = 2;
	}
	else if(argc == 2){
		count = atoi(argv[1]);
		path = 2;
	}
	else{
	 	count = atoi(argv[1]);
		path = atoi(argv[2]);
	}
	adjMatrix = generateAdjMatrix(count, adjMatrix);
	matrixMultiplication(count, path, adjMatrix);	
	return 0;
}

__global__ void multiply(int* matrix, int* multipliedMatrix, int count){
        int element = blockIdx.x*blockDim.x + threadIdx.x;
	int sum = 0;
	int i;
	int col = element % count;
	int row = element / count;
	for(i=0; i < count; i++){
		sum+=matrix[count*i + col]*matrix[row*count + i];
	}
	multipliedMatrix[element] = sum;
}


//Prep for calling the gpu matrix multiplication function

void matrixMultiplication(int count, int path, int* matrix){
	int numThreads = NUMTHREADS;
	int* gpuMatrix;
	int* multipliedMatrix = (int*)malloc(count*count*sizeof(int));
	int* gpuMM;//gpu multiplied matrix
	int numBlocks;
	numBlocks = (count*count)/numThreads + 1;
        cudaMalloc(&gpuMatrix, (count*count*sizeof(int)));
	cudaMalloc(&gpuMM, (count*count*sizeof(int)));
        cudaMemcpy(gpuMatrix, matrix, (count*count*sizeof(int)), cudaMemcpyHostToDevice);

	multiply<<<numBlocks, numThreads>>>(gpuMatrix, gpuMM, count);

	cudaMemcpy(matrix, gpuMatrix, (count*count*sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy(multipliedMatrix, gpuMM, (count*count*sizeof(int)), cudaMemcpyDeviceToHost);
        printAdjMatrix(count, matrix);
	printf("\n");
       // multipliedMatrix = multiplyMatrix(matrix,matrix,path,count);
        printAdjMatrix(count, multipliedMatrix);
	multipliedMatrix = multiplyMatrix(matrix,matrix,path,count);
	printf("\n");
	printAdjMatrix(count, multipliedMatrix);
}

//Creates an adjacency matrix
//	count - the size of the matrix. the size is count X count)
//	matrix - a pointer to an adjacency Matrix
int* generateAdjMatrix(int count, int* matrix){
	matrix = (int *)malloc(count*count*sizeof(int));
	int i, j;

	//Set the random seed to the current time
	srand(time(NULL));

	//Create a random adjacency matrix using rand
	for (i = 0; i < count; i++){
		for(j = 0; j < count; j++){
			if(i != j){
				int randomResult = rand() % 2;
				matrix[(i *count) + j] = randomResult;
				matrix[(j *count) + i] = randomResult;
			}
		}
	}
	return matrix;
}

//Returns a cross multiplied matrix of two matrixies
//	in - the first matrix
//	in2 - the second matrix
//	num - the number of times we do the multiplacation
//	size -
int* multiplyMatrix(int* in,int* in2,int num, int count){
	if(num==0)
		return in2;
	int arr[count];
	int i,j,k;
	int z,n=0;
	int* out = (int *) malloc(sizeof(int)*count*count);
	
	for(i=0; i<count; i++){
		for(j=0; j<count; j++){
			for(k=0;k<count;k++){
				arr[k] = in[(i*count)+k] * in2[(k*count)+j];
			}
			for(z=0;z<count;z++){
				n+=arr[z];	
			}
			out[(i*count)+j] = n;
			n=0;
		}
	}
	return multiplyMatrix(in,out,num-1,count);
}

//Prints the adjacency matrix to stdout
void printAdjMatrix(int count, int* matrix){
	int i;
	for (i = 0; i < count; i++){
		int j;
		for (j = 0; j < count; j++){
			printf("%i  ", matrix[(i * count) + j]);
		} 
		printf("\n");
	}
}

