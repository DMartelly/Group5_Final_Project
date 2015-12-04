#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int* generateAdjMatrix(int count, int* adjMatrix);
void printAdjMatrix(int count, int* adjMatrix);
int* multiplyMatrix(int* in,int* in2, int num,int count);
void CPUMatrixMultiplication(int count, int path, int* matrix);
void GPUMatrixMultiplication(int count, int path, int* matrix);

#define NUMTHREADS 1024;

int fTime = 0;

//This is the main function
int main(int argc, char* argv[]){
	int count;
	int path;
	int* adjMatrix = NULL;
	int gpuOnly=0;
	
	//If there is more than 2 parameters
	if(argc == 1){
		 fprintf(stderr,"Usage:\n%s <node count> <num of paths> [-t]\n%s [-d] [-t]\n", argv[0], argv[0]);
		 return 1;
	}
	//If default '-d' is passed
	if(strncmp(argv[1], "-d", 2) == 0){
	 	count = 10;
	 	path = 2;
		//If time '-t' is passed
		if(argc > 2){
			if(strncmp(argv[2], "-t", 2) == 0){
		 		fTime = 1;
			}
			if(strcmp(argv[2],"-g")==0 || strcmp(argv[3],"-g")==0) gpuOnly =1;
		}
	}
	else{
		count = atoi(argv[1]);
		path = atoi(argv[2]);
		if(argc > 3){
			//If time '-t' is passed
			if(strncmp(argv[3], "-t", 2) == 0){
		 		fTime = 1;
			}
			//If gpu only '-g' is passed
			if(strcmp(argv[3],"-g")==0 || strcmp(argv[4],"-g")==0) gpuOnly=1;
		}
	}
	//adjMatrix now equals a new Random adjancency  Matrix
	adjMatrix = generateAdjMatrix(count, adjMatrix);

	//Print the generated adjancency matrix
	if (!fTime){
		printf("Generated Adjancency Matritx:\n");
		printAdjMatrix(count, adjMatrix);
		printf("\n");
	}

	//Compute the CPU function
	if(!gpuOnly) CPUMatrixMultiplication(count, path, adjMatrix);

	//Compute the GPU function
	GPUMatrixMultiplication(count, path, adjMatrix);	
	return 0;
}

__global__ void multiply(int* matrixA, int* matrixB, int* multipliedMatrix, int count){
        int element = blockIdx.x*blockDim.x + threadIdx.x;
	int sum = 0;
	int i;
	int col = element % count;
	int row = element / count;
	for(i=0; i < count; i++){
		sum+=matrixA[count*i + col]*matrixB[row*count + i];
	}
	multipliedMatrix[element] = sum;
}

//CPU matrix multiplication function
void CPUMatrixMultiplication(int count, int path, int* matrix){
	
	//Create the time interval
	struct timeval start, end;

	//Start time
	gettimeofday(&start, NULL);
	
	//The completed multiplied matrix
	int* multipliedMatrix =  multiplyMatrix(matrix, matrix, path, count);	
	
	//End time
	gettimeofday(&end, NULL);
	long microseconds = end.tv_usec - start.tv_usec;
	
	//Print the multiplied matrix
	printf("CPU Generated matrix:\n");
	if (!fTime){
		printAdjMatrix(count, multipliedMatrix);
	}
	printf("Took %li microseconds to compute\n\n", microseconds);

}
//GPU matrix multiplication function
void GPUMatrixMultiplication(int count, int path, int* matrix){
	
	int numThreads = NUMTHREADS;
	
	//An adjacency matrix on the GPU
	int* gpuMatrix;

	//The multiplied matrix on the GPU
	int* gpuMM;

	//A matrix that will store gpuMM on the CPU
	int* multipliedMatrix = (int*)malloc(count*count*sizeof(int));

	//The number of GPUS
	int numBlocks = (count*count)/numThreads + 1;

	//Allocate the memory on the GPU
        cudaMalloc(&gpuMatrix, (count*count*sizeof(int)));
	cudaMalloc(&gpuMM, (count*count*sizeof(int)));

	//Create the time interval
	struct timeval start, end;

	//Start time
	gettimeofday(&start, NULL);

	//Copy the input matrix from the CPU to the GPU (matrix -> gpuMatrix)
        cudaMemcpy(gpuMatrix, matrix, (count*count*sizeof(int)), cudaMemcpyHostToDevice);

	//Preform the multiplied matrix function on gpuMatrix and store into gpuMM
	multiply<<<numBlocks, numThreads>>>(gpuMatrix, gpuMM, count);
	
	//Copy gpuMM from the GPU to the CPU in multipiedMatrix
	cudaMemcpy(multipliedMatrix, gpuMM, (count*count*sizeof(int)), cudaMemcpyDeviceToHost);

	//End time
	gettimeofday(&end, NULL);
	long microseconds = end.tv_usec - start.tv_usec;
        
	//Print the multiplied matrix, copied earlier from the GPU
        printf("GPU Generated matrix:\n");
	if (!fTime){
		printAdjMatrix(count, multipliedMatrix);
	}
	printf("Took %li microseconds to compute\n", microseconds);
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
			printf("%3i ", matrix[(i * count) + j]);
		} 
		printf("\n");
	}
}

