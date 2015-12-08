#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <curand.h>
#include <unistd.h>
#include <curand_kernel.h>

int* generateAdjMatrix(int count, int* adjMatrix);
void printAdjMatrix(int count, int* adjMatrix);
int* multiplyMatrix(int* in,int* in2, int num,int count);
void CPUMatrixMultiplication(int count, int path, int* matrix);

//this function needs to be renamed since it does not do matrix multiplication
//it just does cuda mem stuff to prep for matrix multiplication
void GPUMatrixMultiplication(int count, int path, int* matrix, int start, int end);

#define NUMTHREADS 1024;
int fTime = 0;


//This is the main function
int main(int argc, char* argv[]){
	int count;
	int path;
	int* adjMatrix = NULL;
	int gpuOnly = 0;

	//start and end of the path
	int start, end;
	
	//If there is more than 2 parameters
	opterr = 0;
	int c;

	while((c = getopt (argc, argv, "dgtc:p:")) != -1){
		switch (c)
		{
			case 'd':
				count = 10;
				path = 2;
				break;
			case 'g':
				gpuOnly = 1;
				break;
			case 't':
				fTime = 1;
				break;
			case 'c':
				count = atoi(optarg);
				break;
			case 'p':
				path = atoi(optarg);
				break;
			case '?':
				if (optopt == 'c' || optopt == 'p'){
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				}
				else{
					fprintf(stderr,"Usage:\n-t: print time only\n-d: default count to 10, path to 2\n-g: preform calculations on GPU only\n-c <num of nodes>\n-p <num of paths>");
				}
				return 1;
			default:
				return 2;
		}
	
	}
	path--;

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
	
	//starting Node is 0 and ending node is 3
	//temporary test please add commandline args for start and end
	start = 0;
	end = 3;
	//Compute the GPU function
	GPUMatrixMultiplication(count, path, adjMatrix, start, end);	
	return 0;
}

//takes in a matrix and returns all paths as an int array
//the int array paths should be a 2d array but for ease of use with cuda it is
//an linear array just like the matrix
__global__ void traverse(int* matrix, int* paths, int count, int start, int end, int length){
	int element = blockIdx.x*blockDim.x + threadIdx.x;

	//curand = cuda random for random number generation
	curandState state;
	curand_init((unsigned long)element, 0, 0, &state);
	//current length of the path
	int currLength = 0;
	//current Node in the graph
	int currNode = start;
	//start is always the first Node
	paths[element*length + currLength] = currNode;
	currLength++;
	while(currLength != length){
		if(currLength == length-1){
			//this case is to assist in our bruteforce algorithm
			//if we can only make one more transition instead of doing 
			//a random transition we try to move to the end point
			if(matrix[currNode * count + end] == 1){
				currNode = end;
				paths[element*length + currLength] = currNode;
				currLength++;
			}else{//if we can't connect to the endpoint we restart
				currLength = 1;
				currNode = start;
				paths[element*length + 0] = currNode;
			}	
		}else{
			int randIdx;
			do{
				randIdx = curand(&state) % count;
			}while(matrix[currNode * count + randIdx] != 1);
		        currNode = randIdx;
			paths[element*length + currLength] = currNode;
        		currLength++;
		}
	}

}
__global__ void multiply(int* matrixA, int* multipliedMatrix, int paths, int count){
        int element = blockIdx.x*blockDim.x + threadIdx.x;
	int i, j;
	for(i=0; i < paths; i++)
	{
		int sum = 0;
		int col = element % count;
		int row = element / count;
		for(j=0; j < count; j++){
			sum+=matrixA[count*j + col]*multipliedMatrix[row*count + j];
		}
		__syncthreads();
		multipliedMatrix[element] = sum;
	}
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
void GPUMatrixMultiplication(int count, int path, int* matrix, int nodeA, int nodeB){

	int numThreads = NUMTHREADS;
		
	//number at index of start*count + end this gets the total number of
	//paths that exist	
	int numPaths;

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
        cudaMemcpy(gpuMM, matrix, (count*count*sizeof(int)), cudaMemcpyHostToDevice);
	
	//Preform the multiplied matrix function on gpuMatrix and store into gpuMM
	multiply<<<numBlocks, numThreads>>>(gpuMatrix, gpuMM, path, count);
	
	//Copy gpuMM from the GPU to the CPU in multipiedMatrix
	cudaMemcpy(multipliedMatrix, gpuMM, (count*count*sizeof(int)), cudaMemcpyDeviceToHost);
	
	
        gettimeofday(&end, NULL);
        long microseconds = end.tv_usec - start.tv_usec;

        //Print the multiplied matrix, copied earlier from the GPU
        printf("GPU Generated matrix:\n");
        if (!fTime){
                printAdjMatrix(count, multipliedMatrix);
        }
        printf("Took %li microseconds to compute\n", microseconds);
	printf("\n");	
	//gets num paths and if no paths exists it shows that and exits
	numPaths = multipliedMatrix[(nodeA * count) + nodeB];
	if (numPaths == 0){
		printf("No paths exist from %d to %d\n", nodeA, nodeB);
		return;
	}else{
		path+=2;
		int* paths = (int *)malloc(numPaths * sizeof(int) * (path));
		int* gpuPaths;
		cudaMalloc(&gpuPaths, (numPaths*path*sizeof(int)));
		traverse<<<numPaths, 1>>>(gpuMatrix, gpuPaths, count, nodeA, nodeB, path);
		cudaMemcpy(paths, gpuPaths, (numPaths*(path)*sizeof(int)), cudaMemcpyDeviceToHost);	
		int i;
		for(i = 0; i < numPaths; i++){
			int j;
			for(j = 0; j < path; j++){
				printf("%d ", paths[i*path + j]);
			}
			printf("\n");
		}
	}
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

