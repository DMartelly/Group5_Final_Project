#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <curand.h>
#include <unistd.h>
#include <curand_kernel.h>

const int NUMTHREADS = 1024;

/*
 * Each matrix is one dimentional, but is treated 2 dimentionaly e.g.
 *
 * 1 0 1
 * 0 1 0
 * 0 1 1
 *
 * EQUALS
 *
 * {1, 0, 1, 0, 1, 0, 0, 1, 1} as an int
 *
 */

//Flags
int fTimeOnly = 0;	//Only print out the computation times
int fGPUOnly = 0;	//Only preform operation on the GPU
int fShowPaths = 0; //List all the generated paths
//------------------------------------------------------------------------------------------

/*
 * generateAdjMatrix - Returns a new random adjacency matrix
 *	count - the size of the matrix. the size is count X count)
 */
long* generateAdjMatrix(int count) {
	long* randomMatrix = (long*)malloc(count*count*sizeof(long));
	int i, j;

	//Set the random seed to the current time
	srand(time(NULL));

	//Create a random adjacency matrix using rand. Nodes do not connect to them selves 
	for (i = 0; i < count; i++){
		for(j = 0; j < count; j++){
			if(i != j){
				long randomResult = rand() % 2;
				randomMatrix[(i *count) + j] = randomResult;
				randomMatrix[(j *count) + i] = randomResult;
			}
		}
	}
	return randomMatrix;
}

/*
 * printAdjMatrix - Prints and adjacency matrix
 *	count - the height of the matrix
 *	matrix - the adjacency matrix
 */
void printAdjMatrix(int count, long* matrix){
	int i;
	for (i = 0; i < count; i++) {
		int j;
		for (j = 0; j < count; j++) {
			printf("%3ld ", matrix[(i * count) + j]);
		}
		printf("\n");
	}
}

/*
 * CPUMultiplyMatrix - copyies the cross multiplied matrix of matrixies 1 and 2, into matrix2(The matrixies must be of the same height and width)
 * The runtime is O(n^3) * t where n = the height of the matrix, t = the number of times to multiply
 *	matrix1 - the first adjacency matrix
 *	matrix2 - the second adjacency matrix
 *	paths - the number of paths (times) to preform matrix multiplacation
 *	count - the height of the matrix
 */
void CPUMultiplyMatrix(long** matrix1, long** matrix2, int paths, int count){
	long* newMatrix = (long*) malloc(sizeof(long)*count*count);
	int i, j, k;
	while(paths > 0) {

		for(i=0; i<count; i++) {
			for(j=0; j<count; j++) {
				for(k=0; k<count; k++) {
					newMatrix[(i*count)+j] += (*matrix1)[(i*count)+k] * (*matrix2)[(k*count)+j];
				}
			}
		}
		//Copy newMatrix to matrix2 and clear newMatrix
		for(i = 0; i < count*count; i++) {
			(*matrix2)[i] = newMatrix[i];
			newMatrix[i] = 0;
		}
		paths--;
	}
	free(newMatrix);
}

/*
 * GPUMultiplyMatrix (GPU Only)- Returns a cross multiplied matrix of two matrixies (The matrixies must be of the same height and width)
 * Each core is calculating an element on the multiplied matrix
 * e.g.
 *		
 *		a b c		1a 1b 1c		2a 2b 2c
 * 		d e f	=	1d 1e 1f	X	2d 2e 2f
 * 		g h i		1g 1h 1i		2g 2h 2i
 *		
 * 	There will be 9 cores (3x3)
 * 	Core 0, the first core will calculate (a) in the final matrix (1a*2a + 1d*2b + 1g*2c)
 * 	Core 1, the first core will calculate (b) in the final matrix (1b*2a + 1e*2b + 1h*2c)
 * 	And so on...
 * 
 * matrix1 - the first adjacency matrix
 * matrix2 - the second adjacency matrix
 * paths - the number of paths (times) to preform matrix multiplication
 * count - the height of the matrix
 */
__global__ void GPUMultiplyMatrix(long* matrix1, long* matrix2, int paths, int count){
	int element = blockIdx.x * blockDim.x + threadIdx.x;
	int i;
	while(paths > 0) {
		long sum = 0;
		int col = element % count;
		int row = element / count;
		for(i=0; i < count; i++) {
			sum += matrix1[count*i + col] * matrix2[row*count + i];
		}
		//Wait till all GPU cores are finished
		__syncthreads();
		matrix2[element] = sum;	

		paths--;
	}
}

/*
 * traverse (GPU Only) - takes in a matrix and returns all paths as an int array the
 *	matrix - The original matrix
 *	paths - An array storing the set of paths
 *	matrix - The multiplied matrix
 * 	count - the height of the matrixies
 *	startNodeNumber - the starting node number (from 0 to count-1)
 *	endNodeNumber - the ending node number (from 0 to count-1)
 */
__global__ void traverse(long* matrix, long* paths, int count, int startNodeNumber, int endNodeNumber, int length, int numPaths) {
	int element = blockIdx.x * blockDim.x + threadIdx.x;

	//curand = cuda random for random number generation
	curandState state;
	curand_init((unsigned long)element, 0, 0, &state);
	//current length of the path
	int currLength = 0;
	//current Node in the graph
	int currNode = startNodeNumber;

	paths[element*length + currLength] = currNode;
	currLength++;
	while(currLength != length) {
		if(currLength == length-1) {
			//this case is to assist in our bruteforce algorithm
			//if we can only make one more transition instead of doing
			//a random transition we try to move to the endNodeNumber point
			if(matrix[currNode * count + endNodeNumber] == 1) {
				currNode = endNodeNumber;
				paths[element*length + currLength] = currNode;
				currLength++;

				//check for duplicates
				//int i;
				//for(i = 0; i < numPaths; i++){
				//	int j;
				//	for(j = 0; j < length; j++){
				//
				//	}
				//}
			} else {//if we can't connect to the endpoint we restart
				currLength = 1;
				currNode = startNodeNumber;
				paths[element*length + 0] = currNode;
			}
		} else {
			int randIdx;
			do {
				randIdx = curand(&state) % count;
			} while(matrix[currNode * count + randIdx] != 1);
			currNode = randIdx;
			paths[element*length + currLength] = currNode;
        	currLength++;
		}
	}
}

/*
 * CPUMatrixMultiplication - Preforms CPU matrix multiplication on an adjacency matrix
 *	count - number of nodes
 *	path - number of paths
 *	matrix - an adjancency matrix
 */
void CPUMatrixMultiplication(int count, int path, long* matrix){
	
	//Array to store the multiplied array 
	long* cpuMM = (long*)malloc(count*count*sizeof(long));

	//Copy matrix to cpuMM
	int i;
	for(i = 0; i < count*count; i++) {
		cpuMM[i] = matrix[i];
	}

	//Create the time interval
	struct timeval start, end;

	//Start time
	gettimeofday(&start, NULL);

	//The completed multiplied matrix
	CPUMultiplyMatrix(&matrix, &cpuMM, path, count);

	//End time
	gettimeofday(&end, NULL);

	//Save the computed time
	unsigned int seconds = end.tv_sec - start.tv_sec;
	unsigned long microseconds = end.tv_usec - start.tv_usec;

	//Print the multiplied matrix
	printf("CPU Generated matrix:\n");
	if (!fTimeOnly)
		printAdjMatrix(count, cpuMM);
	printf("Took %d seconds, and %lu microseconds to compute\n\n", seconds, microseconds % 1000000);
	
	free(cpuMM);
}
/*
 * GPUMatrixMultiplication - Preforms GPU matrix multiplication on an adjacency matrix
 *	count - number of nodes
 *	path - number of paths
 *	matrix - an adjancency matrix
 */
void GPUMatrixMultiplication(int count, int path, long* matrix, int nodeA, int nodeB){

	//paths that exist
	int numPaths;

	//An adjacency matrix on the GPU
	long* gpuMatrix;

	//The multiplied matrix on the GPU
	long* gpuMM;

	//A matrix that will store gpuMM on the CPU
	long* multipliedMatrix = (long*)malloc(count*count*sizeof(long));

	//The number of GPUS needed for matrix multiplcation
	int numBlocks = (count*count)/NUMTHREADS + 1;

	//Allocate the memory on the GPU
    cudaMalloc(&gpuMatrix, (count*count*sizeof(long)));
	cudaMalloc(&gpuMM, (count*count*sizeof(long)));

	//Copy the input matrix from the CPU to the GPU (matrix -> gpuMatrix)
	cudaMemcpy(gpuMatrix, matrix, (count*count*sizeof(long)), cudaMemcpyHostToDevice);
	cudaMemcpy(gpuMM, matrix, (count*count*sizeof(long)), cudaMemcpyHostToDevice);

	//Create the time intervals
	struct timeval start, end;

	//Start time
	gettimeofday(&start, NULL);

	//Preform the multiplied matrix function on gpuMatrix, and gpuMM, and store into gpuMM
	GPUMultiplyMatrix<<<numBlocks, NUMTHREADS>>>(gpuMatrix, gpuMM, path, count);

	//End time
	gettimeofday(&end, NULL);

	//Copy gpuMM from the GPU to the CPU in multipiedMatrix
	cudaMemcpy(multipliedMatrix, gpuMM, (count*count*sizeof(long)), cudaMemcpyDeviceToHost);
	cudaFree(&gpuMM);

	//Calculate time
	long microseconds = end.tv_usec - start.tv_usec;

	//Print the multiplied matrix, copied earlier from the GPU
	printf("GPU Generated matrix:\n");
	if (!fTimeOnly)
		printAdjMatrix(count, multipliedMatrix);
	printf("Took %li microseconds to compute\n", microseconds);
	printf("\n");
	
	//Get the number of paths
	numPaths = multipliedMatrix[(nodeA * count) + nodeB];
	
	//Print the paths
	if (numPaths == 0) {
		printf("No paths exist from Node<%d> to Node<%d>\n", nodeA, nodeB);
		return;
	} else {
		path+=2;
		numBlocks = numPaths / NUMTHREADS + 1;
		long* paths = (long *)malloc(numPaths * sizeof(long) * (path));
		long* gpuPaths;

		cudaMalloc(&gpuPaths, (numPaths*path*sizeof(long)));
		traverse<<<numBlocks, NUMTHREADS>>>(gpuMatrix, gpuPaths, count, nodeA, nodeB, path, numPaths);
		cudaMemcpy(paths, gpuPaths, (numPaths*(path)*sizeof(long)), cudaMemcpyDeviceToHost);

		cudaFree(&gpuMM);
		int i;
		if (fShowPaths) {
			for(i = 0; i < numPaths; i++) {
				int j;
				for(j = 0; j < path; j++) {
					printf("%ld ", paths[i*path + j]);
				}
				printf("\n");
			}
		}
	printf("From Node<%d> to Node<%d>: There are %d paths\n", nodeA, nodeB, numPaths);
	}
}

//Main function
int main(int argc, char* argv[]){
	char usageString[500] = ("Usage:\n-t: Print the Calculation Time only\n-d: Default - Set number of nodes to 10, number of paths to 3\n");
	strcat(usageString,"-g: Preform calculations on GPU only\n-s: Show the paths\n-c <num of nodes>\n-p <num of paths>\n");
	strcat(usageString,"-a <start node number (0 to c-1)>\n-b <end node number (0 to c-1)>\n\n");

	int count; 					//Number of nodes
	int path;					//Number of paths
	long* adjMatrix;

	//start and end of the path
	int startNodeNumber = 0;
	int endNodeNumber = 3;

	//If there is more than 2 parameters
	opterr = 0;
	int c;

	//If no parametters are passed
	if(argc == 1) {
		fprintf(stderr, "%s", usageString);
		return 1;
	}

	while((c = getopt(argc, argv, "dgtsc:p:a:b:")) != -1) {
		switch (c) {
			//Flags
			case 'd':
				count = 10;
				path = 2;
				break;
			case 'g':
				fGPUOnly = 1;
				break;
			case 't':
				fTimeOnly = 1;
				break;
			case 's':
				fShowPaths = 1;
				break;
			//Parameters
			case 'c':
				count = atoi(optarg);
				break;
			case 'p':
				path = atoi(optarg);
				break;
			case 'a':
				startNodeNumber = atoi(optarg);
                break;
			case 'b':
				endNodeNumber = atoi(optarg);
				break;
			case '?':
				if (optopt == 'c' || optopt == 'p' || optopt == 'a' || optopt == 'b') {
					fprintf (stderr, "Option (-%c) requires an argument.\n\n%s\n", optopt, usageString);
				} else {
					fprintf(stderr, "%s", usageString);
				}
				return 1;
			default:
				printf("Error...");
				return 2;
		}
	}

	path--;

	//adjMatrix now equals a new Random adjancency  Matrix
	adjMatrix = generateAdjMatrix(count);

	//Print the generated adjancency matrix
	if (!fTimeOnly){
		printf("Generated Adjancency Matritx:\n");
		printAdjMatrix(count, adjMatrix);
		printf("\n");
	}

	//Compute the CPU function
	if(!fGPUOnly)
		CPUMatrixMultiplication(count, path, adjMatrix);

	//Compute the GPU function
	GPUMatrixMultiplication(count, path, adjMatrix, startNodeNumber, endNodeNumber);
	return 0;
}
