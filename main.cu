#include <stdio.h>
#include <stdlib.h>

int* generateAdjMatrix(int count, int* adjMatrix);
void printAdjMatrix(int count, int* adjMatrix);
int* multiplyMatrix(int* in,int* in2, int num,int count);

//This is the main function
int main(int argc, char* argv[]){
	int* adjMatrix = NULL;
	int* multipliedMatrix = NULL;
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
	
	cudaMalloc(&adjMatrix, (count*count*sizeof(int)));
	
	printAdjMatrix(count, adjMatrix);
	multipliedMatrix = multiplyMatrix(adjMatrix,adjMatrix,path,count);
	printf("\n");
	printAdjMatrix(count, multipliedMatrix);
	return 0;
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

//Square a given matrix: in.
int* multiplyMatrix(int* in,int* in2,int num, int count){
	if(num==0) return in2;
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

