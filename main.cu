#include <stdio.h>
#include <stdlib.h>

const int NUM_OF_NODES = 10;
int* generateAdjMatrix(int NUM_OF_NODES, int* adjMatrix);
void printAdjMatrix(int NUM_OF_NODES, int* adjMatrix);

int main(int argc, char* argv[]){
   int* adjMatrix = NULL;
	 int count;
	 if(argc > 2){
		 fprintf(stderr,"Usage: %s <node count>\n",argv[0]);
		 return 1;
	 }
	 if(argc==1) count = 10;
	 else count = atoi(argv[1]);
   adjMatrix = generateAdjMatrix(NUM_OF_NODES, adjMatrix);
   printAdjMatrix(NUM_OF_NODES, adjMatrix);
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
         if(i != j)
         {
            int randomResult = rand() % 2;
            matrix[(i *count) + j] = randomResult;
            matrix[(j *count) + i] = randomResult;
         }
      }
   }
   return matrix;
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
