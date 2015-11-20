#include <stdio.h>
#include <stdlib.h>

int* generateAdjMatrix(int count, int* adjMatrix);
void printAdjMatrix(int count, int* adjMatrix);

int main(){
   int* adjMatrix = NULL;
   int count = 10;
   adjMatrix = generateAdjMatrix(count, adjMatrix);
   printAdjMatrix(count, adjMatrix);
   return 0;
}

//Creates an adjacency matrix
//	count - the size of the matrix. the size is count X count)
//	adjMatrix - a pointer to an adjacency Matrix
int* generateAdjMatrix(int count, int* adjMatrix){
   adjMatrix = (int *)malloc(count*count*sizeof(int));
   int i;

   //Set the random seed to the current time
   srand(time(NULL));

   //Create a random matrix using rand
   for (i = 0; i < count * count; i++){
      adjMatrix[i] = rand() % 2;
   }
   return adjMatrix;
}

//Prints the adjacency matrix to stdout
void printAdjMatrix(int count, int* adjMatrix){
   int i;
   for (i = 0; i < count; i++){
      int j;
      for (j = 0; j < count; j++){
         printf("%i  ", adjMatrix[(i * count) + j]);
      } 
      printf("\n");
   }
}
