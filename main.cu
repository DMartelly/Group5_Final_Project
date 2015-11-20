#include <stdio.h>
#include <stdlib.h>

void printAdjMatrix(int count, int* adjMatrix);

int main(){
   int* adjMatrix;
   int count = 10;
   adjMatrix = (int *)malloc(sizeof(int));
   int i;
   for (i = 0; i < count * count; i++){
      adjMatrix[i] = rand() % 2;      
   }
   printAdjMatrix(count, adjMatrix);
   return 0;
}

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
