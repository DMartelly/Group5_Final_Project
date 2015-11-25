main.o: main.cu
	nvcc -g -o main main.cu
clean:
	rm -rf main
