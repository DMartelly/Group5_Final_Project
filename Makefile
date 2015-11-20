main.o: main.cu
	nvcc -o main main.cu
clean:
	rm -rf main
