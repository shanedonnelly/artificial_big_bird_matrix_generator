all: big_bird

big_bird: big_bird.c
	gcc -Wall -fopenmp -o big_bird big_bird.c -lm

clean: 
	rm big_bird test