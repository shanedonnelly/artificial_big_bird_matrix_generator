all: big_bird test

big_bird: big_bird.c
	gcc -Wall -fopenmp -o big_bird big_bird.c -lm

test: test.c
	gcc -Wall -o test test.c
clean: 
	rm big_bird test