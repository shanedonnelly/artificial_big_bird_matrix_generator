all: big_bird big_bird_parallel

big_bird: big_bird.c
	gcc -Wall -o big_bird big_bird.c -lm
big_bird_parallel: big_bird_parallel.c
	gcc -Wall -fopenmp -o big_bird_parallel big_bird_parallel.c -lm
clean: 
	rm -f big_bird big_bird_parallel