#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int* get_unique_random_number(int count, int min, int max) {
  int range = max - min;
  int candidates[range];
  int* result = malloc(count * sizeof(int));
  int i;

  for (i = 0; i < range; i++)
    candidates[i] = i + min;

  for (i = 0; i < range - 1; i++) {
    int c = rand() / (RAND_MAX / (range - i) + 1);
    int t = candidates[i];
    candidates[i] = candidates[i + c];
    candidates[i + c] = t;
  }

  for (i = 0; i < count; i++)
    result[i] = candidates[i];

  return result;
}

int main(void) {
  int i;
  int* numbers;

  srand(time(NULL));

  int count = 3 ;int min = 0 ;  int max = 5 ; 
  
  numbers = get_unique_random_number(count, min, max);

  for (i = 0; i < count; i++)
    printf("%i\n", numbers[i]);

  free(numbers);
  return 0;

}