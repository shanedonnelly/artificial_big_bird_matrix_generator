// OpenMP program to print Hello World
// using C language

// OpenMP header
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

//UTILS


static inline int min(int a, int b) {
    return a < b ? a : b;
}

static inline int max(int a, int b) {
    return a > b ? a : b;
}

void print_matrix_int(int* matrix, int length){
    int row, columns;
    for (row=0; row<length; row++)
    {
        for(columns=0; columns<length; columns++)
        {
            printf("%d  ", matrix[row * length + columns]);
        }
        printf("\n");
    }
    printf("\n \n");
}

int* init_matrix_int( int length, int value ){
    int* res = (int *)malloc(sizeof(int) * length * length) ; 
    int row, columns;
    for (row=0; row<length; row++)
    {
        for(columns=0; columns<length; columns++)
        {
         res[row * length + columns] = value;
        }
    }
    return res ; 
}

void print_matrix_bool(bool* matrix, int length){
    int row, columns;
    for (row=0; row<length; row++)
    {
        for(columns=0; columns<length; columns++)
        {
            if (matrix[row * length + columns]){
                printf("T " );
            }
            else {
                printf("F ");
            }
        }
        printf("\n");
    }
    printf("\n \n");
}

bool* init_matrix_bool( int length, bool value ){
    bool* res = (bool *)malloc(sizeof(bool) * length * length) ; 
    int row, columns;
    for (row=0; row<length; row++)
    {
        for(columns=0; columns<length; columns++)
        {
         res[row * length + columns] = value;
        }
    }
    return res ; 
}

void set_boolean_mask_from_range(bool* matrix, int length, int xmin, int xmax, int ymin, int ymax, bool value){
    int row, columns;
    for (row=ymin; row<=ymax; row++)
    {
        for(columns=xmin; columns<=xmax; columns++)
        {
         matrix[row * length + columns] = value;
        }
    }
}

void apply_boolean_mask(int* matrix, bool* b_mask, int length){
    int row, columns;
    for (row=0; row<length; row++)
    {
        for(columns=0; columns<length; columns++)
        {
         if(b_mask[row * length + columns] == false){
            matrix[row * length + columns] = 0 ;
         }
        }
    }
}



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
    
    for (i = 0; i < count; i++){
        result[i] = candidates[i];
    }
    
    return result;
}

// ACTUAL FUNCTIONS

bool* get_random_attention_mask(int length, int nz_per_row){
    bool* res = init_matrix_bool(length, false);
    int* unique_random_number ; 
    int row, columns;
    for (row=0; row < length; row++)
    {
        unique_random_number = get_unique_random_number(nz_per_row, 0, length);
        for(columns = 0; columns < nz_per_row; columns++)
        {
            int indice = unique_random_number[columns] ; 
            res[row * length + indice] = true; 
        }
    }
    free(unique_random_number);
    return res ; 
}

int best_nz_per_row_from_sparsity(int length, double sparsity){
    int res = round((float)length * (1.0 - sparsity));
    return res ; 
}

bool* get_random_attention_mask_with_sparsity(int length, double sparsity){
    int nz_per_row = best_nz_per_row_from_sparsity(length, sparsity); 
    return get_random_attention_mask(length, nz_per_row); 
}

bool* get_window_attention_mask(int length, int diagonal_width){
    bool* res = init_matrix_bool(length, false);
    int row, columns;
    if(diagonal_width > 0){
        int sdw = diagonal_width / 2 ; 
        for (row=0; row < length; row++)
        {
            int min_c = max(0, row - sdw ); 
            int max_c = min(length - 1, row + sdw ); 
            for(columns = min_c; columns <= max_c ; columns++)
            {
                res[row * length + columns] = true ; 
            }
        }
    }
    return res ; 
}

int best_diagonal_width_from_sparsity(int length, double sparsity){
    int n = length;
    double density = 1.0 - sparsity;
    double da = n * n * density;
    double a = -1;
    double b = 2 * n - 1;
    double c = n - da;
    double det = b * b - 4 * a * c;
    double x = (-b + sqrt(det))/(2 * a);
    
    int sdw = round(x);
    
    int dw = 2 * sdw + 1;
    
    if(dw < 0) dw = 0;
    else if(dw > 2*n - 1) dw = 2*n - 1;
    
    return dw;
}

bool* get_window_attention_mask_with_sparsity(int length, double sparsity){
    int dw = best_diagonal_width_from_sparsity(length, sparsity);
    return get_window_attention_mask( length, dw);
}




// TEST

void test_3(){
    int length = 10;
    double sparsity = 1.0 ; 
    bool* bmask = get_window_attention_mask_with_sparsity(length, sparsity); 
    print_matrix_bool(bmask, length); 
    free(bmask); 
}

int main(int argc, char* argv[])
{
    srand(926623761);
    // Beginning of parallel region
    // omp_set_num_threads(9);
    // #pragma omp parallel
    // {
    //     printf("Hello World... from thread = %d\n",
    //            omp_get_thread_num());
    // }
    // Ending of parallel region


    test_3();
 

}