// OpenMP program to print Hello World
// using C language

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

void print_matrix_int(int* matrix, int size){
    int row, columns;
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
            if(matrix[row * size + columns] == 1){
                printf("\033[32m1 \033[0m");
            } else {
                printf("\033[31m0 \033[0m");
            }
        }
        printf("\n");
    }
    printf("\n \n");
}

int* init_matrix_int( int size, int value ){
    int* res = (int *)malloc(sizeof(int) * size * size) ; 
    int row, columns;
    #pragma omp parallel for private(row, columns)collapse(2)
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
         res[row * size + columns] = value;
        }
    }
    return res ; 
}

void print_matrix_bool(bool* bmask, int size){
    int row, columns;
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
            if (bmask[row * size + columns]){
                printf("\033[32mT \033[0m");
            }
            else {
                printf("\033[31mF \033[0m");
            }
        }
        printf("\n");
    }
    printf("\n \n");
}

bool* init_matrix_bool( int size, bool value ){
    bool* res = (bool *)malloc(sizeof(bool) * size * size) ; 
    int row, columns;
    #pragma omp parallel for private(row, columns)collapse(2)
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
         res[row * size + columns] = value;
        }
    }
    return res ; 
}

void set_boolean_mask_from_range(bool* bmask, int size, int ymin, int ymax, int xmin, int xmax, bool value){
    int row, columns;
    #pragma omp parallel for private(row, columns)collapse(2)
    for (row=ymin; row< ymax; row++)
    {
        for(columns=xmin; columns<xmax; columns++)
        {
         bmask[row * size + columns] = value;
        }
    }
}

void apply_boolean_mask(int* matrix, bool* b_mask, int size){
    int row, columns;
    #pragma omp parallel for private(row, columns)collapse(2)
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
         if(b_mask[row * size + columns] == false){
            matrix[row * size + columns] = 0 ;
         }
        }
    }
}



int* get_unique_random_number(int count, int min, int max, unsigned int *seedp) {
    int range = max - min;
    int* candidates = malloc(range * sizeof(int));
    int* result = malloc(count * sizeof(int));
    int i;
    
    for (i = 0; i < range; i++)
    candidates[i] = i + min;
    for (i = 0; i < count; i++) {
        int j = i + (rand_r(seedp) % (range - i));
        int temp = candidates[i];
        candidates[i] = candidates[j];
        candidates[j] = temp;
    }
    for (i = 0; i < count; i++){
        result[i] = candidates[i];
    }
    free(candidates);
    return result;
}

// ACTUAL FUNCTIONS

// RANDOM ATTENTION MASK

bool* get_random_attention_mask(int size, int nz_per_row){
    bool* res = init_matrix_bool(size, false);
    int row;
    #pragma omp parallel for private(row)
    for (row=0; row < size; row++)
    {
        unsigned int seed = omp_get_thread_num() + row; // Each thread/row gets a different seed
        int* unique_random_number = get_unique_random_number(nz_per_row, 0, size, &seed);
        for(int columns = 0; columns < nz_per_row; columns++)
        {
            int indice = unique_random_number[columns] ; 
            res[row * size + indice] = true; 
        }
        free(unique_random_number);
    }
    return res ; 
}

int best_nz_per_row_from_sparsity(int size, double sparsity){
    int res = round((float)size * (1.0 - sparsity));
    return res ; 
}

bool* get_random_attention_mask_with_sparsity(int size, double sparsity){
    int nz_per_row = best_nz_per_row_from_sparsity(size, sparsity); 
    return get_random_attention_mask(size, nz_per_row); 
}

// WINDOW ATTENTION

bool* get_window_attention_mask(int size, int diagonal_width){
    bool* res = init_matrix_bool(size, false);
    int row, columns;
    if(diagonal_width > 0){
        int sdw = diagonal_width / 2 ; 
        #pragma omp parallel for private(row, columns)
        for (row=0; row < size; row++)
        {
            int min_c = max(0, row - sdw ); 
            int max_c = min(size - 1, row + sdw ); 
            for(columns = min_c; columns <= max_c ; columns++)
            {
                res[row * size + columns] = true ; 
            }
        }
    }
    return res ;
}

int best_diagonal_width_from_sparsity(int size, double sparsity){
    int n = size;
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

bool* get_window_attention_mask_with_sparsity(int size, double sparsity){
    int dw = best_diagonal_width_from_sparsity(size, sparsity);
    return get_window_attention_mask( size, dw);
}

// GLOBAL ATTENTION

bool* get_global_attention_mask( int size, int global_width){
    bool* res = init_matrix_bool(size, false) ; 
    set_boolean_mask_from_range(res, size, 0 , global_width, 0, size, true);
    set_boolean_mask_from_range(res, size, global_width , size , 0, global_width, true);
    return res ; 
}

int best_global_width_from_sparsity(int size, double sparsity){
    int n = size;
    double density = 1.0 - sparsity;
    double ga = n * n * density;
    double a = -1;
    double b = 2 * n;
    double c = -ga;
    double det = b * b - 4 * a * c;
    double x = (-b + sqrt(det))/(2 * a);
    int gw = round(x);
    if(gw < 0) gw = 0;
    else if(gw > n * n) gw = n * n;
    return gw;
}

bool* get_global_attention_mask_with_sparsity(int size, double sparsity){
    int gw = best_global_width_from_sparsity(size, sparsity);
    return get_global_attention_mask(size, gw); 
}

// BIG BIRD

bool* combine_all_mask(int size, bool* random_mask, bool* window_mask, bool* global_mask){
    bool* res = (bool *)malloc(sizeof(bool) * size * size) ; 
    int row, columns;
    #pragma omp parallel for private(row, columns)
    for (row=0; row<size; row++)
    {
        for(columns=0; columns<size; columns++)
        {
            res[row * size + columns] = random_mask[row * size + columns] | window_mask[row * size + columns] | global_mask[row * size + columns] ; 
        }
    }
    return res ; 
}

bool* get_big_bird_mask(int size, int nz_per_row, int diagonal_width, int global_width){
    bool* am = get_random_attention_mask(size, nz_per_row);
    bool* wm = get_window_attention_mask(size, diagonal_width);
    bool* gm = get_global_attention_mask(size, global_width);
    bool* res = combine_all_mask(size, am, wm, gm);
    free(am);
    free(wm);
    free(gm);
    return res;
}

bool* get_big_bird_mask_with_sparsity(int size, double random_sparsity, double window_sparsity, double global_sparsity){
    bool* am = get_random_attention_mask_with_sparsity(size, random_sparsity);
    bool* wm = get_window_attention_mask_with_sparsity(size, window_sparsity);
    bool* gm = get_global_attention_mask_with_sparsity(size, global_sparsity);
    bool* res = combine_all_mask(size, am, wm, gm);
    free(am);
    free(wm);
    free(gm);
    return res;
}

double adjust_total_sparsity(double total_sparsity){
    double x = total_sparsity;
    double a = 24.08862473;
    double b = -65.2963488;
    double c = 64.48601296;
    double d = -28.42365239;
    double e = 5.98076684;
    double f = 0.17082526;
    double poly = a * pow(x, 5) + b * pow(x, 4) + c * pow(x, 3) + d * pow(x, 2) + e * x + f;
    if(poly < 0.0){
        poly = 0.0;
    }
    else {
        if(poly > 1.0){
            poly = 1.0;
        }
    }
    return poly;
}

bool* get_big_bird_mask_with_total_sparsity(int size, double total_sparsity, bool adjust){
    if(adjust){
        total_sparsity = adjust_total_sparsity(total_sparsity);
    }
    return get_big_bird_mask_with_sparsity(size, total_sparsity, total_sparsity, total_sparsity);
}

int* generate_big_bird_matrix(int size, double total_sparsity){
    int* res = init_matrix_int(size, 1); 
    bool* b_mask = get_big_bird_mask_with_total_sparsity(size, total_sparsity, true);
    apply_boolean_mask(res, b_mask, size);
    free(b_mask) ; 
    return res ; 
}

// test 1 matrix free(b_mask); and print
void test1(){
    int size = 25;
    double total_sparsity = 0.7;
    int* big_bird_matrix = generate_big_bird_matrix(size, total_sparsity);
    print_matrix_int(big_bird_matrix, size) ;
    free(big_bird_matrix);
}

// benchmark

void test2(){
    double sparsity_values[] = {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0};
    int sizes[] = {10, 50, 100, 250, 500, 1000, 2000};
    int size ;
    double total_sparsity ;
    int* matrix ;
    int count ;

    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 21; j++){
            size = sizes[i] ; 
            total_sparsity =  sparsity_values[j] ; 
            printf("generating bigbird with: size: %d, sparsity: %.2f", size , total_sparsity);
            matrix = generate_big_bird_matrix(size, total_sparsity);
            free(matrix) ;
            count = i * 21 + j  + 1 ;
            printf(" finished [%d /147] \n", count) ;
        } 
    }
}
int main(int argc, char* argv[])
{
    srand(39935753);
    #pragma omp parallel
    {
        # pragma omp master
        printf("NUMBER OF THREADS : %d \n \n", omp_get_num_threads());
    }
    test2();
}