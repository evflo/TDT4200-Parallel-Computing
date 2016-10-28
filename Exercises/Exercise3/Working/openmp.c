// Compile with:
// gcc -std=c99 openmp.c -fopenmp

#include <stdio.h>
#include <omp.h>

int main(){

#pragma omp parallel for
    for(int c = 0; c < 20; c++){
        printf("Iteration: %d, Thread: %d of %d\n", c, omp_get_thread_num(), omp_get_num_threads());
    }
}
