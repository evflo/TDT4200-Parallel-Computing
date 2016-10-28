#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <cblas.h>
#include <x86intrin.h>

extern void chemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta);


float random_float(){
    float f = ((float)rand())/((float)RAND_MAX);
    if(rand() > (RAND_MAX/2)){
        return f;
    }
    return -f;
}


complex float random_complex(){
   return random_float() + random_float() * I;
}

void gemm_atlas(complex float * A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta){
    
    int k = m;

    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, (void*)&alpha, A, k, B, n, (void*)&beta, C, n);
}

// Allocating matrix, and filling it with random complex values
// posix_memalign works like malloc, but returns aligned memory
// neccessary for SSE/AVX instructions
complex float* create_random_matrix(int m, int n){
    complex float* A;
    posix_memalign((void**)&A, 32, sizeof(complex float)*m*n);

    for(int i = 0; i < m*n; i++){
        A[i] = random_complex();
    }
    
    return A;
}
    
// Allocating and filling random Hermitian matrix 
// posix_memalign works like malloc, but returns aligned memory
// neccessary for SSE/AVX instructions
complex float* create_random_hermitian_matrix(int m, int n){
    complex float* A;
    posix_memalign((void**)&A, 32, sizeof(complex float)*m*n);

    for(int y = 0; y < m; y++){
        for(int x = 0; x < n; x++){
            
            if(x == y){
                A[y*n + x] = random_float();
            }
            else if(x > y){
                A[y*n + x] = random_complex();
            }
            else{
                complex float f = A[x*n + y];
                A[y*n + x] = creal(f) - cimag(f)*I;
            }
        }
    }
            
    return A;
}

// Allocating new matrix, and copying content of old matrix into it
// posix_memalign works like malloc, but returns aligned memory
// neccessary for SSE/AVX instructions
complex float* copy_matrix(complex float * A, int m, int n){
    
    complex float* B;
    posix_memalign((void**)&B, 32, sizeof(complex float)*m*n);
    memcpy(B, A, sizeof(complex float) * m * n);
    return B;
}


void print_matrix(complex float* A, int m, int n){

    int max_size = 10;
    if(m > max_size || n > max_size){
        printf("WARNING: matrix too large, only printing part of it\n");
        m = max_size;
        n = max_size;
    }

    for(int y = 0; y < m; y++){
        for(int x = 0; x < n; x++){
            printf("%.4f%+.4fI  ", creal(A[y*n + x]), cimag(A[y*n + x]));
        }
        printf("\n");
    }
    printf("\n");
}

float compare(complex float* A, complex float* B, int m, int n){

    float max = 0;
    for(int i = 0; i < m*n; i++){
        if(fabs(creal(A[i]) - creal(B[i])) > max){
            max = fabs(creal(A[i]) - creal(B[i]));
        }
        if(fabs(cimag(A[i]) - cimag(B[i])) > max){
            max = fabs(cimag(A[i]) - cimag(B[i]));
        }
    }

    return max;
}


int main(int argc, char** argv){

    // Reading command line arguments for matrix sizes
    int m = 2;
    int n = 2;

    if(argc == 1){
        printf("Using default values (n = %d, m = %d)\n", n, m);
    }
    else if(argc == 3){
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        printf("Using n = %d, m = %d\n", n, m);
    }
    else{
        printf("useage: gemm m n k\n");
        exit(-1);
    }

    // Picking random values for constants alpha and beta
    complex float alpha = random_complex();
    complex float beta = random_complex();

    // Allocating matrices
    // A should be hermitian, D is used for checking the answer, and should be equal to C
    complex float* A = create_random_hermitian_matrix(m,m);
    complex float* B = create_random_matrix(m,n);
    complex float* C = create_random_matrix(m,n);
    complex float* D = copy_matrix(C,m,n);

    // For timing
    struct timeval start, end;

    // Running and timing the matrix multiplication
    gettimeofday(&start, NULL);
    chemm(A,B,C,m,n,alpha,beta);
    gettimeofday(&end, NULL);


    long int ms = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));
    double s = ms/1e6;
    printf("Time : %f s\n", s);

    // Checking the answer with plain gemm
    gemm_atlas(A,B,D,m,n,alpha,beta);
    printf("Max error: %f\n", compare(C,D,m,n));
    
    // For debugging, uncomment to print matrices
    // C is your answer, D is the correct answer

     //print_matrix(A, m,m);
     //print_matrix(B, m,n);
     //print_matrix(C, m,n);
     //print_matrix(D, m,n);
}
