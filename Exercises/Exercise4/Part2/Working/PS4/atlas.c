#include <complex.h>
#include <cblas.h>

void chemm(complex float * A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta){
    
    cblas_chemm(CblasRowMajor, CblasLeft, CblasUpper, m, n, (void*)&alpha, A, m, B, n, (void*)&beta, C, n);
}
