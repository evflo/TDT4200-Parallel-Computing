/*

    Speedup in fast.c is achieved by using cache blocking for loop over x and y compared
    to naive.c. The fast implementation also prescales B and C by alpha and beta. The third
    loop over z uses property of A being hermetian only looping over half the matrix compared
    to the naive implementation.

    Speedup using loop reording have been tested but no noticeable speedup was found.

    Speedup using SSE vector intrinsics for complex multiplication have been tested 
    but did not give any speedup, this might be because of non optimal implementation. 
    The implementation is shown in the function complexMult but not used in speedup version.


    A certain speed up is achieved with this fast implementation.

    dhcp-10-22-75-194:PS4 Even$ ./chemm_fast 100 100
    Using n = 100, m = 100
    Time : 0.004507 s
    Max error: 0.000007
    dhcp-10-22-75-194:PS4 Even$ ./chemm_naive 100 100
    Using n = 100, m = 100
    Time : 0.007031 s
    Max error: 0.000007



*/

#include <complex.h>
#include <pmmintrin.h>

void chemm(complex float* A,
        complex float* B,
        complex float* C,
        int m,
        int n,
        complex float alpha,
        complex float beta){

    // Initalize counter variables
    int x;
    int y;
    int z;

    // Allocate scaled B matrix
    complex float* B_scaled = malloc(m*n*sizeof(complex float));

    // Pre scale B matrix by alpha and C matrix by beta
    for(x = 0; x < n; x++){
        for(y = 0; y < m; y++){
            C[y*n + x] *= beta;
            B_scaled[y*n + x] = alpha * B[y*n + x];
       }
    }

    // Set block sizes for cache blocking
    int block_size_x = n/4;
    int block_size_y = m/4;

    // Loop through matrix using cache blocking for a size block_size
    for(x = 0; x < n; x += block_size_x){
        for(y = 0; y < m; y += block_size_y){

            for (int xx = x; xx < x + block_size_x; xx++ ){
                for (int yy = y; yy < y + block_size_y; yy++){
                    
                    C[yy*n + xx] += A[yy*m + yy]*B_scaled[yy*n + xx];

                    // Use property of A hermittian and loop over only half of the array
                    for(z = yy+1; z < m; z += 1){

                        C[yy*n + xx] += A[yy*m + z] * B_scaled[z*n + xx];
                        C[z*n + xx] += conj(A[yy*m+z]) * B_scaled[yy*n + xx];
                        
                    }
                }
            }
        }
    }
    
    free(B_scaled);
}


/* Implementation of complex multiplication using SSE
void complexMult(my_complex* x_mult_elemA,
        my_complex* y_mult_elemZ,
        my_complex* y_mult_elemY,
        float* x_mult,
        float* y_mult,
        float* z_mult){

    x_mult[0] = x_mult_elemA->real;
    x_mult[1] = x_mult_elemA->imag;
    x_mult[2] = x_mult_elemA->real;
    x_mult[3] = (-1) * x_mult_elemA->imag;

    y_mult[0] = y_mult_elemZ->real;
    y_mult[1] = y_mult_elemZ->imag;
    y_mult[2] = y_mult_elemY->real;
    y_mult[3] = y_mult_elemY->imag;

    __m128 x_128,y_128,t_128,t2_128;

    x_128 = _mm_load_ps(x_mult);
    y_128 = _mm_load_ps(y_mult);

    t_128 = _mm_moveldup_ps(x_128);

    t2_128 = _mm_mul_ps(t_128 , y_128); 
    

    y_128 = _mm_shuffle_ps(y_128,y_128,0xb1);


    t_128 = _mm_movehdup_ps(x_128);

    t_128 = _mm_mul_ps(t_128,y_128);

    x_128 = _mm_addsub_ps(t2_128 , t_128);

    _mm_store_ps(z_mult,x_128);


}
*/