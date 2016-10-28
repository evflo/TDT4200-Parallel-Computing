// Compile with:
// gcc -std=c99 -mavx2 simd.c

#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

int main(){
    
    float* a;
    posix_memalign((void**)&a, 32, sizeof(float) * 1024);
    float* b;
    posix_memalign((void**)&b, 32, sizeof(float) * 1024);
    float* c;
    posix_memalign((void**)&c, 32, sizeof(float) * 1024);
    
    for(int i = 0; i < 1024; i++){
        a[i] = i + 0.5;
        b[i] = i - 0.5;
    }
    
    // 128 bit vector, SSE
    for(int i = 0; i < 1024; i += 4){
        __m128 x = _mm_load_ps(&(a[i]));
        __m128 y = _mm_load_ps(&(b[i]));
        
        __m128 z = _mm_mul_ps(x,y);
        
        _mm_store_ps(&(c[i]), z);
    }
    
    printf("\n SSE\n");
    for(int i = 0; i < 10; i++){
        printf("%f * %f = %f\n", a[i], b[i], c[i]);
    }
    
    
    // 256 bit vector, AVX
    for(int i = 0; i < 1024; i += 8){
        __m256 x = _mm256_load_ps(&(a[i]));
        __m256 y = _mm256_load_ps(&(b[i]));
        
        __m256 z = _mm256_mul_ps(x,y);
        
        _mm256_store_ps(&(c[i]), z);
    }
    
    printf("\n AVX\n");
    for(int i = 0; i < 10; i++){
        printf("%f * %f = %f\n", a[i], b[i], c[i]);
    }
}
