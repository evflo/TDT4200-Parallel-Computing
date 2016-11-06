#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void add(int* a, int* b, int* c){
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    
    c[id] = a[id] + b[id];
    
    //if(id == 3){
    //    printf("Inside kernel: %d = %d + %d\n", c[id], a[id], b[id]);
    //}
}


int main(int argc, char** argv){
    
    int* host_a = (int*)malloc(sizeof(int) * 1024);
    int* host_b = (int*)malloc(sizeof(int) * 1024);
    int* host_c = (int*)malloc(sizeof(int) * 1024);
    
    for(int i = 0; i < 1024; i++){
        host_a[i] = i;
        host_b[i] = i;
    }
    
    int* device_a;
    int* device_b;
    int* device_c;
    
    cudaMalloc(&device_a, sizeof(int) * 1024);
    cudaMalloc(&device_b, sizeof(int) * 1024);
    cudaMalloc(&device_c, sizeof(int) * 1024);
    
    cudaMemcpy(device_a, host_a, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(int) * 1024, cudaMemcpyHostToDevice);
    
    add<<<8,128>>>(device_a, device_b, device_c);
    
    cudaMemcpy(host_c, device_c, sizeof(int) * 1024, cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 10; i++){
        printf("%d + %d = %d \n", i, i, host_c[i]);
    }
    printf("...\n");
}
