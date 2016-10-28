// Compile with:
// gcc -std=c99 pthreads.c -pthread

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct{
    int thread_id;
    int p;
} thread_args_t;

void* run_thread(void* arg){
    thread_args_t* thread_args = (thread_args_t*)arg;
    printf("thread_id: %d, p: %d\n", thread_args->thread_id, thread_args->p);
    pthread_exit(NULL);
}

int main(){
    pthread_t threads[4];
    thread_args_t* args[4];

    printf("Launching threads\n");
    for(int i = 0; i < 4; i++){
        args[i] = malloc(sizeof(thread_args_t));
        args[i]->thread_id = i;
        args[i]->p = i*10 + 1;
        pthread_create(&threads[i], NULL, run_thread, (void*)args[i]);
    }
    
    printf("Joining threads\n");
    for(int i = 0; i < 4; i++){
        pthread_join(threads[i], NULL);
    }

    pthread_exit(NULL);
}

