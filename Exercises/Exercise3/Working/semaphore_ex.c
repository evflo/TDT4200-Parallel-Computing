#include <semaphore.h>

int initalValue = 0;
int V = 100;

void main(){
	// Create semaphore
	sem_t s;

	// Initialize semaphore
	sem_init( &m , 0 , initalValue );

	// Entry section
	sem_wait( s );

	// Critical section
	if ( 0 < V ){
		V--;
	}

	// Exit section
	sem_post( s );


}