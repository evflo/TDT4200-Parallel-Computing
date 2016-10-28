
// Serial code
for(int i = 0; i < n; i++){
	calculate(i);
}



// Parallelized code using OpenMP
#pragma omp for 
for(int i = 0; i < n; i++ ){
	calculate(i);
}




