#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>




#define ARRAY_SIZE 10000000

// Struct containing real and imaginary part 
// of complex number
typedef struct complex
{
  double real;
  double imag;
}complex_t;


// Perform complex multiplication
// param a Complex number
// param b Complex number
// param c Complex number product of a * b
complex_t multiply_complex(complex_t a, complex_t b){
  
  // Compute real part of multiplication  
  double real = ( ( a.real * b.real ) - ( a.imag * b.imag ) );
  
  // Compute imaginary part of multiplication
  double imag = ( ( a.real * b.imag ) + ( a.imag * b.real ) );

  // Construct multiplaction number 
  complex_t c = { .real = real, .imag = imag };

  return c;

}

// Perform absolute complex
// param a Complex number
// param absComplex Absolute complex of a
double absolute_complex(complex_t a) {
  // Compute absolute complex of a and return
  return sqrt( ( a.real * a.real ) + ( a.imag * a.imag ) );
}


// Perform square complex
// param a Complex number
// param squareComplex Square complex of a
double square_complex(complex_t a) {
  // Compute power of complex number and return
  return ( ( a.real * a.real ) + ( a.imag * a.imag ));
}

// Create a vector of complex number with random values between [-3,3]
// param size Size of generated vector
// param randComplexArray Generated vector
complex_t* create_random_complex_array(int size){

  // Initalize output array
  complex_t* randComplexArray = ( complex_t* ) calloc( size , sizeof(complex_t));
  
  // Initalize random generator
  srand(time(NULL));

  // Loop through array and fill with random values
  for (int i = 0; i < size; ++i)
  {
    randComplexArray[i].real =  ( rand() % 6 ) - 3;
    randComplexArray[i].imag = ( rand() % 6 ) - 3;
  }

  return randComplexArray;
}

// Find elements in array inside Mandelbot set
// param in Vector of complex numbers
// param size Size of in vector
// return out Returning one or zero indication element in Mandelbot set or not
int* fractal_test_array(complex_t* in, int size){
  
  // Initalize output array
  int* out = ( int* ) calloc( size , sizeof(int));
  
  // Set whole array to zero
  memset(out , 0 , size);

  // Loop through in and determine if the input is in the Mandelbrot set
  // Can use both checks by square_complex and absolute_complex
  for (int i = 0; i < size; ++i)
  {
    // Uncomment to use square_complex method to perform check instead of absolute complex
    /* 
    if ( square_complex(in[i]) >= 4 ) { // ( absolute_complex(in[i]) >= 2 ){
      out[i] = 1;
    }
    */
     
    if ( absolute_complex(in[i]) >= 2 ){
      out[i] = 1;
    } 
    
  }
  
  return out;
}


// Main function of program
// return 0
int main(){
  complex_t a = { .real = -4.0, .imag = 3.0};
  complex_t b = { .real = 10.0, .imag = -8.0};
  complex_t c;

  c = multiply_complex(a, b);
  printf("%f + %fi\n", c.real, c.imag);  // Should print -16 + 62i

  double b_abs;
  b_abs = absolute_complex(b);

  printf("%f\n", b_abs);  // Should print 12.806248

  complex_t* complex_array = create_random_complex_array(ARRAY_SIZE);

  int* fractal_test = fractal_test_array(complex_array, ARRAY_SIZE);


  // This will test the correctness of your implementation of fractal_test_array
  // Remove this before doing your timings.
  complex_t correctness_test_in[20];
  correctness_test_in[0] = (complex_t){.real = -0.201511, .imag = -2.801871};
  correctness_test_in[1] = (complex_t){.real = 2.947470, .imag = 2.131311};
  correctness_test_in[2] = (complex_t){.real = 0.940579, .imag = -1.688483};
  correctness_test_in[3] = (complex_t){.real = 1.673621, .imag = 0.542952};
  correctness_test_in[4] = (complex_t){.real = -0.604970, .imag = 2.271816};
  correctness_test_in[5] = (complex_t){.real = -1.595234, .imag = 2.897538};
  correctness_test_in[6] = (complex_t){.real = 2.916106, .imag = 2.989747};
  correctness_test_in[7] = (complex_t){.real = -1.324724, .imag = 1.368677};
  correctness_test_in[8] = (complex_t){.real = -0.652224, .imag = 0.077847};
  correctness_test_in[9] = (complex_t){.real = 0.369454, .imag = -0.589145};
  correctness_test_in[10] = (complex_t){.real = -1.753048, .imag = 2.524388};
  correctness_test_in[11] = (complex_t){.real = 1.391200, .imag = -0.096644};
  correctness_test_in[12] = (complex_t){.real = 1.710839, .imag = 2.077615};
  correctness_test_in[13] = (complex_t){.real = -1.530207, .imag = -2.185918};
  correctness_test_in[14] = (complex_t){.real = -0.728149, .imag = 2.005854};
  correctness_test_in[15] = (complex_t){.real = -1.605720, .imag = 0.671666};
  correctness_test_in[16] = (complex_t){.real = 2.695829, .imag = 2.797411};
  correctness_test_in[17] = (complex_t){.real = 0.091064, .imag = 0.515436};
  correctness_test_in[18] = (complex_t){.real = -1.075304, .imag = -0.628689};
  correctness_test_in[19] = (complex_t){.real = -0.370658, .imag = -1.642990};

  int* correctness_test_out = fractal_test_array(correctness_test_in, 20);
  int correct_result[20] = {1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                            1, 0, 1, 1, 1, 0, 1, 0, 0, 0};
  for (int i = 0; i < 20; i++) {
    if (correctness_test_out[i] != (int)correct_result[i]) {
      printf("Your program does not seem to run correctly\n");
      break;
    }
  }

  free(correctness_test_out);
  free(complex_array);
  free(fractal_test);

  return 0;
}
