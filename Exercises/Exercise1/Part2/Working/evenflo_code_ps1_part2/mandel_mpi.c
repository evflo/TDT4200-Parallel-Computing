#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <mpi.h>

// Shorthand for less typing 
typedef unsigned char uchar;

// Declarations of output functions
void output();
void fancycolour(uchar *p, int iter);
void savebmp(char *name, uchar *buffer, int x, int y);

// Struct for complex numbers
typedef struct {
  double real, imag;
} complex_t;

// Size of image, in pixels 
const int XSIZE = 2560;
const int YSIZE = 2048;

// Max number of iterations 
const int MAXITER = 255;

// Range in x direction
double xleft = -2.0;
double xright = 1.0;
double ycenter = 0.0;

// Range in y direction, calculated in main
// based on range in x direction and image size
//
double yupper, ylower;

// Distance between numbers
double step;

// Global array for iteration counts/pixels
int* pixel;

// Calculate the number of iterations until divergence for each pixel.
// If divergence never happens, return MAXITER
// param yStart Start of iteration in y-direction
// param yEnd End of iteration in y-direction
// param pixelRegion Region of images to be calculated
void calculate(int yStart,int yEnd, int* pixelRegion){

  // Loop through all elements
  for (int i = 1; i < XSIZE; i++) {
    for (int j = yStart; j <= yEnd; j++) {

      complex_t c, z, temp;
      int iter = 0;

      // Compute complex number
      c.real = (xleft + step * i);
      c.imag = (ylower + step * j);

      z = c;
      // While |z|^2 < 4 iterate iter each loop
      while (z.real * z.real + z.imag * z.imag < 4) {
        temp.real = z.real * z.real - z.imag * z.imag + c.real;
        temp.imag = 2 * z.real * z.imag + c.imag;
        z = temp;

        iter++;

        // Break if iteration passes MAXITER
        if(iter == MAXITER){
            break;
        }
      }

      // Set iteration in pixel element
      pixelRegion[( j - yStart ) * XSIZE + i] = iter;
    }
  }
}

int main(int argc, char **argv) {

  // Initalize timing variables
  double startTime, endTime, elapsedTime, totalElapsedTime;

  // Check input arguments
  if (argc == 1) {
    puts("Usage: MANDEL n");
    puts("n decides whether image should be written to disk (1 = yes, 0 = no)");
    return 0;
  }
  
  // Calculate the range in the y - axis such that we preserve the aspect ratio
  step = (xright - xleft)/XSIZE;
  yupper = ycenter + (step * YSIZE)/2;
  ylower = ycenter - (step * YSIZE)/2;
  
  // Allocate memory for the entire image
  pixel = (int*) malloc(sizeof(int) * XSIZE * YSIZE);
  
  // Initalize MPI variables
  int rank,commSize;
  MPI_Status status;


  // Initalize local decomposistion variables
  int local_YSIZE;
  int* local_pixel;
  int local_YStart, local_YEnd;


  // Initalize Message Passing Interface (MPI)
  MPI_Init(NULL,NULL);

  // Find number of processors
  MPI_Comm_size(MPI_COMM_WORLD, &commSize);

  // Find my rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Set up barrier
  MPI_Barrier(MPI_COMM_WORLD);

  // Start timing
  startTime = MPI_Wtime();

  // Divide imaging depending on processors size
  local_YSIZE = YSIZE/( commSize - 1 );

  // Allocate memory for local image
  local_pixel = (int*) malloc(sizeof(int) * XSIZE * local_YSIZE);

  // Rank larger than 0 will perform calculate
  if ( 0 != rank ){
    
    // Find local decomposition
    local_YStart = ( rank -1 ) * local_YSIZE;
    local_YEnd = ( rank * local_YSIZE ) - 1;

    // Perform calculation
    calculate(local_YStart, local_YEnd, local_pixel);

    // Send computed image region
    MPI_Send(&local_pixel[0], XSIZE * local_YSIZE, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }
  else{ // If rank is 0 combine all calculations

    // Loop through all sending processors and receive local pixelRegion and place in
    // global pixel array
    for (int i = 1; i < commSize; ++i)
    {
      MPI_Recv(&pixel[(i - 1) * XSIZE * local_YSIZE ] , XSIZE * local_YSIZE, MPI_INT, i, 1, MPI_COMM_WORLD, &status);
    }
  
  // Build output image
  if (strtol(argv[1], NULL, 10) != 0) {
      output();
  }
  }

  // End timing
  endTime = MPI_Wtime();

  // Calculate elapsed time
  elapsedTime = endTime - startTime;

  // Reduce elapsed time to the max of all processors
  MPI_Reduce(&elapsedTime, &totalElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);

  // Print out max elapsed time
  if ( 0 == rank){
    printf("Elapsed time: %f seconds\n", totalElapsedTime);
  }

  // Finialize Message Passing Interface (MPI)
  MPI_Finalize();

  return 0;
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down */
void savebmp(char *name, uchar *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  uchar header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, XSIZE * YSIZE * 3, f);
  fclose(f);
}

/* Given iteration number, set a colour */
void fancycolour(uchar *p, int iter) {
  if (iter == MAXITER);
  else if (iter < 8) { p[0] = 128 + iter * 16; p[1] = p[2] = 0; }
  else if (iter < 24) { p[0] = 255; p[1] = p[2] = (iter - 8) * 16; }
  else if (iter < 160) { p[0] = p[1] = 255 - (iter - 24) * 2; p[2] = 255; }
  else { p[0] = p[1] = (iter - 160) * 2; p[2] = 255 - (iter - 160) * 2; }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(){
    unsigned char *buffer = calloc(XSIZE * YSIZE * 3, 1);
    for (int i = 0; i < XSIZE; i++) {
      for (int j = 0; j < YSIZE; j++) {
        int p = ((YSIZE - j - 1) * XSIZE + i) * 3;
        fancycolour(buffer + p, pixel[(i + XSIZE * j)]);
      }
    }
    /* write image to disk */
    savebmp("mandel2.bmp", buffer, XSIZE, YSIZE);
    free(buffer);
}