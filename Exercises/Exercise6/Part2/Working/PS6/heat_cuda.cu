#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <sys/time.h>

/* Functions to be implemented: */
float ftcs_solver_gpu ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_shared ( int step, int block_size_x, int block_size_y );
float ftcs_solver_gpu_texture ( int step, int block_size_x, int block_size_y );
void external_heat_gpu ( int step, int block_size_x, int block_size_y );
void transfer_from_gpu( int step );
void transfer_to_gpu();
void device_allocation();

/* Prototypes for functions found at the end of this file */
void write_temp( int step );
void print_local_temps();
void init_temp_material();
void init_local_temp();
void host_allocation();
void add_time(float time);
void print_time_stats();

/*
 * Physical quantities:
 * k                    : thermal conductivity      [Watt / (meter Kelvin)]
 * rho                  : density                   [kg / meter^3]
 * cp                   : specific heat capacity    [kJ / (kg Kelvin)]
 * rho * cp             : volumetric heat capacity  [Joule / (meter^3 Kelvin)]
 * alpha = k / (rho*cp) : thermal diffusivity       [meter^2 / second]
 *
 * Mercury:
 * cp = 0.140, rho = 13506, k = 8.69
 * alpha = 8.69 / (0.140*13506) =~ 0.0619
 *
 * Copper:
 * cp = 0.385, rho = 8960, k = 401
 * alpha = 401.0 / (0.385 * 8960) =~ 0.120
 *
 * Tin:
 * cp = 0.227, k = 67, rho = 7300
 * alpha = 67.0 / (0.227 * 7300) =~ 0.040
 *
 * Aluminium:
 * cp = 0.897, rho = 2700, k = 237
 * alpha = 237 / (0.897 * 2700) =~ 0.098
 */

const float MERCURY = 0.0619;
const float COPPER = 0.116;
const float TIN = 0.040;
const float ALUMINIUM = 0.098;

/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;

/* Size of the computational grid - 1024x1024 square */
const int GRID_SIZE[2] = {2048, 2048};

int BORDER = 1;

/* Parameters of the simulation: how many steps, and when to cut off the heat */
const int NSTEPS = 10000;
const int CUTOFF = 5000;

/* How often to dump state to file (steps). */
const int SNAPSHOT = 500;

/* For time statistics */
float min_time = -2.0;
float max_time = -2.0;
float avg_time = 0.0;

/* Arrays for the simulation data, on host */
float
    *material,          // Material constants
    *temperature;       // Temperature field

/* Arrays for the simulation data, on device */
float
    *material_device,           // Material constants
    *temperature_device[2];      // Temperature field, 2 arrays
  //  **temperature_device;

texture<float,cudaTextureType2D> texreference;

/* Error handling function */
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

float timing() {
  static struct timeval t;
  gettimeofday(&t,NULL);
  return (t.tv_sec + 1e-6 * t.tv_usec);
}


// Find global temperature array index
__device__ int ti(int x, int y,const int* GRID_SIZE){

	if ( x < 0 ){
		x++;
	}
	if( x >= GRID_SIZE[0] ){
		x--;
	}
	if( y < 0 ){
		y++;
	}
	if( y >= GRID_SIZE[1] ){
		y--;
	}

	return ((y)*GRID_SIZE[0]+x);
}

// Find global material array index
__device__ int mi(int x, int y,const int GRID_SIZE_X){
	return ((y)*(GRID_SIZE_X) + x);
}

// Find local temperature index for shared memory
__device__ int lti(int x, int y,const int* BLOCK_SIZE){
  x++;
  y++;


  return ((y)*(BLOCK_SIZE[0]+2)+x);
}

/* Allocate arrays on GPU */
void device_allocation(){

  // Allocate memory for material
	HANDLE_ERROR(cudaMalloc((void**) &material_device,sizeof(float) * GRID_SIZE[0] * GRID_SIZE[1]));

  // Allocate memory for temperature
	HANDLE_ERROR(cudaMalloc((void**) &temperature_device[0],sizeof(float) * GRID_SIZE[0] * GRID_SIZE[1]));

  HANDLE_ERROR(cudaMalloc((void**) &temperature_device[1],sizeof(float) * GRID_SIZE[0] * GRID_SIZE[1]));

}

/* Transfer input to GPU */
void transfer_to_gpu(){

	// Transfer material array to GPU
	HANDLE_ERROR(cudaMemcpy(material_device,material,sizeof(float)*GRID_SIZE[0]*GRID_SIZE[1],cudaMemcpyHostToDevice));

  // Transfer temperature array to GPU
  HANDLE_ERROR(cudaMemcpy(temperature_device[0],temperature,sizeof(float)*GRID_SIZE[0]*GRID_SIZE[1],cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(temperature_device[1],temperature,sizeof(float)*GRID_SIZE[0]*GRID_SIZE[1],cudaMemcpyHostToDevice));
}

/* Transfer output from GPU to CPU */
void transfer_from_gpu(int step){

  // Copy temperature from GPU -> CPU
  HANDLE_ERROR(cudaMemcpy(temperature,temperature_device[step%2],sizeof(float)*GRID_SIZE[0]*GRID_SIZE[1],cudaMemcpyDeviceToHost));

}

// Plain/global memory only kernel
__global__ void  ftcs_kernel(float* out, float* in, float* material_device, int step, int block_size_x,int block_size_y,int GRID_X,int GRID_Y ){

  // Set grid size
  const int GRID_SIZE[2] = {GRID_X, GRID_Y};

  // Find matrix index
	int x = blockIdx.x * block_size_x + threadIdx.x;
	int y = blockIdx.y * block_size_y + threadIdx.y;

  // Compute ftcs for thread
	out[ti(x,y,GRID_SIZE)] = in[ti(x,y,GRID_SIZE)] + material_device[mi(x,y,GRID_SIZE[0])]*
					(in[ti(x+1,y,GRID_SIZE)] +
					in[ti(x-1,y,GRID_SIZE)] +
					in[ti(x,y+1,GRID_SIZE)] +
					in[ti(x,y-1,GRID_SIZE)] -
					4*in[ti(x,y,GRID_SIZE)]);

}

/* Shared memory kernel */
__global__ void  ftcs_kernel_shared(float*  out,float* in, float* material_device, int step, int block_size_x , int block_size_y, int GRID_X,int GRID_Y ){

  // Compute grid size globally and locally
  const int GRID_SIZE[2] = {GRID_X, GRID_Y};
  const int BLOCK_SIZE[2] = {block_size_x,block_size_y};

  // Initialize shared memory
	extern __shared__ float in_shared[];

  // Find matrix index
	int x = blockIdx.x * block_size_x + threadIdx.x;
	int y = blockIdx.y * block_size_y + threadIdx.y;

  // Find block matrix index
  int local_x = threadIdx.x;
  int local_y = threadIdx.y;

  // Set halo around border in shared memory
  if( local_x == 0 ){
    in_shared[lti(local_x - 1,local_y,BLOCK_SIZE)] = in[ti(x - 1,y,GRID_SIZE)];
  }

  if( local_x == ( block_size_x - 1 ) ){
    in_shared[lti(local_x + 1,local_y,BLOCK_SIZE)] = in[ti(x + 1,y,GRID_SIZE)];
  }

  if( local_y == 0 ){
    in_shared[lti(local_x,local_y-1,BLOCK_SIZE)] = in[ti(x,y - 1,GRID_SIZE)];
  }

  if ( local_y == ( block_size_y - 1 ) ) {
    in_shared[lti(local_x,local_y+1,BLOCK_SIZE)] = in[ti(x,y + 1,GRID_SIZE)];
  }

  // Set data in shared memory
  in_shared[lti(local_x,local_y,BLOCK_SIZE)] = in[ti(x,y,GRID_SIZE)];

  // Synch threads before reading
	__syncthreads();

  // Compute ftcs using shared memory
	out[ti(x,y,GRID_SIZE)] = in_shared[lti(local_x,local_y,BLOCK_SIZE)] + material_device[mi(x,y,GRID_SIZE[0])]*
					(in_shared[lti(local_x+1,local_y,BLOCK_SIZE)] +
					in_shared[lti(local_x-1,local_y,BLOCK_SIZE)] +
					in_shared[lti(local_x,local_y+1,BLOCK_SIZE)] +
					in_shared[lti(local_x,local_y-1,BLOCK_SIZE)] -
					4*in_shared[lti(local_x,local_y,BLOCK_SIZE)]);

}

/* Texture memory kernel */
__global__ void  ftcs_kernel_texture(float* out, float* material_device,int step, int block_size_x, int block_size_y,int GRID_X,int GRID_Y ,size_t offset){

  // Set grid size
  const int GRID_SIZE[2] = {GRID_X, GRID_Y};

	// Find linear index for x and y coordinates
	int x = blockIdx.x * block_size_x + threadIdx.x;
	int y = blockIdx.y * block_size_y + threadIdx.y;

  // Find indices of neighbouring elements
  int x_up = x+1;
  int x_down = x-1;
  int y_up = y+1;
  int y_down = y-1;

  // Edit border values to fit with Neuman boundary condition
  if( x_up >= GRID_SIZE[0] ){
    x_up--;
  }
  if( x_down < 0 ){
    x_down++;
  }
  if( y_up >= GRID_SIZE[1] ){
    y_up--;
  }
  if( y_down < 0 ){
    y_down++;
  }


  offset /= sizeof(float);
  size_t xOffset = offset % GRID_SIZE[0];
  size_t yOffset = offset / GRID_SIZE[1];

	// Fetch data from texture memory
	float in_origin = tex2D(texreference,xOffset + x,yOffset + y);
	float in_up_x = tex2D(texreference,xOffset +x_up,yOffset + y);
	float in_down_x = tex2D(texreference,xOffset +x_down,yOffset + y);
	float in_up_y = tex2D(texreference,xOffset +x,yOffset + y_up);
	float in_down_y = tex2D(texreference,xOffset +x,yOffset + y_down);

  // Compute ftcs using texture memory
	out[ti(x,y,GRID_SIZE)] = in_origin + material_device[mi(x,y,GRID_SIZE[0])]*
									(in_up_x +
									in_down_x +
									in_up_y +
									in_down_y -
									4*in_origin);
}


/* External heat kernel, should do the same work as the external
 * heat function in the serial code
 */
__global__ void external_heat_kernel(float* in, int step, int block_size_x, int block_size_y,int GRID_X, int GRID_Y ){

  // Set grid size
  const int GRID_SIZE[2] = {GRID_X, GRID_Y};

	// Find linear index for x and y coordinates
  int x = blockIdx.x * block_size_x + threadIdx.x;
  int y = blockIdx.y * block_size_y + threadIdx.y;

  // Set value of external heat in array
  if(x>= (GRID_SIZE[0]/4) && x<= (3*GRID_SIZE[0]/4) ){
	   if(y >= (GRID_SIZE[1]/2 - GRID_SIZE[1]/16) && y<= (GRID_SIZE[1]/2 + GRID_SIZE[1]/16)){
	      in[ti(x,y,GRID_SIZE)] = 100;
	}
}
}

/* Set up and call ftcs_kernel
 * should return the execution time of the kernel
 */

float ftcs_solver_gpu( int step, int block_size_x, int block_size_y ){

  // Edit block sizes to be even numbers
  if ( block_size_x % 2){
    block_size_x++;
  }
  if (block_size_y % 2) {
    block_size_y++;
  }

  // Initalize timing
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Compute thread block size
	dim3 gridBlock(GRID_SIZE[0]/block_size_x,GRID_SIZE[1]/block_size_y);
	dim3 threadBlock(block_size_x,block_size_y);

  // Find device temperature arrays
  float* out = temperature_device[(step+1)%2];
  float* in = temperature_device[step%2];

  // Record execution time
  cudaEventRecord(start);

  // Compute global kernel
  ftcs_kernel<<<gridBlock,threadBlock>>>(out,in,material_device,step,block_size_x,block_size_y,GRID_SIZE[0],GRID_SIZE[1]);

  cudaEventRecord(stop);

  // Synch device threads
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );

  // Stop recording
  cudaEventSynchronize(stop);
  float milliseconds = 0;

  // Compute timing
  cudaEventElapsedTime(&milliseconds,start,stop);

  return milliseconds;
}

/* Set up and call ftcs_kernel_shared
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_shared( int step, int block_size_x, int block_size_y ){

  // Edit block sizes to be even numbers
  if ( block_size_x % 2){
    block_size_x++;
  }
  if (block_size_y % 2) {
    block_size_y++;
  }

  // Initalize timing
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Compute thread block size
	dim3 gridBlock(GRID_SIZE[0]/block_size_x,GRID_SIZE[1]/block_size_y);
	dim3 threadBlock(block_size_x,block_size_y);

	// Compute size of shared memory
	int shared_memory_size = (block_size_x+2) * (block_size_y+2) * sizeof(float);

  // Find device temperature arrays
  float* out = temperature_device[(step+1)%2];
  float* in = temperature_device[step%2];

  // Start recording
  cudaEventRecord(start);

	// Compute shared kernel
	ftcs_kernel_shared<<<gridBlock,threadBlock,shared_memory_size>>>(out,in,material_device,step,block_size_x,block_size_y,GRID_SIZE[0],GRID_SIZE[1]);
  cudaEventRecord(stop);

  // Synch device threads
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );

  // Stop timing recording
  cudaEventSynchronize(stop);
  float milliseconds = 0;

  // Compute timing
  cudaEventElapsedTime(&milliseconds,start,stop);

  return milliseconds;
}

/* Set up and call ftcs_kernel_texture
 * should return the execution time of the kernel
 */
float ftcs_solver_gpu_texture( int step, int block_size_x, int block_size_y ){

  // Edit block sizes to be even numbers
  if ( block_size_x % 2){
    block_size_x++;
  }
  if (block_size_y % 2) {
    block_size_y++;
  }

  // Initalize timing
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Compute thread block size
  dim3 gridBlock(GRID_SIZE[0]/block_size_x,GRID_SIZE[1]/block_size_y);
	dim3 threadBlock(block_size_x,block_size_y);

  // Find device memory
  float* out = temperature_device[(step+1)%2];
  float* in = temperature_device[step%2];

  // Create channel for texture memory
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();

  size_t offset;
  //bind texture reference with linear memory
  HANDLE_ERROR(cudaBindTexture2D(&offset,texreference,
    in,desc,GRID_SIZE[1],GRID_SIZE[0],sizeof(float)*GRID_SIZE[0]));

  // Start recording
  cudaEventRecord(start);

	// Compute texture kernel
	ftcs_kernel_texture<<<gridBlock,threadBlock>>>(out,material_device,step,block_size_x,block_size_y,GRID_SIZE[0],GRID_SIZE[1],offset);

  cudaEventRecord(stop);

  // Synch threads
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );

  // Stop recording
  cudaEventSynchronize(stop);

	//Unbind texture reference
	HANDLE_ERROR(cudaUnbindTexture(texreference));

  // Compute timing
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds,start,stop);

  return milliseconds;
}


/* Set up and call external_heat_kernel */
void external_heat_gpu( int step, int block_size_x, int block_size_y ){

	// Compute thread block size
	dim3 gridBlock(GRID_SIZE[0]/block_size_x,GRID_SIZE[1]/block_size_y);
	dim3 threadBlock(block_size_x,block_size_y);

  // Find device temperature array
  float* in = temperature_device[step%2];

  // Compute external heat kernel
	external_heat_kernel<<<gridBlock,threadBlock>>>(in,step, block_size_x, block_size_y,GRID_SIZE[0],GRID_SIZE[1]);

  // Synch threads
  HANDLE_ERROR( cudaDeviceSynchronize() );
  HANDLE_ERROR( cudaPeekAtLastError() );
}

void print_gpu_info(){
  int n_devices;
  cudaGetDeviceCount(&n_devices);
  printf("Number of CUDA devices: %d\n", n_devices);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  printf("CUDA device name: %s\n" , device_prop.name);
  printf("Compute capability: %d.%d\n", device_prop.major, device_prop.minor);
}


int main ( int argc, char **argv ){

    // Parse command line arguments
    int version = 0;
    int block_size_x = 0;
    int block_size_y = 0;
    if(argc != 4){
        printf("Useage: %s <version> <block_size_x> <block_size_y>\n\n<version> can be:\n0: plain\n1: shared memory\n2: texture memory\n", argv[0]);
        exit(0);
    }
    else{
        version = atoi(argv[1]);
        block_size_x = atoi(argv[2]);
        block_size_y = atoi(argv[3]);
    }

    print_gpu_info();

    // Allocate and initialize data on host
    host_allocation();
    init_temp_material();

    // Allocate arrays on device, and transfer inputs
    device_allocation();
    transfer_to_gpu();

    // Main integration loop
    for( int step=0; step<NSTEPS; step += 1 ){

        if( step < CUTOFF ){
            external_heat_gpu ( step, block_size_x, block_size_y );
        }

        float time;
        // Call selected version of ftcs slover
        if(version == 2){
            time = ftcs_solver_gpu_texture( step, block_size_x, block_size_y );
        }
        else if(version == 1){
            time = ftcs_solver_gpu_shared(step, block_size_x, block_size_y);
        }
        else{
            time = ftcs_solver_gpu(step, block_size_x, block_size_y);
        }

        add_time(time);

        if((step % SNAPSHOT) == 0){
            // Transfer output from device, and write to file
            transfer_from_gpu(step);
            write_temp(step);
        }
    }

    print_time_stats();

    exit ( EXIT_SUCCESS );
}


void host_allocation(){
    size_t temperature_size =GRID_SIZE[0]*GRID_SIZE[1];
    temperature = (float*) calloc(temperature_size, sizeof(float));
    size_t material_size = (GRID_SIZE[0])*(GRID_SIZE[1]);
    material = (float*) calloc(material_size, sizeof(float));
}


void init_temp_material(){

    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 10.0;

        }
    }

    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[y * GRID_SIZE[0] + x] = 20.0;
            material[y * GRID_SIZE[0] + x] = MERCURY * (dt/(h*h));
        }
    }

    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = COPPER * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }

    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){
            material[y * GRID_SIZE[0] + x] = TIN * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[y * GRID_SIZE[0] + x] = ALUMINIUM * (dt/(h*h));
            temperature[y * GRID_SIZE[0] + x] = 100.0;
        }
    }
}


void add_time(float time){
    avg_time += time;

    if(time < min_time || min_time < -1.0){
        min_time = time;
    }

    if(time > max_time){
        max_time = time;
    }
}

void print_time_stats(){
    printf("Kernel execution time (min, max, avg): %f %f %f\n", min_time, max_time, avg_time/NSTEPS);
}

/* Save 24 - bits bmp file, buffer must be in bmp format: upside - down
 * Only works for images which dimensions are powers of two
 */
void savebmp(char *name, unsigned char *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
  if (!f) {
    printf("Error writing image to disk.\n");
    return;
  }
  unsigned int size = x * y * 3 + 54;
  unsigned char header[54] = {'B', 'M',
                      size&255,
                      (size >> 8)&255,
                      (size >> 16)&255,
                      size >> 24,
                      0, 0, 0, 0, 54, 0, 0, 0, 40, 0, 0, 0, x&255, x >> 8, 0,
                      0, y&255, y >> 8, 0, 0, 1, 0, 24, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  fwrite(header, 1, 54, f);
  fwrite(buffer, 1, GRID_SIZE[0] * GRID_SIZE[1] * 3, f);
  fclose(f);
}

void fancycolour(unsigned char *p, float temp) {

    if(temp <= 25){
        p[2] = 0;
        p[1] = (unsigned char)((temp/25)*255);
        p[0] = 255;
    }
    else if (temp <= 50){
        p[2] = 0;
        p[1] = 255;
        p[0] = 255 - (unsigned char)(((temp-25)/25) * 255);
    }
    else if (temp <= 75){

        p[2] = (unsigned char)(255* (temp-50)/25);
        p[1] = 255;
        p[0] = 0;
    }
    else{
        p[2] = 255;
        p[1] = 255 -(unsigned char)(255* (temp-75)/25) ;
        p[0] = 0;
    }
}

/* Create nice image from iteration counts. take care to create it upside down (bmp format) */
void output(char* filename){
    unsigned char *buffer = (unsigned char*)calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int j = 0; j < GRID_SIZE[1]; j++) {
        for (int i = 0; i < GRID_SIZE[0]; i++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[j*GRID_SIZE[0] + i]);
      }
    }
    /* write image to disk */
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


void write_temp (int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename );
    printf ( "Snapshot at step %d\n", step );
}
