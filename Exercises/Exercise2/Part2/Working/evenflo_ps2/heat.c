#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

#include <mpi.h>

// Functions implemented by student
void ftcs_solver ( int step );
void border_exchange ( int step );
void gather_temp( int step );
void scatter_temp();
void scatter_material();
void commit_vector_types ();


// Prototypes for functions found at the end of this file 
void external_heat ( int step );
void write_temp ( int step );
void print_local_temps(int step);
void init_temp_material();
void init_local_temp();

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


// Size of the computational grid - 256x256 square
const int GRID_SIZE[2] = {256 , 256};

// Parameters of the simulation: how many steps, and when to cut off the heat
const int NSTEPS = 10000;
const int CUTOFF = 5000;

// How often to dump state to file (steps).
const int SNAPSHOT = 500;

// Border thickness 
const int BORDER = 1;

/* Arrays for the simulation data */
float
    *material,          // Global material constants, on rank 0
    *temperature,       // Global temperature field, on rank 0
    *local_material,    // Local part of the material constants
    *local_temp[2];    // Local part of the temperature (2 buffers)
float    
    *local_material_scatter,
    *recvRowBuf,
    *recvColBuf;
/* Discretization: 5cm square cells, 2.5ms time intervals */
const float
    h  = 5e-2,
    dt = 2.5e-3;

/* Local state */
int
    size, rank,                     // World size, my rank
    dims[2],                        // Size of the cartesian
    periods[2] = { false, false },  // Periodicity of the cartesian
    coords[2],                      // My coordinates in the cartesian
    north, south, east, west,       // Neighbors in the cartesian
    local_grid_size[2],             // Size of local subdomain
    local_origin[2];                // World coordinates of (0,0) local

// Cartesian communicator
MPI_Comm cart;

MPI_Status status;

// MPI datatypes for gather/scater/border exchange
MPI_Datatype
    border_row, border_col, temp_grid, local_temp_grid, material_grid, local_material_grid;

/* Indexing functions, returns linear index for x and y coordinates, compensating for the border */

// temperature
int ti(int x, int y){
    return y*GRID_SIZE[0] + x;
}

// material
int mi(int x, int y){
    return ((y+(BORDER-1))*(GRID_SIZE[0]+2*(BORDER-1)) + x + (BORDER-1));
}

// local_material
int lmi(int x, int y){
    return ((y+(BORDER-1))*(local_grid_size[0]+2*(BORDER-1)) + x + (BORDER-1));
}

// local_temp
int lti(int x, int y){
    return ((y+BORDER)*(local_grid_size[0]+2*BORDER) + x + BORDER);
}

// Check if position is inside local grid
int inside(int x, int y){
    return ( x >= local_origin[0]) &&
    ( x < (local_origin[0] + local_grid_size[0] ) ) &&
    ( y >= local_origin[1]  ) &&
    ( y < ( local_origin[1] + local_grid_size[1] ) );
}



// Compute ftcs 
void ftcs_solver( int step ){
    
    // Initalize temperature[t] and temperature[t+1]
    float *in = local_temp[(step)%2];
    float *out = local_temp[(step + 1)%2];

    // Loop through local grid and calculate temperature[t+1]
    for( int x = 0; x < local_grid_size[0];x++){
        for (int y = 0; y < local_grid_size[1]; y++)
        {   

            out[lti(x,y)] = in[lti(x,y)] + local_material[lmi(x,y)]*(in[lti(x+1,y)] + in[lti(x-1,y)] + in[lti(x,y+1)] + in[lti(x,y-1)] - 4*in[lti(x,y)]);
        }
    }
}

// Commit MPI vector types
void commit_vector_types ( void ){
   
    // Create and commit data type for border rows as a continous array with size of grid in y-direction
    MPI_Type_contiguous(local_grid_size[0],MPI_FLOAT,&border_row);
    MPI_Type_commit(&border_row);

    // Create and commit data type for border cols as a vector which picks one element for every row
    MPI_Type_vector(local_grid_size[1],1,local_grid_size[0] + 2*BORDER,MPI_FLOAT,&border_col);
    MPI_Type_commit(&border_col);

    // Create and commit data type for scattering temperature grid to local grids
    MPI_Type_vector(local_grid_size[1],local_grid_size[0],GRID_SIZE[0],MPI_FLOAT,&temp_grid);
    MPI_Type_commit(&temp_grid);

    // Create and commit data type for receiving local temperature grids from global
    MPI_Type_vector(local_grid_size[1],local_grid_size[0],local_grid_size[0] + 2*BORDER,MPI_FLOAT,&local_temp_grid);
    MPI_Type_commit(&local_temp_grid);


    // Create and commit data type for scattering material grid to local grids
    MPI_Type_vector(local_grid_size[1],local_grid_size[0],GRID_SIZE[0],MPI_FLOAT,&material_grid);
    MPI_Type_commit(&material_grid);

    // Create and commit data type for receiving local material grids from global
    MPI_Type_vector(local_grid_size[1],local_grid_size[0] + (BORDER - 1),local_grid_size[0] + 2*(BORDER-1),MPI_FLOAT,&local_material_grid);
    MPI_Type_commit(&local_material_grid);

}

// Exchange borders (row and cols) between neighbouring grids
void border_exchange ( int step ){

    // Initalize grid to send from and receive to
    float *in = local_temp[(step)%2];

    // Send north row to north process
    if ( north > -1 ){

        MPI_Send(&in[lti(0,0)],1,border_row,north,0,cart);
    }
    // Receive and send row from south process
    if ( south > -1 ){

        MPI_Recv(&in[lti(0,local_grid_size[1])],1, border_row,south,0,
             cart, &status);

        MPI_Send(&in[lti(0,local_grid_size[1]-1)],1,border_row,
                south, 0, cart);

    }

    // Receive row from north process
    if ( north > -1 ){

        MPI_Recv(&in[lti(0,-1)],1,border_row,north,0,cart,&status);
        
    }
    
    // Send column to west process
    if ( west > -1 ){

        MPI_Send(&in[lti(0,0)],1,border_col,west,0,cart);
    }

    // Receive and send column to east process
    if ( east > -1 ){

        MPI_Recv(&in[lti(local_grid_size[0],0)], 1, border_col,east, 0,
            cart, &status);
        MPI_Send(&in[lti(local_grid_size[0]-1,0)],1,border_col,
            east, 0, cart);

    }
    
    // Receive column from west process
    if ( west > -1 ){

        MPI_Recv(&in[lti(-1,0)],1,border_col,west,0,cart,&status);
        
    }

}

// Gather local temperature grids to the global temperature grid
void gather_temp( int step){

    // Init coordinates and data
    int rankCoords[2];
    float *out = local_temp[(step+1)%2];

    if ( rank == 0){
        
        // Copy from local temp to global temperature for rank 0
        for (int x = 0; x < local_grid_size[0]; ++x){
            for (int y = 0; y < local_grid_size[1]; ++y)
            {
                temperature[ti(x,y)] = out[lti(x,y)];
            }
        }

        // Receive from other processes and set in global temperature grid
        for (int i = 1; i < size; ++i)
        {
            MPI_Cart_coords( cart, i, 2, rankCoords );
            MPI_Recv(&temperature[ti(rankCoords[0]*local_grid_size[0],rankCoords[1]*local_grid_size[1])],1,temp_grid,i,0,cart,&status);

        }

    }else{
        // Send local grid from every process to rank 0 
        MPI_Send(&out[lti(0,0)],1,local_temp_grid,0,0,cart);
    }
    
}

// Distribute global temperature grid to local grids
void scatter_temp(){

    // Initalize coordinates
    int rankCoords[2];

    if ( rank == 0){

        // Copy from global temperature to local temp for rank 0
        for (int x = 0; x < local_grid_size[0]; ++x){
            for (int y = 0; y < local_grid_size[1]; ++y){
                local_temp[0][lti(x,y)] = temperature[ti(x,y)];

            }
        }

        // Send grids from global temperature grid to every other process
        for (int i = 1; i < size; ++i)
        {
            MPI_Cart_coords( cart, i, 2, rankCoords );

            MPI_Send(&temperature[ti(rankCoords[0]*local_grid_size[0],rankCoords[1]*local_grid_size[1])],1,temp_grid,i,0,cart);
        }

    }else{

        // Receive from global temperature to local temperature grid in process
        MPI_Recv(&local_temp[0][lti(0,0)],1,local_temp_grid,0,0,cart,&status);

    }

}

// Distribute global material grid to local grids
void scatter_material(){

    // Initalize coordinates
    int rankCoords[2];

    if ( rank == 0){

        // Copy from global material to local material for rank 0
        for (int x = 0; x < local_grid_size[0]; ++x){
            for (int y = 0; y < local_grid_size[1]; ++y){
                local_material[lmi(x,y)] = material[mi(x,y)];

            }
        }

        // Send grids from global material grid to every other process
        for (int i = 1; i < size; ++i)
        {
            MPI_Cart_coords( cart, i, 2, rankCoords );

            MPI_Send(&material[mi(rankCoords[0]*local_grid_size[0],rankCoords[1]*local_grid_size[1])],1,material_grid,i,0,cart);
        }


    }else{

        // Receive from global material to local material grid in process
        MPI_Recv(&local_material[lmi(0,0)],1,local_material_grid,0,0,cart,&status);

    }

}
    

int main ( int argc, char **argv ){
    
    // Initalize MPI setup
    MPI_Init ( &argc, &argv );
    MPI_Comm_size ( MPI_COMM_WORLD, &size );
    MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
    
    // Create cartesian communication
    MPI_Dims_create( size, 2, dims );
    MPI_Cart_create( MPI_COMM_WORLD, 2, dims, periods, 0, &cart );
    MPI_Cart_coords( cart, rank, 2, coords );

    // Set direction of cartesian grid setup
    MPI_Cart_shift( cart, 1, 1, &north, &south );
    MPI_Cart_shift( cart, 0, 1, &west, &east );

    // Find local sizes
    local_grid_size[0] = GRID_SIZE[0] / dims[0];
    local_grid_size[1] = GRID_SIZE[1] / dims[1];
    
    // Find local origin points
    local_origin[0] = coords[0]*local_grid_size[0];
    local_origin[1] = coords[1]*local_grid_size[1];

    // Commit MPI vector types 
    commit_vector_types ();
    

    if(rank == 0){

        // Initalize global arrays
        size_t temperature_size = GRID_SIZE[0]*GRID_SIZE[1];
        temperature = calloc(temperature_size, sizeof(float));

        size_t material_size = (GRID_SIZE[0]+2*(BORDER-1))*(GRID_SIZE[1]+2*(BORDER-1)); 
        material = calloc(material_size, sizeof(float));
        
        init_temp_material();
    }
    
    // Inialize local arrays
    size_t lsize = (local_grid_size[0]+2*(BORDER-1))*(local_grid_size[1]+2*(BORDER-1));
    local_material = calloc( lsize , sizeof(float) );

    size_t lsize_borders = (local_grid_size[0]+2*BORDER)*(local_grid_size[1]+2*BORDER);
    local_temp[0] = calloc( lsize_borders , sizeof(float) );
    local_temp[1] = calloc( lsize_borders , sizeof(float) );
    
    // Initalize local temperature array
    init_local_temp();
        
    // Distribute material data to all processes
    scatter_material();

    // Distribute temperature data to all processes
    scatter_temp();


    // Main integration loop: NSTEPS iterations, impose external heat
    for( int step=0; step<NSTEPS; step += 1 ){
        

        if( step < CUTOFF ){

            // Impose external heat
            external_heat ( step );

        }

        // Exchange borders with neigbouring processes in cartesian grid
        border_exchange( step );
       
        // Solve ftcs
        ftcs_solver( step );
        
        // Print to screen and generate image
        if((step % SNAPSHOT) == 0){
            gather_temp ( step );
            if(rank == 0){
                write_temp(step);
            }
        }
                         
    }


    // Free allocated memory
    if(rank == 0){
        free (temperature);
        free (material);
    }
    free (local_material);
    free (local_temp[0]);
    free (local_temp[1]);

    // Free allocated MPI data types
    MPI_Type_free (&border_row);
    MPI_Type_free (&border_col);
    MPI_Type_free(&material_grid);
    MPI_Type_free(&local_material_grid);
    MPI_Type_free(&temp_grid);
    MPI_Type_free(&local_temp_grid);

    // Finalize MPI
    MPI_Finalize();

    // Exit function
    exit ( EXIT_SUCCESS );
}


void external_heat( int step ){

    // Imposed temperature from outside
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            if(inside(x,y)){
                local_temp[(step)%2][lti(x-local_origin[0], y-local_origin[1] )] = 100.0;
            }
        }
    }
}


void init_local_temp(void){
    
    for(int x=- BORDER; x<local_grid_size[0] + BORDER; x++ ){
        for(int y= - BORDER; y<local_grid_size[1] + BORDER; y++ ){
            local_temp[1][lti(x,y)] = 10.0;
            local_temp[0][lti(x,y)] = 10.0;
        }
    }
}

void init_temp_material(){
    
    for(int x = -(BORDER-1); x < GRID_SIZE[0] + (BORDER-1); x++){
        for(int y = -(BORDER-1); y < GRID_SIZE[1] +(BORDER-1); y++){
            material[mi(x,y)] = MERCURY * (dt/h*h);
        }
    }
    
    for(int x = 0; x < GRID_SIZE[0]; x++){
        for(int y = 0; y < GRID_SIZE[1]; y++){
            temperature[ti(x,y)] = 20.0;
            material[mi(x,y)] = MERCURY * (dt/h*h);
        }
    }
    
    /* Set up the two blocks of copper and tin */
    for(int x=(5*GRID_SIZE[0]/8); x<(7*GRID_SIZE[0]/8); x++ ){
        for(int y=(GRID_SIZE[1]/8); y<(3*GRID_SIZE[1]/8); y++ ){
            material[mi(x,y)] = COPPER * (dt/(h*h));
            temperature[ti(x,y)] = 60.0;
        }
    }
    
    for(int x=(GRID_SIZE[0]/8); x<(GRID_SIZE[0]/2)-(GRID_SIZE[0]/8); x++ ){
        for(int y=(5*GRID_SIZE[1]/8); y<(7*GRID_SIZE[1]/8); y++ ){
            
            material[mi(x,y)] = TIN * (dt/(h*h));
            temperature[ti(x,y)] = 60.0;
        }
    }

    /* Set up the heating element in the middle */
    for(int x=(GRID_SIZE[0]/4); x<=(3*GRID_SIZE[0]/4); x++){
        for(int y=(GRID_SIZE[1]/2)-(GRID_SIZE[1]/16); y<=(GRID_SIZE[1]/2)+(GRID_SIZE[1]/16); y++){
            material[mi(x,y)] = ALUMINIUM * (dt/(h*h));
            temperature[ti(x,y)] = 100.0;
        }
    }
}

void print_local_temps(int step){
    
    MPI_Barrier(cart);
    for(int i = 0; i < size; i++){
        if(rank == i){
            printf("Rank %d step %d\n", i, step);
            for(int y = -BORDER; y < local_grid_size[1] + BORDER; y++){
                for(int x = -BORDER; x < local_grid_size[0] + BORDER; x++){
                    printf("%5.1f ", local_temp[step%2][lti(x,y)]);
                }
                printf("\n");
            }
            printf ("\n");
        }
        fflush(stdout);
        MPI_Barrier(cart);
    }
}

// Save 24 - bits bmp file, buffer must be in bmp format: upside - down 
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

void savebmpLocal(char *name, unsigned char *buffer, int x, int y) {
  FILE *f = fopen(name, "wb");
    int width = (local_grid_size[0]+2*BORDER);
    int heigth = (local_grid_size[1]+2*BORDER);
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
  fwrite(buffer, 1, width * heigth * 3, f);
  fclose(f);
}

// Given iteration number, set a colour
void fancycolour(unsigned char *p, float temp) {
    float r = (temp/101) * 255;
    
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

// Create nice image from iteration counts. take care to create it upside down (bmp format)
void output(char* filename){
    unsigned char *buffer = calloc(GRID_SIZE[0] * GRID_SIZE[1]* 3, 1);
    for (int i = 0; i < GRID_SIZE[0]; i++) {
      for (int j = 0; j < GRID_SIZE[1]; j++) {
        int p = ((GRID_SIZE[1] - j - 1) * GRID_SIZE[0] + i) * 3;
        fancycolour(buffer + p, temperature[(i + GRID_SIZE[0] * j)]);
      }
    }
    // write image to disk
    savebmp(filename, buffer, GRID_SIZE[0], GRID_SIZE[1]);
    free(buffer);
}


// Write output image to file and print to screen
void write_temp ( int step ){
    char filename[15];
    sprintf ( filename, "data/%.4d.bmp", step/SNAPSHOT );

    output ( filename );
    printf ( "Snapshot at step %d\n", step );
}
