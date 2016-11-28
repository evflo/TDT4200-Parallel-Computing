__kernel void mandelbrot( __global int* pixel, float xleft, float ylower, float step,
                         const int MAXITER , const int XSIZE) {
    
    size_t idx = get_global_id(0);
    size_t idy = get_global_id(1);
    
    float c_real, c_imag,z_real, z_imag,temp_imag,temp_real;
    int iter = 0;
    
    c_real = (xleft + step * idx );
    c_imag = (ylower + step * idy );
    
    z_real = c_real;
    z_imag = c_imag;
    
    while( z_real * z_real + z_imag * z_imag < 4 ){
        temp_real = z_real * z_real - z_imag * z_imag + c_real;
        temp_imag = 2 * z_real * z_imag + c_imag;
        z_real = temp_real;
        z_imag = temp_imag;
        
        iter++;
        
        // Break if iteration passes MAXITER
        if(iter == MAXITER){
            break;
        }
    }
    pixel[ idy * XSIZE + idx ] = iter;
}
