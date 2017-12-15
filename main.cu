#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <stdlib.h>
#include <fstream>
#include <time.h>
#include <limits>
#include <curand.h>
#include <curand_kernel.h>

using namespace std;
typedef float real;

#define imin(a,b) (a<b?a:b)

const real K_B = 8.6173324e-2;
const real CONST1 = 0.83739840027; //sqrt(M_PI/4/30/K_B)/PLANK;
const real CONST2 = 0.172346648; //2*K_B;
const real CONST3 = 41.36319552; //16*K_B*30;

//Total number of threads which is also equal to total number of
//Monte Carlo trials
const int N = 1024;

//Threads per block and blocks per grid
const int threadsPerBlock = 32;
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);

//Function for reading input from a file
int input(real *Ea, real *gam, real *l_t, real *t_inc, real *jo, int *trial, real *ax, real *ay, real *az);

int input(real *Ea, real *gam, real *l_t, real *t_inc, real *jo, int *trial, real *ax, real *ay, real *az){

    std::ifstream myinput;
    myinput.open ("input.txt");
    myinput>>*Ea>>*gam>>*l_t>>*t_inc>>*jo>>*trial>>*ax>>*ay>>*az;

    return 0;
}

//CUDA Kernel for seeding CUDA library curand
__global__ void setup_kernel(curandState *state){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, tid, 0, &state[tid]);	
}

//CUDA kernel for calculating diffusion
__global__ void k_diffusion(curandState *state, real *final_diff, real e_max,  real life, real x_sp, real y_sp, real z_sp, real J_o, real gamm, real T, real Ea, real const1, real increment){
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	real rand, fin_x, fin_y, fin_z, del_t, e_i, e_j, r_x, r_y, r_z, r_ij, J_ij, e_diff;
	real decay_t, dwell_t, m_rate;
	int signal=1, i, random;
	int index = threadIdx.x;
	
	//Every block has it's own shared variable. Displacement result from
	//every thread in a block is saved in this array
	__shared__ real diffusion[threadsPerBlock];
	
	for (i=0; i<threadsPerBlock; i++){
		diffusion[i] = 0;
	}

    while (tid <N){
	
	curandState localState = state[tid];
	//Setting time and dipacement values equal to zero at begining of trial
	del_t = 0;
	fin_x = 0;
	fin_y = 0;
	fin_z = 0;
    signal = 1;
	//Exciton keeps diffusing as long as it hops to next site (signal>0)
	//and time is less than lifetime of exciton
	while (signal > 0 && life > del_t){
		//Energy of the current site from normal distribution
		e_i = curand_normal(&localState)*e_max;
                if (del_t > 0)
                    e_i = e_j;
		//For loop for 18 neighbors
		for (i=0; i<18; i++){
			//e_j is energy of neighbor
			e_j = curand_normal(&localState)*e_max;
			random = curand_uniform(&localState)*3;
			//For picking a random site
			if (random == 0){
				r_x = 0;
            		}
            		else if(random == 1){
				r_x = x_sp;
			}
			else {
				r_x = -1*x_sp;
			}
			random = curand_uniform(&localState)*3;
			if (rand == 0){
				r_y = 0;
			}       
			else if(random == 1){
				r_y = y_sp;
			}
			else {
				r_y = -1*y_sp;
			}
			random = curand_uniform(&localState)*3;
			if (random == 0){
				r_z = 0;
			}
			else if(random == 1){
				r_z = z_sp;
			}
			else {
				r_z = -1*z_sp;
			}
			r_ij = sqrt((r_x*r_x)+(r_y*r_y)+(r_z*r_z));
			if (r_ij == 0){
				i -= 1;
			}
			else {
				//Checking if hopping is possible
				J_ij = J_o*exp(-2*gamm*r_ij);
				e_diff = e_i-e_j;
                m_rate = (J_ij*J_ij*const1/sqrt(T))*exp(-(Ea/K_B/T)-((e_diff)/(CONST2*T))-((e_diff*e_diff)/(CONST3*T)));
				rand = curand_uniform(&localState);
				
				decay_t = -log(rand)*life;
				dwell_t = -log(rand)/m_rate;
				
				if(dwell_t < decay_t){
					i=18;
					fin_x += r_x;
					fin_y += r_y;
					fin_z += r_z;
					signal = 1;
				}
				else{
					signal = 0;
				}				
			}
			del_t += increment;
		}
	}
	
	//Calculating final displacement
	fin_x = sqrt((fin_x*fin_x)+(fin_y*fin_y)+(fin_z*fin_z));
	fin_x = (fin_x*fin_x)/life;
	diffusion[index] += fin_x;
	
	tid += blockDim.x*gridDim.x;
}
	__syncthreads();
	
	//Summing results from every thread in a block
	i = blockDim.x/2;
	while (i!=0){
		if(index < i )
			diffusion[index] += diffusion[index+i];
			__syncthreads();
			i /= 2;
	}
	
	//One thread in each block stores the result of sum in final_diff[]
	//which will be copied to host memory
	if (index==0)
	final_diff[blockIdx.x] = diffusion[0];
	
}

int main (void){
	
	remove("graph.plot");
	remove("diffusion.plot");
	
	real E_a, gamma, lt, inc, Jo, a_x, a_y, a_z;
	int trials;
	input(&E_a, &gamma, &lt, &inc, &Jo, &trials, &a_x, &a_y, &a_z);
	
	curandState *devStates;
	real T, max, plot_x;
	real *dev_diff, *host_diff, *ploty, *ploty2; 
	int i, j;
	
	//Allocating memory in host (CPU)
	//host_diff = (real*)malloc(blocksPerGrid*sizeof(real));
	cudaHostAlloc((void **)&host_diff, blocksPerGrid*sizeof(real), cudaHostAllocDefault);
	
	ploty = ((real*)malloc(9*sizeof(real)));
	ploty2 = ((real*)malloc(9*sizeof(real)));
	
	//Allocating memory in device (GPU)
	cudaMalloc((void **)&dev_diff, blocksPerGrid*sizeof(real));
	cudaMalloc((void **)&devStates, blocksPerGrid*threadsPerBlock*sizeof(curandState));
	
	//For loop for temperature
	for (T = 500; T>=30; T -=40.0){
		if(T<=170)
		T += 30.0;
		plot_x = 1/T/T;
		max = 15.0;
		
		//For loop for Energetic Disorder
		for (i=0; i<9; i++){
			
			//Calling CUDA kernel
			k_diffusion<<<blocksPerGrid, threadsPerBlock>>>(devStates, dev_diff, max, lt, a_x, a_y, a_z, Jo, gamma, T, E_a, CONST1, inc);
			
			//Copying results from Device memory to Host memory
			cudaMemcpy (host_diff, dev_diff, blocksPerGrid*sizeof(real), cudaMemcpyDeviceToHost);
			//Summing results from every block
			for (j=0; j<blocksPerGrid; j++){
				ploty[i] += host_diff[j];
			}
			//Averaging over number of trials
			ploty[i] /= N;
			ploty2[i] = ploty[i]/(exp(-E_a/K_B/T));
						
			max += 5.0;
			if (max == 45.0)
			max += 5.0;
			else if (max == 55.0)
			max += 15.0;
			else if (max == 75.0)
            max = max + 25;			
		}
		
		//Writing results in output file
		ofstream plot;
		plot.open("diffusion.plot", ios::app);
        plot<<T<<"  "<<plot_x<<"    "<<ploty[0]<<"     "<<ploty[1]<<"     "<<ploty[2]<<"     "<<ploty[3]<<"		"<<ploty[4]<<"		"<<ploty[5]<<"		"<<ploty[6]<<"		"<<ploty[7]<<"		"<<ploty[8]<<"\r\n";
        plot.close();
        ofstream diffusion;
        diffusion.open("graph.plot", ios::app);
        diffusion<<T<<"  "<<plot_x<<"    "<<ploty2[0]<<"     "<<ploty2[1]<<"     "<<ploty2[2]<<"     "<<ploty2[3]<<"		"<<ploty2[4]<<"		"<<ploty2[5]<<"		"<<ploty2[6]<<"		"<<ploty2[7]<<"		"<<ploty2[8]<<"\r\n";
        diffusion.close();
	}
	
	//Free Device memory
	cudaFree (devStates);
	cudaFree (dev_diff);
	cudaFreeHost(host_diff);	

	//Free host memory
	//free (host_diff);
	free (ploty);
	free (ploty2);
	return 0;
}
